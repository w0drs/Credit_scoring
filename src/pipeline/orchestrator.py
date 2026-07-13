"""
pipeline/orchestrator.py
------------------------
Full retraining pipeline orchestrator.

Wires together all five stages:
  Stage 1 → Data Collection & Validation
  Stage 2 → Feature Engineering
  Stage 3 → Training with Experiment Tracking
  Stage 4 → Evaluation Gate (Champion/Challenger)
  Stage 5 → Deployment with Smoke Test & Rollback

Usage:
  python -m pipeline.orchestrator            # check triggers, retrain if needed
  python -m pipeline.orchestrator --force    # skip trigger check, always retrain
  python -m pipeline.orchestrator --mode static --data-path data/dataset.parquet --force  # train on static dataset

Returns a structured status dict so callers (Prefect, Airflow, CLI) can
interpret the outcome without parsing log output.

Status values:
  "deployed"    — new model is live
  "skipped"     — triggers evaluated, no retraining needed
  "failed"      — a stage raised an unexpected error
  "rejected"    — challenger failed the evaluation gate
  "rolled_back" — smoke test failed after promotion; previous model restored
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.pipeline.data import DataConfig, DataValidationError, load_data, validate_data
from src.pipeline.deploy import promote_artifacts, rollback, smoke_test
from src.pipeline.evaluation import EvaluationGateError, run_evaluation_gate
from src.pipeline.features import FEATURE_COLS, fit_and_save_preprocessor
from src.pipeline.train import run_training
from src.pipeline.triggers import TriggerConfig, should_retrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Path configuration ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PRODUCTION_DIR = BASE_DIR / "model" / "artifacts"
STAGING_DIR = BASE_DIR / "model" / "staging"
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")


def load_feature_psi_from_monitoring():
    """Загружает PSI из файла мониторинга"""
    drift_result_path = BASE_DIR / "data" / "drift_result.json"

    if not drift_result_path.exists():
        logger.debug("Нет файла с дрифтом, используем заглушку")
        return None

    try:
        with open(drift_result_path) as f:
            drift_result = json.load(f)
        return drift_result.get("feature_psi", {})
    except Exception as e:
        logger.warning(f"Ошибка загрузки PSI: {e}")
        return None


def run_pipeline(
    force: bool = False,
    mode: str = 'streaming',
    static_data_path: Optional[Path] = None,
    days_since_training: int = 45,
    current_recall: float = 0.58,
    current_cost_per_1k: float = 16_000.0,
    feature_psi: dict = None,
) -> dict:
    """
    Execute the full retraining pipeline.

    Args:
        force:                Skip trigger check, retrain unconditionally.
        mode:                 'streaming' for incremental data or 'static' for single dataset.
        static_data_path:     Path to static dataset file (required when mode='static').
        days_since_training:  Pull from your model registry in production.
        current_recall:       Pull from your monitoring system in production.
        current_cost_per_1k:  Pull from your monitoring system in production.
        feature_psi:          {feature_name: psi_value} from monitoring.

    Returns:
        Structured status dict — see module docstring for status values.
    """
    run_id = str(uuid.uuid4())[:8]

    if feature_psi is None:
        feature_psi = load_feature_psi_from_monitoring()
        if feature_psi is None:
            feature_psi = {"income": 0.31, "SNA": 0.12}

    logger.info("=" * 60)
    logger.info("Retraining pipeline started | run_id=%s | mode=%s", run_id, mode)
    logger.info("=" * 60)

    # ── Trigger check ──────────────────────────────────────────────
    # Only evaluate triggers for streaming mode (static uses force=True implicitly)
    if mode == 'streaming' and not force:
        logger.info("[Triggers] Evaluating retraining triggers")

        rng = np.random.default_rng(42)
        ref_scores = rng.beta(2, 5, 200)
        cur_scores = rng.beta(3, 4, 200)  # shifted — replace with real score logs

        needs_retrain, reason = should_retrain(
            days_since_training=days_since_training,
            current_recall=current_recall,
            current_cost_per_1k=current_cost_per_1k,
            feature_psi=feature_psi,
            score_distribution=(ref_scores, cur_scores),
            config=TriggerConfig(),
            mode=mode,  # Pass mode to triggers
        )

        if not needs_retrain:
            logger.info("[Triggers] No retraining needed: %s", reason)
            return {"status": "skipped", "reason": reason, "mode": mode}

        logger.info("[Triggers] Retraining triggered: %s", reason)
    elif mode == 'static':
        logger.info("[Triggers] Static mode — skipping trigger evaluation")

    # ── Stage 1: Data ──────────────────────────────────────────────
    logger.info("[Stage 1/5] Data collection and validation (mode=%s)", mode)

    # Build DataConfig based on mode
    if mode == 'static':
        if static_data_path is None:
            raise ValueError(
                "static_data_path must be provided when mode='static'"
            )
        data_config = DataConfig(
            mode='static',
            data_path=static_data_path,
            reference_path=PRODUCTION_DIR / "reference_data.parquet",  # Not used in static mode but required by dataclass
            min_samples=500,
            required_features=FEATURE_COLS + ["label"],
        )
        logger.info("Static dataset path: %s", static_data_path)
    else:  # streaming mode
        data_config = DataConfig(
            mode='streaming',
            data_dir=DATA_DIR,
            reference_path=PRODUCTION_DIR / "reference_data.parquet",  # Used for drift checks in streaming mode
            lookback_days=90,
            min_samples=500,
            required_features=FEATURE_COLS + ["label"],
        )
        logger.info("Streaming data directory: %s (lookback=%d days)", DATA_DIR, 90)

    try:
        df = load_data(data_config)
        validate_data(df, data_config)
    except DataValidationError as exc:
        logger.error("[Stage 1] FAILED: %s", exc)
        return {
            "status": "failed",
            "stage": "data_validation",
            "error": str(exc),
            "run_id": run_id,
            "mode": mode,
        }

    # ── Stage 2: Features ──────────────────────────────────────────
    logger.info("[Stage 2/5] Feature engineering")

    try:
        _, X_transformed, y = fit_and_save_preprocessor(df, STAGING_DIR)
    except Exception as exc:
        logger.error("[Stage 2] FAILED: %s", exc)
        return {
            "status": "failed",
            "stage": "feature_engineering",
            "error": str(exc),
            "run_id": run_id,
            "mode": mode,
        }

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    logger.info(
        "Train: %d | Val: %d | Positive rate: %.2f%%",
        len(X_train), len(X_val), y_val.mean() * 100,
    )

    # ── Stage 3: Training ──────────────────────────────────────────
    logger.info("[Stage 3/5] Training candidates")

    git_commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
    run_metadata = {
        "run_id": run_id,
        "git_commit": git_commit or "unknown",
        "data_window": f"last_{data_config.lookback_days if mode == 'streaming' else 'static'}_days",
        "triggered_at": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
    }

    try:
        run_record = run_training(
            X_train, y_train, X_val, y_val,
            STAGING_DIR, run_metadata,
        )
    except Exception as exc:
        logger.error("[Stage 3] FAILED: %s", exc)
        return {
            "status": "failed",
            "stage": "training",
            "error": str(exc),
            "run_id": run_id,
            "mode": mode,
        }

    # ── Stage 4: Evaluation Gate ───────────────────────────────────
    logger.info("[Stage 4/5] Evaluation gate (champion/challenger)")

    challenger_model = joblib.load(STAGING_DIR / "model.pkl")

    try:
        challenger_metrics = run_evaluation_gate(
            challenger_model,
            X_val, y_val,
            champion_artifact_dir=PRODUCTION_DIR,
        )
    except EvaluationGateError as exc:
        logger.error("[Stage 4] Challenger REJECTED: %s", exc)
        return {
            "status": "rejected",
            "stage": "evaluation_gate",
            "error": str(exc),
            "run_id": run_id,
            "mode": mode,
            "challenger_metrics": run_record.get("val_metrics"),
        }

    # ── Stage 5: Deploy ────────────────────────────────────────────
    logger.info("[Stage 5/5] Promoting artifacts")

    rollback_dir = promote_artifacts(STAGING_DIR, PRODUCTION_DIR)

    # In production: trigger a container restart or call a /reload endpoint here.
    # The smoke test confirms the service picked up the new artifacts.
    time.sleep(1)  # brief pause for service startup in local dev

    if not smoke_test(HEALTH_URL):
        logger.error("[Stage 5] Smoke test failed — rolling back to previous model")
        rollback(rollback_dir, PRODUCTION_DIR)
        return {
            "status": "rolled_back",
            "stage": "smoke_test",
            "error": "Smoke test failed after promotion. Previous model restored.",
            "run_id": run_id,
            "mode": mode,
        }

    logger.info("=" * 60)
    logger.info("Pipeline COMPLETED \u2713 | run_id=%s | mode=%s", run_id, mode)
    logger.info(
        "Deployed: recall=%.4f | ROC-AUC=%.4f | cost/1k=$%.0f",
        challenger_metrics["recall"],
        challenger_metrics["roc_auc"],
        challenger_metrics["cost_per_1k"],
    )
    logger.info("=" * 60)

    return {
        "status": "deployed",
        "run_id": run_id,
        "mode": mode,
        "git_commit": git_commit or "unknown",
        "metrics": challenger_metrics,
        "data_window": run_metadata["data_window"],
    }


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    mode = "streaming"
    static_data_path = None

    # Parse --mode argument
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
            if mode not in ["streaming", "static"]:
                print(f"Error: mode must be 'streaming' or 'static', got '{mode}'")
                sys.exit(1)

    # Parse --data-path argument (required for static mode)
    if "--data-path" in sys.argv:
        idx = sys.argv.index("--data-path")
        if idx + 1 < len(sys.argv):
            static_data_path = Path(sys.argv[idx + 1])

    if mode == "static" and static_data_path is None:
        print("Error: --data-path required for static mode")
        print("Usage: python -m pipeline.orchestrator --mode static --data-path data/dataset.parquet [--force]")
        sys.exit(1)

    result = run_pipeline(
        force=force,
        mode=mode,
        static_data_path=static_data_path,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["status"] in ("deployed", "skipped") else 1)