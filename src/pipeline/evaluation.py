"""
pipeline/evaluation.py
----------------------
Stage 4: Champion/challenger evaluation gate.

This is the stage that prevents regressions from reaching production.
The challenger must clear two bars:
  1. Absolute minimums (recall, ROC-AUC) — always required
  2. No significant regression vs the current champion — required when a
     champion exists

First deployment gets Rule 1 only (relaxed gate).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.pipeline.train import evaluate_model

logger = logging.getLogger(__name__)


class EvaluationGateError(Exception):
    """
    Raised when the challenger fails the evaluation gate.
    Pipeline halts; artifacts are NOT promoted.
    """
    pass


def load_champion_metrics(champion_artifact_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load the current champion's recorded validation metrics.

    Returns None if no champion run log exists (first deployment).
    """
    run_log_path = Path(champion_artifact_dir) / "run_log.json"
    if not run_log_path.exists():
        logger.info(
            "No champion run log at '%s'. "
            "Treating as first deployment — absolute minimums only.",
            run_log_path,
        )
        return None

    with open(run_log_path) as f:
        record = json.load(f)

    metrics = record.get("val_metrics")
    if metrics:
        logger.info(
            "Champion metrics loaded: ROC-AUC=%.4f, recall=%.4f, cost/1k=$%.0f",
            metrics.get("roc_auc", 0),
            metrics.get("recall", 0),
            metrics.get("cost_per_1k", 0),
        )
    return metrics


def run_evaluation_gate(
    challenger_model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    champion_artifact_dir: Path,
    min_recall: float = 0.60,
    min_roc_auc: float = 0.7,
    max_regression_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluate the challenger against absolute minimums and the champion.

    Args:
        challenger_model:      Trained model object (must have predict_proba).
        X_eval, y_eval:        Held-out evaluation set (preprocessed).
        champion_artifact_dir: Path to the current production model artifacts.
        min_recall:            Minimum acceptable recall (default 0.60).
        min_roc_auc:           Minimum acceptable ROC-AUC (default 0.80).
        max_regression_pct:    Max allowed ROC-AUC drop vs champion (default 5%).

    Returns:
        challenger_metrics dict if all checks pass.

    Raises:
        EvaluationGateError listing ALL failures if any check fails.
    """
    challenger_metrics = evaluate_model(challenger_model, X_eval, y_eval)
    champion_metrics = load_champion_metrics(champion_artifact_dir)
    failures = []

    # ── Rule 1: Absolute minimums ─────────────────────────────────
    if challenger_metrics["recall"] < min_recall:
        failures.append(
            f"Recall {challenger_metrics['recall']:.4f} is below "
            f"minimum {min_recall} — model is missing too many positives."
        )

    if challenger_metrics["roc_auc"] < min_roc_auc:
        failures.append(
            f"ROC-AUC {challenger_metrics['roc_auc']:.4f} is below "
            f"minimum {min_roc_auc} — model has insufficient discriminative power."
        )

    # ── Rule 2: No significant regression vs champion ─────────────
    if champion_metrics is not None:
        champ_roc = champion_metrics["roc_auc"]
        chal_roc = challenger_metrics["roc_auc"]
        regression = (champ_roc - chal_roc) / champ_roc

        logger.info(
            "Champion vs Challenger — ROC-AUC: %.4f → %.4f (%+.2f%%)",
            champ_roc, chal_roc,
            (chal_roc - champ_roc) / champ_roc * 100,
        )
        logger.info(
            "Champion vs Challenger — recall:   %.4f → %.4f",
            champion_metrics.get("recall", 0),
            challenger_metrics["recall"],
        )
        logger.info(
            "Champion vs Challenger — cost/1k:  $%.0f → $%.0f",
            champion_metrics.get("cost_per_1k", 0),
            challenger_metrics["cost_per_1k"],
        )

        if regression > max_regression_pct:
            failures.append(
                f"ROC-AUC regression of {regression:.1%} exceeds allowed "
                f"{max_regression_pct:.1%} "
                f"(champion={champ_roc:.4f}, challenger={chal_roc:.4f}). "
                "Investigate data quality or training configuration."
            )

    # ── Raise with all failures listed ────────────────────────────
    if failures:
        raise EvaluationGateError(
            f"Challenger failed evaluation gate ({len(failures)} check(s)):\n" +
            "\n".join(f"  \u2717 {f}" for f in failures)
        )

    logger.info(
        "Evaluation gate PASSED \u2713 — "
        "recall=%.4f, ROC-AUC=%.4f, cost/1k=$%.0f",
        challenger_metrics["recall"],
        challenger_metrics["roc_auc"],
        challenger_metrics["cost_per_1k"],
    )

    return challenger_metrics
