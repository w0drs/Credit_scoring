"""
pipeline/train.py
-----------------
Stage 3: Multi-candidate training with full provenance logging.

Design principles:
- Every run writes a complete run_log.json: data window, git commit,
  all candidate configs, CV results, and the winner's full metrics.
- The winner is chosen by validation ROC-AUC; the full candidate
  comparison is preserved for post-hoc analysis.
- run_record is compatible with MLflow and Weights & Biases logging APIs.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# Candidate model configurations — extend this list to add new architectures.
CANDIDATE_CONFIGS: List[Dict[str, Any]] = [
    {
        "model_class": "GradientBoostingClassifier",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "random_state": 42,
        },
    },
    {
        "model_class": "RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    {
        "model_class": "LGBMClassifier",
        "params": {
            'colsample_bytree': 0.6525305504945635,
            'learning_rate': 0.047127077338180494,
            'max_depth': 3,
            'min_child_samples': 15,
            'n_estimators': 406,
            'num_leaves': 40,
            'reg_alpha': 0.0014341754109660826,
            'reg_lambda': 0.5210082757225214,
            'scale_pos_weight': 11.135712739455382,
            'subsample': 0.8230729607127126,
            'subsample_freq': 2
        },
    },
]

_MODEL_REGISTRY = {
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "LGBMClassifier": lgb.LGBMClassifier
}


def _get_model(config: Dict[str, Any]):
    cls = _MODEL_REGISTRY.get(config["model_class"])
    if cls is None:
        raise ValueError(
            f"Unknown model class '{config['model_class']}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    return cls(**config["params"])


def train_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: Dict[str, Any],
    n_splits: int = 5,
) -> Tuple[Any, np.ndarray, float]:
    """
    Train a single model config with stratified k-fold cross-validation.

    Returns:
        (fitted_model, cv_scores_array, train_duration_seconds)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # CV uses a fresh instance; the final model is fit on the full training set.
    cv_scores = cross_val_score(
        _get_model(model_config),
        X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    logger.info(
        "[%s] CV ROC-AUC: %.4f ± %.4f",
        model_config["model_class"],
        cv_scores.mean(),
        cv_scores.std(),
    )

    model = _get_model(model_config)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    duration = round(time.perf_counter() - t0, 2)

    return model, cv_scores, duration


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.75,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
) -> Dict[str, Any]:
    """
    Full evaluation suite: recall, F1, ROC-AUC, avg precision, cost/1k.

    cost_fp / cost_fn model the asymmetric business cost of false positives
    vs false negatives (e.g. fraud review cost vs fraud loss).
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    cost_per_1k = (fp * cost_fp + fn * cost_fn) / len(y) * 1000

    return {
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_proba)), 4),
        "avg_precision": round(float(average_precision_score(y, y_proba)), 4),
        "cost_per_1k": round(cost_per_1k, 2),
        "threshold": threshold,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "fp": fp,
        "fn": fn,
    }


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    artifact_dir: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train all candidate configs, select the winner by validation ROC-AUC,
    save the winning model artifact, and write a complete run log.

    Args:
        X_train, y_train: Training split (preprocessed).
        X_val, y_val:     Validation split (preprocessed).
        artifact_dir:     Where to save model.pkl and run_log.json.
        run_metadata:     Dict with run_id, git_commit, data_window, etc.

    Returns:
        run_record dict (fed directly into Stage 4 evaluation gate).
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for config in CANDIDATE_CONFIGS:
        logger.info("Training candidate: %s", config["model_class"])
        model, cv_scores, duration = train_with_cv(X_train, y_train, config)
        val_metrics = evaluate_model(model, X_val, y_val)

        results.append({
            "config": config,
            "model": model,
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "val_metrics": val_metrics,
            "train_duration_s": duration,
        })

    # Select winner by validation ROC-AUC
    winner = max(results, key=lambda r: r["val_metrics"]["roc_auc"])
    logger.info(
        "Winner: %s  (val ROC-AUC=%.4f, recall=%.4f, cost/1k=$%.0f)",
        winner["config"]["model_class"],
        winner["val_metrics"]["roc_auc"],
        winner["val_metrics"]["recall"],
        winner["val_metrics"]["cost_per_1k"],
    )

    # Persist winning model
    model_path = artifact_dir / "model.pkl"
    joblib.dump(winner["model"], model_path)
    logger.info("Model saved → %s", model_path)

    # Write complete run log (experiment tracking record)
    run_record = {
        "run_id": run_metadata.get("run_id", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_window": run_metadata.get("data_window"),
        "git_commit": run_metadata.get("git_commit", "unknown"),
        "triggered_at": run_metadata.get("triggered_at"),
        "winner_config": winner["config"],
        "cv_roc_auc_mean": winner["cv_roc_auc_mean"],
        "cv_roc_auc_std": winner["cv_roc_auc_std"],
        "val_metrics": winner["val_metrics"],
        "all_candidates": [
            {
                "config": r["config"],
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "val_roc_auc": r["val_metrics"]["roc_auc"],
                "val_recall": r["val_metrics"]["recall"],
                "val_cost_per_1k": r["val_metrics"]["cost_per_1k"],
                "train_duration_s": r["train_duration_s"],
            }
            for r in results
        ],
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
    }

    run_log_path = artifact_dir / "run_log.json"
    with open(run_log_path, "w") as f:
        json.dump(run_record, f, indent=2)
    logger.info("Run log saved → %s", run_log_path)

    return run_record
