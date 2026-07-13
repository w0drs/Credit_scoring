"""
pipeline/triggers.py
--------------------
Retraining trigger logic: drift-based (leading), performance-based (lagging),
and time-based (failsafe).

Trigger priority:
  1. Drift-based  — catches degradation before metrics move
  2. Performance  — catches degradation that drift missed
  3. Time-based   — guarantees minimum refresh cadence
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class TriggerConfig:
    # Time-based failsafe
    max_days_since_training: int = 30

    # Performance-based (lagging indicators)
    min_recall: float = 0.20
    max_cost_per_1k: float = 15_000.0

    # Drift-based (leading indicators)
    psi_action_threshold: float = 0.25
    ks_action_threshold: float = 0.20


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI < 0.10  → STABLE
    0.10–0.25   → WARNING
    > 0.25      → ACTION (retrain)

    Uses additive smoothing (1e-9) to avoid log(0) on empty bins.
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    ref_pct = (ref_counts + 1e-9) / len(reference)
    cur_pct = (cur_counts + 1e-9) / len(current)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def should_retrain(
    days_since_training: int,
    current_recall: float,
    current_cost_per_1k: float,
    feature_psi: dict,           # {feature_name: psi_value}
    score_distribution: tuple,   # (reference_scores, current_scores)
    config: TriggerConfig,
    mode: str = 'streaming'
) -> tuple:
    """
    Evaluate all triggers in priority order.

    Returns:
        (should_retrain: bool, reason: str)
    """
    # ── 1. Drift-based (только для streaming) ──────────────────────
    if mode == 'streaming':
        flagged = {f: v for f, v in feature_psi.items() if v > config.psi_action_threshold}
        if flagged:
            worst = max(flagged, key=flagged.get)
            return True, f"PSI ACTION on '{worst}' (PSI={flagged[worst]:.3f})"

        ref_scores, cur_scores = score_distribution
        ks_stat, _ = ks_2samp(ref_scores, cur_scores)
        if ks_stat > config.ks_action_threshold:
            return True, f"Score KS statistic elevated ({ks_stat:.3f} > {config.ks_action_threshold})"

    # ── 2. Performance-based (для обоих режимов) ──────────────────
    if current_recall < config.min_recall:
        return True, f"Recall {current_recall:.3f} below minimum {config.min_recall}"

    if current_cost_per_1k > config.max_cost_per_1k:
        return True, (
            f"Cost ${current_cost_per_1k:,.0f}/1k exceeds "
            f"threshold ${config.max_cost_per_1k:,.0f}"
        )

    # ── 3. Time-based failsafe (только для streaming) ─────────────
    if mode == 'streaming' and days_since_training > config.max_days_since_training:
        return True, (
            f"Model age ({days_since_training}d) exceeds "
            f"maximum ({config.max_days_since_training}d)"
        )

    return False, "All checks passed — no retraining needed"
