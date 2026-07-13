"""
pipeline/deploy.py
------------------
Stage 5: Artifact promotion and rollback.

Design principles:
- Promotion is a copy-swap, not a move. Previous artifacts remain on disk
  until the next promotion — rollback is a local file operation with no
  network dependency.
- Smoke test polls /health until model_loaded=True or retries are exhausted.
- Rollback is called automatically on smoke test failure — it is a normal
  pipeline branch, not an emergency procedure.

Manual rollback:
  python -m pipeline.deploy --rollback
"""

import logging
import shutil
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ROLLBACK_DIR_NAME = "previous"


def promote_artifacts(
    challenger_dir: Path,
    production_dir: Path,
) -> Path:
    """
    Promote challenger artifacts to production via copy-swap.

    Steps:
      1. Archive current production → previous/ (rollback slot)
      2. Copy challenger → production

    Args:
        challenger_dir: Path to staged challenger artifacts.
        production_dir: Path to current production artifacts.

    Returns:
        Path to the rollback directory (previous/).

    Note: Uses shutil.copytree rather than os.rename so this works
    across filesystem boundaries (e.g. staging on SSD, production on NFS).
    """
    challenger_dir = Path(challenger_dir)
    production_dir = Path(production_dir)
    rollback_dir = production_dir.parent / ROLLBACK_DIR_NAME

    if production_dir.exists():
        if rollback_dir.exists():
            shutil.rmtree(rollback_dir)
            logger.debug("Cleared stale rollback slot at %s", rollback_dir)

        shutil.copytree(production_dir, rollback_dir)
        logger.info("Current production archived \u2192 %s", rollback_dir)

    if production_dir.exists():
        shutil.rmtree(production_dir)

    shutil.copytree(challenger_dir, production_dir)
    logger.info("Challenger promoted \u2192 %s", production_dir)

    return rollback_dir


def rollback(rollback_dir: Path, production_dir: Path) -> None:
    """
    Restore the previous production artifacts from the rollback slot.

    Called automatically when the smoke test fails after promotion.
    Can also be called manually via: python -m pipeline.deploy --rollback

    Args:
        rollback_dir:   Path to the archived previous artifacts.
        production_dir: Path to the current production artifacts.

    Raises:
        RuntimeError if no rollback artifacts exist.
    """
    rollback_dir = Path(rollback_dir)
    production_dir = Path(production_dir)

    if not rollback_dir.exists():
        raise RuntimeError(
            f"No rollback artifacts found at '{rollback_dir}'. "
            "Cannot roll back — inspect the production directory manually."
        )

    if production_dir.exists():
        shutil.rmtree(production_dir)

    shutil.copytree(rollback_dir, production_dir)
    logger.info(
        "Rolled back: restored '%s' from '%s'",
        production_dir, rollback_dir,
    )


def smoke_test(
    health_url: str,
    max_retries: int = 10,
    delay_s: float = 3.0,
    timeout_s: float = 5.0,
) -> bool:
    """
    Poll the /health endpoint until the service confirms model_loaded=True.

    Args:
        health_url:  URL of the health check endpoint.
        max_retries: Maximum number of polling attempts.
        delay_s:     Seconds to wait between attempts.
        timeout_s:   Per-request timeout in seconds.

    Returns:
        True if the service is healthy within the retry budget.
        False otherwise (pipeline will invoke rollback).
    """
    logger.info(
        "Running smoke test: %s  (max %d retries, %.0fs delay)",
        health_url, max_retries, delay_s,
    )

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(health_url, timeout=timeout_s)
            body = resp.json()

            if resp.status_code == 200 and body.get("model_loaded"):
                logger.info(
                    "Smoke test passed on attempt %d/%d \u2713",
                    attempt, max_retries,
                )
                return True

            logger.warning(
                "Smoke test attempt %d/%d: status=%d, model_loaded=%s",
                attempt, max_retries,
                resp.status_code, body.get("model_loaded"),
            )

        except requests.RequestException as exc:
            logger.warning(
                "Smoke test attempt %d/%d: request failed \u2014 %s",
                attempt, max_retries, exc,
            )

        if attempt < max_retries:
            time.sleep(delay_s)

    logger.error(
        "Smoke test FAILED after %d attempts. Triggering rollback.",
        max_retries,
    )
    return False


# ── Manual rollback entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")

    if "--rollback" in sys.argv:
        from pathlib import Path as _Path
        _BASE = _Path(__file__).parent.parent
        _prod = _BASE / "model" / "artifacts"
        _prev = _prod.parent / ROLLBACK_DIR_NAME

        try:
            rollback(_prev, _prod)
            print(f"Rollback complete. Production restored from {_prev}")
        except RuntimeError as e:
            print(f"Rollback failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python -m pipeline.deploy --rollback")
