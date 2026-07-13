"""
pipeline/scheduled_flow.py
--------------------------
Prefect flow wrapper for the retraining pipeline.

Wraps run_pipeline() in a Prefect @flow so you get:
  - Scheduled execution (cron or interval)
  - Full run history with status and metadata
  - Failure alerting via Prefect Cloud notifications
  - Retry logic on transient failures
  - Concurrency limits to prevent overlapping runs

Usage (local):
  python pipeline/scheduled_flow.py

Deploy to Prefect Cloud:
  prefect deploy pipeline/scheduled_flow.py:retraining_flow \
    --name ml-retraining \
    --cron "0 2 * * *"   # daily at 02:00 UTC
"""
import sys
try:
    from prefect import flow, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

from src.pipeline.orchestrator import run_pipeline


if PREFECT_AVAILABLE:
    @flow(
        name="ml-retraining",
        description="Drift-aware ML retraining pipeline with champion/challenger gate",
        log_prints=True,
        retries=1,
        retry_delay_seconds=60,
    )
    def retraining_flow(
        force: bool = False,
        days_since_training: int = 45,
        current_recall: float = 0.58,
        current_cost_per_1k: float = 16_000.0,
        feature_psi: dict = None,
    ) -> dict:
        """
        Prefect-wrapped retraining pipeline.

        Raises RuntimeError on failure so Prefect marks the run as FAILED
        and sends configured notifications.
        """
        logger = get_run_logger()
        logger.info("Starting retraining flow (force=%s)", force)

        result = run_pipeline(
            force=force,
            days_since_training=days_since_training,
            current_recall=current_recall,
            current_cost_per_1k=current_cost_per_1k,
            feature_psi=feature_psi,
        )

        if result["status"] not in ("deployed", "skipped"):
            raise RuntimeError(
                f"Pipeline {result['status']} at stage "
                f"'{result.get('stage', 'unknown')}': "
                f"{result.get('error', 'no detail')}"
            )

        logger.info("Flow completed: status=%s", result["status"])
        return result

else:
    # Fallback if Prefect is not installed — runs directly
    def retraining_flow(force: bool = False, **kwargs) -> dict:
        """Fallback: run the pipeline directly (Prefect not installed)."""
        print("Note: Prefect not installed. Running pipeline directly.")
        print("Install with: pip install prefect")
        return run_pipeline(force=force, **kwargs)


if __name__ == "__main__":
    force = "--force" in sys.argv
    result = retraining_flow(force=force)
    print(result)
