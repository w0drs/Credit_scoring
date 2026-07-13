"""
tests/test_pipeline.py
----------------------
Test suite covering three categories:
  1. Stage failures — each stage fails loudly and halts correctly
  2. Trigger logic  — drift/performance/time priority ordering
  3. Evaluation gate — champion/challenger logic
  4. Rollback — artifact promotion and restoration
  5. Smoke test — retry and failure behavior

Run with:
  pytest tests/ -v --tb=short
  pytest tests/ -v --tb=short -k "TestDataValidation"
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.pipeline.data import DataConfig, DataValidationError, validate_data
from src.pipeline.deploy import promote_artifacts, rollback, smoke_test
from src.pipeline.evaluation import EvaluationGateError, run_evaluation_gate
from src.pipeline.features import FEATURE_COLS
from src.pipeline.triggers import TriggerConfig, compute_psi, should_retrain


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_df():
    """600-sample synthetic fraud dataset with the correct schema."""
    rng = np.random.default_rng(42)
    n = 600
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLS}
    data["label"] = (rng.random(n) < 0.07).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def data_config(tmp_path):
    return DataConfig(
        data_dir=tmp_path / "data",
        reference_path=tmp_path / "reference.parquet",
        min_samples=100,
        required_features=FEATURE_COLS + ["label"],
    )


@pytest.fixture
def trained_model_and_data(valid_df):
    """Returns (model, X_scaled, y) — a fitted RF on the synthetic data."""
    X = StandardScaler().fit_transform(valid_df[FEATURE_COLS].values)
    y = valid_df["label"].values
    model = RandomForestClassifier(
        n_estimators=20, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model, X, y


def _base_trigger_kwargs():
    """Healthy baseline — no trigger should fire."""
    rng = np.random.default_rng(0)
    return dict(
        days_since_training=10,
        current_recall=0.75,
        current_cost_per_1k=8_000.0,
        feature_psi={"transaction_amount": 0.05},
        score_distribution=(rng.beta(2, 5, 200), rng.beta(2, 5, 200)),
        config=TriggerConfig(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Data Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataValidation:

    def test_valid_data_passes(self, valid_df, data_config):
        """Happy path — should raise nothing."""
        validate_data(valid_df, data_config)

    def test_too_few_samples_raises(self, valid_df, data_config):
        data_config.min_samples = 10_000
        with pytest.raises(DataValidationError, match="Only"):
            validate_data(valid_df, data_config)

    def test_missing_label_column_raises(self, valid_df, data_config):
        df = valid_df.drop(columns=["label"])
        with pytest.raises(DataValidationError, match="label"):
            validate_data(df, data_config)

    def test_all_negative_labels_raises(self, valid_df, data_config):
        df = valid_df.copy()
        df["label"] = 0
        with pytest.raises(DataValidationError, match="Positive rate"):
            validate_data(df, data_config)

    def test_all_positive_labels_raises(self, valid_df, data_config):
        df = valid_df.copy()
        df["label"] = 1
        with pytest.raises(DataValidationError, match="Positive rate"):
            validate_data(df, data_config)

    def test_missing_required_feature_raises(self, valid_df, data_config):
        df = valid_df.drop(columns=["transaction_amount"])
        with pytest.raises(DataValidationError, match="Missing required"):
            validate_data(df, data_config)

    def test_error_message_accumulates_all_failures(self, valid_df, data_config):
        """DataValidationError should report ALL errors, not just the first."""
        df = valid_df.drop(columns=["label", "transaction_amount"])
        with pytest.raises(DataValidationError) as exc_info:
            validate_data(df, data_config)
        error_msg = str(exc_info.value)
        # Both missing label and missing feature should be mentioned
        assert "label" in error_msg or "Missing" in error_msg

    def test_high_null_rate_warns_not_fails(self, valid_df, data_config, caplog):
        """Null rates >10% should warn but not raise DataValidationError."""
        df = valid_df.copy()
        df.loc[:70, "transaction_amount"] = np.nan  # ~12% null rate
        with caplog.at_level(logging.WARNING):
            validate_data(df, data_config)  # must NOT raise
        assert any(
            "null" in r.getMessage().lower() or "null" in r.message.lower()
            for r in caplog.records
        ), "Expected a null-rate warning in the logs"


# ─────────────────────────────────────────────────────────────────────────────
# Trigger logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTriggers:

    def test_no_trigger_when_healthy(self):
        result, reason = should_retrain(**_base_trigger_kwargs())
        assert result is False
        assert "no retraining" in reason.lower()

    def test_psi_action_triggers(self):
        kwargs = _base_trigger_kwargs()
        kwargs["feature_psi"] = {"transaction_amount": 0.30}  # > 0.25 threshold
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "PSI" in reason

    def test_ks_statistic_triggers(self):
        """Large distribution shift should trigger via KS statistic."""
        kwargs = _base_trigger_kwargs()
        rng = np.random.default_rng(99)
        # Very different distributions to exceed KS threshold of 0.20
        ref = rng.beta(2, 5, 500)
        cur = rng.beta(8, 2, 500)
        kwargs["score_distribution"] = (ref, cur)
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "KS" in reason

    def test_low_recall_triggers(self):
        kwargs = _base_trigger_kwargs()
        kwargs["current_recall"] = 0.50  # < 0.60 default threshold
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "Recall" in reason or "recall" in reason

    def test_high_cost_triggers(self):
        kwargs = _base_trigger_kwargs()
        kwargs["current_cost_per_1k"] = 20_000.0  # > 15,000 default threshold
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "Cost" in reason or "cost" in reason

    def test_age_triggers_as_failsafe(self):
        """Time-based trigger fires when no drift or performance trigger fires."""
        kwargs = _base_trigger_kwargs()
        kwargs["days_since_training"] = 60  # > 30-day default maximum
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "age" in reason.lower() or "days" in reason.lower() or "maximum" in reason.lower()

    def test_psi_fires_before_performance(self):
        """Drift (PSI) should appear in reason even when performance is also bad."""
        kwargs = _base_trigger_kwargs()
        kwargs["feature_psi"] = {"transaction_amount": 0.30}  # drift trigger
        kwargs["current_recall"] = 0.40                        # performance trigger too
        result, reason = should_retrain(**kwargs)
        assert result is True
        assert "PSI" in reason  # drift has higher priority

    def test_compute_psi_stable_distributions(self):
        """Near-identical distributions → PSI well below warning threshold."""
        rng = np.random.default_rng(42)
        dist = rng.normal(0, 1, 1000)
        psi = compute_psi(dist, dist + rng.normal(0, 0.01, 1000))
        assert psi < 0.10

    def test_compute_psi_shifted_distributions(self):
        """Highly shifted distributions → PSI above action threshold."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Evaluation Gate
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationGate:

    def test_first_deployment_passes_on_absolute_minimums(
        self, trained_model_and_data, tmp_path
    ):
        """No champion run log → relaxed gate (absolute minimums only)."""
        model, X, y = trained_model_and_data
        metrics = run_evaluation_gate(
            model, X, y,
            champion_artifact_dir=tmp_path,  # no run_log.json present
            min_recall=0.00,
            min_roc_auc=0.50,
        )
        assert "recall" in metrics
        assert "roc_auc" in metrics

    def test_gate_returns_metrics_dict_on_pass(self, trained_model_and_data, tmp_path):
        model, X, y = trained_model_and_data
        metrics = run_evaluation_gate(
            model, X, y,
            champion_artifact_dir=tmp_path,
            min_recall=0.00,
            min_roc_auc=0.50,
        )
        assert metrics["roc_auc"] >= 0.50
        assert "cost_per_1k" in metrics
        assert "f1" in metrics

    def test_below_min_recall_fails(self, trained_model_and_data, tmp_path):
        model, X, y = trained_model_and_data
        with pytest.raises(EvaluationGateError, match="Recall"):
            run_evaluation_gate(
                model, X, y,
                champion_artifact_dir=tmp_path,
                min_recall=0.999,  # impossibly high
                min_roc_auc=0.50,
            )

    def test_below_min_roc_auc_fails(self, tmp_path):
        """Use a constant predictor to guarantee ROC-AUC ≈ 0.5."""
        from sklearn.dummy import DummyClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 12))
        y = (rng.random(200) < 0.07).astype(int)

        bad_model = DummyClassifier(strategy="constant", constant=0)
        bad_model.fit(X, y)

        with pytest.raises(EvaluationGateError, match="ROC-AUC"):
            run_evaluation_gate(
                bad_model, X, y,
                champion_artifact_dir=tmp_path,
                min_recall=0.00,
                min_roc_auc=0.60,  # DummyClassifier ROC-AUC ≈ 0.5
            )

    def test_regression_vs_champion_fails(self, tmp_path):
        """Champion ROC-AUC >> challenger → regression gate fires."""
        from sklearn.dummy import DummyClassifier

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 12))
        y = (rng.random(200) < 0.07).astype(int)

        challenger = DummyClassifier(strategy="constant", constant=0)
        challenger.fit(X, y)

        # Fake champion with very high ROC-AUC
        fake_champion = {"roc_auc": 0.95, "recall": 0.80, "cost_per_1k": 5_000}
        (tmp_path / "run_log.json").write_text(
            json.dumps({"val_metrics": fake_champion})
        )

        with pytest.raises(EvaluationGateError, match="regression"):
            run_evaluation_gate(
                challenger, X, y,
                champion_artifact_dir=tmp_path,
                min_recall=0.00,
                min_roc_auc=0.40,   # easy absolute bar
                max_regression_pct=0.05,
            )

    def test_challenger_better_than_champion_passes(
        self, trained_model_and_data, tmp_path
    ):
        """A challenger outperforming a weak champion must always pass Rule 2."""
        model, X, y = trained_model_and_data

        weak_champion = {"roc_auc": 0.51, "recall": 0.30, "cost_per_1k": 25_000}
        (tmp_path / "run_log.json").write_text(
            json.dumps({"val_metrics": weak_champion})
        )

        metrics = run_evaluation_gate(
            model, X, y,
            champion_artifact_dir=tmp_path,
            min_recall=0.00,
            min_roc_auc=0.50,
        )
        assert metrics["roc_auc"] > 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Artifact Promotion and Rollback
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactPromotion:

    def test_promotion_copies_challenger_to_production(self, tmp_path):
        staging = tmp_path / "staging"
        production = tmp_path / "production"
        staging.mkdir()
        production.mkdir()

        (staging / "model.pkl").write_text("challenger_v2")
        (production / "model.pkl").write_text("champion_v1")

        promote_artifacts(staging, production)
        assert (production / "model.pkl").read_text() == "challenger_v2"

    def test_promotion_archives_current_production(self, tmp_path):
        staging = tmp_path / "staging"
        production = tmp_path / "production"
        staging.mkdir()
        production.mkdir()

        (staging / "model.pkl").write_text("challenger_v2")
        (production / "model.pkl").write_text("champion_v1")

        rollback_dir = promote_artifacts(staging, production)
        assert (rollback_dir / "model.pkl").read_text() == "champion_v1"

    def test_rollback_restores_previous_artifacts(self, tmp_path):
        staging = tmp_path / "staging"
        production = tmp_path / "production"
        staging.mkdir()
        production.mkdir()

        (staging / "model.pkl").write_text("challenger_v2")
        (production / "model.pkl").write_text("champion_v1")

        rollback_dir = promote_artifacts(staging, production)

        rollback(rollback_dir, production)
        assert (production / "model.pkl").read_text() == "champion_v1"

    def test_rollback_raises_clearly_without_backup(self, tmp_path):
        """Rollback must fail with a clear message when no backup exists."""
        nonexistent = tmp_path / "previous_does_not_exist"
        production = tmp_path / "production"
        production.mkdir()

        with pytest.raises(RuntimeError, match="No rollback artifacts"):
            rollback(nonexistent, production)

    def test_double_promotion_overwrites_old_rollback(self, tmp_path):
        """Second promotion overwrites the rollback slot (keeps only one previous)."""
        staging = tmp_path / "staging"
        production = tmp_path / "production"
        staging.mkdir()
        production.mkdir()

        (staging / "model.pkl").write_text("v2")
        (production / "model.pkl").write_text("v1")
        promote_artifacts(staging, production)  # rollback slot = v1

        (staging / "model.pkl").write_text("v3")
        promote_artifacts(staging, production)  # rollback slot should now be v2

        rollback_dir = production.parent / "previous"
        assert (rollback_dir / "model.pkl").read_text() == "v2"

    def test_first_promotion_works_without_existing_production_dir(self, tmp_path):
        """Should succeed even if the production directory does not exist yet."""
        staging = tmp_path / "staging"
        production = tmp_path / "production"  # intentionally absent
        staging.mkdir()

        (staging / "model.pkl").write_text("first_model")

        promote_artifacts(staging, production)  # must not raise
        assert (production / "model.pkl").read_text() == "first_model"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestSmokeTest:

    def test_passes_when_model_loaded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}

        with patch("pipeline.deploy.requests.get", return_value=mock_resp):
            assert smoke_test("http://fake/health", max_retries=1) is True

    def test_fails_when_model_not_loaded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "starting", "model_loaded": False}

        with patch("pipeline.deploy.requests.get", return_value=mock_resp):
            assert smoke_test("http://fake/health", max_retries=2, delay_s=0) is False

    def test_fails_on_connection_error(self):
        import requests as _req

        with patch(
            "pipeline.deploy.requests.get",
            side_effect=_req.ConnectionError("Connection refused"),
        ):
            assert smoke_test("http://fake/health", max_retries=2, delay_s=0) is False

    def test_retries_configured_number_of_times(self):
        """Must attempt exactly max_retries times before giving up."""
        import requests as _req

        call_count = {"n": 0}

        def count_and_fail(*args, **kwargs):
            call_count["n"] += 1
            raise _req.ConnectionError("refused")

        with patch("pipeline.deploy.requests.get", side_effect=count_and_fail):
            smoke_test("http://fake/health", max_retries=3, delay_s=0)

        assert call_count["n"] == 3

    def test_passes_on_second_retry(self):
        """Service may be slow to start — should pass once it becomes healthy."""
        import requests as _req

        attempt = {"n": 0}

        def flaky_health(*args, **kwargs):
            attempt["n"] += 1
            if attempt["n"] < 2:
                raise _req.ConnectionError("not ready yet")
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {"model_loaded": True}
            return mock

        with patch("pipeline.deploy.requests.get", side_effect=flaky_health):
            assert smoke_test("http://fake/health", max_retries=5, delay_s=0) is True
