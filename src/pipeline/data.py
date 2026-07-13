"""
pipeline/data.py
----------------
Stage 1: Data collection and validation.

Design principles:
- Accumulates ALL validation errors before raising (not fail-fast).
  The engineer sees the complete picture in a single run.
- Null-rate check is warning-only: upstream nulls often have legitimate
  causes; silently failing every retrain is worse than logging and continuing.
- Volume and label checks are blocking: training on bad data is worse
  than not training at all.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    reference_path: Path          # original training data — used for drift checks
    lookback_days: int = 90       # how many days of recent data to train on
    min_samples: int = 500        # refuse to train on fewer than this
    min_positive_rate: float = 0.02   # catch label drift to near-zero
    max_positive_rate: float = 0.40   # catch label drift to near-one
    required_features: Optional[List[str]] = field(default_factory=list)
    mode: Literal['streaming', 'static'] = 'streaming' # добавил режим статических и потоковых данных
    data_path: Path = None       # Путь к статическим данным
    data_dir: Path = None        # Путь к потоковым данным
    reference_path: Optional[Path] = None

    def __post_init__(self):
        """Проверяем корректность конфигурации"""
        if self.mode == 'streaming' and self.data_dir is None:
            raise ValueError("data_dir must be provided for streaming mode")
        if self.mode == 'static' and self.data_path is None:
            raise ValueError("data_path must be provided for static mode")


class DataValidationError(Exception):
    """Raised when data fails validation. Pipeline halts."""
    pass

# ДЛЯ ПОТОКОВЫХ ДАННЫХ
def load_training_window(config: DataConfig) -> pd.DataFrame:
    """
    Load the most recent `lookback_days` of labeled data.

    Expects parquet files named YYYY-MM-DD.parquet inside config.data_dir.
    In a real system this would query your feature store or data warehouse.
    """
    cutoff = datetime.now() - timedelta(days=config.lookback_days)
    frames = []

    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise DataValidationError(
            f"Data directory '{data_dir}' does not exist. "
            "Create it and populate with YYYY-MM-DD.parquet files."
        )

    for path in sorted(data_dir.glob("*.parquet")):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d")
        except ValueError:
            logger.debug("Skipping non-date file: %s", path.name)
            continue
        if file_date >= cutoff:
            frames.append(pd.read_parquet(path))

    if not frames:
        raise DataValidationError(
            f"No data files found in '{data_dir}' "
            f"for the past {config.lookback_days} days. "
            "Widen the lookback window or check the upstream pipeline."
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d samples from %d file(s)", len(df), len(frames))
    return df

# ДЛЯ СТАТИЧЕСКИХ ДАННЫХ
def load_static_dataset(data_path: Path) -> pd.DataFrame:
    """
    Загружает статический датасет из одного файла.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Please place your dataset in the data/ directory."
        )

    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix == '.pkl' or data_path.suffix == '.pickle':
        df = pd.read_pickle(data_path)
    else:
        raise ValueError(
            f"Unsupported file format: {data_path.suffix}. "
            "Supported: .parquet, .csv, .pkl"
        )

    logger.info(
        f"Loaded static dataset: {len(df)} samples, "
        f"{len(df.columns)} columns from {data_path}"
    )
    return df


def validate_data(df: pd.DataFrame, config: DataConfig) -> None:
    """
    Validates the training window before any model code runs.

    Accumulates all errors before raising DataValidationError so the engineer
    sees the full picture rather than fixing one issue at a time.

    ML-specific checks (not covered by Great Expectations):
      - Minimum sample volume
      - Label column existence and dtype
      - Label rate bounds (catches collection failures and leakage)
      - Required feature presence
      - Null rate (warning-only, non-blocking)
    """
    errors = []

    # ── Volume check ──────────────────────────────────────────────
    if len(df) < config.min_samples:
        errors.append(
            f"Only {len(df)} samples — minimum is {config.min_samples}. "
            "Widen the lookback window or check the upstream pipeline."
        )

    # ── Label checks ──────────────────────────────────────────────
    if "label" not in df.columns:
        errors.append("Column 'label' not found. Schema may have changed.")
    else:
        if df["label"].dtype not in [np.int64, np.int32, np.float64, np.int8, np.uint8]:
            errors.append(
                f"Label column has unexpected dtype: {df['label'].dtype}. "
                "Expected integer or float."
            )
        else:
            positive_rate = float(df["label"].mean())
            if positive_rate < config.min_positive_rate:
                errors.append(
                    f"Positive rate is {positive_rate:.4f} — suspiciously low "
                    f"(minimum: {config.min_positive_rate}). "
                    "Check if labels are being collected correctly."
                )
            if positive_rate > config.max_positive_rate:
                errors.append(
                    f"Positive rate is {positive_rate:.4f} — suspiciously high "
                    f"(maximum: {config.max_positive_rate}). "
                    "Check for label leakage in the upstream pipeline."
                )

    # ── Feature presence ──────────────────────────────────────────
    if config.required_features:
        missing = set(config.required_features) - set(df.columns)
        if missing:
            errors.append(
                f"Missing required features: {sorted(missing)}. "
                "Schema may have changed upstream."
            )

    if config.mode == 'static':
        # Проверка на дубликаты
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            errors.append(
                f"Found {duplicates} duplicate rows ({duplicates / len(df):.1%})"
            )

        # Проверка на константные колонки
        constant_cols = []
        for col in df.columns:
            if col != 'label' and df[col].nunique() == 1:
                constant_cols.append(col)
        if constant_cols:
            logger.warning(
                f"Constant columns detected (consider dropping): {constant_cols}"
            )

        # Проверка на несбалансированные категориальные колонки
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col != 'label':
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    most_common_pct = value_counts.iloc[0] / len(df)
                    if most_common_pct > 0.95:
                        errors.append(
                            f"Column '{col}' is highly imbalanced: "
                            f"{value_counts.index[0]} = {most_common_pct:.1%}"
                        )

        # Проверка на выбросы для числовых колонок
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != 'label':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
                    if outliers / len(df) > 0.05:
                        errors.append(
                            f"Column '{col}' has {outliers / len(df):.1%} outliers (>3 IQR)"
                        )

    # ── Null rate (warning, not blocking) ─────────────────────────
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.10]
    if not high_null.empty:
        logger.warning(
            "High null rates detected (>10%%) — will be imputed by preprocessor: %s",
            {col: f"{rate:.1%}" for col, rate in high_null.items()},
        )

    # ── Raise with all errors at once ─────────────────────────────
    if errors:
        raise DataValidationError(
            f"Data validation failed ({len(errors)} error(s)):\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        "Data validation passed: %d samples, %.2f%% positive rate",
        len(df),
        float(df["label"].mean()) * 100 if "label" in df.columns else 0,
    )

def load_data(config: DataConfig) -> pd.DataFrame:
    """
    Загружает данные в зависимости от режима.
    """
    if config.mode == 'static':
        return load_static_dataset(config.data_path)
    elif config.mode == 'streaming':
        return load_training_window(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")
