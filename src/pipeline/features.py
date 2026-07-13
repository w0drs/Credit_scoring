"""
pipeline/features.py
--------------------
Stage 2: Feature engineering — preprocessor fit-and-save.

The training-serving skew guard:
  The preprocessor (imputer + scaler) is fit ONCE on the training window
  and saved as an artifact alongside the model. The serving code loads this
  exact artifact. There is no way to accidentally apply different scaling
  parameters at inference time.

  At each retraining, a NEW preprocessor is fit on the new data window
  and versioned together with the new model. Old preprocessor + old model
  are archived in the rollback slot.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from .transformers import (
    TimeTransformer,
    OutlierTransformer,
    CustomStandardScaler,
    ModeImputer,
    MedianImputer,
    CustomOneHotEncoder,
)

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Canonical feature list — must match the serving endpoint's input schema.
# Any change here requires a coordinated update to the serving code.
FEATURE_COLS = [
    'application_dt',
    'education_cd',
    'gender_cd',
    'age',
    'car_own_flg',
    'car_type_flg',
    'appl_rej_cnt',
    'good_work_flg',
    'Score_bki',
    'out_request_cnt',
    'region_rating',
    'home_address_cd',
    'work_address_cd',
    'income',
    'SNA',
    'first_time_cd',
    'Air_flg',
]

NUMERIC_COLS = [
    'age',
    'income',
    'Score_bki',
    'region_rating',
    'SNA',
    'out_request_cnt_gt_10',
    'appl_rej_cnt_gt_5',
]


LOG_COLS = ['income']

CATEGORICAL_COLS = [
    'education_cd',
    'gender_cd',
    'car_own_flg',
    'car_type_flg',
    'Air_flg',
    'first_time_cd',
    'season',
    'home_address_cd',
    'work_address_cd',
]

# Колонки для обработки выбросов
OUTLIER_COLS = [
    ('out_request_cnt', 10),
    ('appl_rej_cnt', 5),
]

# Колонка с датой
DATE_COL = 'application_dt'



def df_to_array(X):
    """Converting DataFrame into numpy array"""
    if isinstance(X, pd.DataFrame):
        return X.values
    return X

def log_transform(X):
    """Логарифмирование колонки income"""
    X_copy = X.copy()
    if 'income' in X_copy.columns:
        # Добавляем 1 чтобы избежать log(0)
        X_copy['income'] = np.log1p(X_copy['income'])
    return X_copy

def build_preprocessor() -> Pipeline:
    """
    Return an unfitted sklearn Pipeline.

    Pipeline steps:
    1. TimeTransformer - извлекает сезон и квартал из даты
    2. OutlierTransformer - создает бинарные признаки для выбросов
    3. ColumnTransformer:
       - Для числовых: MedianImputer + StandardScaler
       - Для категориальных: ModeImputer + OneHotEncoder
    """

    log_transformer = FunctionTransformer(log_transform, validate=False)

    numeric_transformer = Pipeline([
        ('imputer', MedianImputer(columns=NUMERIC_COLS)),
        ('scaler', CustomStandardScaler(columns=NUMERIC_COLS)),
    ])

    categorical_transformer = Pipeline([
        ('imputer', ModeImputer(columns=CATEGORICAL_COLS)),
        ('onehot', CustomOneHotEncoder(columns=CATEGORICAL_COLS, drop_first=True)),
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, NUMERIC_COLS),
        ('categorical', categorical_transformer, CATEGORICAL_COLS),
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('time_features', TimeTransformer(date_col=DATE_COL)),
        ('outliers', OutlierTransformer(conditions=OUTLIER_COLS)),
        ('log_transform', log_transformer),
        ('preprocessor', preprocessor),
        ('to_array', FunctionTransformer(df_to_array, validate=False))
    ])

    return pipeline


def fit_and_save_preprocessor(
    df: pd.DataFrame,
    artifact_dir: Path,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on the training data and persist the artifact.

    Args:
        df:           Raw training DataFrame (must contain FEATURE_COLS + 'label').
        artifact_dir: Directory to write preprocessor.pkl into.

    Returns:
        (fitted_preprocessor, X_transformed, y)
        X_transformed is ready to pass directly into the training stage.

    Raises:
        KeyError if any feature in FEATURE_COLS is missing from df.
        (Stage 1 validation should have caught this already.)
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise KeyError(
            f"Features missing from DataFrame: {sorted(missing)}. "
            "Stage 1 validation should have caught this."
        )

    X = df[FEATURE_COLS].copy()
    y = df["label"].values

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    preprocessor_path = artifact_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(
        "Preprocessor fitted on %d samples and saved → %s",
        len(X), preprocessor_path,
    )

    return preprocessor, X_transformed, y


def fit_and_save_preprocessor2(
        df: pd.DataFrame,
        artifact_dir: Path,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on the training data and persist the artifact.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise KeyError(
            f"Features missing from DataFrame: {sorted(missing)}. "
            "Stage 1 validation should have caught this."
        )

    X = df[FEATURE_COLS].copy()
    y = df["label"].values

    # ===== ОТЛАДКА: применяем трансформеры по шагам =====
    logger.info("=" * 60)
    logger.info("ОТЛАДКА: проверка колонок перед ColumnTransformer")

    # Шаг 1: TimeTransformer
    time_transformer = TimeTransformer(date_col=DATE_COL)
    X_time = time_transformer.transform(X)
    logger.info(f"После TimeTransformer: {list(X_time.columns)}")

    # Шаг 2: OutlierTransformer
    outlier_transformer = OutlierTransformer(conditions=OUTLIER_COLS)
    X_outlier = outlier_transformer.transform(X_time)
    logger.info(f"После OutlierTransformer: {list(X_outlier.columns)}")

    # Шаг 3: log_transform
    X_log = log_transform(X_outlier)
    logger.info(f"После log_transform: {list(X_log.columns)}")

    # Проверяем, какие колонки из NUMERIC_COLS и CATEGORICAL_COLS есть
    available_cols = set(X_log.columns)
    numeric_available = [col for col in NUMERIC_COLS if col in available_cols]
    categorical_available = [col for col in CATEGORICAL_COLS if col in available_cols]

    missing_numeric = set(NUMERIC_COLS) - set(numeric_available)
    missing_categorical = set(CATEGORICAL_COLS) - set(categorical_available)

    if missing_numeric:
        logger.error(f"❌ Отсутствуют числовые колонки: {missing_numeric}")
    if missing_categorical:
        logger.error(f"❌ Отсутствуют категориальные колонки: {missing_categorical}")

    logger.info(f"✅ Доступные числовые колонки: {numeric_available}")
    logger.info(f"✅ Доступные категориальные колонки: {categorical_available}")
    logger.info("=" * 60)

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    preprocessor_path = artifact_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(
        "Preprocessor fitted on %d samples and saved → %s",
        len(X), preprocessor_path,
    )

    return preprocessor, X_transformed, y


def load_preprocessor(artifact_dir: Path) -> Pipeline:
    """Load the fitted preprocessor from an artifact directory."""
    path = Path(artifact_dir) / "preprocessor.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No preprocessor artifact found at '{path}'. "
            "Run the training pipeline first."
        )
    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor
