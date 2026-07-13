import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.triggers import compute_psi
from src.pipeline.features import FEATURE_COLS
from src.utils.data_loader import load_test_data, load_train_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXCLUDE_FROM_PSI = {
    'application_dt',
    'month',
    'quarter',
    'day_of_week',
    'hour',
    'season',
    'year',
    'week',
}

CATEGORICAL_FEATURES = {
    'gender_cd',
    'education_cd',
    'car_type_flg',
    'region_rating',
    'home_address_cd',
    'work_address_cd',
    'first_time_cd',
    'Air_flg',
    'car_own_flg'
}

def compute_psi_categorical(reference, current):
    """
    Считает PSI для категориального признака
    Использует частоты категорий вместо бинов
    """
    all_categories = set(reference) | set(current)

    ref_counts = pd.Series(reference).value_counts(normalize=True)
    cur_counts = pd.Series(current).value_counts(normalize=True)

    epsilon = 1e-9
    psi = 0

    for cat in all_categories:
        ref_pct = ref_counts.get(cat, 0) + epsilon
        cur_pct = cur_counts.get(cat, 0) + epsilon
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return psi


def compute_psi_with_woe(reference, current, target=None):
    """
    Считает PSI на основе WoE (Weight of Evidence)

    WoE показывает, насколько категория влияет на целевую переменную.
    PSI на основе WoE показывает, изменилось ли это влияние.

    Args:
        reference: категории в референсных данных (обучение)
        current: категории в текущих данных (мониторинг)
        target: целевая переменная (0/1) - нужна для WoE

    Returns:
        float: PSI на основе WoE
    """
    # Если нет целевой - используем обычный категориальный PSI
    if target is None:
        return compute_psi_categorical(reference, current)

    # 1. Создаем DataFrame для расчетов
    df_ref = pd.DataFrame({
        'feature': reference,
        'target': target
    })

    # Группируем по категориям
    grouped = df_ref.groupby('feature')['target'].agg(['count', 'sum'])
    grouped.columns = ['count', 'event']
    grouped['non_event'] = grouped['count'] - grouped['event']

    # Считаем WoE для каждой категории
    total_event = grouped['event'].sum()
    total_non_event = grouped['non_event'].sum()

    grouped['event_smoothed'] = grouped['event'] + 0.5
    grouped['non_event_smoothed'] = grouped['non_event'] + 0.5

    grouped['woe'] = np.log(
        (grouped['event_smoothed'] / total_event) /
        (grouped['non_event_smoothed'] / total_non_event)
    )

    # Считаем частоты категорий в текущих данных
    cur_counts = pd.Series(current).value_counts(normalize=True)

    # Считаем PSI на основе WoE
    epsilon = 1e-9
    psi = 0

    for cat in grouped.index:
        # WoE значение для категории
        woe_val = grouped.loc[cat, 'woe']

        # Частота в текущих данных
        cur_pct = cur_counts.get(cat, 0) + epsilon

        # Для референса берем частоту из обучающих данных
        ref_pct = grouped.loc[cat, 'count'] / grouped['count'].sum()

        # PSI на WoE
        psi += (cur_pct - ref_pct) * woe_val

    return psi


def get_feature_psi():
    """
    Собирает PSI для всех фичей (числовых и категориальных)
    """

    ref_df, _ = load_train_data()
    current_df, _ = load_test_data()

    # Считаем PSI для каждой фичи
    feature_psi = {}

    # Определяем, какие фичи будем мониторить
    monitor_features = [col for col in FEATURE_COLS if col not in EXCLUDE_FROM_PSI]

    for col in monitor_features:
        if col not in ref_df.columns or col not in current_df.columns:
            logger.warning(f"Фича {col} отсутствует в данных")
            continue

        # Очищаем данные
        ref_vals = ref_df[col].dropna()
        cur_vals = current_df[col].dropna()

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            logger.warning(f"Нет данных для фичи {col}")
            continue

        try:
            # Выбираем метод расчета в зависимости от типа
            if col in CATEGORICAL_FEATURES:
                # Категориальный PSI
                ref_str = ref_vals.astype(str).values
                cur_str = cur_vals.astype(str).values

                # Если есть целевая переменная - используем WoE
                if 'label' in ref_df.columns:
                    target = ref_df.loc[ref_vals.index, 'label'].values
                    psi = compute_psi_with_woe(ref_str, cur_str, target)
                else:
                    psi = compute_psi_categorical(ref_str, cur_str)

                logger.debug(f"Категориальная фича {col}: PSI={psi:.3f}")
            else:
                # Числовой PSI
                ref_num = ref_vals.values
                cur_num = cur_vals.values
                psi = compute_psi(ref_num, cur_num)
                logger.debug(f"Числовая фича {col}: PSI={psi:.3f}")

            feature_psi[col] = psi

        except Exception as e:
            logger.warning(f"Ошибка PSI для {col}: {e}")

    # Логируем результаты
    logger.info("=" * 60)
    logger.info("PSI по фичам:")

    # Числовые фичи
    numeric_cols = [col for col in feature_psi.keys() if col not in CATEGORICAL_FEATURES]
    if numeric_cols:
        logger.info("Числовые фичи:")
        sorted_numeric = sorted(numeric_cols, key=lambda x: feature_psi[x], reverse=True)
        for col in sorted_numeric[:10]:
            psi = feature_psi[col]
            status = "DRIFT" if psi > 0.25 else "OK"
            logger.info(f"    {col}: {psi:.4f} {status}")

    # Категориальные фичи
    cat_cols = [col for col in feature_psi.keys() if col in CATEGORICAL_FEATURES]
    if cat_cols:
        logger.info("Категориальные фичи:")
        sorted_cat = sorted(cat_cols, key=lambda x: feature_psi[x], reverse=True)
        for col in sorted_cat[:10]:
            psi = feature_psi[col]
            status = "DRIFT" if psi > 0.15 else "OK"
            logger.info(f"    {col}: {psi:.4f} {status}")

    # Пропущенные фичи
    skipped = [col for col in EXCLUDE_FROM_PSI if col in ref_df.columns]
    if skipped:
        logger.info(f"Пропущены временные фичи: {skipped}")

    logger.info("=" * 60)

    return feature_psi