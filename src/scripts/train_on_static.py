# scripts/train_on_static.py
"""
Script for initial model training on static dataset.
Run once to create a baseline model before switching to streaming mode.

Usage:
    python scripts/train_on_static.py
"""

import sys
import logging
import asyncio
from pathlib import Path
import pandas as pd
from sqlalchemy import select
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.db import async_session_maker
from src.models.application import Application
from src.models.client import Client
from src.pipeline.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def load_data_from_db() -> pd.DataFrame:
    """Загрузка данных из БД для обучения"""
    logger.info("Загрузка данных из БД...")

    async with async_session_maker() as session:
        query = select(
            Application.id,
            Application.application_dt,
            Application.education_cd,
            Application.gender_cd,
            Application.age,
            Application.car_own_flg,
            Application.car_type_flg,
            Application.appl_rej_cnt,
            Application.good_work_flg,
            Application.Score_bki,
            Application.out_request_cnt,
            Application.region_rating,
            Application.home_address_cd,
            Application.work_address_cd,
            Application.income,
            Application.SNA,
            Application.first_time_cd,
            Application.Air_flg,
            Application.sample_cd,
            Application.default_flg,
        ).join(Client, Application.client_id == Client.id)

        result = await session.execute(query)
        rows = result.fetchall()

        df = pd.DataFrame([dict(row._mapping) for row in rows])

        logger.info(f"Загружено {len(df)} записей")

        sample_counts = df['sample_cd'].value_counts()
        logger.info(f"Распределение по выборкам:\n{sample_counts}")

        has_labels = df['default_flg'].notna().sum()
        logger.info(f"Записей с метками: {has_labels}")

        return df


async def main():
    """Главная функция"""
    logger.info("=" * 60)
    logger.info("Обучение модели на данных из БД")
    logger.info("=" * 60)

    try:
        df = await load_data_from_db()

        if len(df) == 0:
            logger.error("Нет данных в БД! Сначала запустите миграцию.")
            return

        # Берем только записи с метками
        df_labeled = df[df['default_flg'].notna()].copy()

        if len(df_labeled) == 0:
            logger.error("Нет записей с метками (default_flg)!")
            return

        logger.info(f"Записей с метками: {len(df_labeled)}")

        # Проверяем наличие train и validate
        train_count = len(df_labeled[df_labeled['sample_cd'] == 'train'])
        val_count = len(df_labeled[df_labeled['sample_cd'] == 'validate'])

        if train_count == 0 or val_count == 0:
            logger.error(f"Нет данных для обучения или валидации! Train: {train_count}, Validate: {val_count}")
            return

        # Удаляем служебные колонки (включая sample_cd)
        drop_cols = ['id', 'sample_cd', 'default_flg', 'first_name', 'last_name', 'phone', 'email']

        # X - только признаки (без sample_cd и default_flg)
        X = df_labeled.drop(columns=drop_cols, errors='ignore')
        y = df_labeled['default_flg']
        sample_cd = df_labeled['sample_cd']

        # Преобразуем даты
        if 'application_dt' in X.columns:
            X['application_dt'] = pd.to_datetime(X['application_dt'], format='%Y-%m-%d %H:%M:%S')

        y = y.rename('label')

        # Собираем полный датасет (X + label + sample_cd)
        df_full = pd.concat([X, y, sample_cd], axis=1)

        # Сохраняем в файл
        data_path = PROJECT_ROOT / "data" / "train_data.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_full.to_parquet(data_path)

        logger.info(f"Сохранено {len(df_full)} записей в {data_path}")
        logger.info(f"Train: {train_count}, Validate: {val_count}")
        logger.info(f"Колонки в датасете: {list(df_full.columns)}")

        # Запускаем пайплайн в статическом режиме
        result = run_pipeline(
            force=True,
            mode='static',
            static_data_path=data_path,
        )

        print("\n" + "=" * 60)
        if result["status"] == "deployed":
            print("MODEL TRAINED SUCCESSFULLY!")
            print(f"   Run ID: {result['run_id']}")

            metrics = result.get('metrics', {})
            if metrics:
                print("\n   📊 Metrics:")
                print(f"      Recall:    {metrics.get('recall', 'N/A')}")
                print(f"      ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")
                print(f"      F1:        {metrics.get('f1', 'N/A')}")

            print(f"\n   Model saved to: {PROJECT_ROOT / 'model' / 'artifacts'}")
        else:
            print("TRAINING FAILED!")
            print(f"   Status: {result['status']}")
            print(f"   Error: {result.get('error', 'No error details')}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())