import pandas as pd
from sqlalchemy import create_engine
from src.db.db import DATABASE_URL

def load_data_sync(sample_type: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Синхронная загрузка данных из БД для train, validate или test.

    Args:
        sample_type: 'train', 'validate', или 'test'

    Returns:
        tuple: (X, y) где:
            X - pd.DataFrame с фичами
            y - pd.Series с метками (None для test)
    """
    # Создаем синхронный engine
    sync_url = DATABASE_URL.replace("sqlite+aiosqlite:///", "sqlite:///")
    engine = create_engine(sync_url)

    query = f"""
        SELECT 
            a.id,
            a.application_dt,
            a.education_cd,
            a.gender_cd,
            a.age,
            a.car_own_flg,
            a.car_type_flg,
            a.appl_rej_cnt,
            a.good_work_flg,
            a.Score_bki,
            a.out_request_cnt,
            a.region_rating,
            a.home_address_cd,
            a.work_address_cd,
            a.income,
            a.SNA,
            a.first_time_cd,
            a.Air_flg,
            a.sample_cd,
            a.default_flg
        FROM applications a
        JOIN clients c ON a.client_id = c.id
        WHERE a.sample_cd = '{sample_type}'
    """

    df = pd.read_sql(query, engine)

    if len(df) == 0:
        raise ValueError(f"No data found for sample_type='{sample_type}'")

    # Удаляем служебные колонки
    X = df.drop(columns=['id', 'sample_cd', 'default_flg'], errors='ignore')

    # Преобразуем даты
    if 'application_dt' in X.columns and X['application_dt'].dtype == 'object':
        X['application_dt'] = pd.to_datetime(X['application_dt'], format='%Y-%m-%d %H:%M:%S')

    # Для test меток нет
    if sample_type == 'test':
        return X, None

    # Для train и validate возвращаем метки
    y = df['default_flg']
    return X, y


def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синхронная загрузка тренировочных данных."""
    return load_data_sync('train')


def load_validate_data() -> tuple[pd.DataFrame, pd.Series]:
    """Синхронная загрузка валидационных данных."""
    return load_data_sync('validate')


def load_test_data() -> tuple[pd.DataFrame, None]:
    """Синхронная загрузка тестовых данных (без меток)."""
    return load_data_sync('test')