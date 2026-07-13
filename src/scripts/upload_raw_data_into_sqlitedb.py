import asyncio
import pandas as pd
import random
from datetime import datetime
from typing import Dict, Any
from faker import Faker
from sqlalchemy import select, func
from src.models.base import Base
from src.models.client import Client
from src.models.application import Application
from src.db.db import async_session_maker, engine

fake = Faker('ru_RU')


def generate_client_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """Генерация служебных данных для клиента"""

    gender = row.get('gender_cd', 'M')

    if gender == 'M':
        first_name = fake.first_name_male()
        last_name = fake.last_name_male()
        middle_name = fake.middle_name_male()
    else:
        first_name = fake.first_name_female()
        last_name = fake.last_name_female()
        middle_name = fake.middle_name_female()

    return {
        'id': row['id'],

        # Служебные данные
        'first_name': first_name,
        'last_name': last_name,
        'middle_name': middle_name,
        'phone': fake.phone_number(),
        'email': fake.email(),
        'address': fake.address(),

        # Данные из датасета
        'gender_cd': row.get('gender_cd'),
        'age': row.get('age'),
        'income': row.get('income'),
        'education_cd': row.get('education_cd'),
        'car_own_flg': row.get('car_own_flg'),
        'car_type_flg': row.get('car_type_flg'),
        'home_address_cd': row.get('home_address_cd'),
        'work_address_cd': row.get('work_address_cd'),
        'Score_bki': row.get('Score_bki'),
        'appl_rej_cnt': row.get('appl_rej_cnt', 0),
        'out_request_cnt': row.get('out_request_cnt', 0),
        'good_work_flg': row.get('good_work_flg', 0),
        'SNA': row.get('SNA', 0),
        'first_time_cd': row.get('first_time_cd', 1),
        'region_rating': row.get('region_rating', 1),
        'Air_flg': row.get('Air_flg'),

        'is_active': True,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
    }


def generate_application_data(
        row: Dict[str, Any],
        labels: Dict[int, int]
) -> Dict[str, Any]:
    """Генерация данных для заявки"""

    application_dt = pd.to_datetime(row['application_dt'], format='%d%b%Y')

    # Получаем метку, если есть
    default_flg = labels.get(row['id'])

    return {
        'id': row['id'],
        'client_id': row['id'],

        # Данные из датасета
        'application_dt': application_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'education_cd': row.get('education_cd'),
        'gender_cd': row.get('gender_cd'),
        'age': row.get('age'),
        'car_own_flg': row.get('car_own_flg'),
        'car_type_flg': row.get('car_type_flg'),
        'appl_rej_cnt': row.get('appl_rej_cnt', 0),
        'good_work_flg': row.get('good_work_flg', 0),
        'Score_bki': row.get('Score_bki'),
        'out_request_cnt': row.get('out_request_cnt', 0),
        'region_rating': row.get('region_rating', 1),
        'home_address_cd': row.get('home_address_cd'),
        'work_address_cd': row.get('work_address_cd'),
        'income': row.get('income'),
        'SNA': row.get('SNA', 0),
        'first_time_cd': row.get('first_time_cd', 1),
        'Air_flg': row.get('Air_flg'),

        # Служебные поля
        'sample_cd': row.get('sample_cd'),
        'default_flg': default_flg,  # None для test, 0/1 для train/validate

        # Дополнительные данные для демо
        'requested_amount': random.randint(100000, 5000000),
        'requested_term': random.choice([6, 12, 24, 36, 48, 60]),
        'purpose': random.choice([
            'Покупка автомобиля',
            'Покупка недвижимости',
            'Ремонт',
            'Образование',
            'Лечение',
            'Потребительский кредит',
            'Рефинансирование'
        ]),

        # Результаты скоринга
        'credit_score': None,
        'risk_level': None,
        'decision': None,
        'interest_rate': None,
        'decision_date': None,
        'status': 'pending',

        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
    }


def load_labels(csv_path: str) -> Dict[int, int]:
    """Загрузка меток из файла"""
    print(f"Загрузка меток из {csv_path}...")
    df = pd.read_csv(csv_path)
    labels = dict(zip(df['id'], df['default_flg']))
    print(f"Загружено {len(labels)} меток")
    return labels


async def create_tables():
    """Создание таблиц в БД"""
    print("Создание таблиц...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("Таблицы созданы")


async def migrate_data(
        features_path: str,
        labels_path: str
):
    """Основная функция миграции"""

    # 1. Загружаем признаки
    print(f"Загрузка признаков из {features_path}...")
    df_features = pd.read_csv(features_path)
    print(f"Загружено {len(df_features)} записей")

    # 2. Загружаем метки
    labels = load_labels(labels_path)

    # 3. Преобразуем в список словарей
    rows = df_features.to_dict('records')

    # 4. Подготавливаем данные
    clients_data = []
    applications_data = []

    print("Подготовка данных...")
    for row in rows:
        # Клиент
        client = generate_client_data(row)
        clients_data.append(client)

        # Заявка (с меткой)
        application = generate_application_data(row, labels)
        applications_data.append(application)

    print(f"Подготовлено {len(clients_data)} клиентов")
    print(f"Подготовлено {len(applications_data)} заявок")

    # Считаем статистику по выборкам
    df_app = pd.DataFrame(applications_data)
    stats = df_app['sample_cd'].value_counts()
    print("Распределение по выборкам:")
    for sample, count in stats.items():
        has_labels = df_app[df_app['sample_cd'] == sample]['default_flg'].notna().sum()
        print(f"  {sample}: {count} записей, из них с метками: {has_labels}")

    # 5. Загружаем в БД
    print("Загрузка данных в БД...")
    async with async_session_maker() as session:
        try:
            # Загружаем клиентов
            for client_data in clients_data:
                query = select(Client).where(Client.id == client_data['id'])
                result = await session.execute(query)
                existing = result.scalar_one_or_none()

                if existing:
                    for key, value in client_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    client = Client(**client_data)
                    session.add(client)

            # Загружаем заявки
            for app_data in applications_data:
                query = select(Application).where(Application.id == app_data['id'])
                result = await session.execute(query)
                existing = result.scalar_one_or_none()

                if existing:
                    for key, value in app_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    application = Application(**app_data)
                    session.add(application)

            await session.commit()
            print("Данные загружены")

        except Exception as e:
            await session.rollback()
            print(f"Ошибка при загрузке данных: {e}")
            raise


async def verify_data():
    """Проверка загруженных данных"""
    print("Проверка данных...")

    async with async_session_maker() as session:
        # Общая статистика
        query = select(func.count()).select_from(Application)
        result = await session.execute(query)
        total = result.scalar()

        # Статистика по выборкам
        query = select(
            Application.sample_cd,
            func.count(Application.id),
            func.count(Application.default_flg)
        ).group_by(Application.sample_cd)
        result = await session.execute(query)

        print(f"Всего заявок: {total}")
        print("Распределение по выборкам:")
        for row in result:
            sample, count, has_labels = row
            print(f"  {sample}: {count} записей, с метками: {has_labels}")

        # Пример данных
        query = select(Application).limit(1)
        result = await session.execute(query)
        app = result.scalar_one()

        print("\nПример заявки:")
        print(f"  ID: {app.id}")
        print(f"  sample_cd: {app.sample_cd}")
        print(f"  default_flg: {app.default_flg}")
        print(f"  Возраст: {app.age}")
        print(f"  Доход: {app.income}")
        print(f"  Score_bki: {app.Score_bki}")


async def main():
    """Главная функция"""
    FEATURES_PATH = "../../data/raw/application_info.csv"
    LABELS_PATH = "../../data/raw/default_flg.csv"

    try:
        await create_tables()
        await migrate_data(FEATURES_PATH, LABELS_PATH)
        await verify_data()
        print("Ок")

    except FileNotFoundError as e:
        print(f"Файл не найден: {e}")
    except Exception as e:
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())