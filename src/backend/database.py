from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any
import pandas as pd

Base = declarative_base()


class Database:
    def __init__(self, db_path: str = 'database.db'):
        self.db_url = f'sqlite:///{db_path}' if db_path else 'sqlite:///:memory:'
        self.engine = create_engine(self.db_url)

    def get_client_by_id(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Получение данных о клиенте по ID"""
        query = text("""
            SELECT *
            FROM applications_database
            WHERE id = :client_id
        """)

        with self.engine.connect() as connection:
            result = connection.execute(query, {"client_id": client_id})
            row = result.fetchone()

            if row:
                # Преобразуем в словарь
                return dict(row._mapping)
            return None

    def get_review_status(self, application_id: int) -> Optional[Dict[str, Any]]:
        query = text("SELECT id, review_status FROM applications_database WHERE id = :id")
        with self.engine.connect() as connection:
            row = connection.execute(query, {"id": application_id}).mappings().first()
            return dict(row) if row is not None else None

    def get_all_clients(self, limit: int = 100) -> pd.DataFrame:
        """Получить всех клиентов (для анализа)"""
        query = "SELECT * FROM applications_database LIMIT :limit"
        return pd.read_sql(query, self.engine, params={"limit": limit})