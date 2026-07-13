from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.repository import SQLAlchemyRepository
from src.models.application import Application
from src.db.db import async_session_maker
from sqlalchemy import select
from typing import List


class ApplicationRepository(SQLAlchemyRepository):
    """Репозиторий для работы с заявками"""

    model = Application

    async def get_by_id(self, app_id: int) -> Optional[Dict[str, Any]]:
        """Получение заявки по ID"""
        application = await super().get_by_id(app_id)
        if application:
            return application.to_dict()
        return None

    async def create_application(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание новой заявки (нужно для тестирования модели)
        """
        data['application_dt'] = datetime.now().strftime("%Y-%m-%d")
        data['status'] = 'pending'

        async with async_session_maker() as session:
            application = self.model(**data)
            session.add(application)
            await session.commit()
            await session.refresh(application)
            return application.to_dict()

    async def update_application(self,
                                 app_id: int,
                                 data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Обновление заявки
        """
        async with async_session_maker() as session:
            query = select(self.model).where(self.model.id == app_id)
            result = await session.execute(query)
            application = result.scalar_one_or_none()

            if application:
                for key, value in data.items():
                    if hasattr(application, key):
                        setattr(application, key, value)

                application.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(application)
                return application.to_dict()
            return None

    async def update_scoring_result(
            self,
            app_id: int,
            credit_score: float,
            decision: str) -> Optional[Dict[str, Any]]:
        """
        Обновление результатов скоринга
        """
        data = {
            'credit_score': credit_score,
            'decision': decision,
            'status': 'processed',
            'decision_date': datetime.utcnow()
        }
        return await self.update_application(app_id, data)

    async def find_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение всех заявок"""
        applications = await super().find_all(limit)
        return [app.to_dict() for app in applications]