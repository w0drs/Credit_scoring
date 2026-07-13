from typing import Optional
from src.utils.repository import AbstractRepository
from src.schemas.application import ApplicationSchema


class ApplicationService:
    def __init__(self, application_repo: AbstractRepository):
        self.application_repo: AbstractRepository = application_repo

    async def get_application(self, app_id: int) -> Optional[ApplicationSchema]:
        """Получение заявки"""
        data = await self.application_repo.get_by_id(app_id)
        if data:
            return ApplicationSchema(**data)
        return None