from abc import ABC, abstractmethod
from typing import TypeVar, Type, Optional, List
from sqlalchemy import select
from ..db.db import async_session_maker

ModelType = TypeVar('ModelType')

class AbstractRepository(ABC):
    @abstractmethod
    async def get_by_id(self, entity_id):
        raise NotImplementedError

    @abstractmethod
    async def find_all(self, limit: int = 100):
        raise NotImplementedError


class SQLAlchemyRepository(AbstractRepository):

    model: Type[ModelType] | None = None

    async def get_by_id(self, entity_id: int) -> Optional[ModelType]:
        """Получение записи по ID"""
        async with async_session_maker() as session:
            query = select(self.model).where(self.model.id == entity_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def find_all(self, limit: int = 100) -> List[ModelType]:
        """Получение всех записей"""
        async with async_session_maker() as session:
            query = select(self.model).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

