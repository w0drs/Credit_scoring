from src.models.client import Client
from src.utils.repository import SQLAlchemyRepository
from typing import Optional, Dict, Any



class ClientRepository(SQLAlchemyRepository):
    """Репозиторий для работы с клиентами"""
    model = Client

    async def get_by_id(self, client_id: int) -> Optional[Dict[str, Any]]:
        """Получение клиента по ID в виде словаря"""
        client = await super().get_by_id(client_id)
        if client:
            return client.to_dict()
        return None
