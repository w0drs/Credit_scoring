import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"


class APIClient:
    """Клиент для взаимодействия с бэкендом"""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        self.timeout = 30

    def predict(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Отправка заявки на предсказание"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict/demo",
                json=data,
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ошибка предсказания: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return None

    def get_applications(self, limit: int = 100) -> Optional[list]:
        """Получение списка заявок"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/applications",
                params={"limit": limit},
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ошибка получения заявок: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return None

    def get_application(self, app_id: int) -> Optional[Dict[str, Any]]:
        """Получение заявки по ID"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/application/{app_id}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ошибка получения заявки: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return None

    def health_check(self) -> bool:
        """Проверка доступности бэкенда"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False