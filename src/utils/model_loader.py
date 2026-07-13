# src/utils/model_loader.py
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
from src.utils.yaml_loader import load_yaml_safe

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self._model = None
        self._config = None
        self._threshold = 0.5
        self._preprocessor = None
        self._loaded = False

    def load(self, config_path: str):
        """Загрузка модели."""
        if self._loaded:
            return

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфига не найден: {config_path}")

        logger.info(f"Загрузка конфига из {config_path}")
        self._config = load_yaml_safe(str(config_path))

        if self._config is None:
            raise ValueError(f"Не удалось загрузить конфиг из {config_path}")

        base_dir = config_path.parent.parent
        model_path = base_dir / self._config["model"]["file_path"]
        preprocessor_path = base_dir / self._config["model"]["preprocessor_path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path}\n"
                f"Ожидается по пути: {self._config['model']['file_path']}"
            )

        logger.info(f"Загрузка модели из {model_path}")
        self._model = joblib.load(str(model_path))

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Файл препроцессора не найден: {preprocessor_path}\n"
                f"Ожидается по пути: {self._config['model']['preprocessor_path']}"
            )

        logger.info(f"Загрузка препроцессора из {preprocessor_path}")
        self._preprocessor = joblib.load(str(preprocessor_path))

        self._threshold = self._config["model"].get("threshold", 0.5)
        self._loaded = True
        logger.info("Модель и препроцессор загружены")

    @property
    def model(self):
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
        return self._model

    @property
    def preprocessor(self):
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
        return self._preprocessor

    @property
    def threshold(self):
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
        return self._threshold

    @property
    def config(self):
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
        return self._config

    @property
    def is_loaded(self):
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели."""
        if not self._loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() сначала.")
        return {
            "model_name": self._config["model"].get("name", "unknown"),
            "creation_date": self._config["information"].get("created_at", "unknown"),
            "model_version": self._config["information"].get("version", "unknown"),
            "metrics": self._config.get("evaluate", {})
        }


# Создаем один экземпляр для всего приложения
model_loader = ModelLoader()