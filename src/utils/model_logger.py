import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# Настройка логгера
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Основной логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Форматтер
formatter = logging.Formatter(
    '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler = logging.FileHandler(LOG_DIR / "model_predictions.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ModelLogger:
    """
    Логгер для ML модели.
    Логирует входные данные, выходные данные и метрики.
    """

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Пути к файлам логов
        self.predictions_log = self.log_dir / "predictions.jsonl"  # JSON Lines
        self.drift_log = self.log_dir / "drift.jsonl"
        self.performance_log = self.log_dir / "performance.jsonl"

        self._predictions_buffer = []
        self._buffer_size = 100

    def log_prediction(
            self,
            input_data: Dict[str, Any],
            prediction: int,
            probability: float,
            model_version: str,
            client_id: int,
            application_id: int,
            processing_time_ms: float
    ):
        """
        Логирование одного предсказания.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "client_id": client_id,
            "application_id": application_id,
            "model_version": model_version,
            "input": input_data,
            "prediction": prediction,
            "probability": probability,
            "processing_time_ms": processing_time_ms,
            "threshold": 0.5,  # текущий порог
        }

        # Запись в JSONL файл
        with open(self.predictions_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Логирование в консоль
        logger.info(
            f"Prediction: client={client_id}, app={application_id}, "
            f"pred={prediction}, prob={probability:.4f}, time={processing_time_ms:.1f}ms"
        )

        # Буферизация для batch записи (опционально)
        self._predictions_buffer.append(log_entry)
        if len(self._predictions_buffer) >= self._buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        """Сброс буфера в файл"""
        if not self._predictions_buffer:
            return

        with open(self.predictions_log, "a", encoding="utf-8") as f:
            for entry in self._predictions_buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._predictions_buffer.clear()

    def log_drift(
            self,
            feature_name: str,
            psi_value: float,
            reference_distribution: Dict,
            current_distribution: Dict,
            alert: bool = False
    ):
        """Логирование дрифта фичей"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "feature": feature_name,
            "psi": psi_value,
            "alert": alert,
            "reference": reference_distribution,
            "current": current_distribution,
        }

        with open(self.drift_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_performance(
            self,
            metrics: Dict[str, float],
            dataset: str = "validation"
    ):
        """Логирование метрик производительности"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset,
            "metrics": metrics,
        }

        with open(self.performance_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def get_recent_predictions(self, n: int = 100) -> pd.DataFrame:
        """Получение последних n предсказаний в виде DataFrame"""
        if not self.predictions_log.exists():
            return pd.DataFrame()

        entries = []
        with open(self.predictions_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    continue

        return pd.DataFrame(entries[-n:])