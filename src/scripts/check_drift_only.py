import sys
from datetime import datetime
import json
from psi_calculator import get_feature_psi, PROJECT_ROOT,CATEGORICAL_FEATURES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Главная функция - только проверка и сохранение"""

    feature_psi = get_feature_psi()

    if not feature_psi:
        result = {
            "drift_detected": False,
            "reason": "No data",
            "feature_psi": {},
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        # Проверяем пороги
        has_drift = False
        drift_reason = []

        for col, psi in feature_psi.items():
            threshold = 0.15 if col in CATEGORICAL_FEATURES else 0.25
            if psi > threshold:
                has_drift = True
                drift_reason.append(f"{col}: {psi:.3f} > {threshold}")

        result = {
            "drift_detected": has_drift,
            "reason": "; ".join(drift_reason) if drift_reason else "No drift",
            "feature_psi": feature_psi,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Сохраняем результат
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    result_path = data_dir / "drift_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Выводим в консоль
    logger.info(f"\nPSI сохранен в {result_path}")
    logger.info(f"Дрифт: {result['drift_detected']}")
    logger.info(f"Причина: {result['reason']}")

    return 0 if not result["drift_detected"] else 1


if __name__ == "__main__":
    sys.exit(main())