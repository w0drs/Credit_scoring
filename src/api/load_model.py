from src import state
import joblib
import logging
import json

logger = logging.getLogger(__name__)

def load_model(production_dir):
    """Загружает модель из production"""
    model_path = production_dir / "model.pkl"
    preprocessor_path = production_dir / "preprocessor.pkl"

    if model_path.exists() and preprocessor_path.exists():
        state.model = joblib.load(model_path)
        state.preprocessor = joblib.load(preprocessor_path)
        state.model_version = str(model_path.stat().st_mtime)
        state.is_loaded = True

        # Пробуем загрузить порог из конфига
        config_path = production_dir / "run_log.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                state.threshold = config.get("threshold", 0.5)

        logger.info(f"Model loaded: version {state.model_version}")
        logger.info(f"Threshold: {state.threshold}")
        return True
    else:
        logger.warning("Model files not found")
        logger.warning(f"Expected: {model_path}")
        logger.warning(f"Expected: {preprocessor_path}")
        return False