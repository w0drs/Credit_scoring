import uvicorn
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import router
from src.api.load_model import load_model
from src import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Путь к модели
PRODUCTION_DIR = Path(__file__).parent / "model" / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan контекст для загрузки модели при старте."""
    logger.info("=" * 60)
    logger.info("Запуск Credit Scoring API")
    logger.info("=" * 60)

    load_model(PRODUCTION_DIR)

    yield

    logger.info("Завершение работы Credit Scoring API")

app = FastAPI(
    title="Credit Scoring API",
    description="API для кредитного скоринга",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint для smoke test.
    Пайплайн проверяет этот эндпоинт после деплоя.
    Пайплайн: src.pipline...
    """
    return {
        "status": "ok",
        "model_loaded": state.is_loaded,
        "model_version": state.model_version,
        "preprocessor_loaded": state.preprocessor is not None,
        "threshold": state.threshold,
    }


@app.get("/reload")
async def reload_model():
    """
    Эндпоинт для перезагрузки модели (webhook).
    Пайплайн может вызвать его после деплоя.
    Пайплайн: src.pipline...
    """
    success = load_model(PRODUCTION_DIR)
    return {
        "status": "reloaded" if success else "failed",
        "model_loaded": state.is_loaded,
        "model_version": state.model_version,
    }


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Credit Scoring API",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running",
        "model_loaded": state.is_loaded,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )