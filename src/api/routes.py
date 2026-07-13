from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from src.services.prediction import PredictionService
from src.schemas.prediction import PredictionRequest
from src import state

router = APIRouter(prefix="/api/v1", tags=["prediction"])


def get_prediction_service() -> PredictionService:
    return PredictionService()


@router.post("/predict")
async def predict(
        request: PredictionRequest,
        service: PredictionService = Depends(get_prediction_service)
):
    """Предсказание по ID клиента"""
    result = await service.predict_by_id(request.client_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Добавляем информацию о модели
    result["model_info"] = {
        "model_loaded": state.is_loaded,
        "model_version": state.model_version,
        "threshold": state.threshold,
    }

    return result


@router.post("/predict/demo")
async def predict_demo(
        form_data: Dict[str, Any],
        service: PredictionService = Depends(get_prediction_service)
):
    """Демо-предсказание по форме"""
    result = await service.predict_from_form(form_data)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/client/{client_id}")
async def get_client(
        client_id: int,
        service: PredictionService = Depends(get_prediction_service)
):
    """Получение данных клиента"""
    client = await service.client_repo.get_by_id(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
    return client


@router.get("/client/{client_id}/applications")
async def get_client_applications(
        client_id: int,
        limit: int = 10,
        service: PredictionService = Depends(get_prediction_service)
):
    """Получение всех заявок клиента"""
    applications = await service.application_repo.get_by_client_id(client_id, limit)
    if not applications:
        raise HTTPException(status_code=404, detail=f"No applications for client {client_id}")
    return applications


@router.get("/application/{app_id}")
async def get_application(
        app_id: int,
        service: PredictionService = Depends(get_prediction_service)
):
    """Получение заявки по ID"""
    application = await service.application_repo.get_by_id(app_id)
    if not application:
        raise HTTPException(status_code=404, detail=f"Application {app_id} not found")
    return application


@router.get("/applications")
async def get_applications(
    limit: int = 100,
    offset: int = 0,
    service: PredictionService = Depends(get_prediction_service)
):
    """Получение списка всех заявок"""
    applications = await service.application_repo.find_all(limit=limit)
    return applications