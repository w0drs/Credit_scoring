from pydantic import BaseModel
from typing import Optional, Dict, Any


class PredictionRequest(BaseModel):
    """Запрос на предсказание"""
    client_id: int

    class Config:
        from_attributes = True


class ModelMetrics(BaseModel):
    """Метрики модели"""
    train_roc_auc: float
    val_roc_auc: float
    test_roc_auc: float


class ModelInfo(BaseModel):
    """Информация о модели"""
    model_name: str
    creation_date: str
    model_version: str
    metrics: ModelMetrics


class PredictionResponse(BaseModel):
    """Ответ на предсказание"""
    prediction: int
    predict_proba: float
    processing_time_ms: float
    client_data: Dict[str, Any]
    model_info: ModelInfo
    error: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    error: str
    data_columns: Optional[list] = None
    processing_time_ms: float


class PredictionRequestForm(BaseModel):
    """Схема для предсказания из формы"""
    # Клиентские поля
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    gender_cd: Optional[str] = None
    age: Optional[int] = None
    income: Optional[int] = None
    education_cd: Optional[str] = None
    car_own_flg: Optional[str] = None
    car_type_flg: Optional[str] = None
    home_address_cd: Optional[int] = None
    work_address_cd: Optional[int] = None
    Score_bki: Optional[float] = None
    appl_rej_cnt: Optional[int] = 0
    out_request_cnt: Optional[int] = 0
    good_work_flg: Optional[int] = 0
    SNA: Optional[int] = 0
    first_time_cd: Optional[int] = 1
    region_rating: Optional[int] = 1
    Air_flg: Optional[str] = None
    # Заявочные поля
    requested_amount: float = 0
    requested_term: int = 12
    purpose: Optional[str] = None