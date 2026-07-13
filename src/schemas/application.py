from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
from src.config.features import MODEL_FEATURES


class ApplicationData(BaseModel):
    """Данные заявки для предсказания"""
    id: int
    application_dt: str
    education_cd: Optional[str] = None
    gender_cd: Optional[str] = None
    age: Optional[int] = None
    car_own_flg: Optional[str] = None
    car_type_flg: Optional[str] = None
    appl_rej_cnt: Optional[int] = 0
    good_work_flg: Optional[int] = 0
    Score_bki: Optional[float] = None
    out_request_cnt: Optional[int] = 0
    region_rating: Optional[int] = 1
    home_address_cd: Optional[int] = None
    work_address_cd: Optional[int] = None
    income: Optional[int] = None
    SNA: Optional[int] = 0
    first_time_cd: Optional[int] = 1
    Air_flg: Optional[str] = None

    @classmethod
    def from_client(cls, client: Dict[str, Any]) -> "ApplicationData":
        """Создание из данных клиента"""
        data = {
            'id': client.get('id', 0),
            'application_dt': datetime.now().strftime("%Y-%m-%d"),
        }
        for field in MODEL_FEATURES:
            data[field] = client.get(field)
        return cls(**data)

    def to_dataframe(self) -> pd.DataFrame:
        """Преобразование в DataFrame для модели"""
        data = {field: getattr(self, field) for field in MODEL_FEATURES}
        return pd.DataFrame([data])

    class Config:
        from_attributes = True


class ApplicationCreate(BaseModel):
    """Создание заявки (для демо)"""
    client_id: int
    requested_amount: float
    requested_term: int
    purpose: Optional[str] = None

    class Config:
        from_attributes = True


class ApplicationResponse(BaseModel):
    """Ответ с данными заявки"""
    id: int
    client_id: int
    application_dt: str
    requested_amount: Optional[float] = None
    requested_term: Optional[int] = None
    purpose: Optional[str] = None
    credit_score: Optional[float] = None
    decision: Optional[str] = None
    status: str
    sample_cd: Optional[str] = None
    default_flg: Optional[int] = None

    class Config:
        from_attributes = True


class ApplicationSchema(BaseModel):
    """Полная схема заявки"""
    id: int
    client_id: int
    application_dt: str
    education_cd: Optional[str] = None
    gender_cd: Optional[str] = None
    age: Optional[int] = None
    car_own_flg: Optional[str] = None
    car_type_flg: Optional[str] = None
    appl_rej_cnt: Optional[int] = 0
    good_work_flg: Optional[int] = 0
    Score_bki: Optional[float] = None
    out_request_cnt: Optional[int] = 0
    region_rating: Optional[int] = 1
    home_address_cd: Optional[int] = None
    work_address_cd: Optional[int] = None
    income: Optional[int] = None
    SNA: Optional[int] = 0
    first_time_cd: Optional[int] = 1
    Air_flg: Optional[str] = None
    sample_cd: Optional[str] = None
    default_flg: Optional[int] = None
    requested_amount: Optional[float] = None
    requested_term: Optional[int] = None
    purpose: Optional[str] = None
    credit_score: Optional[float] = None
    decision: Optional[str] = None
    status: str = "pending"

    class Config:
        from_attributes = True