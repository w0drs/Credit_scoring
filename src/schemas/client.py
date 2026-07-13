from pydantic import BaseModel
from typing import Optional


class ClientData(BaseModel):
    """Данные клиента для предсказания"""
    id: int
    full_name: Optional[str] = None

    class Config:
        from_attributes = True


class ClientResponse(BaseModel):
    """Ответ с данными клиента"""
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    income: Optional[int] = None
    gender_cd: Optional[str] = None
    education_cd: Optional[str] = None
    Score_bki: Optional[float] = None

    class Config:
        from_attributes = True


class ClientCreate(BaseModel):
    """Схема для создания клиента"""
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