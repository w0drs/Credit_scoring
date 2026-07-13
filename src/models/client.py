from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, Boolean
from datetime import datetime
from typing import Optional, List
from .base import Base


class Client(Base):
    """Модель клиента"""
    __tablename__ = 'clients'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    first_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    middle_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Эти поля дублируются для удобства, но основные данные в application
    gender_cd: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    income: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    education_cd: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    car_own_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    car_type_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    home_address_cd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    work_address_cd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    Score_bki: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    appl_rej_cnt: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    out_request_cnt: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    good_work_flg: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    SNA: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    first_time_cd: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    region_rating: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    Air_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)

    applications: Mapped[List["Application"]] = relationship(
        "Application",
        back_populates="client",
        cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Преобразование в словарь"""
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'middle_name': self.middle_name,
            'phone': self.phone,
            'email': self.email,
            'address': self.address,
            'gender_cd': self.gender_cd,
            'age': self.age,
            'income': self.income,
            'education_cd': self.education_cd,
            'car_own_flg': self.car_own_flg,
            'car_type_flg': self.car_type_flg,
            'home_address_cd': self.home_address_cd,
            'work_address_cd': self.work_address_cd,
            'Score_bki': self.Score_bki,
            'appl_rej_cnt': self.appl_rej_cnt,
            'out_request_cnt': self.out_request_cnt,
            'good_work_flg': self.good_work_flg,
            'SNA': self.SNA,
            'first_time_cd': self.first_time_cd,
            'region_rating': self.region_rating,
            'Air_flg': self.Air_flg,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }