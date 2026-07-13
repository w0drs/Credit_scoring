from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text
from datetime import datetime
from typing import Optional
from .base import Base


class Application(Base):
    """Модель заявки"""
    __tablename__ = 'applications'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    client_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('clients.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    application_dt: Mapped[str] = mapped_column(String(50), nullable=False)
    education_cd: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    gender_cd: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    car_own_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    car_type_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    appl_rej_cnt: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    good_work_flg: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    Score_bki: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    out_request_cnt: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    region_rating: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    home_address_cd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    work_address_cd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    income: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    SNA: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    first_time_cd: Mapped[Optional[int]] = mapped_column(Integer, default=1)
    Air_flg: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)

    sample_cd: Mapped[str] = mapped_column(String(20), nullable=True, index=True)  # 'train', 'validate', 'test'
    default_flg: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # NULL для test, 0/1 для train/validate

    requested_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    requested_term: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    credit_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    interest_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    decision_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)

    client: Mapped["Client"] = relationship("Client", back_populates="applications")

    def to_dict(self) -> dict:
        """Преобразование в словарь"""
        return {
            'id': self.id,
            'client_id': self.client_id,
            'application_dt': self.application_dt,
            'education_cd': self.education_cd,
            'gender_cd': self.gender_cd,
            'age': self.age,
            'car_own_flg': self.car_own_flg,
            'car_type_flg': self.car_type_flg,
            'appl_rej_cnt': self.appl_rej_cnt,
            'good_work_flg': self.good_work_flg,
            'Score_bki': self.Score_bki,
            'out_request_cnt': self.out_request_cnt,
            'region_rating': self.region_rating,
            'home_address_cd': self.home_address_cd,
            'work_address_cd': self.work_address_cd,
            'income': self.income,
            'SNA': self.SNA,
            'first_time_cd': self.first_time_cd,
            'Air_flg': self.Air_flg,
            'sample_cd': self.sample_cd,
            'default_flg': self.default_flg,
            'requested_amount': self.requested_amount,
            'requested_term': self.requested_term,
            'purpose': self.purpose,
            'credit_score': self.credit_score,
            'risk_level': self.risk_level,
            'decision': self.decision,
            'interest_rate': self.interest_rate,
            'decision_date': self.decision_date.isoformat() if self.decision_date else None,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_dict_for_scoring(self) -> dict:
        """Только поля для ML модели"""
        return {
            'education_cd': self.education_cd,
            'gender_cd': self.gender_cd,
            'age': self.age,
            'car_own_flg': self.car_own_flg,
            'car_type_flg': self.car_type_flg,
            'appl_rej_cnt': self.appl_rej_cnt,
            'good_work_flg': self.good_work_flg,
            'Score_bki': self.Score_bki,
            'out_request_cnt': self.out_request_cnt,
            'region_rating': self.region_rating,
            'home_address_cd': self.home_address_cd,
            'work_address_cd': self.work_address_cd,
            'income': self.income,
            'SNA': self.SNA,
            'first_time_cd': self.first_time_cd,
            'Air_flg': self.Air_flg,
        }