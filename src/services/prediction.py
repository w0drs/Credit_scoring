from typing import Dict, Any, Tuple
import timeit
import pandas as pd
import numpy as np
import logging
from src.repositories.client import ClientRepository
from src.repositories.application import ApplicationRepository
from src.utils.model_logger import ModelLogger
from src.models.client import Client
from src.db.db import async_session_maker
from src.schemas.prediction import PredictionRequestForm
from src.pipeline.features import (
    DATE_COL,
    TimeTransformer,
    OutlierTransformer,
    OUTLIER_COLS,
    log_transform,
)
from src import state

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.client_repo = ClientRepository()
        self.application_repo = ApplicationRepository()

        # Берем модель из глобального состояния
        self.model = state.model
        self.preprocessor = state.preprocessor
        self.threshold = state.threshold

        # Логирование модели
        self.model_logger = ModelLogger()

        # Трансформеры
        self.time_transformer = TimeTransformer(date_col=DATE_COL)
        self.outlier_transformer = OutlierTransformer(conditions=OUTLIER_COLS)

    def _prepare_features(self, client_data: Dict[str, Any]) -> np.ndarray:
        """
        Подготовка фичей для модели.
        Полностью повторяет pipeline из обучения.
        """
        features = {
            'application_dt': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'education_cd': client_data.get('education_cd'),
            'gender_cd': client_data.get('gender_cd'),
            'age': client_data.get('age'),
            'car_own_flg': client_data.get('car_own_flg'),
            'car_type_flg': client_data.get('car_type_flg'),
            'appl_rej_cnt': client_data.get('appl_rej_cnt', 0),
            'good_work_flg': client_data.get('good_work_flg', 0),
            'Score_bki': client_data.get('Score_bki'),
            'out_request_cnt': client_data.get('out_request_cnt', 0),
            'region_rating': client_data.get('region_rating', 1),
            'home_address_cd': client_data.get('home_address_cd'),
            'work_address_cd': client_data.get('work_address_cd'),
            'income': client_data.get('income'),
            'SNA': client_data.get('SNA', 0),
            'first_time_cd': client_data.get('first_time_cd', 1),
            'Air_flg': client_data.get('Air_flg'),
        }

        df = pd.DataFrame([features])

        df = self.time_transformer.transform(df)
        df = self.outlier_transformer.transform(df)
        df = log_transform(df)

        X_transformed = self.preprocessor.transform(df)

        return X_transformed

    def _predict(self, client_data: Dict[str, Any]) -> Tuple[int, float]:
        """
        Реальное предсказание модели.
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        X = self._prepare_features(client_data)
        proba = self.model.predict_proba(X)[0, 1]
        prediction = 1 if proba > self.threshold else 0
        return prediction, proba

    async def predict_by_id(self, client_id: int) -> Dict[str, Any]:
        start_time = timeit.default_timer()

        client = await self.client_repo.get_by_id(client_id)
        if not client:
            return {"error": f"Client {client_id} not found"}

        prediction, proba = self._predict(client)

        application = await self.application_repo.create_application({
            'client_id': client_id,
            'requested_amount': 0,
            'requested_term': 0,
        })

        decision = 'approved' if prediction == 0 else 'rejected'
        await self.application_repo.update_scoring_result(
            app_id=application['id'],
            credit_score=proba,
            decision=decision
        )

        processing_time = (timeit.default_timer() - start_time) * 1000

        self.model_logger.log_prediction(
            input_data=client,
            prediction=prediction,
            probability=proba,
            model_version=state.model_version or "unknown",
            client_id=client_id,
            application_id=application['id'],
            processing_time_ms=processing_time
        )

        return {
            'client_id': client_id,
            'application_id': application['id'],
            'prediction': prediction,
            'probability': proba,
            'decision': decision,
            'processing_time_ms': processing_time,
            'client_data': client
        }

    async def predict_from_form(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = timeit.default_timer()

        # Валидируем данные через Pydantic
        validated_data = PredictionRequestForm(**form_data)

        # Преобразуем в словарь и убираем None
        data_dict = validated_data.model_dump(exclude_none=True)

        # Создаем клиента (берем все поля, кроме заявочных)
        client_fields = ['first_name', 'last_name', 'middle_name', 'gender_cd', 'age',
                         'income', 'education_cd', 'car_own_flg', 'car_type_flg',
                         'home_address_cd', 'work_address_cd', 'Score_bki', 'appl_rej_cnt',
                         'out_request_cnt', 'good_work_flg', 'SNA', 'first_time_cd',
                         'region_rating', 'Air_flg']

        client_data = {k: v for k, v in data_dict.items() if k in client_fields}
        application_data = {k: v for k, v in data_dict.items() if k not in client_fields}

        async with async_session_maker() as session:
            client = Client(**client_data)
            session.add(client)
            await session.commit()
            await session.refresh(client)
            client_dict = client.to_dict()

        prediction, proba = self._predict(client_dict)

        application_data['client_id'] = client_dict['id']
        application = await self.application_repo.create_application(application_data)

        decision = 'approved' if prediction == 0 else 'rejected'
        await self.application_repo.update_scoring_result(
            app_id=application['id'],
            credit_score=proba,
            decision=decision
        )

        processing_time = (timeit.default_timer() - start_time) * 1000

        self.model_logger.log_prediction(
            input_data=client_dict,
            prediction=prediction,
            probability=proba,
            model_version=state.model_version or "unknown",
            client_id=client_dict['id'],
            application_id=application['id'],
            processing_time_ms=processing_time
        )

        return {
            'client_id': client_dict['id'],
            'application_id': application['id'],
            'prediction': prediction,
            'probability': proba,
            'decision': decision,
            'processing_time_ms': processing_time,
            'client_data': client_dict
        }