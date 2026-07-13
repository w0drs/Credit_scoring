import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='application_dt'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        if self.date_col in df_temp.columns:
            dates = pd.to_datetime(df_temp[self.date_col])

            if dates.isna().any():
                dates = pd.to_datetime(df_temp[self.date_col], format='%d%b%Y', errors='coerce')

            month = dates.dt.month
            df_temp['season'] = month.apply(self._get_season)
            df_temp['quarter'] = dates.dt.quarter
            df_temp = df_temp.drop(self.date_col, axis=1)

        return df_temp

    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'


# transformers.py
class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для обработки колонок с выбросами"""

    def __init__(self, conditions: list[tuple[str, int]]):
        self.conditions = conditions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        for condition in self.conditions:
            column, num = condition[0], condition[1]
            if column in df_temp.columns:
                df_temp[f'{column}_gt_{num}'] = (df_temp[column] > num).astype(int)

        return df_temp


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Стандартный скейлер для указанных колонок.
    Остальные колонки остаются без изменений.
    """

    def __init__(self, columns=None):
        self.columns = columns  # Список колонок для скейлинга
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.columns:
            self.scaler.fit(X[self.columns])
            self._fitted = True
        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        if self.columns and self._fitted:
            # Скейлим только указанные колонки
            scaled_data = self.scaler.transform(df_temp[self.columns])
            df_temp[self.columns] = scaled_data

        return df_temp


class ModeImputer(BaseEstimator, TransformerMixin):
    """
    Заполнение пропусков модой для указанных колонок.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.modes = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                self.modes[col] = X[col].mode()[0] if not X[col].mode().empty else None

        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        for col in self.columns:
            if col in df_temp.columns and col in self.modes:
                mode_val = self.modes[col]
                if mode_val is not None:
                    df_temp[col] = df_temp[col].fillna(mode_val)

        return df_temp


class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Заполнение пропусков медианой для указанных колонок.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.medians = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                self.medians[col] = X[col].median()

        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        for col in self.columns:
            if col in df_temp.columns and col in self.medians:
                median_val = self.medians[col]
                if median_val is not None:
                    df_temp[col] = df_temp[col].fillna(median_val)

        return df_temp


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot кодирование для указанных категориальных колонок.
    """

    def __init__(self, columns=None, drop_first=False, handle_unknown='ignore'):
        self.columns = columns
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.categories_ = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                # Получаем все уникальные значения
                categories = X[col].dropna().unique()
                if self.drop_first and len(categories) > 1:
                    categories = categories[1:]  # Удаляем первую категорию
                self.categories_[col] = sorted(categories)

        # Сохраняем названия новых колонок
        self.feature_names_ = []
        for col, cats in self.categories_.items():
            for cat in cats:
                self.feature_names_.append(f"{col}_{cat}")

        return self

    def transform(self, X):
        df_temp = X.copy(deep=True)

        for col in self.columns:
            if col not in df_temp.columns:
                continue

            # Получаем категории для этой колонки
            cats = self.categories_.get(col, [])

            # Создаем one-hot колонки
            for cat in cats:
                new_col = f"{col}_{cat}"
                df_temp[new_col] = (df_temp[col] == cat).astype(int)

            # Удаляем оригинальную колонку
            df_temp = df_temp.drop(columns=[col])

        return df_temp

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена колонок после трансформации"""
        return self.feature_names_