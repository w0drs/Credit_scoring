import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='application_dt'):
        self.date_col = date_col  # Название колонки с датой

    def fit(self, X, y=None):
        return self  # Ничего не учим, просто преобразуем

    def transform(self, X):
        df_temp = X.copy()

        if self.date_col in df_temp.columns:
            dates = pd.to_datetime(df_temp[self.date_col], format='%d%b%Y')
            month = dates.dt.month

            # ОСТАВЛЯЕМ ТОЛЬКО ВАЖНЫЕ:
            # 1. Сезон
            df_temp['season'] = month.apply(self._get_season)

            # 2. Квартал года
            df_temp['quarter'] = dates.dt.quarter  # 1-4

            # Удаляем исходную дату
            df_temp = df_temp.drop(self.date_col, axis=1)

        return df_temp

    def _get_season(self, month):
        """Определяем сезон по месяцу"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для обработки колонок с выбросами"""
    
    def __init__(self, conditions: list[tuple[str, int]]):
        self.conditions = conditions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df_temp = X.copy()
        for condition in self.conditions:
            column, num = condition[0], condition[1] 
            if column in df_temp.columns:
                df_temp[f'{column}_gt_{num}'] = (df_temp[column] > num).astype(int)
                df_temp = df_temp.drop(column, axis=1)
        
        return df_temp