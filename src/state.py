"""
Глобальное состояние приложения.
Используется для хранения загруженной модели и препроцессора.
"""

model = None
preprocessor = None
model_version = None
threshold = 0.5
is_loaded = False