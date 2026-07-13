from datetime import datetime

def datetime_to_bank_format(dt: datetime = None) -> str:
    """
    Преобразует datetime в формат банка: 01JAN2014

    Примеры:
    01JAN2014 - 1 января 2014
    21MAY2014 - 21 мая 2014
    07DEC2024 - 7 декабря 2024
    """
    if dt is None:
        dt = datetime.now()

    # Получаем компоненты даты
    day = dt.day
    month_en = dt.strftime('%b').upper()  # JAN, FEB, MAR...
    year = dt.year

    # Форматируем: день (без ведущего нуля) + месяц + год
    return f"{day:02d}{month_en}{year}"