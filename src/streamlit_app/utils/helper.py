import pandas as pd


def format_currency(value: float) -> str:
    """Форматирование валюты"""
    return f"₽ {value:,.0f}".replace(",", " ")


def format_date(date_str: str) -> str:
    """Форматирование даты"""
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime("%d.%m.%Y %H:%M")
    except:
        return date_str


def get_status_color(status: str) -> str:
    """Цвет статуса"""
    colors = {
        'pending': '#ffa500',
        'processed': '#1f77b4',
        'approved': '#2ca02c',
        'rejected': '#d62728'
    }
    return colors.get(status, '#808080')


def get_status_icon(status: str) -> str:
    """Иконка статуса"""
    icons = {
        'pending': '🟡',
        'processed': '🔵',
        'approved': '🟢',
        'rejected': '🔴'
    }
    return icons.get(status, '⚪')