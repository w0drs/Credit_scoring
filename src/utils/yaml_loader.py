import yaml
import os

def load_yaml_safe(file_path):
    """
    Загружает YAML файл с проверкой существования и обработкой ошибок
    """
    try:
        # Проверяем существует ли файл
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден")
            return None

        # Открываем и читаем файл
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        return data

    except yaml.YAMLError as e:
        print(f"Ошибка парсинга YAML в файле {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None