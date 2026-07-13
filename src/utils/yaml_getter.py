import os
from pathlib import Path


def get_yaml_files(folder_path: str) -> dict:
    """
    Получает все YAML файлы в папке.

    Args:
        folder_path: Путь до папки (относительный или абсолютный)

    Returns:
        Словарь {имя_файла_без_расширения: полный_путь_к_файлу}
    """
    result = {}
    folder = Path(folder_path)

    # Проверяем, существует ли папка
    if not folder.exists():
        print(f"Папка {folder_path} не существует!")
        return result

    # Перебираем все файлы в папке
    for file_path in folder.iterdir():
        # Проверяем, что это файл и имеет расширение .yaml или .yml
        if file_path.is_file() and file_path.suffix.lower() in ['.yaml', '.yml']:
            # Получаем имя файла без расширения
            file_name = file_path.stem
            # Добавляем в словарь
            result[file_name] = str(file_path.absolute())

    return result