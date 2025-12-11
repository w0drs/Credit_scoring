from src.tools.yaml_getter import get_yaml_files
from src.tools.yaml_loader import load_yaml_safe


def test_load_models_config():
    list_of_models = get_yaml_files("../../config/")
    models_list = dict()
    for name, path in list_of_models.items():
        model_setting = load_yaml_safe(path)
        models_list[name] = model_setting
    assert models_list
