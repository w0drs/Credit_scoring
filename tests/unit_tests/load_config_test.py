from src.tools.yaml_loader import load_yaml_safe

def test_yaml_load():
    model_config = load_yaml_safe("../../config/model_config_1_0_1.yaml")
    assert model_config