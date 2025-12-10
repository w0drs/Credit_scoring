from src.ml.data_transformers.transformers import TimeTransformer, OutlierTransformer
import sys
sys.modules['__main__'].OutlierTransformer = OutlierTransformer
sys.modules['__main__'].TimeTransformer = TimeTransformer
from lightgbm import LGBMClassifier
from src.tools.time import datetime_to_bank_format
from src.tools.yaml_loader import load_yaml_safe
import joblib
from fastapi import FastAPI
import uvicorn
from database import Database
from schema import ClientData
import pandas as pd
import timeit


app = FastAPI()
# загрузка yaml файла
model_config = load_yaml_safe("../../config/model_config_1_0_1.yaml")
threshold = model_config["model"]["threshold"]
# ML модель
model = joblib.load(model_config["model"]["file_path"])
# загрузка данных
db = Database(model_config["data"]["database"])
#list_of_models = get_yaml_files("../../config/")
print("✅ Импорт успешен!")

@app.post("/predict")
def predict(client: ClientData):
    start = timeit.default_timer()
    client_name, client_id = client.full_name, client.id
    client_data = db.get_client_by_id(client_id)
    client_data["application_dt"] = datetime_to_bank_format()
    client_dataframe = pd.DataFrame([client_data])
    try:
        # Предсказание модели
        predict_proba = model.predict_proba(client_dataframe.drop(columns=['id']))[0, 1]
        # Метка класса по порогу
        prediction = 1 if predict_proba > threshold else 0
        end = timeit.default_timer()
        processing_time = end - start
        return {
            "prediction": int(prediction),
            "predict_proba": float(predict_proba),
            "processing_time_ms": processing_time * 1000,
            "client_data": client_data,
            "model_info": {
                "model_name": model_config["model"]["name"],
                "creation_date": model_config["information"]["created_at"],
                "model_version": model_config["information"]["version"],
                "metrics": {
                    "train_roc_auc": model_config["evaluate"]["train_roc_auc"],
                    "val_roc_auc": model_config["evaluate"]["validation_roc_auc"],
                    "test_roc_auc": model_config["evaluate"]["test_roc_auc"]
                }
            }
        }

    except Exception as e:
        end = timeit.default_timer()
        processing_time = end - start
        return {
            "error": str(e),
            "data_columns": client_dataframe.columns.tolist(),
            "processing_time_ms": processing_time * 1000
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)