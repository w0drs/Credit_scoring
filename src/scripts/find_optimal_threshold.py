"""
Находит оптимальный порог для модели
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def find_optimal_threshold():
    """Находит порог, который дает нужный recall"""

    PRODUCTION_DIR = PROJECT_ROOT / "model" / "artifacts"

    # Загружаем модель и препроцессор
    model = joblib.load(PRODUCTION_DIR / "model.pkl")
    preprocessor = joblib.load(PRODUCTION_DIR / "preprocessor.pkl")

    # Загружаем валидационные данные
    data_path = PROJECT_ROOT / "data" / "train_data.parquet"
    df = pd.read_parquet(data_path)
    df_val = df[df['sample_cd'] == 'validate'].copy()

    X_val = df_val.drop(columns=['label', 'sample_cd'], errors='ignore')
    y_val = df_val['label'].values

    X_val_processed = preprocessor.transform(X_val)

    # Проверяем, что получилось
    print(f"Данные трансформированы: {X_val_processed.shape}")

    # Получаем вероятности
    y_proba = model.predict_proba(X_val_processed)[:, 1]

    # Перебираем пороги
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        results.append({
            'threshold': threshold,
            'recall': recall_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
        })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("Поиск оптимального порога")
    print("=" * 60)
    print(df_results.round(4))

    # Ищем порог для recall = 0.50
    recall_50 = df_results[df_results['recall'] >= 0.50]
    if not recall_50.empty:
        best = recall_50.loc[recall_50['f1'].idxmax()]
        print(f"\nПорог для recall >= 0.50:")
        print(f"   Порог: {best['threshold']:.2f}")
        print(f"   Recall: {best['recall']:.4f}")
        print(f"   Precision: {best['precision']:.4f}")
        print(f"   F1: {best['f1']:.4f}")

    # Ищем порог для recall = 0.60
    recall_60 = df_results[df_results['recall'] >= 0.60]
    if not recall_60.empty:
        best = recall_60.loc[recall_60['f1'].idxmax()]
        print(f"\nПорог для recall >= 0.60:")
        print(f"   Порог: {best['threshold']:.2f}")
        print(f"   Recall: {best['recall']:.4f}")
        print(f"   Precision: {best['precision']:.4f}")
        print(f"   F1: {best['f1']:.4f}")

    # Ищем порог для recall = 0.70
    recall_70 = df_results[df_results['recall'] >= 0.70]
    if not recall_70.empty:
        best = recall_70.loc[recall_70['f1'].idxmax()]
        print(f"\nПорог для recall >= 0.70:")
        print(f"   Порог: {best['threshold']:.2f}")
        print(f"   Recall: {best['recall']:.4f}")
        print(f"   Precision: {best['precision']:.4f}")
        print(f"   F1: {best['f1']:.4f}")

    # Находим порог с максимальным F1
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    print(f"\nЛучший порог по F1:")
    print(f"   Порог: {best_f1['threshold']:.2f}")
    print(f"   Recall: {best_f1['recall']:.4f}")
    print(f"   Precision: {best_f1['precision']:.4f}")
    print(f"   F1: {best_f1['f1']:.4f}")

    print("=" * 60)

    return df_results


if __name__ == "__main__":
    find_optimal_threshold()