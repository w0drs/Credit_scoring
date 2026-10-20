import streamlit as st
import requests

BACKEND_URL = "http://host.docker.internal:8000"


st.set_page_config(page_title="Кредитный скоринг", layout="centered")

# Заголовок
# st.title("🏦 Проверка кредитного скоринга")
st.title("🐋 Проверка на deepseek")

# Форма ввода
with st.form("scoring_form"):
    col1, col2 = st.columns(2)

    with col1:
        full_name = st.text_input("ФИО клиента")

    with col2:
        client_id = st.number_input("ID клиента", min_value=1, step=1)

    submitted = st.form_submit_button("Проверить", type="primary")

model_info: dict | None = None
# Обработка формы
if submitted:
    if not full_name:
        st.warning("Введите ФИО клиента")
    elif not client_id:
        st.warning("Введите ID клиента")
    else:
        with st.spinner("Проверяем данные..."):
            try:
                features = {
                    "full_name": full_name,
                    "id": client_id
                }
                response = requests.post(f"{BACKEND_URL}/predict", json=features)
                if response.json().get("error", None) is not None:
                    st.error(f"Клиент с ID {client_id} не найден")
                else:
                    predict = response.json().get('prediction', None)
                    predict_prob = response.json().get('predict_proba', None)
                    client_data = response.json().get('client_data', None)
                    processing_time = response.json().get('processing_time_ms', None)
                    model_info = response.json().get('model_info', None)

                    # Результат
                    st.markdown("---")
                    st.subheader("Результат проверки")

                    if predict == 0:
                        st.success(f"✅ Кредит одобрен для {full_name}")
                        st.balloons()
                    elif predict == 1:
                        st.error(f"❌ Кредит отклонен для {full_name}")
                    else:
                        st.error(f"Ошибка. Пользователя {full_name} с id равным {client_id} нет в базе")

                    st.metric(
                        label="Вероятность одобрения",
                        value=f"Predict: {predict}",
                        delta=f"Predict_proba: {predict_prob}"
                    )

                    # Дополнительно
                    with st.expander("Детали"):
                        if client_data is not None:
                            st.write(f"**ФИО:** {full_name}")
                            for key, value in client_data.items():
                                st.write(f"**{key}:** {value}")
                            st.write(f"**Решение:** {'Одобрено' if predict == 0 else 'Отклонено'}")
                            st.write(f"**Уверенность модели:** {predict_prob}")
                            st.write(f"**Время отклика модели:** {processing_time}")
                        else:
                            st.error(f"Ошибка. Пользователя {full_name} с id равным {client_id} нет в базе")

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

# Информация в сайдбаре
with st.sidebar:
    st.header("📊 Статистика")
    if model_info is not None:
        st.info(f"""
            **Текущая модель:**
            - Название: {model_info["model_name"]}
            - Дата обучения: {model_info["creation_date"]}
            - Версия: {model_info["model_version"]}
            - AUC: 
                - train auc: {model_info["metrics"]["train_roc_auc"]}
                - val auc: {model_info["metrics"]["val_roc_auc"]}
                - val auc: {model_info["metrics"]["test_roc_auc"]}
            - Обучена на 200k+ заявок
        """)
    else:
        st.info(f"""
            **Модель еще не загружена**
        """)
