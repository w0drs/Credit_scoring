import streamlit as st
import requests
import logging

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"


def render():
    st.markdown("### 📝 Новая заявка на кредит")
    st.caption("Заполните все поля формы для подачи заявки")

    with st.form("application_form", clear_on_submit=True):
        # Личные данные
        st.markdown("#### 👤 Личные данные")
        col1, col2, col3 = st.columns(3)
        with col1:
            first_name = st.text_input("Имя*", placeholder="Иван")
        with col2:
            last_name = st.text_input("Фамилия*", placeholder="Иванов")
        with col3:
            middle_name = st.text_input("Отчество", placeholder="Иванович")

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Пол", ["", "M", "F"])
        with col2:
            age = st.number_input("Возраст*", min_value=18, max_value=100, step=1)
        with col3:
            education = st.selectbox("Образование", ["", "SCH", "GRD", "UGR", "PGR", "ACD"])

        # Финансовые данные
        st.markdown("#### 💰 Финансовые данные")
        col1, col2, col3 = st.columns(3)
        with col1:
            income = st.number_input("Доход (руб)*", min_value=0, step=10000)
        with col2:
            car_own = st.selectbox("Наличие авто", ["", "Y", "N"])
        with col3:
            car_type = st.selectbox("Иномарка", ["", "Y", "N"])

        # Кредитная история
        st.markdown("#### 📊 Кредитная история")
        col1, col2, col3 = st.columns(3)
        with col1:
            appl_rej_cnt = st.number_input("Отказов по заявкам", min_value=0, max_value=20, step=1)
        with col2:
            out_request_cnt = st.number_input("Запросов в БКИ", min_value=0, max_value=50, step=1)
        with col3:
            good_work = st.selectbox("Хорошая работа", ["", 0, 1], format_func=lambda x: "Да" if x == 1 else "Нет" if x == 0 else "")

        col1, col2, col3 = st.columns(3)
        with col1:
            sna = st.number_input("SNA", min_value=0, max_value=10, step=1)
        with col2:
            first_time = st.number_input("First time CD", min_value=0, max_value=10, step=1)
        with col3:
            air_flg = st.selectbox("Загранпаспорт", ["", "Y", "N"])

        # Данные заявки
        st.markdown("#### 💳 Данные заявки")
        col1, col2 = st.columns(2)
        with col1:
            requested_amount = st.number_input("Сумма кредита (руб)*", min_value=10000, step=50000)
        with col2:
            requested_term = st.selectbox("Срок кредита (мес)", [6, 12, 24, 36, 48, 60])

        purpose = st.text_area("Цель кредита", placeholder="Например: Покупка автомобиля")

        # Остальные поля (скрытые или с дефолтными значениями)
        st.markdown("#### 📍 Дополнительные данные")
        col1, col2, col3 = st.columns(3)
        with col1:
            home_address_cd = st.number_input("Код адреса", min_value=0, max_value=100, step=1, value=1)
        with col2:
            work_address_cd = st.number_input("Код работы", min_value=0, max_value=100, step=1, value=1)
        with col3:
            region_rating = st.number_input("Рейтинг региона", min_value=0, max_value=100, step=1, value=50)

        # Score_bki (скрытое поле)
        score_bki = st.number_input("Score BKI", value=0.0, step=0.01, format="%.2f", help="Балл из БКИ")

        st.markdown("---")
        submitted = st.form_submit_button("📤 Отправить заявку", type="primary", use_container_width=True)

        if submitted:
            # Валидация
            errors = []
            if not first_name:
                errors.append("Имя обязательно")
            if not last_name:
                errors.append("Фамилия обязательна")
            if age < 18:
                errors.append("Возраст должен быть не менее 18 лет")
            if income <= 0:
                errors.append("Доход должен быть больше 0")
            if requested_amount <= 0:
                errors.append("Сумма кредита должна быть больше 0")

            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                return

            # Формируем данные для отправки
            form_data = {
                "first_name": first_name,
                "last_name": last_name,
                "middle_name": middle_name or None,
                "gender_cd": gender or None,
                "age": age,
                "income": income,
                "education_cd": education or None,
                "car_own_flg": car_own or None,
                "car_type_flg": car_type or None,
                "home_address_cd": home_address_cd,
                "work_address_cd": work_address_cd,
                "Score_bki": score_bki,
                "appl_rej_cnt": appl_rej_cnt,
                "out_request_cnt": out_request_cnt,
                "good_work_flg": good_work,
                "SNA": sna,
                "first_time_cd": first_time,
                "region_rating": region_rating,
                "Air_flg": air_flg or None,
                "requested_amount": requested_amount,
                "requested_term": requested_term,
                "purpose": purpose or None,
            }

            with st.spinner("⏳ Отправка заявки..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/api/v1/predict/demo",
                        json=form_data,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Заявка успешно отправлена!")

                        # Показываем результат
                        st.markdown("---")
                        st.markdown("### 📊 Результат")

                        col1, col2 = st.columns(2)
                        with col1:
                            status_color = "green" if result.get("decision") == "approved" else "red"
                            st.markdown(f"""
                                <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; text-align: center;">
                                    <h3 style="color: white;">{'✅ ОДОБРЕНО' if result.get('decision') == 'approved' else '❌ ОТКАЗАНО'}</h3>
                                    <p style="color: white;">Заявка #{result.get('application_id')}</p>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.metric("Вероятность", f"{result.get('probability', 0):.2%}")
                            st.metric("Время обработки", f"{result.get('processing_time_ms', 0):.0f} мс")

                        with st.expander("📋 Детали заявки"):
                            st.json(result)

                    else:
                        st.error(f"❌ Ошибка при отправке заявки: {response.status_code}")
                        st.json(response.json())

                except requests.exceptions.ConnectionError:
                    st.error("❌ Не удалось подключиться к серверу. Убедитесь, что бэкенд запущен.")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    logger.exception("Ошибка при отправке заявки")