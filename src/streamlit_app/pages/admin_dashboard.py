import streamlit as st
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"


def render():
    st.markdown("### 📊 Панель администратора")
    st.caption("Просмотр и управление заявками")

    # Загрузка списка заявок
    if st.button("🔄 Обновить список заявок", use_container_width=True):
        with st.spinner("Загрузка заявок..."):
            try:
                # Получаем список заявок (эндпоинт для получения всех заявок)
                response = requests.get(f"{BACKEND_URL}/api/v1/applications", timeout=10)
                if response.status_code == 200:
                    applications = response.json()
                    if applications:
                        df = pd.DataFrame(applications)
                        df['created_at'] = pd.to_datetime(df['created_at'])
                        df = df.sort_values('created_at', ascending=False)
                        st.session_state['applications_df'] = df
                        st.success(f"✅ Загружено {len(df)} заявок")
                    else:
                        st.info("Заявок пока нет")
                else:
                    st.error(f"Ошибка загрузки: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    if 'applications_df' not in st.session_state:
        st.info("Нажмите 'Обновить список заявок' для загрузки данных")
        return

    df = st.session_state['applications_df']

    # Фильтры
    st.markdown("#### 🔍 Фильтры")
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Статус",
            options=['pending', 'processed', 'approved', 'rejected'],
            default=[]
        )
    with col2:
        date_range = st.date_input("Период", value=[])
    with col3:
        search = st.text_input("Поиск по ID", placeholder="ID заявки")

    # Применяем фильтры
    filtered_df = df.copy()
    if status_filter:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    if search:
        filtered_df = filtered_df[filtered_df['id'].astype(str).str.contains(search)]

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) &
            (filtered_df['created_at'].dt.date <= end_date)
        ]

    # Отображение таблицы
    st.markdown("#### 📋 Список заявок")
    if filtered_df.empty:
        st.info("Нет заявок по заданным фильтрам")
        return

    # Выбираем колонки для отображения
    display_cols = ['id', 'client_id', 'application_dt', 'requested_amount', 'requested_term',
                    'status', 'decision', 'credit_score', 'created_at']

    # Добавляем статус с цветом
    def color_status(status):
        colors = {
            'pending': '🟡',
            'processed': '🔵',
            'approved': '🟢',
            'rejected': '🔴'
        }
        return colors.get(status, '⚪')

    filtered_df['status_display'] = filtered_df['status'].apply(color_status) + ' ' + filtered_df['status']

    # Отображаем таблицу
    st.dataframe(
        filtered_df[display_cols],
        column_config={
            "id": "ID",
            "client_id": "Клиент ID",
            "application_dt": "Дата заявки",
            "requested_amount": st.column_config.NumberColumn("Сумма", format="₽ %.0f"),
            "requested_term": "Срок (мес)",
            "status": "Статус",
            "decision": "Решение",
            "credit_score": st.column_config.NumberColumn("Score", format="%.2f"),
            "created_at": "Создана",
        },
        use_container_width=True,
        height=400
    )

    # Детальный просмотр заявки
    st.markdown("---")
    st.markdown("#### 📄 Детальный просмотр заявки")

    selected_id = st.selectbox(
        "Выберите ID заявки для просмотра",
        options=filtered_df['id'].tolist(),
        format_func=lambda x: f"Заявка #{x}"
    )

    if selected_id:
        application = filtered_df[filtered_df['id'] == selected_id].iloc[0].to_dict()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Информация о заявке**")
            st.json({
                "ID": application.get('id'),
                "Клиент ID": application.get('client_id'),
                "Дата": application.get('application_dt'),
                "Сумма": f"₽ {application.get('requested_amount', 0):,.0f}",
                "Срок": f"{application.get('requested_term', 0)} мес",
                "Статус": application.get('status'),
                "Решение": application.get('decision', 'Нет решения'),
            })

        with col2:
            st.markdown("**Результаты скоринга**")
            st.json({
                "Credit Score": application.get('credit_score'),
                "Вероятность": f"{application.get('credit_score', 0):.2%}" if application.get('credit_score') else "N/A",
            })

        # Кнопки управления
        if application.get('status') == 'pending':
            st.markdown("---")
            st.markdown("**Действия**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Одобрить заявку", use_container_width=True):
                    _update_application_status(selected_id, 'approved')
            with col2:
                if st.button("❌ Отклонить заявку", use_container_width=True):
                    _update_application_status(selected_id, 'rejected')


def _update_application_status(application_id: int, decision: str):
    """Обновление статуса заявки через API"""
    with st.spinner(f"Обновление статуса на {decision}..."):
        try:
            # В реальности здесь должен быть эндпоинт для обновления заявки
            # Пока просто показываем сообщение
            st.info(f"Заявка #{application_id} обновлена до {decision}")
            # Обновляем локальные данные
            if 'applications_df' in st.session_state:
                idx = st.session_state['applications_df'][
                    st.session_state['applications_df']['id'] == application_id
                ].index
                if not idx.empty:
                    st.session_state['applications_df'].loc[idx, 'status'] = 'processed'
                    st.session_state['applications_df'].loc[idx, 'decision'] = decision
                    st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")