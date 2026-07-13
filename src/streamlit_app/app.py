import streamlit as st
from src.streamlit_app.pages import create_application, admin_dashboard

st.set_page_config(
    page_title="Кредитный скоринг",
    page_icon="🏦",
    layout="wide"
)

# Стили
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<p class="main-header">🏦 Система кредитного скоринга</p>', unsafe_allow_html=True)

# Создание вкладок
tab1, tab2 = st.tabs(["📝 Создание заявки", "📊 Панель администратора"])

with tab1:
    create_application.render()

with tab2:
    admin_dashboard.render()