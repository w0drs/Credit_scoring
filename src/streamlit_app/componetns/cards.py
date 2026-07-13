import streamlit as st


def status_card(status: str, title: str, value: str, icon: str = "📊"):
    """Карточка со статусом"""
    colors = {
        'success': '#2ca02c',
        'error': '#d62728',
        'warning': '#ffa500',
        'info': '#1f77b4'
    }
    color = colors.get(status, '#808080')

    st.markdown(f"""
        <div style="
            background-color: {color}20;
            border-left: 4px solid {color};
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        ">
            <div style="font-size: 2rem; float: left; margin-right: 10px;">{icon}</div>
            <div style="font-size: 0.9rem; color: #666;">{title}</div>
            <div style="font-size: 1.2rem; font-weight: 600;">{value}</div>
        </div>
    """, unsafe_allow_html=True)