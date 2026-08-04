"""Streamlit application router."""

from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()

st.set_page_config(
    page_title=settings.app_name,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Overview": [
        st.Page(
            "pages/00_Executive_Overview.py",
            title="Executive Overview",
            icon=":material/dashboard:",
            default=True,
        ),
    ],
    "Analysis": [
        st.Page(
            "pages/01_Data_and_PR_Explorer.py",
            title="Data & PR Explorer",
            icon=":material/table_view:",
        ),
        st.Page(
            "pages/02_PR_Intelligence.py",
            title="PR Intelligence",
            icon=":material/insights:",
        ),
    ],
}

navigation = st.navigation(
    pages,
    position="sidebar",
)

logger.info(
    "Streamlit application router loaded. Selected page: %s",
    navigation.title,
)

navigation.run()