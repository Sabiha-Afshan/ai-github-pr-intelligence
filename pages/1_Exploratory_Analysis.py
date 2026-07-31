"""Streamlit exploratory-analysis page entry point."""

import streamlit as st

from src.ui.eda_page import (
    render_eda_page,
)

st.set_page_config(
    page_title=("Exploratory Analysis | GitHub PR Intelligence"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_eda_page()
