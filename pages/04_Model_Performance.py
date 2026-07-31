"""Model-performance page."""

import streamlit as st

st.set_page_config(
    page_title="📈 Model Performance",
    page_icon="📈",
    layout="wide",
)

st.title("Model Performance")

st.info(
    "This page will display model comparison, "
    "threshold analysis, explainability, fairness "
    "and error analysis."
)
