"""Pull-request intelligence page."""

import streamlit as st

st.set_page_config(
    page_title="🧠 PR Intelligence",
    page_icon="🧠",
    layout="wide",
)

st.title("PR Intelligence")

st.info(
    "This page will combine ML predictions, risk rules, "
    "RAG evidence, LLM analysis and the final agent report."
)

st.warning("No pull request is currently selected.")
