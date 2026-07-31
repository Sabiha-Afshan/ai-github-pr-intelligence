"""AI-governance page."""

import streamlit as st

st.set_page_config(
    page_title="⚖️ AI Governance",
    page_icon="⚖️",
    layout="wide",
)

st.title("AI Governance")

st.info(
    "This page will explain model selection, fairness, "
    "leakage controls, agent restrictions and evidence rules."
)

st.markdown(
    """
    ### Current governance decisions

    - The primary merge model excludes author association.
    - Predictions are decision-support signals.
    - Repository content is treated as untrusted input.
    - Agent outputs require evidence verification.
    - The agent cannot perform GitHub write actions.
    """
)
