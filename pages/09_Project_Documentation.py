"""Project documentation page."""

import streamlit as st

st.set_page_config(
    page_title="📄 Project Documentation",
    page_icon="📄",
    layout="wide",
)

st.title("Project Documentation")

st.markdown(
    """
    This page will explain:

    - the business problem;
    - project architecture;
    - data collection;
    - feature engineering;
    - model selection;
    - LLM and RAG design;
    - governed agentic RAG;
    - evaluation;
    - limitations;
    - future improvements.
    """
)
