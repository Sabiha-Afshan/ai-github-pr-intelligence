"""RAG knowledge-base page."""

import streamlit as st

st.set_page_config(
    page_title="📚 RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
)

st.title("RAG Knowledge Base")

st.info(
    "This page will display indexed repository documents, "
    "retrieved chunks, similarity scores and source citations."
)
