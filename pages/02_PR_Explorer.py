"""Pull-request explorer page."""

import streamlit as st

st.set_page_config(
    page_title="🔎 PR Explorer",
    page_icon="🔎",
    layout="wide",
)

st.title("Pull Request Explorer")

st.info(
    "This page will allow users to search, filter and select historical pull requests."
)

search_text = st.text_input("Search by PR number or title")

st.write(
    "Current search:",
    search_text or "No search entered",
)
