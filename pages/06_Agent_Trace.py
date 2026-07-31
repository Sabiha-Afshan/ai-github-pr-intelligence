"""Agent trace page."""

import streamlit as st

st.set_page_config(
    page_title="🧭 Agent Trace",
    page_icon="🧭",
    layout="wide",
)

st.title("Governed Agent Trace")

st.info(
    "This page will show each tool used by the agent, "
    "its result, execution time, evidence and errors."
)
