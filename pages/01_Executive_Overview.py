"""Executive overview page."""

import streamlit as st

st.set_page_config(
    page_title="📊 Executive Overview",
    page_icon="📊",
    layout="wide",
)

st.title("Executive Overview")

st.info(
    "This page will present repository level KPIs, "
    "outcome patterns, delay trends and risk indicators."
)

metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "PRs analysed",
        "Pending",
    )

with metric_columns[1]:
    st.metric(
        "Merge rate",
        "Pending",
    )

with metric_columns[2]:
    st.metric(
        "Delayed PR rate",
        "Pending",
    )

with metric_columns[3]:
    st.metric(
        "High-risk PRs",
        "Pending",
    )
