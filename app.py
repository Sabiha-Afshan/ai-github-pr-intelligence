"""Main Streamlit entry point."""

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

logger.info("Streamlit application landing page loaded.")


st.title("AI GitHub Pull Request Intelligence")

st.subheader(
    "Governed machine learning, LLM, RAG and agentic RAG for pull request analysis"
)

st.info(
    "It demonstrates AI assisted pull request analysis "
    "and does not perform autonomous GitHub actions."
)


st.markdown(
    """
    ## Project capabilities

    This application combines:

    - merge-outcome prediction;
    - merge-delay prediction;
    - explainable machine learning;
    - fairness and shortcut analysis;
    - pull-request risk scoring;
    - test-gap detection;
    - security-review recommendations;
    - repository-policy checks;
    - local LLM-generated summaries;
    - repository-document RAG;
    - governed agentic RAG;
    - evidence verification;
    - AI evaluation and governance.
    """
)


st.markdown("## Project workflow")

workflow_columns = st.columns(5)

with workflow_columns[0]:
    st.metric(
        label="1. Data",
        value="GitHub PRs",
    )

with workflow_columns[1]:
    st.metric(
        label="2. ML",
        value="Predictions",
    )

with workflow_columns[2]:
    st.metric(
        label="3. Rules",
        value="Risk checks",
    )

with workflow_columns[3]:
    st.metric(
        label="4. RAG + LLM",
        value="Evidence",
    )

with workflow_columns[4]:
    st.metric(
        label="5. Agent",
        value="PR report",
    )


st.markdown(
    """
    ## Responsible-use principle

    The system provides decision-support recommendations.

    It does not:

    - merge pull requests;
    - close pull requests;
    - approve or reject code;
    - post GitHub comments automatically;
    - replace maintainers or reviewers.
    """
)


with st.sidebar:
    st.header("Project")

    st.write(f"**Application:** {settings.app_name}")

    st.write(f"**Repository:** {settings.github_repository_full_name}")

    st.write(f"**Environment:** {settings.app_env}")

    st.divider()

    st.caption("Use the navigation above to explore the project modules.")
