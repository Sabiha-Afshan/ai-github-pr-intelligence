"""Project documentation page."""

from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.ui.streamlit_data import (
    apply_global_page_style,
    render_project_sidebar,
)
from src.utils.logging import get_logger
from src.utils.paths import create_required_directories


settings = get_settings()
logger = get_logger(__name__)

create_required_directories()
apply_global_page_style()
render_project_sidebar(
    settings=settings,
    current_page="Project Documentation",
)

logger.info("Project Documentation page loaded.")

st.title("Project Documentation")
st.caption(
    "Architecture, methodology, technology stack, repository structure, "
    "execution instructions, governance and limitations."
)


st.markdown(
    """
    ## Problem statement

    Pull-request review requires maintainers to combine repository history,
    change evidence, review readiness, model predictions and policy checks.
    This project consolidates those signals into a governed decision-support
    workflow.

    The system analyses historical pull requests, estimates merge outcome and
    merge-delay risk, applies deterministic policy rules, creates a retrievable
    knowledge base and answers repository questions through an evidence-
    validated local LLM.
    """
)


st.markdown("## End-to-end architecture")

st.code(
    """
GitHub API and historical PR data
        ↓
Validation, reconciliation and feature engineering
        ↓
Merge-outcome model + merge-delay model
        ↓
Deterministic policy rules
        ↓
Unified PR intelligence
        ↓
Section-aware RAG knowledge base
        ↓
Sentence-transformer embeddings + FAISS
        ↓
Query understanding + governed hybrid retrieval
        ↓
Local Ollama LLM generation
        ↓
Sentence citation validation
        ↓
Claim-to-evidence groundedness validation
        ↓
Deterministic citation repair and revalidation
        ↓
Safe answer release or fail-closed abstention
    """,
    language=None,
)


st.markdown("## Core technologies")

technology_columns = st.columns(4)

technology_columns[0].markdown(
    """
    **Data and analytics**

    - Python
    - Pandas
    - NumPy
    - JSON and CSV artefacts
    """
)

technology_columns[1].markdown(
    """
    **Machine learning**

    - scikit-learn
    - contributor-neutral features
    - calibration and thresholds
    - explainability analysis
    """
)

technology_columns[2].markdown(
    """
    **LLM and RAG**

    - Ollama
    - qwen2.5-coder:3b
    - sentence-transformers
    - FAISS
    - hybrid retrieval
    """
)

technology_columns[3].markdown(
    """
    **Application and quality**

    - Streamlit
    - Pytest
    - Ruff
    - structured logging
    - Git and GitHub
    """
)


st.markdown("## Final application pages")

st.markdown(
    """
    1. **Executive Overview** — portfolio summary, KPIs and key outcomes.
    2. **Data & PR Explorer** — exploratory patterns, filtering and record inspection.
    3. **PR Intelligence** — predictions, policy risks and governed review action.
    4. **Model & System Evaluation** — ML, retrieval and production-AI evaluation.
    5. **AI PR Analyst** — natural-language questions with governed evidence.
    6. **Project Documentation** — architecture, setup, limitations and responsible use.
    """
)


st.markdown("## Repository structure")

st.code(
    """
AI GitHub PR Intelligence/
├── app.py
├── pages/
│   ├── 01_Data_and_PR_Explorer.py
│   ├── 02_PR_Intelligence.py
│   ├── 03_Model_and_System_Evaluation.py
│   ├── 04_AI_PR_Analyst.py
│   └── 05_Project_Documentation.py
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── governance/
│   ├── intelligence/
│   ├── models/
│   ├── policies/
│   ├── rag/
│   ├── ui/
│   └── utils/
├── scripts/
├── tests/
├── data/
│   ├── processed/
│   ├── reports/
│   ├── knowledge_base/
│   └── vector_store/
├── models/
├── policies/
├── README.md
├── ARCHITECTURE.md
├── GOVERNANCE.md
├── LIMITATIONS.md
└── MODEL_CARD.md
    """,
    language=None,
)


st.markdown("## Run locally")

st.code(
    """
# Activate the virtual environment
.\\.venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Confirm the local model
ollama list

# Pull the model when required
ollama pull qwen2.5-coder:3b

# Launch the application
streamlit run app.py
    """,
    language="powershell",
)


st.markdown("## Validated project outcomes")

outcome_columns = st.columns(4)

outcome_columns[0].metric(
    "PRs analysed",
    "600",
)

outcome_columns[1].metric(
    "Knowledge-base chunks",
    "3,986",
)

outcome_columns[2].metric(
    "Governed retrieval Hit@5",
    "100%",
)

outcome_columns[3].metric(
    "Safe production pipeline",
    "100%",
)

st.markdown(
    """
    The final production evaluation released three of four generated
    in-domain answers, safely withheld one unsupported answer, correctly
    abstained from an out-of-domain request and repaired one missing citation
    deterministically in milliseconds.
    """
)


st.markdown("## Responsible-use controls")

st.markdown(
    """
    The system:

    - is read-only;
    - requires retrieved evidence for factual answers;
    - validates sentence citations;
    - validates PR numbers, dates, percentages, decimals, Booleans and authors;
    - allows deterministic repair only when one unique evidence item supports
      the uncited sentence;
    - revalidates every repaired answer;
    - withholds unsupported, ambiguous or insufficiently cited outputs;
    - skips LLM generation for out-of-domain questions;
    - leaves all merge, approval and review decisions to maintainers.
    """
)


st.markdown("## Limitations")

st.markdown(
    """
    - The project uses historical data from the `pallets/flask` repository.
    - Predictions may not transfer directly to repositories with different
      engineering practices.
    - The local 3B model is practical but slower and less capable than larger
      hosted models.
    - Retrieval and validation reduce hallucination risk but cannot guarantee
      perfect semantic understanding.
    - The current system does not execute repository writes or autonomous
      actions.
    - Results should be interpreted as decision support, not as approval,
      rejection or merge authority.
    """
)
