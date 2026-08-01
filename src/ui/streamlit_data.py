"""Shared Streamlit data and display helpers."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNIFIED_INTELLIGENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_pr_intelligence.csv"
)

CANONICAL_POPULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pallets_flask_canonical_600_pr_population.csv"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "reports"
)


def apply_global_page_style() -> None:
    """Apply a consistent, restrained visual style."""

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.20);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            padding: 0.8rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 8px;
            overflow: hidden;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_project_sidebar(
    settings: Any,
    current_page: str,
) -> None:
    """Render project metadata without duplicating page content."""

    with st.sidebar:
        st.header("Project")
        st.write(f"**Page:** {current_page}")
        st.write(f"**Application:** {settings.app_name}")
        st.write(
            f"**Repository:** "
            f"{settings.github_repository_full_name}"
        )
        st.write(f"**Environment:** {settings.app_env}")
        st.divider()
        st.caption(
            "Read-only decision support. No autonomous GitHub writes."
        )


@st.cache_data(show_spinner=False)
def load_unified_intelligence() -> pd.DataFrame:
    """Load the Stage 7B unified intelligence dataset."""

    if UNIFIED_INTELLIGENCE_PATH.exists():
        return pd.read_csv(
            UNIFIED_INTELLIGENCE_PATH,
            low_memory=False,
        )

    if CANONICAL_POPULATION_PATH.exists():
        return pd.read_csv(
            CANONICAL_POPULATION_PATH,
            low_memory=False,
        )

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv(
    relative_path: str,
) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def load_json(
    relative_path: str,
) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        return payload

    return {"value": payload}


def available_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """Return the first available column from an ordered alias list."""

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def numeric_series(
    series: pd.Series,
) -> pd.Series:
    """Convert numbers and percentage strings to numeric values."""

    text = series.astype(str).str.strip()

    contains_percent = text.str.contains(
        "%",
        regex=False,
        na=False,
    ).any()

    numeric = pd.to_numeric(
        text.str.replace("%", "", regex=False),
        errors="coerce",
    )

    if contains_percent:
        numeric = numeric / 100.0

    return numeric


def boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Normalise common Boolean, binary and outcome representations."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .fillna(0)
            .astype(float)
            .ne(0)
        )

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "merged",
            "delayed",
            "required",
            "positive",
        }
    )


def safe_divide(
    numerator: float | int,
    denominator: float | int,
) -> float:
    if not denominator:
        return 0.0

    return float(numerator) / float(denominator)


def safe_text(
    value: Any,
    default: str = "Not available",
) -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return text if text else default


def safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    try:
        text = str(value).strip().replace("%", "")
        number = float(text)

        if "%" in str(value):
            return number / 100.0

        return number
    except (TypeError, ValueError):
        return default


def parse_list_value(
    value: Any,
) -> list[Any]:
    """Parse Python-list strings, JSON arrays or delimited text."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return []

    for parser in (
        json.loads,
        ast.literal_eval,
    ):
        try:
            parsed = parser(text)

            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, tuple):
                return list(parsed)
        except (
            json.JSONDecodeError,
            SyntaxError,
            ValueError,
            TypeError,
        ):
            continue

    delimiter = "|" if "|" in text else ","

    return [
        item.strip()
        for item in text.split(delimiter)
        if item.strip()
    ]


def select_pr_number_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return available_column(
        dataframe,
        ["pr_number", "number", "pull_request_number"],
    )


def select_title_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return available_column(
        dataframe,
        ["title", "pr_title"],
    )


def select_repository_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return available_column(
        dataframe,
        ["repository", "repository_full_name", "repo"],
    )


def select_created_column(
    dataframe: pd.DataFrame,
) -> str | None:
    return available_column(
        dataframe,
        ["created_at", "created_time", "pr_created_at"],
    )


def format_probability(
    value: Any,
) -> str:
    number = safe_number(value)

    if number > 1:
        number = number / 100.0

    return f"{number:.1%}"


def format_score(
    value: Any,
    suffix: str = "",
) -> str:
    number = safe_number(value)

    if suffix:
        return f"{number:.2f}{suffix}"

    return f"{number:.2f}"


def extract_report_status(
    report: dict[str, Any],
) -> str:
    status = report.get("status")

    if status is None:
        status = report.get(
            "validation_passed"
        )

    if isinstance(status, bool):
        return "PASSED" if status else "FAILED"

    return str(status or "Not available").upper()


def flatten_metrics(
    payload: dict[str, Any],
) -> dict[str, float]:
    """Recursively extract numeric scalar fields from a JSON report."""

    flattened: dict[str, float] = {}

    def visit(
        value: Any,
        prefix: str,
    ) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                child_prefix = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )

                visit(
                    nested_value,
                    child_prefix,
                )

        elif isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        ):
            flattened[prefix] = float(value)

    visit(payload, "")

    return flattened
