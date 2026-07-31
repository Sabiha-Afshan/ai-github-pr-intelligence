"""Streamlit exploratory-analysis page."""

from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.eda_charts import (
    build_all_eda_charts,
)
from src.analytics.eda_comparisons import (
    calculate_complexity_summary,
    calculate_file_category_metrics,
    calculate_lifecycle_bands,
    calculate_outcome_comparisons,
    calculate_outlier_summary,
)
from src.analytics.eda_metrics import (
    calculate_missing_value_summary,
    calculate_outcome_metrics,
    calculate_quarterly_metrics,
    calculate_size_band_metrics,
    calculate_summary_metrics,
)
from src.data.eda_loader import (
    describe_eda_dataset,
    load_eda_dataset,
)

METRIC_LABELS = {
    "total_prs": "Total PRs",
    "merged_prs": "Merged PRs",
    "unmerged_prs": "Unmerged PRs",
    "merge_rate_percent": "Merge Rate",
    "median_total_changes": "Median Changes",
    "median_changed_files": "Median Files Changed",
    "median_commit_count": "Median Commits",
    "median_lifecycle_days": "Median Lifecycle",
}


def format_integer(
    value: Any,
) -> str:
    """Format an integer-like dashboard value."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{int(round(float(value))):,}"


def format_decimal(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format a decimal dashboard value."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}"


def format_percentage(
    value: Any,
    decimals: int = 1,
) -> str:
    """Format a percentage dashboard value."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}%"


def format_duration_days(
    value: Any,
) -> str:
    """Format a duration measured in days."""

    if value is None or pd.isna(value):
        return "N/A"

    numeric_value = float(value)

    if numeric_value < 1 / 24:
        minutes = numeric_value * 24 * 60
        return f"{minutes:.1f} min"

    if numeric_value < 1:
        hours = numeric_value * 24
        return f"{hours:.1f} hrs"

    return f"{numeric_value:.2f} days"


def build_filter_options(
    dataframe: pd.DataFrame,
) -> dict[str, list[Any]]:
    """Build reusable Streamlit filter options."""

    options: dict[str, list[Any]] = {
        "outcomes": [],
        "years": [],
        "quarters": [],
        "size_bands": [],
    }

    if "merge_outcome" in dataframe.columns:
        options["outcomes"] = sorted(
            dataframe["merge_outcome"].dropna().astype(str).unique().tolist()
        )

    if "created_year" in dataframe.columns:
        options["years"] = sorted(
            dataframe["created_year"].dropna().astype(int).unique().tolist()
        )

    if "created_quarter" in dataframe.columns:
        options["quarters"] = sorted(
            dataframe["created_quarter"].dropna().astype(str).unique().tolist()
        )

    if "change_size_band" in dataframe.columns:
        options["size_bands"] = [
            str(value)
            for value in dataframe["change_size_band"].dropna().unique().tolist()
        ]

    return options


def apply_eda_filters(
    dataframe: pd.DataFrame,
    selected_outcomes: list[str] | None = None,
    selected_years: list[int] | None = None,
    selected_quarters: list[str] | None = None,
    selected_size_bands: list[str] | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters to an EDA dataframe."""

    filtered = dataframe.copy()

    if selected_outcomes:
        filtered = filtered[
            filtered["merge_outcome"].astype(str).isin(selected_outcomes)
        ]

    if selected_years:
        filtered = filtered[filtered["created_year"].isin(selected_years)]

    if selected_quarters:
        filtered = filtered[
            filtered["created_quarter"].astype(str).isin(selected_quarters)
        ]

    if selected_size_bands:
        filtered = filtered[
            filtered["change_size_band"].astype(str).isin(selected_size_bands)
        ]

    return filtered.reset_index(drop=True)


def build_eda_outputs(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Build all tables and charts required by the EDA page."""

    summary_metrics = calculate_summary_metrics(dataframe)

    outcome_metrics = calculate_outcome_metrics(dataframe)

    quarterly_metrics = calculate_quarterly_metrics(dataframe)

    size_band_metrics = calculate_size_band_metrics(dataframe)

    outcome_comparisons = calculate_outcome_comparisons(dataframe)

    outlier_summary = calculate_outlier_summary(dataframe)

    file_category_metrics = calculate_file_category_metrics(dataframe)

    lifecycle_metrics = calculate_lifecycle_bands(dataframe)

    complexity_summary = calculate_complexity_summary(dataframe)

    missing_value_summary = calculate_missing_value_summary(dataframe)

    figures = build_all_eda_charts(
        dataframe=dataframe,
        quarterly_metrics=quarterly_metrics,
        size_band_metrics=size_band_metrics,
        outcome_comparisons=outcome_comparisons,
        file_category_metrics=(file_category_metrics),
        lifecycle_metrics=(lifecycle_metrics),
    )

    return {
        "summary_metrics": (summary_metrics),
        "outcome_metrics": (outcome_metrics),
        "quarterly_metrics": (quarterly_metrics),
        "size_band_metrics": (size_band_metrics),
        "outcome_comparisons": (outcome_comparisons),
        "outlier_summary": (outlier_summary),
        "file_category_metrics": (file_category_metrics),
        "lifecycle_metrics": (lifecycle_metrics),
        "complexity_summary": (complexity_summary),
        "missing_value_summary": (missing_value_summary),
        "figures": figures,
    }


def render_summary_metrics(
    summary_metrics: dict[str, Any],
) -> None:
    """Render top-level EDA metrics."""

    first_row = st.columns(4)

    first_row[0].metric(
        METRIC_LABELS["total_prs"],
        format_integer(summary_metrics["total_prs"]),
    )

    first_row[1].metric(
        METRIC_LABELS["merged_prs"],
        format_integer(summary_metrics["merged_prs"]),
    )

    first_row[2].metric(
        METRIC_LABELS["unmerged_prs"],
        format_integer(summary_metrics["unmerged_prs"]),
    )

    first_row[3].metric(
        METRIC_LABELS["merge_rate_percent"],
        format_percentage(summary_metrics["merge_rate_percent"]),
    )

    second_row = st.columns(4)

    second_row[0].metric(
        METRIC_LABELS["median_total_changes"],
        format_decimal(
            summary_metrics["median_total_changes"],
            decimals=1,
        ),
    )

    second_row[1].metric(
        METRIC_LABELS["median_changed_files"],
        format_decimal(
            summary_metrics["median_changed_files"],
            decimals=1,
        ),
    )

    second_row[2].metric(
        METRIC_LABELS["median_commit_count"],
        format_decimal(
            summary_metrics["median_commit_count"],
            decimals=1,
        ),
    )

    second_row[3].metric(
        METRIC_LABELS["median_lifecycle_days"],
        format_duration_days(summary_metrics["median_lifecycle_days"]),
    )


def render_filter_sidebar(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Render sidebar controls and return filtered data."""

    options = build_filter_options(dataframe)

    st.sidebar.header("EDA Filters")

    selected_outcomes = st.sidebar.multiselect(
        "PR outcome",
        options=options["outcomes"],
        default=options["outcomes"],
    )

    selected_years = st.sidebar.multiselect(
        "Created year",
        options=options["years"],
        default=options["years"],
    )

    selected_size_bands = st.sidebar.multiselect(
        "Change-size band",
        options=options["size_bands"],
        default=options["size_bands"],
    )

    available_quarters = (
        dataframe.loc[
            dataframe["created_year"].isin(selected_years),
            "created_quarter",
        ]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
        if selected_years
        else options["quarters"]
    )

    selected_quarters = st.sidebar.multiselect(
        "Created quarter",
        options=available_quarters,
        default=available_quarters,
    )

    filtered = apply_eda_filters(
        dataframe=dataframe,
        selected_outcomes=(selected_outcomes),
        selected_years=(selected_years),
        selected_quarters=(selected_quarters),
        selected_size_bands=(selected_size_bands),
    )

    st.sidebar.caption(f"{len(filtered):,} of {len(dataframe):,} PRs selected")

    return filtered


def render_overview_tab(
    outputs: dict[str, Any],
) -> None:
    """Render overview charts and tables."""

    figures = outputs["figures"]

    first_column, second_column = st.columns(2)

    first_column.plotly_chart(
        figures["outcome_distribution"],
        use_container_width=True,
    )

    second_column.plotly_chart(
        figures["size_band_merge_rate"],
        use_container_width=True,
    )

    st.plotly_chart(
        figures["quarterly_volume"],
        use_container_width=True,
    )

    st.subheader("Outcome Summary")

    st.dataframe(
        outputs["outcome_metrics"],
        use_container_width=True,
        hide_index=True,
    )


def render_complexity_tab(
    outputs: dict[str, Any],
) -> None:
    """Render complexity and outlier analysis."""

    figures = outputs["figures"]

    first_column, second_column = st.columns(2)

    first_column.plotly_chart(
        figures["total_changes_boxplot"],
        use_container_width=True,
    )

    second_column.plotly_chart(
        figures["commit_count_boxplot"],
        use_container_width=True,
    )

    st.subheader("Complexity Percentiles")

    st.dataframe(
        outputs["complexity_summary"],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("IQR Outlier Summary")

    st.dataframe(
        outputs["outlier_summary"],
        use_container_width=True,
        hide_index=True,
    )


def render_outcome_tab(
    outputs: dict[str, Any],
) -> None:
    """Render merged-versus-unmerged comparisons."""

    st.plotly_chart(
        outputs["figures"]["outcome_effect_sizes"],
        use_container_width=True,
    )

    st.subheader("Outcome Comparison Table")

    st.dataframe(
        outputs["outcome_comparisons"],
        use_container_width=True,
        hide_index=True,
    )


def render_lifecycle_tab(
    outputs: dict[str, Any],
) -> None:
    """Render lifecycle analysis."""

    st.plotly_chart(
        outputs["figures"]["lifecycle_merge_rate"],
        use_container_width=True,
    )

    st.dataframe(
        outputs["lifecycle_metrics"],
        use_container_width=True,
        hide_index=True,
    )


def render_file_category_tab(
    outputs: dict[str, Any],
) -> None:
    """Render changed-file-category analysis."""

    st.plotly_chart(
        outputs["figures"]["file_category_merge_rate"],
        use_container_width=True,
    )

    st.dataframe(
        outputs["file_category_metrics"],
        use_container_width=True,
        hide_index=True,
    )


def render_data_quality_tab(
    dataframe: pd.DataFrame,
    outputs: dict[str, Any],
) -> None:
    """Render dataset metadata and missing values."""

    description = describe_eda_dataset(dataframe)

    metadata = pd.DataFrame(
        [
            {
                "metric": ("Rows"),
                "value": description["row_count"],
            },
            {
                "metric": ("Columns"),
                "value": description["column_count"],
            },
            {
                "metric": ("Unique PRs"),
                "value": description["unique_pr_count"],
            },
            {
                "metric": ("Duplicate rows"),
                "value": description["duplicate_row_count"],
            },
            {
                "metric": ("Total missing cells"),
                "value": description["missing_value_count"],
            },
            {
                "metric": ("First PR date"),
                "value": description["minimum_created_at"],
            },
            {
                "metric": ("Latest PR date"),
                "value": description["maximum_created_at"],
            },
        ]
    )

    st.dataframe(
        metadata,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing Values by Column")

    missing_summary = outputs["missing_value_summary"]

    st.dataframe(
        missing_summary[missing_summary["missing_count"] > 0],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Missing merged-at values are expected for pull requests that were not merged."
    )


def render_eda_page() -> None:
    """Render the complete Streamlit EDA page."""

    st.title("GitHub Pull Request Exploratory Analysis")

    st.caption("Interactive analysis of the authoritative 600-PR Flask dataset.")

    try:
        dataframe = load_eda_dataset()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    filtered_dataframe = render_filter_sidebar(dataframe)

    if filtered_dataframe.empty:
        st.warning("No pull requests match the selected filters.")
        st.stop()

    outputs = build_eda_outputs(filtered_dataframe)

    render_summary_metrics(outputs["summary_metrics"])

    st.divider()

    (
        overview_tab,
        complexity_tab,
        outcome_tab,
        lifecycle_tab,
        file_category_tab,
        quality_tab,
    ) = st.tabs(
        [
            "Overview",
            "Complexity & Outliers",
            "Outcome Comparison",
            "Lifecycle",
            "File Categories",
            "Data Quality",
        ]
    )

    with overview_tab:
        render_overview_tab(outputs)

    with complexity_tab:
        render_complexity_tab(outputs)

    with outcome_tab:
        render_outcome_tab(outputs)

    with lifecycle_tab:
        render_lifecycle_tab(outputs)

    with file_category_tab:
        render_file_category_tab(outputs)

    with quality_tab:
        render_data_quality_tab(
            filtered_dataframe,
            outputs,
        )

    st.divider()

    st.caption(
        "EDA statistics are descriptive associations "
        "and should not be interpreted as causal effects."
    )
