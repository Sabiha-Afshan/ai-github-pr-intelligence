"""Tests for reusable EDA charts."""

import pandas as pd
import plotly.graph_objects as go

from src.analytics.eda_charts import (
    build_all_eda_charts,
    create_effect_size_chart,
    create_file_category_chart,
    create_lifecycle_merge_rate_chart,
    create_log_complexity_boxplot,
    create_outcome_distribution_chart,
    create_quarterly_volume_chart,
    create_size_band_merge_rate_chart,
    describe_chart_collection,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a small prepared PR dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "merge_outcome": [
                "Merged",
                "Unmerged",
                "Merged",
                "Unmerged",
            ],
            "was_merged_numeric": [
                1,
                0,
                1,
                0,
            ],
            "total_changes": [
                10,
                20,
                100,
                1000,
            ],
            "commit_count": [
                1,
                2,
                3,
                20,
            ],
        }
    )


def create_quarterly_metrics() -> pd.DataFrame:
    """Create sample quarterly metrics."""

    return pd.DataFrame(
        {
            "period": [
                "2024Q1",
                "2024Q2",
            ],
            "total_prs": [
                2,
                2,
            ],
            "merged_prs": [
                1,
                1,
            ],
            "unmerged_prs": [
                1,
                1,
            ],
            "merge_rate_percent": [
                50.0,
                50.0,
            ],
        }
    )


def create_size_band_metrics() -> pd.DataFrame:
    """Create sample size-band metrics."""

    return pd.DataFrame(
        {
            "change_size_band": [
                "Very small",
                "Small",
                "Medium",
                "Large",
                "Very large",
            ],
            "total_prs": [
                10,
                10,
                10,
                10,
                10,
            ],
            "merged_prs": [
                5,
                6,
                7,
                8,
                4,
            ],
            "unmerged_prs": [
                5,
                4,
                3,
                2,
                6,
            ],
            "median_total_changes": [
                5,
                40,
                200,
                900,
                5000,
            ],
            "merge_rate_percent": [
                50.0,
                60.0,
                70.0,
                80.0,
                40.0,
            ],
        }
    )


def create_outcome_comparisons() -> pd.DataFrame:
    """Create sample effect-size data."""

    return pd.DataFrame(
        {
            "metric": [
                "total_changes",
                "commit_count",
            ],
            "standardized_mean_difference": [
                -0.5,
                0.3,
            ],
            "absolute_effect_size": [
                0.5,
                0.3,
            ],
            "effect_size_label": [
                "Moderate",
                "Small",
            ],
        }
    )


def create_file_category_metrics() -> pd.DataFrame:
    """Create sample file-category data."""

    return pd.DataFrame(
        {
            "file_category": [
                "Tests",
                "Documentation",
            ],
            "merge_rate_with_category": [
                60.0,
                70.0,
            ],
            "merge_rate_without_category": [
                45.0,
                40.0,
            ],
        }
    )


def create_lifecycle_metrics() -> pd.DataFrame:
    """Create sample lifecycle data."""

    return pd.DataFrame(
        {
            "lifecycle_band": [
                "Under 1 hour",
                "1-24 hours",
                "1-7 days",
                "8-30 days",
                "31-90 days",
                "Over 90 days",
            ],
            "total_prs": [
                10,
                10,
                10,
                10,
                10,
                10,
            ],
            "merge_rate_percent": [
                40.0,
                45.0,
                55.0,
                60.0,
                65.0,
                50.0,
            ],
        }
    )


def test_individual_chart_functions() -> None:
    """Confirm every chart returns a Plotly figure."""

    dataframe = create_sample_dataframe()

    figures = [
        create_outcome_distribution_chart(dataframe),
        create_quarterly_volume_chart(create_quarterly_metrics()),
        create_size_band_merge_rate_chart(create_size_band_metrics()),
        create_lifecycle_merge_rate_chart(create_lifecycle_metrics()),
        create_file_category_chart(create_file_category_metrics()),
        create_effect_size_chart(create_outcome_comparisons()),
        create_log_complexity_boxplot(
            dataframe,
            metric="total_changes",
        ),
    ]

    assert all(
        isinstance(
            figure,
            go.Figure,
        )
        for figure in figures
    )

    assert all(len(figure.data) > 0 for figure in figures)


def test_chart_collection() -> None:
    """Confirm the full chart collection is created."""

    figures = build_all_eda_charts(
        dataframe=(create_sample_dataframe()),
        quarterly_metrics=(create_quarterly_metrics()),
        size_band_metrics=(create_size_band_metrics()),
        outcome_comparisons=(create_outcome_comparisons()),
        file_category_metrics=(create_file_category_metrics()),
        lifecycle_metrics=(create_lifecycle_metrics()),
    )

    assert len(figures) == 8

    description = describe_chart_collection(figures)

    assert description["chart_count"] == 8

    assert description["all_figures_valid"] is True

    assert all(trace_count > 0 for trace_count in description["trace_counts"].values())
