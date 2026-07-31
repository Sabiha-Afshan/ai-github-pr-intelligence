"""Tests for Streamlit EDA-page utilities."""

import pandas as pd

from src.ui.eda_page import (
    apply_eda_filters,
    build_eda_outputs,
    build_filter_options,
    format_decimal,
    format_duration_days,
    format_integer,
    format_percentage,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a prepared EDA dataframe."""

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
            "created_at": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-02-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                ],
                utc=True,
            ),
            "created_year": [
                2024,
                2024,
                2025,
                2025,
            ],
            "created_quarter": [
                "2024Q1",
                "2024Q1",
                "2025Q1",
                "2025Q1",
            ],
            "change_size_band": [
                "Very small",
                "Small",
                "Medium",
                "Large",
            ],
            "total_changes": [
                10,
                50,
                200,
                1000,
            ],
            "additions": [
                7,
                30,
                150,
                800,
            ],
            "deletions": [
                3,
                20,
                50,
                200,
            ],
            "changed_files": [
                1,
                2,
                5,
                10,
            ],
            "commit_count": [
                1,
                2,
                3,
                4,
            ],
            "comments": [
                0,
                1,
                2,
                3,
            ],
            "review_comments": [
                0,
                1,
                1,
                2,
            ],
            "lifecycle_days": [
                0.01,
                0.5,
                5.0,
                50.0,
            ],
            "title_length": [
                20,
                30,
                40,
                50,
            ],
            "body_length": [
                100,
                200,
                300,
                400,
            ],
            "label_count": [
                0,
                1,
                0,
                1,
            ],
            "requested_reviewer_count": [
                0,
                0,
                0,
                0,
            ],
            "test_files_changed": [
                1,
                0,
                1,
                0,
            ],
            "documentation_files_changed": [
                0,
                1,
                0,
                1,
            ],
            "configuration_files_changed": [
                0,
                0,
                1,
                1,
            ],
            "security_sensitive_files_changed": [
                0,
                0,
                0,
                1,
            ],
        }
    )


def test_dashboard_formatters() -> None:
    """Confirm dashboard values are formatted."""

    assert format_integer(1234) == "1,234"

    assert (
        format_decimal(
            12.345,
            decimals=2,
        )
        == "12.35"
    )

    assert format_percentage(50.0) == "50.0%"

    assert format_duration_days(0.5) == "12.0 hrs"

    assert format_duration_days(2) == "2.00 days"


def test_filter_options() -> None:
    """Confirm filter options are generated."""

    dataframe = create_sample_dataframe()

    options = build_filter_options(dataframe)

    assert options["outcomes"] == [
        "Merged",
        "Unmerged",
    ]

    assert options["years"] == [
        2024,
        2025,
    ]

    assert options["quarters"] == [
        "2024Q1",
        "2025Q1",
    ]


def test_apply_eda_filters() -> None:
    """Confirm multiple filters are applied."""

    dataframe = create_sample_dataframe()

    filtered = apply_eda_filters(
        dataframe=dataframe,
        selected_outcomes=[
            "Merged",
        ],
        selected_years=[
            2025,
        ],
        selected_quarters=[
            "2025Q1",
        ],
        selected_size_bands=[
            "Medium",
        ],
    )

    assert len(filtered) == 1

    assert (
        int(
            filtered.loc[
                0,
                "pr_number",
            ]
        )
        == 3
    )


def test_build_eda_outputs() -> None:
    """Confirm all dashboard outputs are built."""

    dataframe = create_sample_dataframe()

    outputs = build_eda_outputs(dataframe)

    required_outputs = {
        "summary_metrics",
        "outcome_metrics",
        "quarterly_metrics",
        "size_band_metrics",
        "outcome_comparisons",
        "outlier_summary",
        "file_category_metrics",
        "lifecycle_metrics",
        "complexity_summary",
        "missing_value_summary",
        "figures",
    }

    assert required_outputs.issubset(outputs)

    assert len(outputs["figures"]) == 8

    assert outputs["summary_metrics"]["total_prs"] == 4
