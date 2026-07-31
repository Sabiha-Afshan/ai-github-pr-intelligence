"""Tests for reusable EDA metrics."""

import pandas as pd

from src.analytics.eda_metrics import (
    calculate_outcome_metrics,
    calculate_quarterly_metrics,
    calculate_summary_metrics,
    safe_percentage,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a small prepared EDA dataframe."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
            ],
            "was_merged_numeric": [
                1,
                0,
                1,
                0,
            ],
            "merge_outcome": [
                "Merged",
                "Unmerged",
                "Merged",
                "Unmerged",
            ],
            "created_quarter": [
                "2024Q1",
                "2024Q1",
                "2024Q2",
                "2024Q2",
            ],
            "total_changes": [
                10,
                20,
                30,
                40,
            ],
            "changed_files": [
                1,
                2,
                3,
                4,
            ],
            "commit_count": [
                1,
                2,
                3,
                4,
            ],
            "lifecycle_days": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )


def test_safe_percentage() -> None:
    """Confirm percentage calculation."""

    assert (
        safe_percentage(
            1,
            4,
        )
        == 25.0
    )

    assert (
        safe_percentage(
            1,
            0,
        )
        == 0.0
    )


def test_summary_metrics() -> None:
    """Confirm top-level metrics."""

    dataframe = create_sample_dataframe()

    result = calculate_summary_metrics(dataframe)

    assert result["total_prs"] == 4
    assert result["unique_prs"] == 4
    assert result["merged_prs"] == 2
    assert result["unmerged_prs"] == 2

    assert result["merge_rate_percent"] == 50.0

    assert result["median_total_changes"] == 25.0


def test_outcome_metrics() -> None:
    """Confirm outcome grouping."""

    dataframe = create_sample_dataframe()

    result = calculate_outcome_metrics(dataframe)

    assert len(result) == 2

    assert int(result["pr_count"].sum()) == 4


def test_quarterly_metrics() -> None:
    """Confirm quarterly grouping."""

    dataframe = create_sample_dataframe()

    result = calculate_quarterly_metrics(dataframe)

    assert len(result) == 2

    assert int(result["total_prs"].sum()) == 4

    assert (result["merge_rate_percent"] == 50.0).all()
