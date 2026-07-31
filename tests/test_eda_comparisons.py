"""Tests for comparative EDA utilities."""

import pandas as pd

from src.analytics.eda_comparisons import (
    calculate_file_category_metrics,
    calculate_iqr_bounds,
    calculate_lifecycle_bands,
    calculate_outcome_comparisons,
    calculate_outlier_summary,
    effect_size_label,
    standardized_mean_difference,
    winsorized_mean,
)


def create_sample_dataframe() -> pd.DataFrame:
    """Create a prepared comparison dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "was_merged_numeric": [
                1,
                1,
                1,
                0,
                0,
                0,
            ],
            "total_changes": [
                10,
                20,
                30,
                40,
                50,
                1000,
            ],
            "changed_files": [
                1,
                2,
                3,
                4,
                5,
                30,
            ],
            "commit_count": [
                1,
                1,
                2,
                3,
                4,
                20,
            ],
            "comments": [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            "review_comments": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "lifecycle_days": [
                0.01,
                0.5,
                2.0,
                10.0,
                60.0,
                120.0,
            ],
            "test_files_changed": [
                1,
                0,
                1,
                0,
                0,
                1,
            ],
            "documentation_files_changed": [
                0,
                1,
                0,
                1,
                0,
                0,
            ],
            "configuration_files_changed": [
                0,
                0,
                0,
                1,
                1,
                0,
            ],
            "security_sensitive_files_changed": [
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        }
    )


def test_iqr_bounds() -> None:
    """Confirm IQR boundaries are calculated."""

    series = pd.Series(
        [
            1,
            2,
            3,
            4,
            100,
        ]
    )

    result = calculate_iqr_bounds(series)

    assert result["q1"] == 2.0
    assert result["median"] == 3.0
    assert result["q3"] == 4.0
    assert result["upper_bound"] == 7.0


def test_winsorized_mean_limits_extreme_values() -> None:
    """Confirm winsorization reduces outlier impact."""

    series = pd.Series(
        [
            1,
            2,
            3,
            4,
            1000,
        ]
    )

    result = winsorized_mean(series)

    assert result is not None
    assert result < series.mean()


def test_standardized_mean_difference() -> None:
    """Confirm outcome effect size is calculated."""

    first = pd.Series(
        [
            1,
            2,
            3,
        ]
    )

    second = pd.Series(
        [
            5,
            6,
            7,
        ]
    )

    result = standardized_mean_difference(
        first,
        second,
    )

    assert result is not None
    assert result < 0


def test_effect_size_labels() -> None:
    """Confirm effect-size labels."""

    assert effect_size_label(0.1) == "Negligible"

    assert effect_size_label(0.3) == "Small"

    assert effect_size_label(0.6) == "Moderate"

    assert effect_size_label(1.0) == "Large"


def test_outcome_comparisons() -> None:
    """Confirm merged and unmerged groups are compared."""

    dataframe = create_sample_dataframe()

    result = calculate_outcome_comparisons(dataframe)

    assert not result.empty

    assert "standardized_mean_difference" in result.columns

    total_changes = result[result["metric"] == "total_changes"].iloc[0]

    assert total_changes["merged_median"] == 20.0

    assert total_changes["unmerged_median"] == 50.0


def test_outlier_summary_detects_outlier() -> None:
    """Confirm extreme values are identified."""

    dataframe = create_sample_dataframe()

    result = calculate_outlier_summary(
        dataframe,
        columns=("total_changes",),
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "outlier_count",
        ]
        >= 1
    )


def test_file_category_metrics() -> None:
    """Confirm file categories are summarized."""

    dataframe = create_sample_dataframe()

    result = calculate_file_category_metrics(dataframe)

    assert len(result) == 4

    assert "merge_rate_with_category" in result.columns


def test_lifecycle_bands_cover_all_rows() -> None:
    """Confirm lifecycle groups preserve row count."""

    dataframe = create_sample_dataframe()

    result = calculate_lifecycle_bands(dataframe)

    assert int(result["total_prs"].sum()) == len(dataframe)
