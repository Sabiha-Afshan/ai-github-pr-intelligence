"""Tests for PR data-quality checks."""

import pandas as pd

from src.data.quality_checks import (
    check_change_totals,
    check_merge_consistency,
    check_population_integrity,
    check_target_quality,
)


def create_valid_dataframe() -> pd.DataFrame:
    """Create a small valid PR dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                True,
                False,
            ],
            "merged_at": [
                "2025-01-02T00:00:00Z",
                None,
            ],
            "additions": [
                10,
                3,
            ],
            "deletions": [
                2,
                1,
            ],
            "total_changes": [
                12,
                4,
            ],
        }
    )


def test_population_integrity_passes() -> None:
    """Confirm a unique population passes."""

    dataframe = create_valid_dataframe()

    results = check_population_integrity(
        dataframe,
        expected_rows=2,
    )

    assert all(result["passed"] for result in results)


def test_merge_consistency_passes() -> None:
    """Confirm merged_at matches was_merged."""

    dataframe = create_valid_dataframe()

    results = check_merge_consistency(dataframe)

    assert results[0]["passed"] is True


def test_change_totals_pass() -> None:
    """Confirm valid total changes pass."""

    dataframe = create_valid_dataframe()

    results = check_change_totals(dataframe)

    assert results[0]["passed"] is True


def test_incorrect_change_total_fails() -> None:
    """Confirm an invalid total is detected."""

    dataframe = create_valid_dataframe()

    dataframe.loc[
        0,
        "total_changes",
    ] = 100

    results = check_change_totals(dataframe)

    assert results[0]["passed"] is False


def test_balanced_target_check() -> None:
    """Confirm the target validator detects imbalance."""

    dataframe = pd.DataFrame(
        {
            "was_merged": [
                True,
                True,
            ]
        }
    )

    results = check_target_quality(dataframe)

    balanced_check = next(
        result
        for result in results
        if result["check_name"] == "balanced_target_distribution"
    )

    assert balanced_check["passed"] is False
