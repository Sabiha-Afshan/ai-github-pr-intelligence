"""Tests for time-matched sampling validation."""

import pandas as pd

from src.data.sampling_validation import (
    compare_population_membership,
    normalize_outcome,
    validate_sampling_plan,
    validate_selected_population,
)


def test_outcome_normalization() -> None:
    """Confirm outcomes are normalized."""

    series = pd.Series(
        [
            True,
            False,
            "Merged",
            "Closed without merge",
        ]
    )

    result = normalize_outcome(series)

    assert result.tolist() == [
        "merged",
        "unmerged",
        "merged",
        "unmerged",
    ]


def test_population_membership_alignment() -> None:
    """Confirm equal PR sets align."""

    first = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
            ]
        }
    )

    second = pd.DataFrame(
        {
            "pr_number": [
                3,
                2,
                1,
            ]
        }
    )

    result = compare_population_membership(
        first,
        second,
    )

    assert result["exact_alignment"] is True
    assert result["validation_passed"] is True


def test_sampling_plan_passes() -> None:
    """Confirm a valid sampling plan passes."""

    dataframe = pd.DataFrame(
        {
            "available_merged": [
                20,
                30,
            ],
            "available_unmerged": [
                20,
                30,
            ],
            "selected_merged": [
                10,
                290,
            ],
            "selected_unmerged": [
                10,
                290,
            ],
        }
    )

    result = validate_sampling_plan(dataframe)

    assert result["validation_passed"] is False


def test_small_selected_population() -> None:
    """Confirm a balanced small population passes."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                True,
                False,
            ],
            "period": [
                "2025Q1",
                "2025Q1",
            ],
        }
    )

    result = validate_selected_population(
        dataframe,
        expected_records=2,
    )

    assert result["row_count"] == 2
    assert result["duplicate_pr_count"] == 0
