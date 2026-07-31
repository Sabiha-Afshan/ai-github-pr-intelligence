"""Tests for split assignment validation."""

import pandas as pd

from src.data.split_validation import (
    validate_split_assignments,
)


def test_valid_split_assignments_pass() -> None:
    """Confirm non-overlapping splits pass."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
        }
    )

    result = validate_split_assignments(
        dataframe,
        expected_pr_count=6,
    )

    assert result["validation_passed"] is True

    assert result["total_overlap_count"] == 0


def test_duplicate_pr_fails_validation() -> None:
    """Confirm duplicate PR membership fails."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                1,
                2,
            ],
            "split": [
                "train",
                "validation",
                "test",
            ],
        }
    )

    result = validate_split_assignments(
        dataframe,
        expected_pr_count=2,
    )

    assert result["validation_passed"] is False

    assert result["duplicate_pr_count"] == 1
