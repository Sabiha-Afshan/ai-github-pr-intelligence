"""Tests for chronological split verification."""

import pandas as pd

from src.data.chronological_validation import (
    period_sort_value,
    validate_chronological_order,
    validate_complete_quarter_assignment,
)


def create_valid_split_dataframe() -> pd.DataFrame:
    """Create a valid chronological split dataset."""

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
            "period": [
                "2023Q1",
                "2023Q1",
                "2023Q2",
                "2023Q2",
                "2023Q3",
                "2023Q3",
            ],
            "split": [
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
            "created_at": [
                "2023-01-01T00:00:00Z",
                "2023-02-01T00:00:00Z",
                "2023-04-01T00:00:00Z",
                "2023-05-01T00:00:00Z",
                "2023-07-01T00:00:00Z",
                "2023-08-01T00:00:00Z",
            ],
        }
    )


def test_period_sort_value() -> None:
    """Confirm quarters sort chronologically."""

    assert period_sort_value("2023Q1") < period_sort_value("2023Q2")

    assert period_sort_value("2023Q4") < period_sort_value("2024Q1")


def test_complete_quarters_pass() -> None:
    """Confirm a quarter belongs to one split."""

    dataframe = create_valid_split_dataframe()

    result = validate_complete_quarter_assignment(dataframe)

    assert result["validation_passed"] is True

    assert result["divided_quarter_count"] == 0


def test_divided_quarter_fails() -> None:
    """Confirm a quarter split across sets fails."""

    dataframe = create_valid_split_dataframe()

    dataframe.loc[
        1,
        "split",
    ] = "validation"

    result = validate_complete_quarter_assignment(dataframe)

    assert result["validation_passed"] is False

    assert result["divided_quarter_count"] == 1


def test_chronological_order_passes() -> None:
    """Confirm correctly ordered splits pass."""

    dataframe = create_valid_split_dataframe()

    result = validate_chronological_order(dataframe)

    assert result["train_before_validation"] is True

    assert result["validation_before_test"] is True

    assert result["validation_passed"] is True
