"""Tests for quarter-level validation."""

import pandas as pd

from src.data.quarter_validation import (
    build_quarterly_outcome_summary,
    validate_assigned_periods,
)


def test_assigned_periods_match() -> None:
    """Confirm assigned quarters match created dates."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "created_at": [
                "2025-01-10T00:00:00Z",
                "2025-04-15T00:00:00Z",
            ],
            "period": [
                "2025Q1",
                "2025Q2",
            ],
        }
    )

    result = validate_assigned_periods(dataframe)

    assert result["validation_passed"] is True

    assert result["period_mismatch_count"] == 0


def test_quarterly_outcomes_are_summarized() -> None:
    """Confirm outcomes are counted by quarter."""

    dataframe = pd.DataFrame(
        {
            "period": [
                "2025Q1",
                "2025Q1",
                "2025Q2",
                "2025Q2",
            ],
            "was_merged": [
                True,
                False,
                True,
                False,
            ],
        }
    )

    result = build_quarterly_outcome_summary(dataframe)

    assert len(result) == 2
    assert result["is_time_matched"].all()
