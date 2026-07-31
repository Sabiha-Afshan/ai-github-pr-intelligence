"""Tests for dataset completeness utilities."""

import pandas as pd

from src.data.completeness import (
    identify_target_column,
    normalize_boolean_target,
)


def test_target_column_is_identified() -> None:
    """Confirm was_merged is detected."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [1, 2],
            "was_merged": [
                True,
                False,
            ],
        }
    )

    assert identify_target_column(dataframe) == "was_merged"


def test_boolean_target_is_normalized() -> None:
    """Confirm Boolean outcomes become zero and one."""

    series = pd.Series(
        [
            True,
            False,
            True,
        ]
    )

    result = normalize_boolean_target(series)

    assert result.tolist() == [
        1,
        0,
        1,
    ]


def test_text_target_is_normalized() -> None:
    """Confirm textual outcomes become zero and one."""

    series = pd.Series(
        [
            "Merged",
            "Closed without merge",
        ]
    )

    result = normalize_boolean_target(series)

    assert result.tolist() == [
        1,
        0,
    ]
