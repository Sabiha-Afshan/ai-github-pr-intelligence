"""Tests for the EDA dataset loader."""

import pandas as pd

from src.data.eda_loader import (
    describe_eda_dataset,
    prepare_eda_dataframe,
)


def test_prepare_eda_dataframe() -> None:
    """Confirm derived EDA fields are created."""

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
            "created_at": [
                "2024-01-01T00:00:00Z",
                "2024-04-01T00:00:00Z",
            ],
            "closed_at": [
                "2024-01-03T00:00:00Z",
                "2024-04-05T00:00:00Z",
            ],
            "merged_at": [
                "2024-01-02T00:00:00Z",
                None,
            ],
            "additions": [
                10,
                20,
            ],
            "deletions": [
                5,
                10,
            ],
            "total_changes": [
                15,
                30,
            ],
            "changed_files": [
                1,
                3,
            ],
        }
    )

    result = prepare_eda_dataframe(dataframe)

    assert "was_merged_numeric" in result.columns

    assert "merge_outcome" in result.columns

    assert "created_quarter" in result.columns

    assert "lifecycle_days" in result.columns

    assert "merge_duration_days" in result.columns

    assert "change_size_band" in result.columns

    assert (
        result.loc[
            0,
            "created_quarter",
        ]
        == "2024Q1"
    )

    assert (
        result.loc[
            1,
            "created_quarter",
        ]
        == "2024Q2"
    )


def test_describe_eda_dataset() -> None:
    """Confirm dataset metadata is returned."""

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "created_at": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-02-01T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    result = describe_eda_dataset(dataframe)

    assert result["row_count"] == 2
    assert result["column_count"] == 2
    assert result["unique_pr_count"] == 2
    assert result["duplicate_row_count"] == 0
