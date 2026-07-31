"""Tests for processed dataset validation."""

import pandas as pd

from src.data.processed_dataset_validation import (
    compare_pr_membership,
    compare_target_values,
    inspect_processed_dataset,
    normalize_target,
)


def test_target_normalization() -> None:
    """Confirm mixed target values normalize correctly."""

    series = pd.Series(
        [
            True,
            False,
            "1",
            "0",
            "Merged",
            "Closed without merge",
        ]
    )

    result = normalize_target(series)

    assert result.tolist() == [
        1,
        0,
        1,
        0,
        1,
        0,
    ]


def test_pr_membership_matches() -> None:
    """Confirm equal PR sets pass."""

    reference = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
            ]
        }
    )

    comparison = pd.DataFrame(
        {
            "pr_number": [
                3,
                2,
                1,
            ]
        }
    )

    result = compare_pr_membership(
        reference_dataframe=reference,
        comparison_dataframe=comparison,
        reference_name="Reference",
        comparison_name="Comparison",
    )

    assert result["validation_passed"] is True

    assert result["matched_pr_count"] == 3


def test_target_values_match() -> None:
    """Confirm equal targets pass."""

    reference = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                True,
                False,
            ],
        }
    )

    comparison = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                1,
                0,
            ],
        }
    )

    result = compare_target_values(
        reference_dataframe=reference,
        comparison_dataframe=comparison,
        reference_name="Reference",
        comparison_name="Comparison",
    )

    assert result["validation_passed"] is True

    assert result["target_mismatch_count"] == 0


def test_target_mismatch_is_detected() -> None:
    """Confirm changed target values fail."""

    reference = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                True,
                False,
            ],
        }
    )

    comparison = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "was_merged": [
                False,
                False,
            ],
        }
    )

    result = compare_target_values(
        reference_dataframe=reference,
        comparison_dataframe=comparison,
        reference_name="Reference",
        comparison_name="Comparison",
    )

    assert result["validation_passed"] is False

    assert result["target_mismatch_count"] == 1


def test_dataset_without_target_can_be_validated(
    tmp_path,
) -> None:
    """Confirm split datasets do not require a target."""

    file_path = tmp_path / "split.csv"

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ],
            "split": [
                "train",
                "test",
            ],
        }
    )

    dataframe.to_csv(
        file_path,
        index=False,
    )

    result = inspect_processed_dataset(
        dataset_name="Split assignments",
        file_path=file_path,
        expected_rows=2,
        require_target=False,
    )

    assert result["validation_passed"] is True

    assert result["target_required"] is False

    assert result["target_column_present"] is False


def test_dataset_without_required_target_fails(
    tmp_path,
) -> None:
    """Confirm processed datasets require the target."""

    file_path = tmp_path / "processed.csv"

    dataframe = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ]
        }
    )

    dataframe.to_csv(
        file_path,
        index=False,
    )

    result = inspect_processed_dataset(
        dataset_name="Processed dataset",
        file_path=file_path,
        expected_rows=2,
        require_target=True,
    )

    assert result["validation_passed"] is False

    assert result["target_required"] is True
