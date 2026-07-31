"""Tests for model feature validation."""

import numpy as np
import pandas as pd

from src.models.feature_validation import (
    build_numeric_feature_matrix,
    merge_features_with_splits,
    normalize_split_series,
    validate_feature_matrix,
    validate_splits,
)


def create_feature_dataframe() -> pd.DataFrame:
    """Create a small feature dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
            ],
            "repository": [
                "pallets/flask",
                "pallets/flask",
                "pallets/flask",
            ],
            "title": [
                "A",
                "B",
                "C",
            ],
            "was_merged": [
                True,
                False,
                True,
            ],
            "merged_at": [
                "2024-01-02T00:00:00Z",
                None,
                "2024-01-04T00:00:00Z",
            ],
            "total_changes": [
                10,
                20,
                30,
            ],
            "changed_files": [
                1,
                2,
                3,
            ],
            "has_tests": [
                True,
                False,
                True,
            ],
            "constant_feature": [
                1,
                1,
                1,
            ],
        }
    )


def create_split_dataframe() -> pd.DataFrame:
    """Create a small split dataset."""

    return pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
                3,
            ],
            "split": [
                "train",
                "validation",
                "test",
            ],
        }
    )


def test_split_normalization() -> None:
    """Confirm split aliases normalize."""

    result = normalize_split_series(
        pd.Series(
            [
                "Training",
                "Val",
                "Testing",
            ]
        )
    )

    assert result.tolist() == [
        "train",
        "validation",
        "test",
    ]


def test_feature_split_merge() -> None:
    """Confirm split assignments attach correctly."""

    result = merge_features_with_splits(
        create_feature_dataframe(),
        create_split_dataframe(),
    )

    assert len(result) == 3

    assert result["split"].tolist() == [
        "train",
        "validation",
        "test",
    ]


def test_numeric_feature_selection() -> None:
    """Confirm leakage and identifiers are excluded."""

    dataframe = merge_features_with_splits(
        create_feature_dataframe(),
        create_split_dataframe(),
    )

    matrix, audit = build_numeric_feature_matrix(dataframe)

    assert "total_changes" in matrix.columns

    assert "changed_files" in matrix.columns

    assert "has_tests" in matrix.columns

    assert "was_merged" not in matrix.columns

    assert "merged_at" not in matrix.columns

    assert "pr_number" not in matrix.columns

    assert "constant_feature" not in matrix.columns

    selected = audit[audit["status"] == "selected"]["column"].tolist()

    assert "total_changes" in selected


def test_feature_matrix_validation() -> None:
    """Confirm a clean feature matrix passes."""

    source = pd.DataFrame(
        {
            "pr_number": [
                1,
                2,
            ]
        }
    )

    matrix = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
            ],
            "feature_b": [
                0.0,
                np.nan,
            ],
        }
    )

    result = validate_feature_matrix(
        feature_matrix=matrix,
        source_dataframe=source,
    )

    assert result["validation_passed"] is True

    assert result["feature_count"] == 2

    assert result["infinite_value_count"] == 0


def test_invalid_split_distribution_fails() -> None:
    """Confirm unexpected split counts fail."""

    dataframe = pd.DataFrame(
        {
            "split": [
                "train",
                "validation",
                "test",
            ]
        }
    )

    result = validate_splits(dataframe)

    assert result["validation_passed"] is False
