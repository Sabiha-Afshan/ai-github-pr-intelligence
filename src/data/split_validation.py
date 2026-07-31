"""Validate chronological model split assignments."""

from typing import Any

import pandas as pd

from src.data.completeness import (
    identify_split_column,
)
from src.data.population_validation import (
    standardize_pr_number_column,
)


VALID_SPLIT_NAMES = {
    "train",
    "training",
    "validation",
    "validate",
    "val",
    "test",
    "testing",
}


def normalize_split_name(
    split_value: object,
) -> str | None:
    """Normalize train-validation-test labels."""

    if split_value is None:
        return None

    normalized = str(
        split_value
    ).strip().lower()

    mapping = {
        "training": "train",
        "train": "train",
        "validation": "validation",
        "validate": "validation",
        "val": "validation",
        "testing": "test",
        "test": "test",
    }

    return mapping.get(normalized)


def validate_split_assignments(
    dataframe: pd.DataFrame,
    expected_pr_count: int = 600,
) -> dict[str, Any]:
    """Validate chronological split coverage."""

    normalized_dataframe = (
        standardize_pr_number_column(
            dataframe
        )
    )

    split_column = identify_split_column(
        normalized_dataframe
    )

    if split_column is None:
        return {
            "validation_passed": False,
            "reason": (
                "No recognized split column was found."
            ),
            "available_columns": list(
                normalized_dataframe.columns
            ),
        }

    normalized_dataframe[
        "_normalized_split"
    ] = normalized_dataframe[
        split_column
    ].map(normalize_split_name)

    invalid_split_count = int(
        normalized_dataframe[
            "_normalized_split"
        ].isna()
        .sum()
    )

    duplicate_pr_count = int(
        normalized_dataframe.duplicated(
            subset=["pr_number"]
        ).sum()
    )

    split_distribution = {
        str(key): int(value)
        for key, value in (
            normalized_dataframe[
                "_normalized_split"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ).items()
    }

    split_sets = {
        split_name: set(
            normalized_dataframe.loc[
                normalized_dataframe[
                    "_normalized_split"
                ]
                == split_name,
                "pr_number",
            ].astype(int)
        )
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    }

    train_validation_overlap = (
        split_sets["train"]
        & split_sets["validation"]
    )

    train_test_overlap = (
        split_sets["train"]
        & split_sets["test"]
    )

    validation_test_overlap = (
        split_sets["validation"]
        & split_sets["test"]
    )

    total_overlap_count = len(
        train_validation_overlap
        | train_test_overlap
        | validation_test_overlap
    )

    unique_pr_count = int(
        normalized_dataframe[
            "pr_number"
        ].nunique()
    )

    required_splits_present = all(
        split_sets[split_name]
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    )

    validation_passed = (
        unique_pr_count == expected_pr_count
        and duplicate_pr_count == 0
        and invalid_split_count == 0
        and total_overlap_count == 0
        and required_splits_present
    )

    return {
        "split_column": split_column,
        "row_count": len(
            normalized_dataframe
        ),
        "unique_pr_count": unique_pr_count,
        "expected_pr_count": expected_pr_count,
        "duplicate_pr_count": duplicate_pr_count,
        "invalid_split_count": (
            invalid_split_count
        ),
        "split_distribution": (
            split_distribution
        ),
        "train_validation_overlap_count": len(
            train_validation_overlap
        ),
        "train_test_overlap_count": len(
            train_test_overlap
        ),
        "validation_test_overlap_count": len(
            validation_test_overlap
        ),
        "total_overlap_count": (
            total_overlap_count
        ),
        "required_splits_present": (
            required_splits_present
        ),
        "validation_passed": (
            validation_passed
        ),
    }