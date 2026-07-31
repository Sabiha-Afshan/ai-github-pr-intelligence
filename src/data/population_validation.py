"""Validation utilities for the modelling population."""

from typing import Any

import pandas as pd


PR_NUMBER_CANDIDATES = (
    "pr_number",
    "number",
    "pull_request_number",
)


def identify_pr_number_column(
    dataframe: pd.DataFrame,
) -> str:
    """Identify the PR-number column."""

    lowercase_mapping = {
        column.lower(): column
        for column in dataframe.columns
    }

    for candidate in PR_NUMBER_CANDIDATES:
        if candidate in lowercase_mapping:
            return lowercase_mapping[
                candidate
            ]

    raise ValueError(
        "Dataset does not contain a recognized "
        "pull-request number column."
    )


def standardize_pr_number_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize the PR-number column name."""

    normalized_dataframe = dataframe.copy()

    original_column = (
        identify_pr_number_column(
            normalized_dataframe
        )
    )

    if original_column != "pr_number":
        normalized_dataframe = (
            normalized_dataframe.rename(
                columns={
                    original_column: "pr_number"
                }
            )
        )

    normalized_dataframe[
        "pr_number"
    ] = pd.to_numeric(
        normalized_dataframe["pr_number"],
        errors="coerce",
    )

    normalized_dataframe = (
        normalized_dataframe.dropna(
            subset=["pr_number"]
        )
    )

    normalized_dataframe[
        "pr_number"
    ] = normalized_dataframe[
        "pr_number"
    ].astype(int)

    return normalized_dataframe


def validate_population(
    dataframe: pd.DataFrame,
    expected_records: int = 600,
) -> dict[str, Any]:
    """Validate a PR modelling population."""

    normalized_dataframe = (
        standardize_pr_number_column(
            dataframe
        )
    )

    duplicate_pr_count = int(
        normalized_dataframe.duplicated(
            subset=[
                "pr_number",
            ]
        ).sum()
    )

    unique_pr_count = int(
        normalized_dataframe[
            "pr_number"
        ].nunique()
    )

    target_column = None

    for candidate in [
        "merge_target",
        "was_merged",
        "merged",
    ]:
        if candidate in normalized_dataframe.columns:
            target_column = candidate
            break

    class_distribution = {}

    if target_column is not None:
        class_distribution = (
            normalized_dataframe[
                target_column
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    required_count_matches = (
        unique_pr_count == expected_records
    )

    return {
        "row_count": len(
            normalized_dataframe
        ),
        "unique_pr_count": unique_pr_count,
        "expected_record_count": (
            expected_records
        ),
        "record_count_matches": (
            required_count_matches
        ),
        "duplicate_pr_count": (
            duplicate_pr_count
        ),
        "target_column": target_column,
        "class_distribution": (
            class_distribution
        ),
        "minimum_pr_number": (
            int(
                normalized_dataframe[
                    "pr_number"
                ].min()
            )
            if not normalized_dataframe.empty
            else None
        ),
        "maximum_pr_number": (
            int(
                normalized_dataframe[
                    "pr_number"
                ].max()
            )
            if not normalized_dataframe.empty
            else None
        ),
        "validation_passed": (
            required_count_matches
            and duplicate_pr_count == 0
        ),
    }