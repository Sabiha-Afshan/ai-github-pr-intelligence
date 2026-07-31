"""Dataset completeness and alignment checks."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.population_validation import (
    standardize_pr_number_column,
)


TARGET_CANDIDATES = (
    "was_merged",
    "merge_target",
    "merged",
)


SPLIT_COLUMN_CANDIDATES = (
    "split",
    "dataset_split",
    "model_split",
    "split_assignment",
    "set",
)


def identify_target_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Identify the merge-outcome target column."""

    lowercase_mapping = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in TARGET_CANDIDATES:
        if candidate in lowercase_mapping:
            return lowercase_mapping[candidate]

    return None


def identify_split_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Identify the train-validation-test column."""

    lowercase_mapping = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in SPLIT_COLUMN_CANDIDATES:
        if candidate in lowercase_mapping:
            return lowercase_mapping[candidate]

    return None


def normalize_boolean_target(
    series: pd.Series,
) -> pd.Series:
    """Normalize a binary merge-outcome target."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        ).astype("Int64")

    normalized_text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "merged": 1,
        "closed without merge": 0,
        "closed_unmerged": 0,
        "unmerged": 0,
    }

    return normalized_text.map(mapping).astype(
        "Int64"
    )


def inspect_pr_dataset(
    file_path: Path,
) -> dict[str, Any]:
    """Inspect one PR-level dataset."""

    dataframe = load_dataset(file_path)

    if dataframe.empty:
        return {
            "file_path": str(file_path),
            "exists": True,
            "row_count": 0,
            "column_count": len(
                dataframe.columns
            ),
            "contains_pr_number": False,
            "unique_pr_count": 0,
            "duplicate_pr_count": 0,
            "missing_pr_number_count": 0,
            "target_column": None,
            "target_missing_count": None,
            "target_distribution": {},
        }

    normalized_dataframe = (
        standardize_pr_number_column(
            dataframe
        )
    )

    target_column = identify_target_column(
        normalized_dataframe
    )

    target_distribution: dict[str, int] = {}
    target_missing_count: int | None = None

    if target_column is not None:
        normalized_target = (
            normalize_boolean_target(
                normalized_dataframe[
                    target_column
                ]
            )
        )

        target_missing_count = int(
            normalized_target.isna().sum()
        )

        target_distribution = {
            str(key): int(value)
            for key, value in (
                normalized_target.value_counts(
                    dropna=False
                ).to_dict()
            ).items()
        }

    return {
        "file_path": str(file_path),
        "exists": True,
        "row_count": len(
            normalized_dataframe
        ),
        "column_count": len(
            normalized_dataframe.columns
        ),
        "contains_pr_number": True,
        "unique_pr_count": int(
            normalized_dataframe[
                "pr_number"
            ].nunique()
        ),
        "duplicate_pr_count": int(
            normalized_dataframe.duplicated(
                subset=["pr_number"]
            ).sum()
        ),
        "missing_pr_number_count": int(
            normalized_dataframe[
                "pr_number"
            ].isna()
            .sum()
        ),
        "minimum_pr_number": int(
            normalized_dataframe[
                "pr_number"
            ].min()
        ),
        "maximum_pr_number": int(
            normalized_dataframe[
                "pr_number"
            ].max()
        ),
        "target_column": target_column,
        "target_missing_count": (
            target_missing_count
        ),
        "target_distribution": (
            target_distribution
        ),
    }


def compare_pr_coverage(
    reference_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Compare PR-number coverage between datasets."""

    reference = standardize_pr_number_column(
        load_dataset(reference_path)
    )

    comparison = standardize_pr_number_column(
        load_dataset(comparison_path)
    )

    reference_prs = set(
        reference["pr_number"].astype(int)
    )

    comparison_prs = set(
        comparison["pr_number"].astype(int)
    )

    missing_from_comparison = (
        reference_prs - comparison_prs
    )

    outside_reference = (
        comparison_prs - reference_prs
    )

    return {
        "reference_file": str(
            reference_path
        ),
        "comparison_file": str(
            comparison_path
        ),
        "reference_unique_pr_count": len(
            reference_prs
        ),
        "comparison_unique_pr_count": len(
            comparison_prs
        ),
        "matched_pr_count": len(
            reference_prs & comparison_prs
        ),
        "missing_pr_count": len(
            missing_from_comparison
        ),
        "outside_reference_count": len(
            outside_reference
        ),
        "coverage_rate": (
            len(
                reference_prs
                & comparison_prs
            )
            / len(reference_prs)
            if reference_prs
            else 0.0
        ),
        "missing_pr_numbers": sorted(
            missing_from_comparison
        ),
        "outside_reference_pr_numbers": sorted(
            outside_reference
        ),
        "exact_pr_alignment": (
            reference_prs == comparison_prs
        ),
    }


def inspect_failure_report(
    file_path: Path,
) -> dict[str, Any]:
    """Inspect an extraction-failure CSV."""

    if not file_path.exists():
        return {
            "file_path": str(file_path),
            "exists": False,
            "failure_count": None,
            "status": "missing",
        }

    dataframe = load_dataset(file_path)

    return {
        "file_path": str(file_path),
        "exists": True,
        "failure_count": len(dataframe),
        "status": (
            "no_failures"
            if dataframe.empty
            else "failures_present"
        ),
    }