"""Validate processed datasets across the data pipeline."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.population_validation import (
    standardize_pr_number_column,
)


TARGET_COLUMN = "was_merged"

IMPORTANT_SHARED_COLUMNS = (
    "repository",
    "pr_number",
    "was_merged",
    "created_at",
    "period",
)


def normalize_target(
    series: pd.Series,
) -> pd.Series:
    """Normalize merge-outcome values to zero and one."""

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

    return normalized_text.map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "merged": 1,
            "unmerged": 0,
            "closed without merge": 0,
        }
    ).astype("Int64")


def inspect_processed_dataset(
    dataset_name: str,
    file_path: Path,
    expected_rows: int = 600,
    require_target: bool = True,
) -> dict[str, Any]:
    """Inspect one processed PR dataset."""

    if not file_path.exists():
        return {
            "dataset_name": dataset_name,
            "file_path": str(file_path),
            "exists": False,
            "status": "missing",
            "validation_passed": False,
        }

    dataframe = standardize_pr_number_column(
        load_dataset(file_path)
    )

    if "pr_number" not in dataframe.columns:
        return {
            "dataset_name": dataset_name,
            "file_path": str(file_path),
            "exists": True,
            "status": "review_required",
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "unique_pr_count": 0,
            "duplicate_pr_count": None,
            "missing_pr_count": None,
            "target_required": require_target,
            "target_column_present": (
                TARGET_COLUMN in dataframe.columns
            ),
            "target_missing_count": None,
            "target_distribution": {},
            "reason": "pr_number column is missing.",
            "validation_passed": False,
        }

    duplicate_pr_count = int(
        dataframe.duplicated(
            subset=["pr_number"]
        ).sum()
    )

    missing_pr_count = int(
        dataframe["pr_number"].isna().sum()
    )

    unique_pr_count = int(
        dataframe["pr_number"].nunique()
    )

    target_column_present = (
        TARGET_COLUMN in dataframe.columns
    )

    target_missing_count: int | None = None
    target_distribution: dict[str, int] = {}

    if target_column_present:
        normalized_target = normalize_target(
            dataframe[TARGET_COLUMN]
        )

        target_missing_count = int(
            normalized_target.isna().sum()
        )

        target_distribution = {
            str(key): int(value)
            for key, value in (
                normalized_target.value_counts(
                    dropna=False
                )
                .sort_index()
                .to_dict()
            ).items()
        }

    population_valid = (
        len(dataframe) == expected_rows
        and unique_pr_count == expected_rows
        and duplicate_pr_count == 0
        and missing_pr_count == 0
    )

    target_valid = (
        target_column_present
        and target_missing_count == 0
        and target_distribution.get("0", 0)
        == expected_rows // 2
        and target_distribution.get("1", 0)
        == expected_rows // 2
    )

    validation_passed = (
        population_valid
        and (
            target_valid
            if require_target
            else True
        )
    )

    return {
        "dataset_name": dataset_name,
        "file_path": str(file_path),
        "exists": True,
        "status": (
            "verified"
            if validation_passed
            else "review_required"
        ),
        "row_count": len(dataframe),
        "column_count": len(
            dataframe.columns
        ),
        "unique_pr_count": unique_pr_count,
        "duplicate_pr_count": (
            duplicate_pr_count
        ),
        "missing_pr_count": missing_pr_count,
        "target_required": require_target,
        "target_column_present": (
            target_column_present
        ),
        "target_missing_count": (
            target_missing_count
        ),
        "target_distribution": (
            target_distribution
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def compare_pr_membership(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    reference_name: str,
    comparison_name: str,
) -> dict[str, Any]:
    """Compare PR-number membership across datasets."""

    reference = standardize_pr_number_column(
        reference_dataframe
    )

    comparison = standardize_pr_number_column(
        comparison_dataframe
    )

    if (
        "pr_number" not in reference.columns
        or "pr_number"
        not in comparison.columns
    ):
        return {
            "reference_dataset": reference_name,
            "comparison_dataset": comparison_name,
            "validation_passed": False,
            "reason": (
                "One or both datasets are missing pr_number."
            ),
        }

    reference_prs = set(
        pd.to_numeric(
            reference["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    comparison_prs = set(
        pd.to_numeric(
            comparison["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    missing_from_comparison = (
        reference_prs - comparison_prs
    )

    additional_in_comparison = (
        comparison_prs - reference_prs
    )

    return {
        "reference_dataset": reference_name,
        "comparison_dataset": comparison_name,
        "reference_unique_pr_count": len(
            reference_prs
        ),
        "comparison_unique_pr_count": len(
            comparison_prs
        ),
        "matched_pr_count": len(
            reference_prs & comparison_prs
        ),
        "missing_from_comparison_count": len(
            missing_from_comparison
        ),
        "additional_in_comparison_count": len(
            additional_in_comparison
        ),
        "missing_from_comparison": sorted(
            missing_from_comparison
        ),
        "additional_in_comparison": sorted(
            additional_in_comparison
        ),
        "exact_alignment": (
            reference_prs == comparison_prs
        ),
        "validation_passed": (
            reference_prs == comparison_prs
        ),
    }


def compare_target_values(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    reference_name: str,
    comparison_name: str,
) -> dict[str, Any]:
    """Confirm target values remain unchanged."""

    reference = standardize_pr_number_column(
        reference_dataframe
    )

    comparison = standardize_pr_number_column(
        comparison_dataframe
    )

    if (
        TARGET_COLUMN
        not in reference.columns
        or TARGET_COLUMN
        not in comparison.columns
    ):
        return {
            "reference_dataset": reference_name,
            "comparison_dataset": comparison_name,
            "validation_passed": False,
            "reason": (
                "One or both datasets are missing was_merged."
            ),
        }

    reference_target = reference[
        [
            "pr_number",
            TARGET_COLUMN,
        ]
    ].copy()

    comparison_target = comparison[
        [
            "pr_number",
            TARGET_COLUMN,
        ]
    ].copy()

    reference_target[
        "_reference_target"
    ] = normalize_target(
        reference_target[TARGET_COLUMN]
    )

    comparison_target[
        "_comparison_target"
    ] = normalize_target(
        comparison_target[TARGET_COLUMN]
    )

    merged = reference_target[
        [
            "pr_number",
            "_reference_target",
        ]
    ].merge(
        comparison_target[
            [
                "pr_number",
                "_comparison_target",
            ]
        ],
        on="pr_number",
        how="outer",
        indicator=True,
    )

    missing_membership_count = int(
        (
            merged["_merge"] != "both"
        ).sum()
    )

    target_mismatch_mask = (
        merged["_merge"].eq("both")
        & merged["_reference_target"].ne(
            merged["_comparison_target"]
        )
    )

    target_mismatch_count = int(
        target_mismatch_mask.sum()
    )

    mismatch_records = merged.loc[
        target_mismatch_mask,
        [
            "pr_number",
            "_reference_target",
            "_comparison_target",
        ],
    ].to_dict(
        orient="records"
    )

    validation_passed = (
        missing_membership_count == 0
        and target_mismatch_count == 0
    )

    return {
        "reference_dataset": reference_name,
        "comparison_dataset": comparison_name,
        "missing_membership_count": (
            missing_membership_count
        ),
        "target_mismatch_count": (
            target_mismatch_count
        ),
        "mismatch_records": mismatch_records,
        "validation_passed": (
            validation_passed
        ),
    }


def compare_shared_identity_values(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    reference_name: str,
    comparison_name: str,
) -> list[dict[str, Any]]:
    """Compare important shared columns by PR number."""

    reference = standardize_pr_number_column(
        reference_dataframe
    )

    comparison = standardize_pr_number_column(
        comparison_dataframe
    )

    available_columns = [
        column
        for column in IMPORTANT_SHARED_COLUMNS
        if (
            column in reference.columns
            and column in comparison.columns
            and column != "pr_number"
        )
    ]

    results: list[dict[str, Any]] = []

    for column in available_columns:
        reference_values = reference[
            [
                "pr_number",
                column,
            ]
        ].rename(
            columns={
                column: "_reference_value",
            }
        )

        comparison_values = comparison[
            [
                "pr_number",
                column,
            ]
        ].rename(
            columns={
                column: "_comparison_value",
            }
        )

        merged = reference_values.merge(
            comparison_values,
            on="pr_number",
            how="inner",
        )

        if column == TARGET_COLUMN:
            merged["_reference_value"] = (
                normalize_target(
                    merged["_reference_value"]
                )
            )

            merged["_comparison_value"] = (
                normalize_target(
                    merged["_comparison_value"]
                )
            )

        elif column == "created_at":
            merged["_reference_value"] = (
                pd.to_datetime(
                    merged["_reference_value"],
                    errors="coerce",
                    utc=True,
                )
            )

            merged["_comparison_value"] = (
                pd.to_datetime(
                    merged["_comparison_value"],
                    errors="coerce",
                    utc=True,
                )
            )

        else:
            merged["_reference_value"] = (
                merged["_reference_value"]
                .astype("string")
                .str.strip()
            )

            merged["_comparison_value"] = (
                merged["_comparison_value"]
                .astype("string")
                .str.strip()
            )

        mismatch_mask = ~(
            merged["_reference_value"].eq(
                merged["_comparison_value"]
            )
            | (
                merged[
                    "_reference_value"
                ].isna()
                & merged[
                    "_comparison_value"
                ].isna()
            )
        )

        mismatch_count = int(
            mismatch_mask.sum()
        )

        results.append(
            {
                "reference_dataset": (
                    reference_name
                ),
                "comparison_dataset": (
                    comparison_name
                ),
                "column": column,
                "compared_row_count": len(
                    merged
                ),
                "mismatch_count": (
                    mismatch_count
                ),
                "validation_passed": (
                    mismatch_count == 0
                ),
            }
        )

    return results