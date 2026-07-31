"""Target construction and validation for merge-delay prediction."""

from typing import Any

import numpy as np
import pandas as pd

PR_IDENTIFIER_COLUMN = "pr_number"
MERGE_OUTCOME_COLUMN = "was_merged"
MERGE_DURATION_COLUMN = "merge_hours"
SPLIT_COLUMN = "split"
PRIMARY_DELAY_TARGET = "merge_delay_target"
PRIMARY_DELAY_THRESHOLD_HOURS = 48.0

ALTERNATIVE_DELAY_THRESHOLDS = {
    "delayed_over_24_hours": 24.0,
    "merge_delay_target": 48.0,
    "delayed_over_72_hours": 72.0,
    "delayed_over_168_hours": 168.0,
}

TARGET_LEAKAGE_COLUMNS = {
    "merge_hours",
    "merged_at",
    "closed_at",
    "resolution_hours",
    "was_merged",
    "merge_target",
    "delayed_merge",
    "merge_delay_target",
    "delayed_over_24_hours",
    "delayed_over_72_hours",
    "delayed_over_168_hours",
}


def normalize_boolean_target(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """Normalize a Boolean-like column into integer zero and one."""

    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric_values.notna().all():
        integer_values = numeric_values.astype(int)

        invalid_values = sorted(set(integer_values.unique()) - {0, 1})

        if invalid_values:
            raise ValueError(f"{column_name} contains invalid values: {invalid_values}")

        return integer_values

    normalized_text = values.astype("string").str.strip().str.lower()

    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "merged": 1,
        "unmerged": 0,
    }

    mapped_values = normalized_text.map(mapping)

    if mapped_values.isna().any():
        invalid_examples = (
            normalized_text.loc[mapped_values.isna()].dropna().unique().tolist()
        )

        raise ValueError(
            f"{column_name} could not be normalized. Examples: {invalid_examples[:10]}"
        )

    return mapped_values.astype(int)


def validate_source_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the source dataset before target construction."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        MERGE_OUTCOME_COLUMN,
        MERGE_DURATION_COLUMN,
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    duplicate_pr_count = (
        int(dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    validation_passed = not missing_columns and duplicate_pr_count == 0

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "missing_required_columns": (missing_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "validation_passed": (validation_passed),
    }


def create_merged_population(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only valid merged PRs with a usable merge duration."""

    source_validation = validate_source_dataset(dataframe)

    if not source_validation["validation_passed"]:
        raise ValueError(f"Source dataset failed validation: {source_validation}")

    working = dataframe.copy()

    working[MERGE_OUTCOME_COLUMN] = normalize_boolean_target(
        working[MERGE_OUTCOME_COLUMN],
        MERGE_OUTCOME_COLUMN,
    )

    working[MERGE_DURATION_COLUMN] = pd.to_numeric(
        working[MERGE_DURATION_COLUMN],
        errors="coerce",
    )

    merged_population = working.loc[working[MERGE_OUTCOME_COLUMN] == 1].copy()

    invalid_duration_mask = (
        merged_population[MERGE_DURATION_COLUMN].isna()
        | (merged_population[MERGE_DURATION_COLUMN] < 0)
        | ~np.isfinite(merged_population[MERGE_DURATION_COLUMN])
    )

    invalid_duration_count = int(invalid_duration_mask.sum())

    if invalid_duration_count > 0:
        raise ValueError(
            "Merged population contains "
            f"{invalid_duration_count} invalid merge durations."
        )

    merged_population = merged_population.sort_values(
        [
            "created_at",
            PR_IDENTIFIER_COLUMN,
        ]
        if "created_at" in merged_population.columns
        else [PR_IDENTIFIER_COLUMN],
        kind="stable",
    ).reset_index(drop=True)

    return merged_population


def add_delay_targets(
    merged_population: pd.DataFrame,
) -> pd.DataFrame:
    """Add primary and alternative delay targets."""

    if MERGE_DURATION_COLUMN not in (merged_population.columns):
        raise ValueError(f"{MERGE_DURATION_COLUMN} is missing.")

    result = merged_population.copy()

    merge_hours = pd.to_numeric(
        result[MERGE_DURATION_COLUMN],
        errors="raise",
    )

    for (
        target_column,
        threshold_hours,
    ) in ALTERNATIVE_DELAY_THRESHOLDS.items():
        result[target_column] = (merge_hours > threshold_hours).astype(int)

    return result


def build_threshold_summary(
    delay_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize delay prevalence at each candidate threshold."""

    records = []

    row_count = len(delay_dataframe)

    for (
        target_column,
        threshold_hours,
    ) in ALTERNATIVE_DELAY_THRESHOLDS.items():
        target = delay_dataframe[target_column].astype(int)

        delayed_count = int(target.sum())

        non_delayed_count = int(row_count - delayed_count)

        delayed_rate = delayed_count / row_count if row_count else np.nan

        minority_count = min(
            delayed_count,
            non_delayed_count,
        )

        records.append(
            {
                "target_column": (target_column),
                "threshold_hours": (threshold_hours),
                "population_count": (row_count),
                "non_delayed_count": (non_delayed_count),
                "delayed_count": (delayed_count),
                "delayed_rate": (delayed_rate),
                "minority_class_count": (minority_count),
                "both_classes_present": (delayed_count > 0 and non_delayed_count > 0),
                "suitable_for_modelling": (minority_count >= 30),
                "is_primary_target": (target_column == PRIMARY_DELAY_TARGET),
            }
        )

    return pd.DataFrame(records).sort_values("threshold_hours").reset_index(drop=True)


def attach_split_assignments(
    delay_dataframe: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach and normalize existing chronological split assignments."""

    required_split_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = sorted(required_split_columns - set(split_assignments.columns))

    if missing_columns:
        raise ValueError(f"Split assignments are missing: {missing_columns}")

    split_data = split_assignments[
        [
            PR_IDENTIFIER_COLUMN,
            SPLIT_COLUMN,
        ]
    ].copy()

    duplicate_split_count = int(
        split_data.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum()
    )

    if duplicate_split_count > 0:
        raise ValueError(
            f"Split assignments contain {duplicate_split_count} duplicate PR numbers."
        )

    normalized_splits = (
        split_data[SPLIT_COLUMN].astype("string").str.strip().str.lower()
    )

    split_alias_mapping = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "validate": "validation",
        "valid": "validation",
        "val": "validation",
        "test": "test",
        "testing": "test",
    }

    split_data[SPLIT_COLUMN] = normalized_splits.map(split_alias_mapping)

    unknown_split_mask = split_data[SPLIT_COLUMN].isna()

    if unknown_split_mask.any():
        unknown_values = sorted(
            normalized_splits.loc[unknown_split_mask].dropna().unique().tolist()
        )

        raise ValueError(f"Unexpected split values: {unknown_values}")

    result = delay_dataframe.drop(
        columns=[SPLIT_COLUMN],
        errors="ignore",
    ).merge(
        split_data,
        on=PR_IDENTIFIER_COLUMN,
        how="left",
        validate="one_to_one",
    )

    missing_split_count = int(result[SPLIT_COLUMN].isna().sum())

    if missing_split_count > 0:
        missing_pr_examples = (
            result.loc[
                result[SPLIT_COLUMN].isna(),
                PR_IDENTIFIER_COLUMN,
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{missing_split_count} merged PRs "
            "do not have split assignments. "
            f"Example PR numbers: {missing_pr_examples}"
        )

    return result


def build_split_target_summary(
    delay_dataframe: pd.DataFrame,
    target_column: str = (PRIMARY_DELAY_TARGET),
) -> pd.DataFrame:
    """Summarize the primary delay target in each split."""

    required_columns = {
        SPLIT_COLUMN,
        target_column,
    }

    missing_columns = sorted(required_columns - set(delay_dataframe.columns))

    if missing_columns:
        raise ValueError(f"Delay dataset is missing columns: {missing_columns}")

    summary = (
        delay_dataframe.groupby(
            SPLIT_COLUMN,
            observed=False,
        )
        .agg(
            row_count=(
                target_column,
                "size",
            ),
            delayed_count=(
                target_column,
                "sum",
            ),
        )
        .reset_index()
    )

    summary["non_delayed_count"] = summary["row_count"] - summary["delayed_count"]

    summary["delayed_rate"] = summary["delayed_count"] / summary["row_count"]

    summary["both_classes_present"] = (summary["delayed_count"] > 0) & (
        summary["non_delayed_count"] > 0
    )

    split_order = pd.Categorical(
        summary[SPLIT_COLUMN],
        categories=[
            "train",
            "validation",
            "test",
        ],
        ordered=True,
    )

    summary = (
        summary.assign(_split_order=split_order)
        .sort_values("_split_order")
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )

    return summary


def build_merge_duration_summary(
    merged_population: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate descriptive statistics for merge duration."""

    merge_hours = pd.to_numeric(
        merged_population[MERGE_DURATION_COLUMN],
        errors="raise",
    )

    quantiles = merge_hours.quantile(
        [
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    return {
        "row_count": len(merge_hours),
        "minimum_hours": float(merge_hours.min()),
        "mean_hours": float(merge_hours.mean()),
        "median_hours": float(quantiles.loc[0.50]),
        "maximum_hours": float(merge_hours.max()),
        "p25_hours": float(quantiles.loc[0.25]),
        "p75_hours": float(quantiles.loc[0.75]),
        "p90_hours": float(quantiles.loc[0.90]),
        "p95_hours": float(quantiles.loc[0.95]),
        "p99_hours": float(quantiles.loc[0.99]),
    }


def validate_delay_dataset(
    delay_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the completed Model 2 population."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        MERGE_OUTCOME_COLUMN,
        MERGE_DURATION_COLUMN,
        PRIMARY_DELAY_TARGET,
        SPLIT_COLUMN,
    }

    missing_columns = sorted(required_columns - set(delay_dataframe.columns))

    duplicate_pr_count = (
        int(delay_dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())
        if PR_IDENTIFIER_COLUMN in delay_dataframe.columns
        else None
    )

    non_merged_count = (
        int((delay_dataframe[MERGE_OUTCOME_COLUMN] != 1).sum())
        if MERGE_OUTCOME_COLUMN in delay_dataframe.columns
        else None
    )

    invalid_duration_count = (
        int(
            (
                delay_dataframe[MERGE_DURATION_COLUMN].isna()
                | (delay_dataframe[MERGE_DURATION_COLUMN] < 0)
                | ~np.isfinite(delay_dataframe[MERGE_DURATION_COLUMN])
            ).sum()
        )
        if MERGE_DURATION_COLUMN in delay_dataframe.columns
        else None
    )

    invalid_target_count = (
        int((~delay_dataframe[PRIMARY_DELAY_TARGET].isin([0, 1])).sum())
        if PRIMARY_DELAY_TARGET in delay_dataframe.columns
        else None
    )

    target_values = (
        sorted(delay_dataframe[PRIMARY_DELAY_TARGET].astype(int).unique().tolist())
        if PRIMARY_DELAY_TARGET in delay_dataframe.columns
        else []
    )

    split_values = (
        sorted(
            delay_dataframe[SPLIT_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        if SPLIT_COLUMN in delay_dataframe.columns
        else []
    )

    expected_split_values = [
        "test",
        "train",
        "validation",
    ]

    validation_passed = (
        not missing_columns
        and duplicate_pr_count == 0
        and non_merged_count == 0
        and invalid_duration_count == 0
        and invalid_target_count == 0
        and target_values == [0, 1]
        and split_values == expected_split_values
    )

    return {
        "row_count": len(delay_dataframe),
        "column_count": len(delay_dataframe.columns),
        "missing_required_columns": (missing_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "non_merged_pr_count": (non_merged_count),
        "invalid_merge_duration_count": (invalid_duration_count),
        "invalid_primary_target_count": (invalid_target_count),
        "primary_target_values": (target_values),
        "split_values": (split_values),
        "validation_passed": (validation_passed),
    }


def audit_target_leakage_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Document columns that cannot be used as Model 2 features."""

    records = []

    for column in sorted(TARGET_LEAKAGE_COLUMNS):
        records.append(
            {
                "column": column,
                "exists_in_delay_dataset": (column in dataframe.columns),
                "allowed_as_model_feature": (False),
                "reason": (
                    "Outcome, duration or target-derived "
                    "information unavailable for unbiased "
                    "merge-delay prediction."
                ),
            }
        )

    return pd.DataFrame(records)
