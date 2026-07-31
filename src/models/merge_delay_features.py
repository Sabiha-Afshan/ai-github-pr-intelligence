"""Feature approval and leakage auditing for merge-delay prediction."""

from typing import Any

import numpy as np
import pandas as pd

from src.models.merge_delay_target import (
    MERGE_DURATION_COLUMN,
    MERGE_OUTCOME_COLUMN,
    PR_IDENTIFIER_COLUMN,
    PRIMARY_DELAY_TARGET,
    SPLIT_COLUMN,
)

MODEL_METADATA_COLUMNS = {
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    PRIMARY_DELAY_TARGET,
}

DIRECT_TARGET_LEAKAGE_COLUMNS = {
    MERGE_DURATION_COLUMN,
    MERGE_OUTCOME_COLUMN,
    "merged_at",
    "closed_at",
    "resolution_hours",
    "resolution_hours_iqr_outlier",
    "merge_target",
    "delayed_merge",
    "delayed_over_24_hours",
    "delayed_over_72_hours",
    "delayed_over_168_hours",
}

CONTRIBUTOR_IDENTITY_COLUMNS = {
    "author_association",
    "author_login",
    "author_id",
    "user_login",
    "user_id",
    "created_by",
    "merged_by",
    "merged_by_login",
    "closed_by",
}

CHRONOLOGICAL_SHORTCUT_COLUMNS = {
    "created_at",
    "updated_at",
    "created_year",
    "created_month",
    "created_quarter",
    "created_date",
    "created_week",
    "created_period",
    "period",
    "quarter",
    "year",
    "month",
}

RAW_TEXT_COLUMNS = {
    "title",
    "body",
    "description",
    "labels",
    "changed_file_paths",
    "changed_files_json",
    "commits_json",
    "files_json",
}

EXTRACTION_METADATA_COLUMNS = {
    "file_records_returned",
    "files_requested",
    "files_response_count",
    "commit_records_returned",
    "commits_requested",
    "commits_response_count",
    "extraction_complete",
    "extraction_failure",
    "extraction_failed",
    "extraction_status",
    "data_quality_status",
    "missing_file_records",
    "missing_commit_records",
    "api_page_count",
}

KNOWN_IDENTIFIER_COLUMNS = {
    "id",
    "node_id",
    "repository_id",
    "repository_name",
    "repo_name",
    "repo_full_name",
    "html_url",
    "api_url",
    "url",
}

BLOCKED_FEATURE_COLUMNS = (
    DIRECT_TARGET_LEAKAGE_COLUMNS
    | CONTRIBUTOR_IDENTITY_COLUMNS
    | CHRONOLOGICAL_SHORTCUT_COLUMNS
    | RAW_TEXT_COLUMNS
    | EXTRACTION_METADATA_COLUMNS
    | KNOWN_IDENTIFIER_COLUMNS
)

TARGET_LEAKAGE_NAME_PATTERNS = (
    "merge_hours",
    "resolution_hours",
    "merge_delay_target",
    "delayed_over_",
    "delayed_merge",
    "merged_at",
    "closed_at",
)


def has_target_leakage_name(
    column_name: str,
) -> bool:
    """Detect direct and derived target-leakage column names."""

    normalized_name = column_name.strip().lower()

    return any(pattern in normalized_name for pattern in TARGET_LEAKAGE_NAME_PATTERNS)


ALLOWED_TARGET_VALUES = {0, 1}
ALLOWED_SPLITS = {
    "train",
    "validation",
    "test",
}


def normalize_split_values(
    values: pd.Series,
) -> pd.Series:
    """Normalize common chronological split aliases."""

    normalized = values.astype("string").str.strip().str.lower()

    mapping = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "validate": "validation",
        "valid": "validation",
        "val": "validation",
        "test": "test",
        "testing": "test",
    }

    result = normalized.map(mapping)

    if result.isna().any():
        unknown_values = sorted(
            normalized.loc[result.isna()].dropna().unique().tolist()
        )

        raise ValueError(f"Unexpected split values: {unknown_values}")

    return result.astype(str)


def normalize_binary_target(
    values: pd.Series,
) -> pd.Series:
    """Normalize the merge-delay target into integer zero and one."""

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():
        invalid_count = int(numeric.isna().sum())

        raise ValueError(
            f"{PRIMARY_DELAY_TARGET} contains {invalid_count} non-numeric values."
        )

    integer_target = numeric.astype(int)

    invalid_values = sorted(set(integer_target.unique()) - ALLOWED_TARGET_VALUES)

    if invalid_values:
        raise ValueError(
            f"{PRIMARY_DELAY_TARGET} contains invalid values: {invalid_values}"
        )

    return integer_target


def validate_delay_population_source(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the Stage 6A population before feature approval."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        PRIMARY_DELAY_TARGET,
    }

    missing_required_columns = sorted(required_columns - set(dataframe.columns))

    duplicate_pr_count = (
        int(dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    validation_passed = (
        not missing_required_columns and duplicate_pr_count == 0 and len(dataframe) > 0
    )

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "missing_required_columns": (missing_required_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "validation_passed": (validation_passed),
    }


def classify_feature_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> dict[str, Any]:
    """Classify one source column for Model 2 feature eligibility."""

    series = dataframe[column_name]

    if column_name in MODEL_METADATA_COLUMNS:
        return {
            "column": column_name,
            "decision": "metadata",
            "approved_as_feature": False,
            "reason": ("Required identifier, split or target column."),
            "source_dtype": str(series.dtype),
        }

    if column_name in DIRECT_TARGET_LEAKAGE_COLUMNS or has_target_leakage_name(
        column_name
    ):
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Direct or derived outcome, duration or target leakage."),
            "source_dtype": str(series.dtype),
        }

    if column_name in CONTRIBUTOR_IDENTITY_COLUMNS:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Contributor identity or relationship shortcut."),
            "source_dtype": str(series.dtype),
        }

    if column_name in CHRONOLOGICAL_SHORTCUT_COLUMNS:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Chronological period shortcut or raw timestamp."),
            "source_dtype": str(series.dtype),
        }

    if column_name in RAW_TEXT_COLUMNS:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": (
                "Raw text or serialized content requires a "
                "separate controlled text pipeline."
            ),
            "source_dtype": str(series.dtype),
        }

    if column_name in EXTRACTION_METADATA_COLUMNS:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Extraction-quality or API-response metadata."),
            "source_dtype": str(series.dtype),
        }

    if column_name in KNOWN_IDENTIFIER_COLUMNS:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Repository, record or URL identifier."),
            "source_dtype": str(series.dtype),
        }

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_missing_source_count = int(series.notna().sum())

    numeric_count = int(numeric_values.notna().sum())

    fully_numeric = numeric_count == non_missing_source_count

    if not fully_numeric:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": (
                "Non-numeric feature not supported by the "
                "current contributor-neutral tabular pipeline."
            ),
            "source_dtype": str(series.dtype),
        }

    finite_values = numeric_values.dropna()

    infinite_count = int((~np.isfinite(finite_values)).sum())

    if infinite_count > 0:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": ("Feature contains infinite values."),
            "source_dtype": str(series.dtype),
        }

    unique_non_missing_count = int(numeric_values.nunique(dropna=True))

    if unique_non_missing_count <= 1:
        return {
            "column": column_name,
            "decision": "blocked",
            "approved_as_feature": False,
            "reason": (
                "Constant or near-empty feature within the merged-PR population."
            ),
            "source_dtype": str(series.dtype),
        }

    return {
        "column": column_name,
        "decision": "approved",
        "approved_as_feature": True,
        "reason": (
            "Contributor-neutral numeric snapshot feature "
            "with variation in the Model 2 population."
        ),
        "source_dtype": str(series.dtype),
    }


def build_feature_review(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Review every Model 2 source column."""

    records = [
        classify_feature_column(
            dataframe=dataframe,
            column_name=column_name,
        )
        for column_name in dataframe.columns
    ]

    review = pd.DataFrame(records)

    review["decision_order"] = review["decision"].map(
        {
            "approved": 1,
            "metadata": 2,
            "blocked": 3,
        }
    )

    review = (
        review.sort_values(
            [
                "decision_order",
                "column",
            ]
        )
        .drop(columns=["decision_order"])
        .reset_index(drop=True)
    )

    return review


def get_approved_feature_names(
    feature_review: pd.DataFrame,
) -> list[str]:
    """Return approved Model 2 feature names."""

    required_columns = {
        "column",
        "approved_as_feature",
    }

    missing_columns = sorted(required_columns - set(feature_review.columns))

    if missing_columns:
        raise ValueError(f"Feature review is missing columns: {missing_columns}")

    approved_features = (
        feature_review.loc[
            feature_review["approved_as_feature"].astype(bool),
            "column",
        ]
        .astype(str)
        .tolist()
    )

    if not approved_features:
        raise ValueError("No Model 2 features were approved.")

    return approved_features


def build_modelling_dataset(
    delay_population: pd.DataFrame,
    approved_features: list[str],
) -> pd.DataFrame:
    """Create the leakage-safe Model 2 modelling dataset."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        PRIMARY_DELAY_TARGET,
        *approved_features,
    }

    missing_columns = sorted(required_columns - set(delay_population.columns))

    if missing_columns:
        raise ValueError(
            f"Delay population is missing approved columns: {missing_columns}"
        )

    modelling_dataset = delay_population[
        [
            PR_IDENTIFIER_COLUMN,
            SPLIT_COLUMN,
            PRIMARY_DELAY_TARGET,
            *approved_features,
        ]
    ].copy()

    modelling_dataset[SPLIT_COLUMN] = normalize_split_values(
        modelling_dataset[SPLIT_COLUMN]
    )

    modelling_dataset[PRIMARY_DELAY_TARGET] = normalize_binary_target(
        modelling_dataset[PRIMARY_DELAY_TARGET]
    )

    for feature in approved_features:
        modelling_dataset[feature] = pd.to_numeric(
            modelling_dataset[feature],
            errors="coerce",
        )

    return modelling_dataset


def build_feature_group_summary(
    feature_review: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize approved and blocked feature decisions."""

    summary = (
        feature_review.groupby(
            [
                "decision",
                "reason",
            ],
            observed=False,
        )
        .agg(
            column_count=(
                "column",
                "size",
            )
        )
        .reset_index()
    )

    return summary.sort_values(
        [
            "decision",
            "column_count",
            "reason",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


def validate_modelling_dataset(
    dataframe: pd.DataFrame,
    approved_features: list[str],
) -> dict[str, Any]:
    """Validate the leakage-safe Model 2 modelling dataset."""

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        SPLIT_COLUMN,
        PRIMARY_DELAY_TARGET,
        *approved_features,
    }

    missing_columns = sorted(required_columns - set(dataframe.columns))

    unexpected_columns = sorted(set(dataframe.columns) - required_columns)

    duplicate_pr_count = (
        int(dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum())
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    target_values = (
        sorted(dataframe[PRIMARY_DELAY_TARGET].astype(int).unique().tolist())
        if PRIMARY_DELAY_TARGET in dataframe.columns
        else []
    )

    split_values = (
        sorted(dataframe[SPLIT_COLUMN].astype(str).unique().tolist())
        if SPLIT_COLUMN in dataframe.columns
        else []
    )

    blocked_columns_present = sorted(set(dataframe.columns) & BLOCKED_FEATURE_COLUMNS)

    numeric_feature_count = 0
    non_numeric_features: list[str] = []
    infinite_value_count = 0

    for feature in approved_features:
        if feature not in dataframe.columns:
            continue

        numeric_values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

        source_non_missing_count = int(dataframe[feature].notna().sum())

        numeric_non_missing_count = int(numeric_values.notna().sum())

        if source_non_missing_count != numeric_non_missing_count:
            non_numeric_features.append(feature)
        else:
            numeric_feature_count += 1

        finite_values = numeric_values.dropna()

        infinite_value_count += int((~np.isfinite(finite_values)).sum())

    validation_passed = (
        not missing_columns
        and not unexpected_columns
        and duplicate_pr_count == 0
        and target_values == [0, 1]
        and split_values
        == [
            "test",
            "train",
            "validation",
        ]
        and not blocked_columns_present
        and not non_numeric_features
        and infinite_value_count == 0
        and numeric_feature_count == len(approved_features)
    )

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "approved_feature_count": len(approved_features),
        "numeric_feature_count": (numeric_feature_count),
        "missing_required_columns": (missing_columns),
        "unexpected_columns": (unexpected_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "target_values": (target_values),
        "split_values": (split_values),
        "blocked_columns_present": (blocked_columns_present),
        "non_numeric_features": (non_numeric_features),
        "infinite_value_count": (infinite_value_count),
        "validation_passed": (validation_passed),
    }
