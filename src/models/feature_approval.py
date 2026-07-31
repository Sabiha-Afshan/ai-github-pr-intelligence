"""Manual feature approval and leakage-review utilities."""

from typing import Any

import pandas as pd

from src.models.feature_schema import (
    CHRONOLOGICAL_PERIOD_COLUMNS,
    DIRECT_LEAKAGE_COLUMNS,
    NON_MODEL_COLUMNS,
    POST_OUTCOME_COLUMNS,
    TARGET_COLUMNS,
    normalize_column_name,
)


MANUALLY_BLOCKED_FEATURES = {
    "resolution_hours",
    "resolution_hours_iqr_outlier",
    "merge_hours",
    "merge_hours_iqr_outlier",
    "merge_target",
    "created_year",
    "created_month",
    "created_quarter",
}


FEATURE_GROUPS = {
    "PR structure": {
        "draft",
        "title_length",
        "body_length",
        "body_word_count",
        "has_description",
        "has_detailed_description",
        "short_title",
        "very_long_description",
        "label_count",
        "requested_reviewer_count",
    },
    "Code change": {
        "additions",
        "deletions",
        "total_changes",
        "changed_files",
        "change_density_per_file",
        "deletion_ratio",
        "has_zero_code_change",
    },
    "Commit activity": {
        "commit_count",
        "has_multiple_commits",
    },
    "File composition": {
        "file_records_returned",
        "test_files_changed",
        "documentation_files_changed",
        "configuration_files_changed",
        "security_sensitive_files_changed",
        "files_added",
        "files_modified",
        "files_removed",
        "files_renamed",
        "has_test_changes",
        "has_documentation_changes",
        "has_configuration_changes",
        "has_security_sensitive_changes",
    },
    "Submission timing": {
        "created_day_of_week",
        "created_hour_utc",
        "created_on_weekend",
    },
    "Outlier indicators": {
        "commit_count_iqr_outlier",
        "configuration_files_changed_iqr_outlier",
        "test_files_changed_iqr_outlier",
        "deletions_iqr_outlier",
        "files_modified_iqr_outlier",
        "additions_iqr_outlier",
        "total_changes_iqr_outlier",
        "body_length_iqr_outlier",
        "body_word_count_iqr_outlier",
        "files_added_iqr_outlier",
        "changed_files_iqr_outlier",
        "documentation_files_changed_iqr_outlier",
        "security_sensitive_files_changed_iqr_outlier",
        "files_removed_iqr_outlier",
        "files_renamed_iqr_outlier",
        "title_length_iqr_outlier",
        "requested_reviewer_count_iqr_outlier",
        "iqr_outlier_feature_count",
        "has_any_iqr_outlier",
    },
    "Log transformations": {
        "log1p_body_length",
        "log1p_body_word_count",
        "log1p_additions",
        "log1p_deletions",
        "log1p_total_changes",
        "log1p_changed_files",
        "log1p_commit_count",
    },
}


def determine_exclusion_reason(
    feature_name: str,
) -> str | None:
    """Return the reason a feature must be excluded."""

    normalized_name = normalize_column_name(
        feature_name
    )

    if normalized_name in TARGET_COLUMNS:
        return "Target or target-derived feature"

    if normalized_name in DIRECT_LEAKAGE_COLUMNS:
        return "Direct post-outcome leakage"

    if normalized_name in POST_OUTCOME_COLUMNS:
        return "Available only after PR resolution"

    if normalized_name in MANUALLY_BLOCKED_FEATURES:
        return (
            "Manually blocked because the feature "
            "contains or derives from leakage"
        )

    if (
        normalized_name
        in CHRONOLOGICAL_PERIOD_COLUMNS
    ):
        return (
            "Historical period field excluded to "
            "reduce temporal memorization"
        )

    if normalized_name in NON_MODEL_COLUMNS:
        return "Non-model identifier or metadata"

    return None


def identify_feature_group(
    feature_name: str,
) -> str:
    """Assign a selected feature to a business group."""

    normalized_name = normalize_column_name(
        feature_name
    )

    for group_name, features in (
        FEATURE_GROUPS.items()
    ):
        if normalized_name in features:
            return group_name

    return "Other approved feature"


def review_candidate_features(
    candidate_features: list[str],
) -> pd.DataFrame:
    """Review candidate features and return approval decisions."""

    records: list[dict[str, Any]] = []

    for feature_name in candidate_features:
        exclusion_reason = (
            determine_exclusion_reason(
                feature_name
            )
        )

        approved = exclusion_reason is None

        records.append(
            {
                "feature": feature_name,
                "approved": approved,
                "decision": (
                    "approved"
                    if approved
                    else "excluded"
                ),
                "feature_group": (
                    identify_feature_group(
                        feature_name
                    )
                    if approved
                    else "Excluded"
                ),
                "reason": (
                    "Approved pre-outcome feature"
                    if approved
                    else exclusion_reason
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def get_approved_features(
    review_dataframe: pd.DataFrame,
) -> list[str]:
    """Return approved feature names."""

    if review_dataframe.empty:
        return []

    return (
        review_dataframe.loc[
            review_dataframe["approved"],
            "feature",
        ]
        .astype(str)
        .tolist()
    )


def validate_approved_features(
    approved_features: list[str],
) -> dict[str, Any]:
    """Validate the manually approved feature list."""

    blocked_features_present = sorted(
        set(approved_features)
        & MANUALLY_BLOCKED_FEATURES
    )

    duplicate_features = sorted(
        pd.Series(
            approved_features
        )[
            pd.Series(
                approved_features
            ).duplicated()
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    suspicious_derived_features = sorted(
        feature
        for feature in approved_features
        if (
            feature.startswith(
                "resolution_"
            )
            or feature.startswith(
                "merge_hours"
            )
            or feature.startswith(
                "merge_target"
            )
        )
    )

    validation_passed = (
        len(approved_features) > 0
        and not blocked_features_present
        and not duplicate_features
        and not suspicious_derived_features
    )

    return {
        "approved_feature_count": len(
            approved_features
        ),
        "blocked_features_present": (
            blocked_features_present
        ),
        "duplicate_features": (
            duplicate_features
        ),
        "suspicious_derived_features": (
            suspicious_derived_features
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def build_feature_group_summary(
    review_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize approved features by feature group."""

    approved_rows = review_dataframe[
        review_dataframe["approved"]
    ]

    if approved_rows.empty:
        return pd.DataFrame(
            columns=[
                "feature_group",
                "feature_count",
                "features",
            ]
        )

    summary = (
        approved_rows.groupby(
            "feature_group",
            dropna=False,
        )["feature"]
        .agg(
            feature_count="count",
            features=lambda values: ", ".join(
                sorted(
                    values.astype(str)
                )
            ),
        )
        .reset_index()
        .sort_values(
            [
                "feature_count",
                "feature_group",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return summary