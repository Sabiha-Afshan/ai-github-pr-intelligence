"""Feature-schema definitions for merge-outcome modelling."""

from typing import Final


TARGET_COLUMN: Final[str] = "was_merged"

PR_IDENTIFIER_COLUMN: Final[str] = "pr_number"

SPLIT_COLUMN: Final[str] = "split"


IDENTIFIER_COLUMNS: Final[set[str]] = {
    "repository",
    "repository_name",
    "repository_full_name",
    "owner",
    "pr_number",
    "pull_request_number",
    "node_id",
    "html_url",
    "api_url",
    "url",
    "author_login",
    "author",
    "title",
    "body",
    "state",
    "period",
    "year",
    "quarter",
}


TIMESTAMP_COLUMNS: Final[set[str]] = {
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
}


TARGET_COLUMNS: Final[set[str]] = {
    "was_merged",
    "merge_outcome",
    "was_merged_numeric",
    "actual_outcome",
    "expected_outcome",
    "merge_target",
}


DIRECT_LEAKAGE_COLUMNS: Final[set[str]] = {
    "merged",
    "merge_status",
    "merge_state",
    "merge_result",
    "merged_at",
    "merged_by",
    "merged_by_login",
    "merge_commit_sha",
    "merge_duration_hours",
    "merge_duration_days",
    "merge_hours",
    "time_to_merge_hours",
    "time_to_merge_days",
    "hours_to_merge",
    "days_to_merge",
    "merged_timestamp",
}


POST_OUTCOME_COLUMNS: Final[set[str]] = {
    "closed_at",
    "resolution_hours",
    "resolution_days",
    "lifecycle_hours",
    "lifecycle_days",
    "time_to_close_hours",
    "time_to_close_days",
    "closed_without_merge",
    "is_closed",
}


CHRONOLOGICAL_PERIOD_COLUMNS: Final[set[str]] = {
    "created_year",
    "created_month",
    "created_quarter",
}


SPLIT_COLUMNS: Final[set[str]] = {
    "split",
    "dataset_split",
    "model_split",
    "split_assignment",
    "time_split",
    "set",
}


NON_MODEL_COLUMNS: Final[set[str]] = (
    IDENTIFIER_COLUMNS
    | TIMESTAMP_COLUMNS
    | TARGET_COLUMNS
    | DIRECT_LEAKAGE_COLUMNS
    | POST_OUTCOME_COLUMNS
    | CHRONOLOGICAL_PERIOD_COLUMNS
    | SPLIT_COLUMNS
)


APPROVED_TIMING_FEATURES: Final[set[str]] = {
    "created_day_of_week",
    "created_hour_utc",
    "created_on_weekend",
}


BOOLEAN_TEXT_VALUES: Final[dict[str, int]] = {
    "true": 1,
    "false": 0,
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    "1": 1,
    "0": 0,
}


def normalize_column_name(
    column_name: str,
) -> str:
    """Normalize a column name for schema comparisons."""

    return (
        str(column_name)
        .strip()
        .lower()
    )


def is_excluded_feature(
    column_name: str,
) -> bool:
    """Return whether a column must be excluded from model inputs."""

    normalized_name = normalize_column_name(
        column_name
    )

    return normalized_name in NON_MODEL_COLUMNS


def identify_leakage_columns(
    columns: list[str],
) -> list[str]:
    """Identify known target, direct and post-outcome leakage."""

    leakage_names = (
        DIRECT_LEAKAGE_COLUMNS
        | POST_OUTCOME_COLUMNS
        | TARGET_COLUMNS
    )

    return sorted(
        column
        for column in columns
        if normalize_column_name(
            column
        )
        in leakage_names
    )


def identify_chronological_period_columns(
    columns: list[str],
) -> list[str]:
    """Identify historical period fields excluded from training."""

    return sorted(
        column
        for column in columns
        if normalize_column_name(
            column
        )
        in CHRONOLOGICAL_PERIOD_COLUMNS
    )