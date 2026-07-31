"""Formal data-quality checks for PR datasets."""

from typing import Any

import pandas as pd


NON_NEGATIVE_COLUMNS = (
    "additions",
    "deletions",
    "total_changes",
    "changed_files",
    "commit_count",
    "title_length",
    "body_length",
    "body_word_count",
    "label_count",
    "requested_reviewer_count",
    "file_records_returned",
    "test_files_changed",
    "documentation_files_changed",
    "configuration_files_changed",
    "security_sensitive_files_changed",
    "files_added",
    "files_modified",
    "files_removed",
    "files_renamed",
)


TIMESTAMP_COLUMNS = (
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
)


def add_check(
    checks: list[dict[str, Any]],
    check_name: str,
    passed: bool,
    observed_value: Any,
    expected_value: Any,
    severity: str = "error",
    details: str = "",
) -> None:
    """Append one standardized quality-check result."""

    checks.append(
        {
            "check_name": check_name,
            "passed": bool(passed),
            "severity": severity,
            "observed_value": observed_value,
            "expected_value": expected_value,
            "details": details,
        }
    )


def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> list[dict[str, Any]]:
    """Check whether required columns exist."""

    checks: list[dict[str, Any]] = []

    missing_columns = sorted(
        required_columns
        - set(dataframe.columns)
    )

    add_check(
        checks=checks,
        check_name="required_columns_present",
        passed=not missing_columns,
        observed_value=missing_columns,
        expected_value="No missing required columns",
        details=(
            "Columns required for dataset identity, "
            "target validation and core PR analysis."
        ),
    )

    return checks


def check_population_integrity(
    dataframe: pd.DataFrame,
    expected_rows: int = 600,
) -> list[dict[str, Any]]:
    """Check row count, uniqueness and PR identifiers."""

    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "row_count",
        len(dataframe) == expected_rows,
        len(dataframe),
        expected_rows,
    )

    duplicate_pr_count = int(
        dataframe.duplicated(
            subset=["pr_number"]
        ).sum()
    )

    add_check(
        checks,
        "duplicate_pr_numbers",
        duplicate_pr_count == 0,
        duplicate_pr_count,
        0,
    )

    missing_pr_number_count = int(
        dataframe["pr_number"].isna().sum()
    )

    add_check(
        checks,
        "missing_pr_numbers",
        missing_pr_number_count == 0,
        missing_pr_number_count,
        0,
    )

    unique_pr_count = int(
        dataframe["pr_number"].nunique()
    )

    add_check(
        checks,
        "unique_pr_count",
        unique_pr_count == expected_rows,
        unique_pr_count,
        expected_rows,
    )

    return checks


def check_target_quality(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate the merge-outcome target."""

    checks: list[dict[str, Any]] = []

    target = dataframe["was_merged"]

    missing_target_count = int(
        target.isna().sum()
    )

    add_check(
        checks,
        "missing_was_merged",
        missing_target_count == 0,
        missing_target_count,
        0,
    )

    normalized_target = (
        target.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    invalid_target_count = int(
        normalized_target.isna().sum()
    )

    add_check(
        checks,
        "valid_binary_target",
        invalid_target_count == 0,
        invalid_target_count,
        0,
    )

    distribution = (
        normalized_target.value_counts()
        .sort_index()
        .to_dict()
    )

    add_check(
        checks,
        "balanced_target_distribution",
        distribution == {
            0: 300,
            1: 300,
        },
        distribution,
        {
            0: 300,
            1: 300,
        },
    )

    return checks


def check_timestamp_quality(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate timestamps and event ordering."""

    checks: list[dict[str, Any]] = []

    parsed: dict[str, pd.Series] = {}

    for column in TIMESTAMP_COLUMNS:
        if column not in dataframe.columns:
            continue

        parsed[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
            utc=True,
        )

        original_non_missing = (
            dataframe[column].notna()
        )

        invalid_count = int(
            (
                original_non_missing
                & parsed[column].isna()
            ).sum()
        )

        add_check(
            checks,
            f"valid_timestamp_{column}",
            invalid_count == 0,
            invalid_count,
            0,
        )

    if {
        "created_at",
        "updated_at",
    }.issubset(parsed):
        invalid_order_count = int(
            (
                parsed["updated_at"]
                < parsed["created_at"]
            )
            .fillna(False)
            .sum()
        )

        add_check(
            checks,
            "updated_at_not_before_created_at",
            invalid_order_count == 0,
            invalid_order_count,
            0,
        )

    if {
        "created_at",
        "closed_at",
    }.issubset(parsed):
        invalid_order_count = int(
            (
                parsed["closed_at"]
                < parsed["created_at"]
            )
            .fillna(False)
            .sum()
        )

        add_check(
            checks,
            "closed_at_not_before_created_at",
            invalid_order_count == 0,
            invalid_order_count,
            0,
        )

    return checks


def check_numeric_quality(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check numeric count fields for invalid negatives."""

    checks: list[dict[str, Any]] = []

    for column in NON_NEGATIVE_COLUMNS:
        if column not in dataframe.columns:
            continue

        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        negative_count = int(
            (numeric_series < 0).sum()
        )

        add_check(
            checks,
            f"non_negative_{column}",
            negative_count == 0,
            negative_count,
            0,
        )

    return checks


def check_merge_consistency(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check merge outcome against merged_at."""

    checks: list[dict[str, Any]] = []

    was_merged = (
        dataframe["was_merged"]
        .astype("string")
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    merged_at_present = (
        dataframe["merged_at"].notna()
    )

    inconsistent_count = int(
        (
            was_merged
            != merged_at_present
        ).sum()
    )

    add_check(
        checks,
        "was_merged_matches_merged_at",
        inconsistent_count == 0,
        inconsistent_count,
        0,
    )

    return checks


def check_change_totals(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check additions plus deletions against total changes."""

    checks: list[dict[str, Any]] = []

    required = {
        "additions",
        "deletions",
        "total_changes",
    }

    if not required.issubset(
        dataframe.columns
    ):
        return checks

    additions = pd.to_numeric(
        dataframe["additions"],
        errors="coerce",
    )

    deletions = pd.to_numeric(
        dataframe["deletions"],
        errors="coerce",
    )

    total_changes = pd.to_numeric(
        dataframe["total_changes"],
        errors="coerce",
    )

    comparable_rows = (
        additions.notna()
        & deletions.notna()
        & total_changes.notna()
    )

    mismatch_count = int(
        (
            total_changes[comparable_rows]
            != (
                additions[comparable_rows]
                + deletions[comparable_rows]
            )
        ).sum()
    )

    add_check(
        checks,
        "total_changes_equals_additions_plus_deletions",
        mismatch_count == 0,
        mismatch_count,
        0,
    )

    return checks


def run_all_quality_checks(
    dataframe: pd.DataFrame,
    expected_rows: int = 600,
) -> pd.DataFrame:
    """Run the complete validated-dataset quality suite."""

    required_columns = {
        "repository",
        "pr_number",
        "title",
        "created_at",
        "closed_at",
        "merged_at",
        "was_merged",
        "additions",
        "deletions",
        "total_changes",
        "changed_files",
        "commit_count",
    }

    checks: list[dict[str, Any]] = []

    checks.extend(
        check_required_columns(
            dataframe,
            required_columns,
        )
    )

    if "pr_number" in dataframe.columns:
        checks.extend(
            check_population_integrity(
                dataframe,
                expected_rows=expected_rows,
            )
        )

    if "was_merged" in dataframe.columns:
        checks.extend(
            check_target_quality(dataframe)
        )

    checks.extend(
        check_timestamp_quality(dataframe)
    )

    checks.extend(
        check_numeric_quality(dataframe)
    )

    if {
        "was_merged",
        "merged_at",
    }.issubset(dataframe.columns):
        checks.extend(
            check_merge_consistency(dataframe)
        )

    checks.extend(
        check_change_totals(dataframe)
    )

    return pd.DataFrame(checks)