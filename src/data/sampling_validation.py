"""Validation utilities for time-matched PR sampling."""

from typing import Any

import pandas as pd


OUTCOME_COLUMN_CANDIDATES = (
    "was_merged",
    "expected_outcome",
    "actual_outcome",
    "outcome",
)

PERIOD_COLUMN_CANDIDATES = (
    "period",
    "quarter",
    "year_quarter",
)

AVAILABLE_MERGED_CANDIDATES = (
    "available_merged",
    "merged_available",
    "available_merged_count",
)

AVAILABLE_UNMERGED_CANDIDATES = (
    "available_unmerged",
    "unmerged_available",
    "available_unmerged_count",
)

SELECTED_MERGED_CANDIDATES = (
    "selected_merged",
    "merged_selected",
    "selected_merged_count",
    "planned_per_outcome",
)

SELECTED_UNMERGED_CANDIDATES = (
    "selected_unmerged",
    "unmerged_selected",
    "selected_unmerged_count",
    "planned_per_outcome",
)


def identify_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Identify one column using case-insensitive candidates."""

    lowercase_mapping = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in lowercase_mapping:
            return lowercase_mapping[candidate]

    return None


def normalize_outcome(
    series: pd.Series,
) -> pd.Series:
    """Normalize merged and unmerged outcome values."""

    if pd.api.types.is_bool_dtype(series):
        return series.map(
            {
                True: "merged",
                False: "unmerged",
            }
        ).astype("string")

    if pd.api.types.is_numeric_dtype(series):
        numeric_values = pd.to_numeric(
            series,
            errors="coerce",
        )

        return numeric_values.map(
            {
                1: "merged",
                0: "unmerged",
            }
        ).astype("string")

    normalized_text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized_text.map(
        {
            "true": "merged",
            "false": "unmerged",
            "1": "merged",
            "0": "unmerged",
            "merged": "merged",
            "unmerged": "unmerged",
            "closed without merge": "unmerged",
            "closed_without_merge": "unmerged",
            "closed-unmerged": "unmerged",
        }
    ).astype("string")


def normalize_period(
    series: pd.Series,
) -> pd.Series:
    """Normalize quarterly period labels to YYYYQ#."""

    normalized = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(" ", "", regex=False)
    )

    normalized = normalized.str.replace(
        r"^(\d{4})[-_/]?Q?([1-4])$",
        r"\1Q\2",
        regex=True,
    )

    return normalized


def validate_selected_population(
    dataframe: pd.DataFrame,
    expected_records: int = 600,
) -> dict[str, Any]:
    """Validate the selected 600-PR population."""

    if "pr_number" not in dataframe.columns:
        return {
            "validation_passed": False,
            "reason": "pr_number column is missing.",
        }

    outcome_column = identify_column(
        dataframe,
        OUTCOME_COLUMN_CANDIDATES,
    )

    period_column = identify_column(
        dataframe,
        PERIOD_COLUMN_CANDIDATES,
    )

    if outcome_column is None:
        return {
            "validation_passed": False,
            "reason": "No recognized outcome column was found.",
            "available_columns": list(
                dataframe.columns
            ),
        }

    normalized_outcome = normalize_outcome(
        dataframe[outcome_column]
    )

    duplicate_pr_count = int(
        dataframe.duplicated(
            subset=["pr_number"]
        ).sum()
    )

    missing_pr_count = int(
        dataframe["pr_number"].isna().sum()
    )

    invalid_outcome_count = int(
        normalized_outcome.isna().sum()
    )

    outcome_distribution = {
        str(key): int(value)
        for key, value in (
            normalized_outcome.value_counts(
                dropna=False
            ).to_dict()
        ).items()
    }

    period_distribution: dict[str, int] = {}

    invalid_period_count: int | None = None

    if period_column is not None:
        normalized_period = normalize_period(
            dataframe[period_column]
        )

        invalid_period_count = int(
            normalized_period.isna().sum()
        )

        period_distribution = {
            str(key): int(value)
            for key, value in (
                normalized_period.value_counts(
                    dropna=False
                ).sort_index()
                .to_dict()
            ).items()
        }

    validation_passed = (
        len(dataframe) == expected_records
        and int(
            dataframe["pr_number"].nunique()
        )
        == expected_records
        and duplicate_pr_count == 0
        and missing_pr_count == 0
        and invalid_outcome_count == 0
        and outcome_distribution.get(
            "merged",
            0,
        )
        == 300
        and outcome_distribution.get(
            "unmerged",
            0,
        )
        == 300
    )

    return {
        "row_count": len(dataframe),
        "expected_record_count": expected_records,
        "unique_pr_count": int(
            dataframe["pr_number"].nunique()
        ),
        "duplicate_pr_count": duplicate_pr_count,
        "missing_pr_count": missing_pr_count,
        "outcome_column": outcome_column,
        "invalid_outcome_count": (
            invalid_outcome_count
        ),
        "outcome_distribution": (
            outcome_distribution
        ),
        "period_column": period_column,
        "invalid_period_count": (
            invalid_period_count
        ),
        "period_distribution": (
            period_distribution
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_sampling_plan(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate quarterly availability and selected counts."""

    available_merged_column = identify_column(
        dataframe,
        AVAILABLE_MERGED_CANDIDATES,
    )

    available_unmerged_column = identify_column(
        dataframe,
        AVAILABLE_UNMERGED_CANDIDATES,
    )

    selected_merged_column = identify_column(
        dataframe,
        SELECTED_MERGED_CANDIDATES,
    )

    selected_unmerged_column = identify_column(
        dataframe,
        SELECTED_UNMERGED_CANDIDATES,
    )

    required_columns = {
        "available_merged": (
            available_merged_column
        ),
        "available_unmerged": (
            available_unmerged_column
        ),
        "selected_merged": (
            selected_merged_column
        ),
        "selected_unmerged": (
            selected_unmerged_column
        ),
    }

    missing_roles = [
        role
        for role, column
        in required_columns.items()
        if column is None
    ]

    if missing_roles:
        return {
            "validation_passed": False,
            "reason": (
                "Sampling-plan columns could not be identified."
            ),
            "missing_column_roles": missing_roles,
            "available_columns": list(
                dataframe.columns
            ),
        }

    available_merged = pd.to_numeric(
        dataframe[available_merged_column],
        errors="coerce",
    )

    available_unmerged = pd.to_numeric(
        dataframe[available_unmerged_column],
        errors="coerce",
    )

    selected_merged = pd.to_numeric(
        dataframe[selected_merged_column],
        errors="coerce",
    )

    selected_unmerged = pd.to_numeric(
        dataframe[selected_unmerged_column],
        errors="coerce",
    )

    invalid_numeric_count = int(
        (
            available_merged.isna()
            | available_unmerged.isna()
            | selected_merged.isna()
            | selected_unmerged.isna()
        ).sum()
    )

    merged_over_selection_count = int(
        (
            selected_merged
            > available_merged
        ).fillna(False).sum()
    )

    unmerged_over_selection_count = int(
        (
            selected_unmerged
            > available_unmerged
        ).fillna(False).sum()
    )

    negative_count = int(
        (
            (available_merged < 0)
            | (available_unmerged < 0)
            | (selected_merged < 0)
            | (selected_unmerged < 0)
        ).fillna(False).sum()
    )

    selected_merged_total = int(
        selected_merged.fillna(0).sum()
    )

    selected_unmerged_total = int(
        selected_unmerged.fillna(0).sum()
    )

    unequal_quarter_selection_count = int(
        (
            selected_merged
            != selected_unmerged
        ).fillna(False).sum()
    )

    validation_passed = (
        invalid_numeric_count == 0
        and merged_over_selection_count == 0
        and unmerged_over_selection_count == 0
        and negative_count == 0
        and selected_merged_total == 300
        and selected_unmerged_total == 300
        and unequal_quarter_selection_count == 0
    )

    return {
        "row_count": len(dataframe),
        "available_merged_column": (
            available_merged_column
        ),
        "available_unmerged_column": (
            available_unmerged_column
        ),
        "selected_merged_column": (
            selected_merged_column
        ),
        "selected_unmerged_column": (
            selected_unmerged_column
        ),
        "invalid_numeric_count": (
            invalid_numeric_count
        ),
        "merged_over_selection_count": (
            merged_over_selection_count
        ),
        "unmerged_over_selection_count": (
            unmerged_over_selection_count
        ),
        "negative_count": negative_count,
        "selected_merged_total": (
            selected_merged_total
        ),
        "selected_unmerged_total": (
            selected_unmerged_total
        ),
        "unequal_quarter_selection_count": (
            unequal_quarter_selection_count
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def compare_population_membership(
    first_dataframe: pd.DataFrame,
    second_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Compare PR membership between two datasets."""

    if (
        "pr_number" not in first_dataframe.columns
        or "pr_number"
        not in second_dataframe.columns
    ):
        return {
            "validation_passed": False,
            "reason": (
                "One or both datasets are missing pr_number."
            ),
        }

    first_prs = set(
        pd.to_numeric(
            first_dataframe["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    second_prs = set(
        pd.to_numeric(
            second_dataframe["pr_number"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    missing_from_second = (
        first_prs - second_prs
    )

    additional_in_second = (
        second_prs - first_prs
    )

    return {
        "first_unique_pr_count": len(
            first_prs
        ),
        "second_unique_pr_count": len(
            second_prs
        ),
        "matched_pr_count": len(
            first_prs & second_prs
        ),
        "missing_from_second_count": len(
            missing_from_second
        ),
        "additional_in_second_count": len(
            additional_in_second
        ),
        "missing_from_second": sorted(
            missing_from_second
        ),
        "additional_in_second": sorted(
            additional_in_second
        ),
        "exact_alignment": (
            first_prs == second_prs
        ),
        "validation_passed": (
            first_prs == second_prs
        ),
    }