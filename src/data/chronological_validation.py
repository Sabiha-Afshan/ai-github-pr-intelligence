"""Chronological train-validation-test verification."""

from typing import Any

import pandas as pd

from src.data.sampling_validation import normalize_period


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "validate": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


def normalize_split_values(
    series: pd.Series,
) -> pd.Series:
    """Normalize split names."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(SPLIT_ALIASES)
        .astype("string")
    )


def period_sort_value(
    period: str,
) -> int:
    """Convert YYYYQ# into a sortable integer."""

    normalized_period = str(period).strip().upper()

    if (
        len(normalized_period) != 6
        or normalized_period[4] != "Q"
    ):
        raise ValueError(
            f"Invalid quarterly period: {period}"
        )

    year = int(normalized_period[:4])
    quarter = int(normalized_period[-1])

    if quarter not in {
        1,
        2,
        3,
        4,
    }:
        raise ValueError(
            f"Invalid quarterly period: {period}"
        )

    return year * 4 + quarter


def identify_period_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Identify the quarterly period column."""

    candidates = (
        "period",
        "quarter",
        "year_quarter",
        "created_quarter",
    )

    lowercase_mapping = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in lowercase_mapping:
            return lowercase_mapping[candidate]

    return None


def identify_split_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Identify the split-assignment column."""

    candidates = (
        "split",
        "dataset_split",
        "model_split",
        "split_assignment",
        "time_split",
        "set",
    )

    lowercase_mapping = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in lowercase_mapping:
            return lowercase_mapping[candidate]

    return None


def prepare_split_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize split and period fields."""

    if "pr_number" not in dataframe.columns:
        raise ValueError(
            "Split dataset is missing pr_number."
        )

    split_column = identify_split_column(
        dataframe
    )

    if split_column is None:
        raise ValueError(
            "No recognized split column was found."
        )

    period_column = identify_period_column(
        dataframe
    )

    if period_column is None:
        raise ValueError(
            "No recognized period column was found."
        )

    prepared = dataframe.copy()

    prepared["pr_number"] = pd.to_numeric(
        prepared["pr_number"],
        errors="coerce",
    )

    prepared["_split"] = normalize_split_values(
        prepared[split_column]
    )

    prepared["_period"] = normalize_period(
        prepared[period_column]
    )

    prepared["_period_sort"] = (
        prepared["_period"].map(
            lambda value: (
                period_sort_value(value)
                if pd.notna(value)
                else pd.NA
            )
        )
    )

    return prepared


def validate_basic_split_integrity(
    dataframe: pd.DataFrame,
    expected_rows: int = 600,
) -> dict[str, Any]:
    """Validate split counts and membership."""

    prepared = prepare_split_dataframe(
        dataframe
    )

    invalid_pr_count = int(
        prepared["pr_number"].isna().sum()
    )

    duplicate_pr_count = int(
        prepared.duplicated(
            subset=["pr_number"]
        ).sum()
    )

    invalid_split_count = int(
        prepared["_split"].isna().sum()
    )

    invalid_period_count = int(
        prepared["_period_sort"].isna().sum()
    )

    split_distribution = {
        str(key): int(value)
        for key, value in (
            prepared["_split"]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ).items()
    }

    expected_distribution = {
        "train": 440,
        "validation": 84,
        "test": 76,
    }

    validation_passed = (
        len(prepared) == expected_rows
        and int(
            prepared["pr_number"].nunique()
        )
        == expected_rows
        and invalid_pr_count == 0
        and duplicate_pr_count == 0
        and invalid_split_count == 0
        and invalid_period_count == 0
        and split_distribution
        == expected_distribution
    )

    return {
        "row_count": len(prepared),
        "expected_row_count": expected_rows,
        "unique_pr_count": int(
            prepared["pr_number"].nunique()
        ),
        "invalid_pr_count": invalid_pr_count,
        "duplicate_pr_count": duplicate_pr_count,
        "invalid_split_count": (
            invalid_split_count
        ),
        "invalid_period_count": (
            invalid_period_count
        ),
        "split_distribution": (
            split_distribution
        ),
        "expected_distribution": (
            expected_distribution
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_complete_quarter_assignment(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Confirm no quarter appears in multiple splits."""

    prepared = prepare_split_dataframe(
        dataframe
    )

    quarter_split_counts = (
        prepared.groupby(
            "_period",
            dropna=False,
        )["_split"]
        .nunique(
            dropna=True
        )
    )

    divided_quarters = (
        quarter_split_counts[
            quarter_split_counts > 1
        ]
        .index.astype(str)
        .tolist()
    )

    quarter_assignments = (
        prepared.groupby(
            "_period",
            dropna=False,
        )
        .agg(
            split=(
                "_split",
                "first",
            ),
            pr_count=(
                "pr_number",
                "count",
            ),
            unique_split_count=(
                "_split",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_period": "period",
            }
        )
        .sort_values(
            "period"
        )
    )

    return {
        "quarter_count": len(
            quarter_assignments
        ),
        "divided_quarter_count": len(
            divided_quarters
        ),
        "divided_quarters": (
            divided_quarters
        ),
        "quarter_assignments": (
            quarter_assignments.to_dict(
                orient="records"
            )
        ),
        "validation_passed": (
            len(divided_quarters) == 0
        ),
    }


def validate_chronological_order(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Confirm train precedes validation and test."""

    prepared = prepare_split_dataframe(
        dataframe
    )

    boundaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_rows = prepared[
            prepared["_split"]
            == split_name
        ]

        if split_rows.empty:
            boundaries[split_name] = {
                "row_count": 0,
                "minimum_period": None,
                "maximum_period": None,
                "minimum_period_sort": None,
                "maximum_period_sort": None,
            }

            continue

        minimum_index = (
            split_rows["_period_sort"]
            .astype(int)
            .idxmin()
        )

        maximum_index = (
            split_rows["_period_sort"]
            .astype(int)
            .idxmax()
        )

        boundaries[split_name] = {
            "row_count": len(split_rows),
            "minimum_period": (
                split_rows.loc[
                    minimum_index,
                    "_period",
                ]
            ),
            "maximum_period": (
                split_rows.loc[
                    maximum_index,
                    "_period",
                ]
            ),
            "minimum_period_sort": int(
                split_rows[
                    "_period_sort"
                ].min()
            ),
            "maximum_period_sort": int(
                split_rows[
                    "_period_sort"
                ].max()
            ),
        }

    required_splits_present = all(
        boundaries[split_name][
            "row_count"
        ]
        > 0
        for split_name in (
            "train",
            "validation",
            "test",
        )
    )

    train_before_validation = False
    validation_before_test = False

    if required_splits_present:
        train_before_validation = (
            boundaries["train"][
                "maximum_period_sort"
            ]
            < boundaries["validation"][
                "minimum_period_sort"
            ]
        )

        validation_before_test = (
            boundaries["validation"][
                "maximum_period_sort"
            ]
            < boundaries["test"][
                "minimum_period_sort"
            ]
        )

    validation_passed = (
        required_splits_present
        and train_before_validation
        and validation_before_test
    )

    return {
        "boundaries": boundaries,
        "required_splits_present": (
            required_splits_present
        ),
        "train_before_validation": (
            train_before_validation
        ),
        "validation_before_test": (
            validation_before_test
        ),
        "validation_passed": (
            validation_passed
        ),
    }


def validate_timestamp_ordering(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Check exact created-at ordering across splits."""

    prepared = prepare_split_dataframe(
        dataframe
    )

    if "created_at" not in prepared.columns:
        return {
            "validation_passed": False,
            "reason": (
                "created_at column is missing."
            ),
        }

    prepared["_created_at"] = pd.to_datetime(
        prepared["created_at"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamp_count = int(
        prepared["_created_at"].isna().sum()
    )

    timestamp_boundaries: dict[
        str,
        dict[str, Any],
    ] = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_dates = prepared.loc[
            prepared["_split"]
            == split_name,
            "_created_at",
        ]

        timestamp_boundaries[
            split_name
        ] = {
            "minimum_created_at": (
                split_dates.min()
                if not split_dates.empty
                else None
            ),
            "maximum_created_at": (
                split_dates.max()
                if not split_dates.empty
                else None
            ),
        }

    train_before_validation = (
        timestamp_boundaries["train"][
            "maximum_created_at"
        ]
        <= timestamp_boundaries[
            "validation"
        ][
            "minimum_created_at"
        ]
    )

    validation_before_test = (
        timestamp_boundaries[
            "validation"
        ][
            "maximum_created_at"
        ]
        <= timestamp_boundaries["test"][
            "minimum_created_at"
        ]
    )

    return {
        "invalid_timestamp_count": (
            invalid_timestamp_count
        ),
        "timestamp_boundaries": (
            timestamp_boundaries
        ),
        "train_before_validation": (
            bool(train_before_validation)
        ),
        "validation_before_test": (
            bool(validation_before_test)
        ),
        "validation_passed": (
            invalid_timestamp_count == 0
            and train_before_validation
            and validation_before_test
        ),
    }


def compare_split_feature_membership(
    split_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Compare split and feature dataset PR coverage."""

    split_prs = set(
        pd.to_numeric(
            split_dataframe[
                "pr_number"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    feature_prs = set(
        pd.to_numeric(
            feature_dataframe[
                "pr_number"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    missing_from_features = (
        split_prs - feature_prs
    )

    missing_from_splits = (
        feature_prs - split_prs
    )

    return {
        "split_unique_pr_count": len(
            split_prs
        ),
        "feature_unique_pr_count": len(
            feature_prs
        ),
        "matched_pr_count": len(
            split_prs & feature_prs
        ),
        "missing_from_features_count": len(
            missing_from_features
        ),
        "missing_from_splits_count": len(
            missing_from_splits
        ),
        "missing_from_features": sorted(
            missing_from_features
        ),
        "missing_from_splits": sorted(
            missing_from_splits
        ),
        "exact_alignment": (
            split_prs == feature_prs
        ),
        "validation_passed": (
            split_prs == feature_prs
        ),
    }