"""Quarter-level consistency checks."""

from typing import Any

import pandas as pd

from src.data.sampling_validation import (
    identify_column,
    normalize_outcome,
    normalize_period,
    OUTCOME_COLUMN_CANDIDATES,
    PERIOD_COLUMN_CANDIDATES,
)


def derive_period_from_created_at(
    created_at: pd.Series,
) -> pd.Series:
    """Convert timestamps into YYYYQ# labels."""

    parsed_dates = pd.to_datetime(
        created_at,
        errors="coerce",
        utc=True,
    )

    timezone_naive_dates = (
        parsed_dates.dt.tz_localize(None)
    )

    return timezone_naive_dates.dt.to_period(
        "Q"
    ).astype("string")


def validate_assigned_periods(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Check assigned periods against created-at quarters."""

    period_column = identify_column(
        dataframe,
        PERIOD_COLUMN_CANDIDATES,
    )

    if period_column is None:
        return {
            "validation_passed": False,
            "reason": (
                "No recognized period column was found."
            ),
            "available_columns": list(
                dataframe.columns
            ),
        }

    if "created_at" not in dataframe.columns:
        return {
            "validation_passed": False,
            "reason": "created_at column is missing.",
        }

    assigned_period = normalize_period(
        dataframe[period_column]
    )

    derived_period = derive_period_from_created_at(
        dataframe["created_at"]
    )

    valid_rows = (
        assigned_period.notna()
        & derived_period.notna()
    )

    mismatch_mask = (
        assigned_period[valid_rows]
        != derived_period[valid_rows]
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    invalid_created_at_count = int(
        derived_period.isna().sum()
    )

    invalid_assigned_period_count = int(
        assigned_period.isna().sum()
    )

    mismatch_records = dataframe.loc[
        valid_rows
        & assigned_period.ne(
            derived_period
        ),
        [
            column
            for column in [
                "pr_number",
                "created_at",
                period_column,
            ]
            if column in dataframe.columns
        ],
    ].copy()

    if not mismatch_records.empty:
        mismatch_records[
            "derived_period"
        ] = derived_period.loc[
            mismatch_records.index
        ]

    return {
        "period_column": period_column,
        "row_count": len(dataframe),
        "invalid_created_at_count": (
            invalid_created_at_count
        ),
        "invalid_assigned_period_count": (
            invalid_assigned_period_count
        ),
        "period_mismatch_count": (
            mismatch_count
        ),
        "mismatch_records": (
            mismatch_records.to_dict(
                orient="records"
            )
        ),
        "validation_passed": (
            invalid_created_at_count == 0
            and invalid_assigned_period_count == 0
            and mismatch_count == 0
        ),
    }


def build_quarterly_outcome_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected outcomes by quarter."""

    period_column = identify_column(
        dataframe,
        PERIOD_COLUMN_CANDIDATES,
    )

    outcome_column = identify_column(
        dataframe,
        OUTCOME_COLUMN_CANDIDATES,
    )

    if period_column is None:
        raise ValueError(
            "No recognized period column was found."
        )

    if outcome_column is None:
        raise ValueError(
            "No recognized outcome column was found."
        )

    working_dataframe = dataframe.copy()

    working_dataframe[
        "_normalized_period"
    ] = normalize_period(
        working_dataframe[period_column]
    )

    working_dataframe[
        "_normalized_outcome"
    ] = normalize_outcome(
        working_dataframe[outcome_column]
    )

    summary = (
        working_dataframe.groupby(
            [
                "_normalized_period",
                "_normalized_outcome",
            ],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
        .rename(
            columns={
                "_normalized_period": "period",
                "merged": "merged_count",
                "unmerged": "unmerged_count",
            }
        )
    )

    for column in [
        "merged_count",
        "unmerged_count",
    ]:
        if column not in summary.columns:
            summary[column] = 0

    summary["total_selected"] = (
        summary["merged_count"]
        + summary["unmerged_count"]
    )

    summary["is_time_matched"] = (
        summary["merged_count"]
        == summary["unmerged_count"]
    )

    return summary.sort_values(
        "period"
    ).reset_index(
        drop=True
    )