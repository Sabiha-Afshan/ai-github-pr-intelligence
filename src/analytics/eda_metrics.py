"""Reusable exploratory-analysis metrics for GitHub PR data."""

from typing import Any

import numpy as np
import pandas as pd


def safe_percentage(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Calculate a percentage without division errors."""

    if denominator == 0:
        return 0.0

    return round(
        float(numerator) / float(denominator) * 100,
        2,
    )


def safe_median(
    series: pd.Series,
) -> float | None:
    """Calculate a numeric median."""

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return None

    return round(
        float(numeric_series.median()),
        2,
    )


def safe_mean(
    series: pd.Series,
) -> float | None:
    """Calculate a numeric mean."""

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return None

    return round(
        float(numeric_series.mean()),
        2,
    )


def safe_percentile(
    series: pd.Series,
    percentile: float,
) -> float | None:
    """Calculate a percentile for a numeric series."""

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric_series.empty:
        return None

    return round(
        float(
            np.percentile(
                numeric_series,
                percentile,
            )
        ),
        2,
    )


def calculate_summary_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate top-level portfolio-ready PR metrics."""

    total_prs = len(dataframe)

    unique_prs = (
        int(dataframe["pr_number"].nunique())
        if "pr_number" in dataframe.columns
        else total_prs
    )

    merged_count = 0
    unmerged_count = 0

    if "was_merged_numeric" in dataframe.columns:
        merged_count = int((dataframe["was_merged_numeric"] == 1).sum())

        unmerged_count = int((dataframe["was_merged_numeric"] == 0).sum())

    merge_rate = safe_percentage(
        merged_count,
        merged_count + unmerged_count,
    )

    created_minimum = (
        dataframe["created_at"].min() if "created_at" in dataframe.columns else None
    )

    created_maximum = (
        dataframe["created_at"].max() if "created_at" in dataframe.columns else None
    )

    metrics: dict[str, Any] = {
        "total_prs": total_prs,
        "unique_prs": unique_prs,
        "merged_prs": merged_count,
        "unmerged_prs": unmerged_count,
        "merge_rate_percent": merge_rate,
        "date_range_start": created_minimum,
        "date_range_end": created_maximum,
    }

    optional_metrics = {
        "median_total_changes": (
            "total_changes",
            safe_median,
        ),
        "average_total_changes": (
            "total_changes",
            safe_mean,
        ),
        "p90_total_changes": (
            "total_changes",
            lambda series: safe_percentile(
                series,
                90,
            ),
        ),
        "median_changed_files": (
            "changed_files",
            safe_median,
        ),
        "average_changed_files": (
            "changed_files",
            safe_mean,
        ),
        "median_commit_count": (
            "commit_count",
            safe_median,
        ),
        "average_commit_count": (
            "commit_count",
            safe_mean,
        ),
        "median_lifecycle_days": (
            "lifecycle_days",
            safe_median,
        ),
        "average_lifecycle_days": (
            "lifecycle_days",
            safe_mean,
        ),
        "median_merge_duration_days": (
            "merge_duration_days",
            safe_median,
        ),
        "average_merge_duration_days": (
            "merge_duration_days",
            safe_mean,
        ),
    }

    for metric_name, (
        column,
        function,
    ) in optional_metrics.items():
        metrics[metric_name] = (
            function(dataframe[column]) if column in dataframe.columns else None
        )

    return metrics


def calculate_outcome_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics separately for merged and unmerged PRs."""

    if "merge_outcome" not in dataframe.columns:
        raise ValueError("merge_outcome column is missing.")

    records: list[dict[str, Any]] = []

    for outcome, group in dataframe.groupby(
        "merge_outcome",
        dropna=False,
    ):
        record: dict[str, Any] = {
            "merge_outcome": str(outcome),
            "pr_count": len(group),
            "percentage_of_dataset": (
                safe_percentage(
                    len(group),
                    len(dataframe),
                )
            ),
        }

        for column in (
            "total_changes",
            "changed_files",
            "commit_count",
            "comments",
            "review_comments",
            "lifecycle_days",
        ):
            if column not in group.columns:
                continue

            record[f"median_{column}"] = safe_median(group[column])

            record[f"average_{column}"] = safe_mean(group[column])

        records.append(record)

    return pd.DataFrame(records).sort_values("merge_outcome").reset_index(drop=True)


def calculate_quarterly_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate quarterly PR volume and outcome metrics."""

    if "created_quarter" not in dataframe.columns:
        raise ValueError("created_quarter column is missing.")

    working = dataframe.copy()

    grouped = (
        working.groupby(
            "created_quarter",
            dropna=False,
        )
        .agg(
            total_prs=(
                "pr_number",
                "count",
            ),
            merged_prs=(
                "was_merged_numeric",
                lambda series: int((series == 1).sum()),
            ),
            unmerged_prs=(
                "was_merged_numeric",
                lambda series: int((series == 0).sum()),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "created_quarter": "period",
            }
        )
    )

    grouped["merge_rate_percent"] = grouped.apply(
        lambda row: safe_percentage(
            row["merged_prs"],
            row["total_prs"],
        ),
        axis=1,
    )

    return grouped.sort_values("period").reset_index(drop=True)


def calculate_size_band_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate merge outcomes by PR change-size band."""

    if "change_size_band" not in dataframe.columns:
        raise ValueError("change_size_band column is missing.")

    grouped = (
        dataframe.groupby(
            "change_size_band",
            observed=False,
            dropna=False,
        )
        .agg(
            total_prs=(
                "pr_number",
                "count",
            ),
            merged_prs=(
                "was_merged_numeric",
                lambda series: int((series == 1).sum()),
            ),
            unmerged_prs=(
                "was_merged_numeric",
                lambda series: int((series == 0).sum()),
            ),
            median_total_changes=(
                "total_changes",
                "median",
            ),
        )
        .reset_index()
    )

    grouped["merge_rate_percent"] = grouped.apply(
        lambda row: safe_percentage(
            row["merged_prs"],
            row["total_prs"],
        ),
        axis=1,
    )

    return grouped


def calculate_missing_value_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize missing values by column."""

    missing_count = dataframe.isna().sum()

    summary = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": (missing_count.values),
        }
    )

    summary["missing_percent"] = summary["missing_count"].apply(
        lambda value: safe_percentage(
            value,
            len(dataframe),
        )
    )

    return summary.sort_values(
        [
            "missing_count",
            "column",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)
