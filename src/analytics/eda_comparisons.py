"""Comparative and outlier-aware EDA utilities."""

from typing import Any

import numpy as np
import pandas as pd

COMPARISON_COLUMNS = (
    "total_changes",
    "additions",
    "deletions",
    "changed_files",
    "commit_count",
    "comments",
    "review_comments",
    "lifecycle_days",
    "title_length",
    "body_length",
    "label_count",
    "requested_reviewer_count",
    "test_files_changed",
    "documentation_files_changed",
    "configuration_files_changed",
    "security_sensitive_files_changed",
)


def to_numeric_series(
    series: pd.Series,
) -> pd.Series:
    """Convert a series to clean numeric values."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return numeric.dropna()


def safe_float(
    value: Any,
) -> float | None:
    """Convert a finite numeric value to float."""

    if value is None:
        return None

    try:
        converted = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(converted):
        return None

    return round(
        converted,
        4,
    )


def calculate_iqr_bounds(
    series: pd.Series,
    multiplier: float = 1.5,
) -> dict[str, float | None]:
    """Calculate quartiles and IQR outlier boundaries."""

    numeric = to_numeric_series(series)

    if numeric.empty:
        return {
            "q1": None,
            "median": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
        }

    q1 = float(numeric.quantile(0.25))

    median = float(numeric.quantile(0.50))

    q3 = float(numeric.quantile(0.75))

    iqr = q3 - q1

    return {
        "q1": round(q1, 4),
        "median": round(
            median,
            4,
        ),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(
            q1 - multiplier * iqr,
            4,
        ),
        "upper_bound": round(
            q3 + multiplier * iqr,
            4,
        ),
    }


def calculate_outlier_summary(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...] = COMPARISON_COLUMNS,
) -> pd.DataFrame:
    """Summarize IQR outliers for numeric columns."""

    records: list[dict[str, Any]] = []

    for column in columns:
        if column not in dataframe.columns:
            continue

        numeric = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        valid = numeric.dropna()

        bounds = calculate_iqr_bounds(valid)

        lower_bound = bounds["lower_bound"]

        upper_bound = bounds["upper_bound"]

        if lower_bound is None or upper_bound is None:
            outlier_count = 0
        else:
            outlier_count = int(
                ((numeric < lower_bound) | (numeric > upper_bound)).sum()
            )

        records.append(
            {
                "column": column,
                "valid_count": int(valid.count()),
                "missing_count": int(numeric.isna().sum()),
                "minimum": (safe_float(valid.min()) if not valid.empty else None),
                "q1": bounds["q1"],
                "median": bounds["median"],
                "q3": bounds["q3"],
                "maximum": (safe_float(valid.max()) if not valid.empty else None),
                "iqr": bounds["iqr"],
                "lower_bound": (lower_bound),
                "upper_bound": (upper_bound),
                "outlier_count": (outlier_count),
                "outlier_percent": (
                    round(
                        outlier_count / len(valid) * 100,
                        2,
                    )
                    if len(valid) > 0
                    else 0.0
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "outlier_percent",
                "column",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def winsorized_mean(
    series: pd.Series,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> float | None:
    """Calculate a winsorized mean without SciPy."""

    numeric = to_numeric_series(series)

    if numeric.empty:
        return None

    lower_value = float(numeric.quantile(lower_quantile))

    upper_value = float(numeric.quantile(upper_quantile))

    clipped = numeric.clip(
        lower=lower_value,
        upper=upper_value,
    )

    return round(
        float(clipped.mean()),
        4,
    )


def standardized_mean_difference(
    first_series: pd.Series,
    second_series: pd.Series,
) -> float | None:
    """Calculate pooled-standard-deviation effect size."""

    first = to_numeric_series(first_series)

    second = to_numeric_series(second_series)

    if len(first) < 2 or len(second) < 2:
        return None

    first_variance = float(first.var(ddof=1))

    second_variance = float(second.var(ddof=1))

    pooled_variance = (
        ((len(first) - 1) * first_variance) + ((len(second) - 1) * second_variance)
    ) / (len(first) + len(second) - 2)

    if pooled_variance <= 0 or not np.isfinite(pooled_variance):
        return 0.0

    pooled_standard_deviation = pooled_variance**0.5

    effect_size = (
        float(first.mean()) - float(second.mean())
    ) / pooled_standard_deviation

    return round(
        effect_size,
        4,
    )


def effect_size_label(
    absolute_effect_size: float | None,
) -> str:
    """Label standardized mean-difference magnitude."""

    if absolute_effect_size is None:
        return "Unavailable"

    if absolute_effect_size < 0.2:
        return "Negligible"

    if absolute_effect_size < 0.5:
        return "Small"

    if absolute_effect_size < 0.8:
        return "Moderate"

    return "Large"


def calculate_outcome_comparisons(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...] = COMPARISON_COLUMNS,
) -> pd.DataFrame:
    """Compare merged and unmerged PR characteristics."""

    if "was_merged_numeric" not in dataframe.columns:
        raise ValueError("was_merged_numeric column is missing.")

    merged_rows = dataframe[dataframe["was_merged_numeric"] == 1]

    unmerged_rows = dataframe[dataframe["was_merged_numeric"] == 0]

    records: list[dict[str, Any]] = []

    for column in columns:
        if column not in dataframe.columns:
            continue

        merged_values = merged_rows[column]

        unmerged_values = unmerged_rows[column]

        merged_numeric = to_numeric_series(merged_values)

        unmerged_numeric = to_numeric_series(unmerged_values)

        merged_mean = (
            safe_float(merged_numeric.mean()) if not merged_numeric.empty else None
        )

        unmerged_mean = (
            safe_float(unmerged_numeric.mean()) if not unmerged_numeric.empty else None
        )

        merged_median = (
            safe_float(merged_numeric.median()) if not merged_numeric.empty else None
        )

        unmerged_median = (
            safe_float(unmerged_numeric.median())
            if not unmerged_numeric.empty
            else None
        )

        effect_size = standardized_mean_difference(
            merged_numeric,
            unmerged_numeric,
        )

        records.append(
            {
                "metric": column,
                "merged_valid_count": len(merged_numeric),
                "unmerged_valid_count": len(unmerged_numeric),
                "merged_mean": (merged_mean),
                "unmerged_mean": (unmerged_mean),
                "mean_difference": (
                    safe_float(merged_mean - unmerged_mean)
                    if (merged_mean is not None and unmerged_mean is not None)
                    else None
                ),
                "merged_median": (merged_median),
                "unmerged_median": (unmerged_median),
                "median_difference": (
                    safe_float(merged_median - unmerged_median)
                    if (merged_median is not None and unmerged_median is not None)
                    else None
                ),
                "merged_winsorized_mean": (winsorized_mean(merged_numeric)),
                "unmerged_winsorized_mean": (winsorized_mean(unmerged_numeric)),
                "standardized_mean_difference": (effect_size),
                "absolute_effect_size": (
                    abs(effect_size) if effect_size is not None else None
                ),
                "effect_size_label": (
                    effect_size_label(
                        abs(effect_size) if effect_size is not None else None
                    )
                ),
            }
        )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    return result.sort_values(
        [
            "absolute_effect_size",
            "metric",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)


def calculate_file_category_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize file-category presence and merge rates."""

    category_columns = {
        "Tests": ("test_files_changed"),
        "Documentation": ("documentation_files_changed"),
        "Configuration": ("configuration_files_changed"),
        "Security-sensitive": ("security_sensitive_files_changed"),
    }

    records: list[dict[str, Any]] = []

    for category, column in category_columns.items():
        if column not in dataframe.columns:
            continue

        numeric = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).fillna(0)

        category_present = numeric > 0

        category_rows = dataframe[category_present]

        category_absent_rows = dataframe[~category_present]

        present_count = len(category_rows)

        absent_count = len(category_absent_rows)

        present_merged = int((category_rows["was_merged_numeric"] == 1).sum())

        absent_merged = int((category_absent_rows["was_merged_numeric"] == 1).sum())

        present_merge_rate = (
            round(
                present_merged / present_count * 100,
                2,
            )
            if present_count > 0
            else 0.0
        )

        absent_merge_rate = (
            round(
                absent_merged / absent_count * 100,
                2,
            )
            if absent_count > 0
            else 0.0
        )

        records.append(
            {
                "file_category": (category),
                "source_column": column,
                "pr_count_with_category": (present_count),
                "dataset_percent_with_category": (
                    round(
                        present_count / len(dataframe) * 100,
                        2,
                    )
                    if len(dataframe) > 0
                    else 0.0
                ),
                "merge_rate_with_category": (present_merge_rate),
                "pr_count_without_category": (absent_count),
                "merge_rate_without_category": (absent_merge_rate),
                "merge_rate_difference": (
                    round(
                        present_merge_rate - absent_merge_rate,
                        2,
                    )
                ),
                "total_files_changed": (safe_float(numeric.sum())),
                "median_files_changed_when_present": (
                    safe_float(numeric[category_present].median())
                    if present_count > 0
                    else None
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "pr_count_with_category",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_lifecycle_bands(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize merge outcomes by lifecycle-duration band."""

    if "lifecycle_days" not in dataframe.columns:
        raise ValueError("lifecycle_days column is missing.")

    working = dataframe.copy()

    working["lifecycle_band"] = pd.cut(
        pd.to_numeric(
            working["lifecycle_days"],
            errors="coerce",
        ),
        bins=[
            -np.inf,
            1 / 24,
            1,
            7,
            30,
            90,
            np.inf,
        ],
        labels=[
            "Under 1 hour",
            "1-24 hours",
            "1-7 days",
            "8-30 days",
            "31-90 days",
            "Over 90 days",
        ],
    )

    grouped = (
        working.groupby(
            "lifecycle_band",
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
                lambda values: int((values == 1).sum()),
            ),
            unmerged_prs=(
                "was_merged_numeric",
                lambda values: int((values == 0).sum()),
            ),
            median_lifecycle_days=(
                "lifecycle_days",
                "median",
            ),
            average_lifecycle_days=(
                "lifecycle_days",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["merge_rate_percent"] = grouped.apply(
        lambda row: (
            round(
                row["merged_prs"] / row["total_prs"] * 100,
                2,
            )
            if row["total_prs"] > 0
            else 0.0
        ),
        axis=1,
    )

    return grouped


def calculate_complexity_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create portfolio-ready complexity metrics."""

    metrics = (
        "total_changes",
        "changed_files",
        "commit_count",
        "comments",
        "review_comments",
        "lifecycle_days",
    )

    records: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in dataframe.columns:
            continue

        numeric = to_numeric_series(dataframe[metric])

        if numeric.empty:
            continue

        records.append(
            {
                "metric": metric,
                "valid_count": len(numeric),
                "minimum": safe_float(numeric.min()),
                "mean": safe_float(numeric.mean()),
                "winsorized_mean": (winsorized_mean(numeric)),
                "median": safe_float(numeric.median()),
                "p75": safe_float(numeric.quantile(0.75)),
                "p90": safe_float(numeric.quantile(0.90)),
                "p95": safe_float(numeric.quantile(0.95)),
                "p99": safe_float(numeric.quantile(0.99)),
                "maximum": safe_float(numeric.max()),
                "standard_deviation": (safe_float(numeric.std(ddof=1))),
            }
        )

    return pd.DataFrame(records)
