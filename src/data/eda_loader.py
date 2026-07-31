"""Load and prepare the authoritative dataset for EDA."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.processed_dataset_validation import (
    normalize_target,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
)


DEFAULT_EDA_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "pallets_flask_corrected_600_feature_engineered.csv"
)

TIMESTAMP_COLUMNS = (
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
)

NUMERIC_COLUMNS = (
    "additions",
    "deletions",
    "total_changes",
    "changed_files",
    "commit_count",
    "comments",
    "review_comments",
    "title_length",
    "body_length",
    "body_word_count",
    "label_count",
    "requested_reviewer_count",
    "test_files_changed",
    "documentation_files_changed",
    "configuration_files_changed",
    "security_sensitive_files_changed",
    "files_added",
    "files_modified",
    "files_removed",
    "files_renamed",
)


def prepare_eda_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare a PR dataframe for exploratory analysis."""

    prepared = dataframe.copy()

    if "pr_number" in prepared.columns:
        prepared["pr_number"] = pd.to_numeric(
            prepared["pr_number"],
            errors="coerce",
        ).astype("Int64")

    if "was_merged" in prepared.columns:
        prepared["was_merged_numeric"] = (
            normalize_target(
                prepared["was_merged"]
            )
        )

        prepared["merge_outcome"] = (
            prepared[
                "was_merged_numeric"
            ].map(
                {
                    1: "Merged",
                    0: "Unmerged",
                }
            ).astype("string")
        )

    for column in TIMESTAMP_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(
                prepared[column],
                errors="coerce",
                utc=True,
            )

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    if "created_at" in prepared.columns:
        prepared["created_year"] = (
            prepared["created_at"].dt.year
        ).astype("Int64")

        prepared["created_month"] = (
            prepared["created_at"].dt.month
        ).astype("Int64")

        prepared["created_month_name"] = (
            prepared[
                "created_at"
            ].dt.month_name()
        ).astype("string")

        prepared["created_quarter"] = (
            prepared["created_at"]
            .dt.tz_localize(None)
            .dt.to_period("Q")
            .astype("string")
        )

        prepared["created_day_name"] = (
            prepared[
                "created_at"
            ].dt.day_name()
        ).astype("string")

    if {
        "created_at",
        "closed_at",
    }.issubset(prepared.columns):
        prepared["lifecycle_hours"] = (
            (
                prepared["closed_at"]
                - prepared["created_at"]
            ).dt.total_seconds()
            / 3600
        )

        prepared["lifecycle_days"] = (
            prepared["lifecycle_hours"]
            / 24
        )

    if {
        "created_at",
        "merged_at",
    }.issubset(prepared.columns):
        prepared["merge_duration_hours"] = (
            (
                prepared["merged_at"]
                - prepared["created_at"]
            ).dt.total_seconds()
            / 3600
        )

        prepared["merge_duration_days"] = (
            prepared["merge_duration_hours"]
            / 24
        )

    if {
        "additions",
        "deletions",
    }.issubset(prepared.columns):
        prepared["calculated_total_changes"] = (
            prepared["additions"].fillna(0)
            + prepared["deletions"].fillna(0)
        )

    if "total_changes" in prepared.columns:
        prepared["change_size_band"] = pd.cut(
            prepared["total_changes"],
            bins=[
                -1,
                20,
                100,
                500,
                2000,
                float("inf"),
            ],
            labels=[
                "Very small",
                "Small",
                "Medium",
                "Large",
                "Very large",
            ],
        )

    if "changed_files" in prepared.columns:
        prepared["file_count_band"] = pd.cut(
            prepared["changed_files"],
            bins=[
                -1,
                1,
                5,
                15,
                50,
                float("inf"),
            ],
            labels=[
                "1 file",
                "2-5 files",
                "6-15 files",
                "16-50 files",
                "51+ files",
            ],
        )

    return prepared


@lru_cache(maxsize=4)
def load_eda_dataset_cached(
    file_path_string: str,
    modified_time: float,
) -> pd.DataFrame:
    """Load and cache the EDA dataset using its modification time."""

    del modified_time

    file_path = Path(file_path_string)

    dataframe = load_dataset(
        file_path
    )

    return prepare_eda_dataframe(
        dataframe
    )


def load_eda_dataset(
    file_path: Path | None = None,
) -> pd.DataFrame:
    """Load the final feature-engineered dataset for EDA."""

    resolved_path = (
        file_path
        if file_path is not None
        else DEFAULT_EDA_DATASET_PATH
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "EDA dataset was not found: "
            f"{resolved_path}"
        )

    modified_time = (
        resolved_path.stat().st_mtime
    )

    return load_eda_dataset_cached(
        str(resolved_path.resolve()),
        modified_time,
    ).copy()


def describe_eda_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return basic metadata for an EDA dataframe."""

    return {
        "row_count": len(dataframe),
        "column_count": len(
            dataframe.columns
        ),
        "unique_pr_count": (
            int(
                dataframe[
                    "pr_number"
                ].nunique()
            )
            if "pr_number"
            in dataframe.columns
            else None
        ),
        "minimum_created_at": (
            dataframe[
                "created_at"
            ].min()
            if "created_at"
            in dataframe.columns
            else None
        ),
        "maximum_created_at": (
            dataframe[
                "created_at"
            ].max()
            if "created_at"
            in dataframe.columns
            else None
        ),
        "missing_value_count": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_row_count": int(
            dataframe.duplicated().sum()
        ),
    }