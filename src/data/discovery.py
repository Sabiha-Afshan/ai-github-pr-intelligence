"""Discover and inspect project datasets."""

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.logging import get_logger


logger = get_logger(__name__)


SUPPORTED_DATASET_SUFFIXES = {
    ".csv",
    ".parquet",
    ".pq",
}


def discover_dataset_files(
    root_directory: Path,
) -> list[Path]:
    """Find CSV and Parquet files recursively."""

    if not root_directory.exists():
        return []

    dataset_files = [
        file_path
        for file_path in root_directory.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_DATASET_SUFFIXES
        )
    ]

    return sorted(dataset_files)


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """Load one supported dataset."""

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except EmptyDataError:
            return pd.DataFrame()

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported dataset format: {suffix}"
    )


def calculate_duplicate_row_count(
    dataframe: pd.DataFrame,
) -> int | None:
    """
    Calculate duplicate rows when values are hashable.

    Return None when the dataset contains arrays, lists,
    dictionaries or other unhashable objects.
    """

    if dataframe.empty:
        return 0

    try:
        return int(
            dataframe.duplicated().sum()
        )
    except TypeError:
        logger.warning(
            "Full-row duplicate checking was skipped "
            "because the dataset contains unhashable values."
        )

        return None


def inspect_dataset(
    file_path: Path,
) -> dict[str, Any]:
    """Return metadata about one dataset."""

    try:
        dataframe = load_dataset(file_path)

    except Exception as error:
        logger.exception(
            "Unable to inspect dataset %s.",
            file_path,
        )

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "file_size_bytes": (
                file_path.stat().st_size
            ),
        }

    pr_number_columns = [
        column
        for column in dataframe.columns
        if str(column).lower()
        in {
            "pr_number",
            "number",
            "pull_request_number",
        }
    ]

    status = (
        "empty"
        if dataframe.empty
        and len(dataframe.columns) == 0
        else "loaded"
    )

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "status": status,
        "row_count": len(dataframe),
        "column_count": len(
            dataframe.columns
        ),
        "columns": [
            str(column)
            for column in dataframe.columns
        ],
        "pr_number_columns": [
            str(column)
            for column in pr_number_columns
        ],
        "contains_pr_number": bool(
            pr_number_columns
        ),
        "duplicate_row_count": (
            calculate_duplicate_row_count(
                dataframe
            )
        ),
        "file_size_bytes": (
            file_path.stat().st_size
        ),
    }


def inspect_all_datasets(
    root_directory: Path,
) -> list[dict[str, Any]]:
    """Inspect all supported datasets."""

    dataset_files = discover_dataset_files(
        root_directory
    )

    return [
        inspect_dataset(file_path)
        for file_path in dataset_files
    ]