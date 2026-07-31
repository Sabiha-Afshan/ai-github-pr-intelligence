"""Local storage helpers for collected datasets."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def records_to_dataframe(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert records into a DataFrame."""

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def save_records_csv(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save normalized records to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = records_to_dataframe(
        records
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def save_records_parquet(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save normalized records to Parquet."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = records_to_dataframe(
        records
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )


def save_json_records(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save records to JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            records,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def load_existing_records(
    file_path: Path,
) -> list[dict[str, Any]]:
    """Load existing CSV or Parquet records."""

    if not file_path.exists():
        return []

    if file_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(
            file_path
        )
    elif file_path.suffix.lower() in {
        ".parquet",
        ".pq",
    }:
        dataframe = pd.read_parquet(
            file_path
        )
    else:
        raise ValueError(
            "Unsupported dataset format: "
            f"{file_path.suffix}"
        )

    return dataframe.to_dict(
        orient="records"
    )