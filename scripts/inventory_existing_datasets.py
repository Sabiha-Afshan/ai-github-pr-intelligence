"""Create an inventory of existing project datasets."""

import json
import sys

import pandas as pd

from src.data.discovery import (
    inspect_all_datasets,
)
from src.utils.paths import (
    DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)


def main() -> int:
    """Inspect existing data files."""

    inspection_records = inspect_all_datasets(DATA_DIRECTORY)

    if not inspection_records:
        print("FAIL: No CSV or Parquet datasets were found.")
        return 1

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_csv = REPORTS_DIRECTORY / "existing_dataset_inventory.csv"

    inventory_json = REPORTS_DIRECTORY / "existing_dataset_inventory.json"

    inventory_dataframe = pd.DataFrame(inspection_records)

    inventory_dataframe.to_csv(
        inventory_csv,
        index=False,
        encoding="utf-8",
    )

    with inventory_json.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            inspection_records,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    display_columns = [
        column
        for column in [
            "file_name",
            "status",
            "row_count",
            "column_count",
            "contains_pr_number",
        ]
        if column in inventory_dataframe.columns
    ]

    print(inventory_dataframe[display_columns].to_string(index=False))

    print()
    print("Inventory CSV:")
    print(inventory_csv)

    print()
    print("Inventory JSON:")
    print(inventory_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
