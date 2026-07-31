"""Reconcile the existing 600-PR population."""

import json
import sys

from src.data.reconciliation import (
    reconcile_pr_datasets,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)


def main() -> int:
    """Create canonical reconciliation outputs."""

    population_path = RAW_DATA_DIRECTORY / (
        "pallets_flask_time_matched_600_pr_population.csv"
    )

    summary_path = RAW_DATA_DIRECTORY / "pallets_flask_closed_pr_summaries.parquet"

    detailed_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_time_matched_600_detailed.csv"
    )

    required_files = [
        population_path,
        summary_path,
        detailed_path,
    ]

    missing_files = [
        file_path for file_path in required_files if not file_path.exists()
    ]

    if missing_files:
        print("FAIL: Required files are missing:")

        for file_path in missing_files:
            print(file_path)

        return 1

    (
        canonical_dataframe,
        coverage_dataframe,
        missing_lists,
        reconciliation_summary,
    ) = reconcile_pr_datasets(
        population_path=population_path,
        summary_path=summary_path,
        detailed_path=detailed_path,
    )

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_csv_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_canonical_600_pr_population.csv"
    )

    canonical_parquet_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_canonical_600_pr_population.parquet"
    )

    coverage_path = REPORTS_DIRECTORY / "pallets_flask_dataset_coverage.csv"

    missing_path = REPORTS_DIRECTORY / "pallets_flask_missing_pr_numbers.json"

    summary_path_output = (
        REPORTS_DIRECTORY / "pallets_flask_reconciliation_summary.json"
    )

    canonical_dataframe.to_csv(
        canonical_csv_path,
        index=False,
        encoding="utf-8",
    )

    canonical_dataframe.to_parquet(
        canonical_parquet_path,
        index=False,
    )

    coverage_dataframe.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8",
    )

    with missing_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            missing_lists,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    with summary_path_output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            reconciliation_summary,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("Dataset reconciliation completed")
    print("-" * 60)

    print(coverage_dataframe.to_string(index=False))

    print()
    print("Population validation:")
    print(
        json.dumps(
            reconciliation_summary["population_validation"],
            indent=2,
            default=str,
        )
    )

    print()
    print("Canonical CSV:")
    print(canonical_csv_path)

    print()
    print("Canonical Parquet:")
    print(canonical_parquet_path)

    print()
    print("Missing PR report:")
    print(missing_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
