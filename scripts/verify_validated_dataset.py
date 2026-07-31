"""Run Stage 3A formal data-quality verification."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.dataset_comparison import (
    compare_dataset_populations,
    compare_shared_columns,
)
from src.data.discovery import load_dataset
from src.data.quality_checks import (
    run_all_quality_checks,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)


def save_json(
    payload: Any,
    output_path: Path,
) -> None:
    """Save a JSON report atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    temporary_path.replace(output_path)


def main() -> int:
    """Run all Stage 3A checks."""

    detailed_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_time_matched_600_detailed.csv"
    )

    validated_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_validated.csv"
    )

    required_paths = [
        detailed_path,
        validated_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required datasets are missing:")

        for path in missing_paths:
            print(path)

        return 1

    detailed_dataframe = load_dataset(detailed_path)

    validated_dataframe = load_dataset(validated_path)

    quality_results = run_all_quality_checks(
        validated_dataframe,
        expected_rows=600,
    )

    population_comparison = compare_dataset_populations(
        source_dataframe=(detailed_dataframe),
        validated_dataframe=(validated_dataframe),
    )

    missing_value_comparison = compare_shared_columns(
        source_dataframe=(detailed_dataframe),
        validated_dataframe=(validated_dataframe),
    )

    error_checks = quality_results[quality_results["severity"] == "error"]

    failed_error_checks = error_checks[~error_checks["passed"]]

    quality_passed = failed_error_checks.empty

    population_passed = bool(population_comparison["exact_population_alignment"])

    overall_passed = quality_passed and population_passed

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_output = REPORTS_DIRECTORY / "stage_3a_quality_checks.csv"

    missing_output = REPORTS_DIRECTORY / "stage_3a_missing_value_comparison.csv"

    population_output = REPORTS_DIRECTORY / "stage_3a_population_comparison.json"

    completion_output = REPORTS_DIRECTORY / "stage_3a_completion_report.json"

    quality_results.to_csv(
        quality_output,
        index=False,
        encoding="utf-8",
    )

    missing_value_comparison.to_csv(
        missing_output,
        index=False,
        encoding="utf-8",
    )

    save_json(
        population_comparison,
        population_output,
    )

    save_json(
        {
            "generated_at": datetime.now(UTC),
            "stage": "Stage 3A",
            "validated_dataset": str(validated_path),
            "quality_check_count": len(quality_results),
            "failed_error_check_count": len(failed_error_checks),
            "quality_checks_passed": (quality_passed),
            "population_alignment_passed": (population_passed),
            "overall_verification_passed": (overall_passed),
        },
        completion_output,
    )

    print("Stage 3A data-quality verification")
    print("=" * 72)

    print(
        quality_results[
            [
                "check_name",
                "passed",
                "observed_value",
                "expected_value",
            ]
        ].to_string(index=False)
    )

    print()
    print("Population comparison:")
    print(
        json.dumps(
            population_comparison,
            indent=2,
            default=str,
        )
    )

    print()
    print(
        "Overall Stage 3A verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_output)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
