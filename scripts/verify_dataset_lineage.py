"""Verify extraction completeness and dataset lineage."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.authoritative_sources import (
    AuthoritativeDataset,
    get_authoritative_datasets,
)
from src.data.completeness import (
    compare_pr_coverage,
    inspect_failure_report,
    inspect_pr_dataset,
)
from src.data.discovery import load_dataset
from src.data.split_validation import (
    validate_split_assignments,
)
from src.utils.paths import (
    EVALUATION_DIRECTORY,
    REPORTS_DIRECTORY,
)


def build_dataset_status(
    definition: AuthoritativeDataset,
) -> dict[str, Any]:
    """Inspect one authoritative dataset."""

    if not definition.path.exists():
        return {
            "dataset_name": (definition.dataset_name),
            "stage": definition.stage,
            "file_path": str(definition.path),
            "exists": False,
            "status": "missing",
            "authoritative_for": (definition.authoritative_for),
            "description": (definition.description),
            "validation_passed": False,
        }

    inspection = inspect_pr_dataset(definition.path)

    expected_rows_match = (
        definition.expected_rows is None
        or inspection["row_count"] == definition.expected_rows
    )

    expected_coverage_match = (
        definition.expected_pr_coverage is None
        or inspection["unique_pr_count"] == definition.expected_pr_coverage
    )

    validation_passed = (
        expected_rows_match
        and expected_coverage_match
        and inspection["duplicate_pr_count"] == 0
    )

    return {
        "dataset_name": (definition.dataset_name),
        "stage": definition.stage,
        "file_path": str(definition.path),
        "exists": True,
        "status": ("verified" if validation_passed else "review_required"),
        "expected_rows": (definition.expected_rows),
        "actual_rows": inspection["row_count"],
        "expected_pr_coverage": (definition.expected_pr_coverage),
        "actual_pr_coverage": inspection["unique_pr_count"],
        "duplicate_pr_count": inspection["duplicate_pr_count"],
        "column_count": inspection["column_count"],
        "target_column": inspection["target_column"],
        "target_distribution": inspection["target_distribution"],
        "authoritative_for": (definition.authoritative_for),
        "description": (definition.description),
        "validation_passed": (validation_passed),
    }


def save_json(
    payload: Any,
    output_path: Path,
) -> None:
    """Save JSON safely."""

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
    """Run the full Stage 2F verification."""

    authoritative_datasets = get_authoritative_datasets()

    population_definition = authoritative_datasets[0]

    detailed_definition = authoritative_datasets[1]

    split_definition = authoritative_datasets[-1]

    dataset_statuses = [
        build_dataset_status(definition) for definition in authoritative_datasets
    ]

    population_path = population_definition.path

    coverage_results = []

    if population_path.exists():
        for definition in authoritative_datasets[1:]:
            if definition.path.exists():
                coverage_results.append(
                    {
                        "dataset_name": (definition.dataset_name),
                        **compare_pr_coverage(
                            reference_path=(population_path),
                            comparison_path=(definition.path),
                        ),
                    }
                )

    failure_reports = [
        inspect_failure_report(
            EVALUATION_DIRECTORY / "time_matched_600_extraction_failures.csv"
        ),
        inspect_failure_report(
            EVALUATION_DIRECTORY / "pallets_flask_pr_pilot_failures.csv"
        ),
    ]

    split_validation: dict[str, Any]

    if split_definition.path.exists():
        split_dataframe = load_dataset(split_definition.path)

        split_validation = validate_split_assignments(
            split_dataframe,
            expected_pr_count=600,
        )
    else:
        split_validation = {
            "validation_passed": False,
            "reason": ("Split assignment file is missing."),
        }

    detailed_target_valid = False
    detailed_target_distribution: dict[str, int] = {}

    if detailed_definition.path.exists():
        detailed_inspection = inspect_pr_dataset(detailed_definition.path)

        detailed_target_distribution = detailed_inspection["target_distribution"]

        detailed_target_valid = (
            detailed_inspection["target_column"] == "was_merged"
            and detailed_inspection["target_missing_count"] == 0
            and detailed_target_distribution.get(
                "0",
                0,
            )
            == 300
            and detailed_target_distribution.get(
                "1",
                0,
            )
            == 300
        )

    all_authoritative_files_verified = all(
        status["validation_passed"] for status in dataset_statuses
    )

    all_coverage_exact = all(
        result["exact_pr_alignment"] for result in coverage_results
    )

    zero_extraction_failures = all(
        report["failure_count"] == 0 for report in failure_reports if report["exists"]
    )

    overall_verification_passed = (
        all_authoritative_files_verified
        and all_coverage_exact
        and zero_extraction_failures
        and detailed_target_valid
        and split_validation.get(
            "validation_passed",
            False,
        )
    )

    recollection_required = not (
        detailed_definition.path.exists()
        and any(
            result["dataset_name"] == detailed_definition.dataset_name
            and result["exact_pr_alignment"]
            for result in coverage_results
        )
        and detailed_target_valid
    )

    final_report = {
        "generated_at": datetime.now(UTC),
        "project_stage": "Stage 2F",
        "dataset_statuses": (dataset_statuses),
        "coverage_results": (coverage_results),
        "failure_reports": (failure_reports),
        "split_validation": (split_validation),
        "detailed_target_validation": {
            "target_column": "was_merged",
            "expected_distribution": {
                "0": 300,
                "1": 300,
            },
            "actual_distribution": (detailed_target_distribution),
            "validation_passed": (detailed_target_valid),
        },
        "source_of_truth": {
            "population_membership": str(population_definition.path),
            "raw_detailed_attributes": str(detailed_definition.path),
            "validated_analysis": str(authoritative_datasets[2].path),
            "outlier_analysis": str(authoritative_datasets[3].path),
            "model_features": str(authoritative_datasets[4].path),
            "split_assignments": str(split_definition.path),
        },
        "recollection_required": (recollection_required),
        "overall_verification_passed": (overall_verification_passed),
    }

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_csv_path = REPORTS_DIRECTORY / "authoritative_dataset_status.csv"

    coverage_csv_path = REPORTS_DIRECTORY / "authoritative_dataset_coverage.csv"

    lineage_json_path = REPORTS_DIRECTORY / "dataset_lineage_report.json"

    completion_json_path = REPORTS_DIRECTORY / "stage_2f_completion_report.json"

    pd.DataFrame(dataset_statuses).to_csv(
        status_csv_path,
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "missing_pr_numbers",
                    "outside_reference_pr_numbers",
                }
            }
            for result in coverage_results
        ]
    ).to_csv(
        coverage_csv_path,
        index=False,
        encoding="utf-8",
    )

    save_json(
        {
            "generated_at": datetime.now(UTC),
            "lineage": [
                {
                    "order": index,
                    "dataset_name": (definition.dataset_name),
                    "stage": definition.stage,
                    "file_path": str(definition.path),
                    "authoritative_for": (definition.authoritative_for),
                    "description": (definition.description),
                }
                for index, definition in enumerate(
                    authoritative_datasets,
                    start=1,
                )
            ],
        },
        lineage_json_path,
    )

    save_json(
        final_report,
        completion_json_path,
    )

    print("Stage 2F verification")
    print("=" * 70)

    display_columns = [
        "dataset_name",
        "status",
        "actual_rows",
        "actual_pr_coverage",
        "duplicate_pr_count",
        "validation_passed",
    ]

    status_dataframe = pd.DataFrame(dataset_statuses)

    available_columns = [
        column for column in display_columns if column in status_dataframe.columns
    ]

    print(status_dataframe[available_columns].to_string(index=False))

    print()
    print("Detailed target:")
    print(
        json.dumps(
            final_report["detailed_target_validation"],
            indent=2,
        )
    )

    print()
    print("Split validation:")
    print(
        json.dumps(
            split_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print(
        "GitHub recollection required:",
        recollection_required,
    )

    print(
        "Overall verification passed:",
        overall_verification_passed,
    )

    print()
    print("Completion report:")
    print(completion_json_path)

    return 0 if overall_verification_passed else 1


if __name__ == "__main__":
    sys.exit(main())
