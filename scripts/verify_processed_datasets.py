"""Verify final processed datasets and complete Stage 3."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.processed_dataset_validation import (
    compare_pr_membership,
    compare_shared_identity_values,
    compare_target_values,
    inspect_processed_dataset,
)
from src.data.report_validation import (
    validate_stage_report,
)
from src.utils.paths import (
    DATA_EVALUATION_DIRECTORY,
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
    """Run Stage 3D verification."""

    dataset_definitions = {
        "Detailed dataset": (
            PROCESSED_DATA_DIRECTORY / "pallets_flask_time_matched_600_detailed.csv"
        ),
        "Validated dataset": (
            PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_validated.csv"
        ),
        "Outlier-flagged dataset": (
            PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_outlier_flagged.csv"
        ),
        "Feature-engineered dataset": (
            PROCESSED_DATA_DIRECTORY
            / "pallets_flask_corrected_600_feature_engineered.csv"
        ),
        "Split assignments": (
            DATA_EVALUATION_DIRECTORY
            / "corrected_model1_time_based_split_assignments.csv"
        ),
    }

    dataset_statuses = [
        inspect_processed_dataset(
            dataset_name=dataset_name,
            file_path=file_path,
            expected_rows=600,
            require_target=(dataset_name != "Split assignments"),
        )
        for dataset_name, file_path in dataset_definitions.items()
    ]

    missing_datasets = [
        status["dataset_name"]
        for status in dataset_statuses
        if not status.get(
            "exists",
            False,
        )
    ]

    if missing_datasets:
        print("FAIL: Required datasets are missing:")

        for dataset_name in missing_datasets:
            print(dataset_name)

        return 1

    dataframes = {
        dataset_name: load_dataset(file_path)
        for dataset_name, file_path in dataset_definitions.items()
    }

    reference_name = "Detailed dataset"

    reference_dataframe = dataframes[reference_name]

    membership_results = []

    target_results = []

    shared_value_results = []

    for comparison_name in (
        "Validated dataset",
        "Outlier-flagged dataset",
        "Feature-engineered dataset",
        "Split assignments",
    ):
        comparison_dataframe = dataframes[comparison_name]

        membership_results.append(
            compare_pr_membership(
                reference_dataframe=(reference_dataframe),
                comparison_dataframe=(comparison_dataframe),
                reference_name=reference_name,
                comparison_name=(comparison_name),
            )
        )

    for comparison_name in (
        "Validated dataset",
        "Outlier-flagged dataset",
        "Feature-engineered dataset",
    ):
        comparison_dataframe = dataframes[comparison_name]

        target_results.append(
            compare_target_values(
                reference_dataframe=(reference_dataframe),
                comparison_dataframe=(comparison_dataframe),
                reference_name=reference_name,
                comparison_name=(comparison_name),
            )
        )

        shared_value_results.extend(
            compare_shared_identity_values(
                reference_dataframe=(reference_dataframe),
                comparison_dataframe=(comparison_dataframe),
                reference_name=reference_name,
                comparison_name=(comparison_name),
            )
        )

    previous_stage_reports = [
        validate_stage_report(
            stage_name="Stage 3A",
            report_path=(REPORTS_DIRECTORY / "stage_3a_completion_report.json"),
        ),
        validate_stage_report(
            stage_name="Stage 3B",
            report_path=(REPORTS_DIRECTORY / "stage_3b_completion_report.json"),
        ),
        validate_stage_report(
            stage_name="Stage 3C",
            report_path=(REPORTS_DIRECTORY / "stage_3c_completion_report.json"),
        ),
    ]

    dataset_status_passed = all(
        status.get(
            "validation_passed",
            False,
        )
        for status in dataset_statuses
    )

    membership_passed = all(
        result.get(
            "validation_passed",
            False,
        )
        for result in membership_results
    )

    target_passed = all(
        result.get(
            "validation_passed",
            False,
        )
        for result in target_results
    )

    shared_values_passed = all(
        result.get(
            "validation_passed",
            False,
        )
        for result in shared_value_results
    )

    previous_reports_passed = all(
        result.get(
            "validation_passed",
            False,
        )
        for result in previous_stage_reports
    )

    overall_passed = all(
        [
            dataset_status_passed,
            membership_passed,
            target_passed,
            shared_values_passed,
            previous_reports_passed,
        ]
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_status_path = REPORTS_DIRECTORY / "stage_3d_dataset_status.csv"

    membership_path = REPORTS_DIRECTORY / "stage_3d_membership_comparison.csv"

    target_path = REPORTS_DIRECTORY / "stage_3d_target_comparison.csv"

    shared_values_path = REPORTS_DIRECTORY / "stage_3d_shared_value_comparison.csv"

    stage_report_path = REPORTS_DIRECTORY / "stage_3_completion_report.json"

    pd.DataFrame(dataset_statuses).to_csv(
        dataset_status_path,
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
                    "missing_from_comparison",
                    "additional_in_comparison",
                }
            }
            for result in membership_results
        ]
    ).to_csv(
        membership_path,
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "mismatch_records"}
            for result in target_results
        ]
    ).to_csv(
        target_path,
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame(shared_value_results).to_csv(
        shared_values_path,
        index=False,
        encoding="utf-8",
    )

    final_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 3D",
        "stage_3_status": ("complete" if overall_passed else "review_required"),
        "dataset_status_passed": (dataset_status_passed),
        "membership_alignment_passed": (membership_passed),
        "target_consistency_passed": (target_passed),
        "shared_value_consistency_passed": (shared_values_passed),
        "previous_stage_reports_passed": (previous_reports_passed),
        "previous_stage_reports": (previous_stage_reports),
        "authoritative_sources": {
            "detailed_data": str(dataset_definitions["Detailed dataset"]),
            "validated_data": str(dataset_definitions["Validated dataset"]),
            "outlier_data": str(dataset_definitions["Outlier-flagged dataset"]),
            "model_features": str(dataset_definitions["Feature-engineered dataset"]),
            "split_assignments": str(dataset_definitions["Split assignments"]),
        },
        "dataset_counts": {
            status["dataset_name"]: {
                "rows": status.get("row_count"),
                "columns": status.get("column_count"),
                "unique_prs": status.get("unique_pr_count"),
            }
            for status in dataset_statuses
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        final_report,
        stage_report_path,
    )

    print("Stage 3D processed dataset verification")

    print("=" * 76)

    print()
    print("Dataset status:")

    print(
        pd.DataFrame(dataset_statuses)[
            [
                "dataset_name",
                "status",
                "row_count",
                "column_count",
                "unique_pr_count",
                "validation_passed",
            ]
        ].to_string(index=False)
    )

    print()
    print("Population membership:")

    print(
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "missing_from_comparison",
                        "additional_in_comparison",
                    }
                }
                for result in membership_results
            ]
        ).to_string(index=False)
    )

    print()
    print("Target consistency:")

    print(
        pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key != "mismatch_records"
                }
                for result in target_results
            ]
        ).to_string(index=False)
    )

    print()
    print("Shared identity consistency:")

    if shared_value_results:
        print(pd.DataFrame(shared_value_results).to_string(index=False))
    else:
        print("No shared identity columns were available for comparison.")

    print()
    print("Previous Stage 3 reports:")

    print(pd.DataFrame(previous_stage_reports).to_string(index=False))

    print()
    print(
        "Overall Stage 3 verification passed:",
        overall_passed,
    )

    print()
    print("Stage 3 completion report:")
    print(stage_report_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
