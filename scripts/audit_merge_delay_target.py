"""Build and verify the Model 2 merge-delay population."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.models.merge_delay_target import (
    PRIMARY_DELAY_TARGET,
    PRIMARY_DELAY_THRESHOLD_HOURS,
    add_delay_targets,
    attach_split_assignments,
    audit_target_leakage_columns,
    build_merge_duration_summary,
    build_split_target_summary,
    build_threshold_summary,
    create_merged_population,
    validate_delay_dataset,
    validate_source_dataset,
)
from src.utils.paths import (
    DATA_EVALUATION_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)

SOURCE_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_feature_engineered.csv"
)

FALLBACK_SOURCE_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_outlier_flagged.csv"
)

SPLIT_ASSIGNMENT_PATH = (
    DATA_EVALUATION_DIRECTORY / "corrected_model1_time_based_split_assignments.csv"
)

OUTPUT_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "model2_merge_delay_population.csv"

THRESHOLD_REPORT_PATH = REPORTS_DIRECTORY / "stage_6a_delay_threshold_summary.csv"

SPLIT_REPORT_PATH = REPORTS_DIRECTORY / "stage_6a_delay_split_summary.csv"

LEAKAGE_REPORT_PATH = REPORTS_DIRECTORY / "stage_6a_delay_leakage_audit.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6a_completion_report.json"


def save_json(
    payload: Any,
    output_path: Path,
) -> None:
    """Save JSON atomically."""

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


def resolve_source_dataset() -> Path:
    """Return the best available authoritative dataset."""

    for candidate_path in (
        SOURCE_DATASET_PATH,
        FALLBACK_SOURCE_DATASET_PATH,
    ):
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Neither authoritative Model 2 source dataset exists. "
        f"Checked: {SOURCE_DATASET_PATH} and "
        f"{FALLBACK_SOURCE_DATASET_PATH}"
    )


def validate_chronological_split_order(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Check chronological ordering between split date ranges."""

    if "created_at" not in dataframe.columns:
        return {
            "created_at_exists": False,
            "validation_passed": False,
            "reason": ("created_at is required to verify chronological ordering."),
        }

    working = dataframe.copy()

    working["created_at"] = pd.to_datetime(
        working["created_at"],
        utc=True,
        errors="coerce",
    )

    invalid_created_at_count = int(working["created_at"].isna().sum())

    split_ranges = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_rows = working.loc[
            working["split"] == split_name,
            "created_at",
        ]

        split_ranges[split_name] = {
            "row_count": len(split_rows),
            "minimum_created_at": (split_rows.min()),
            "maximum_created_at": (split_rows.max()),
        }

    each_split_present = all(
        split_ranges[split_name]["row_count"] > 0
        for split_name in (
            "train",
            "validation",
            "test",
        )
    )

    chronological_order_valid = (
        each_split_present
        and split_ranges["train"]["maximum_created_at"]
        < split_ranges["validation"]["minimum_created_at"]
        and split_ranges["validation"]["maximum_created_at"]
        < split_ranges["test"]["minimum_created_at"]
    )

    validation_passed = invalid_created_at_count == 0 and chronological_order_valid

    return {
        "created_at_exists": True,
        "invalid_created_at_count": (invalid_created_at_count),
        "split_ranges": (split_ranges),
        "each_split_present": (each_split_present),
        "chronological_order_valid": (chronological_order_valid),
        "validation_passed": (validation_passed),
    }


def main() -> int:
    """Run Stage 6A merge-delay target auditing."""

    try:
        source_path = resolve_source_dataset()
    except FileNotFoundError as error:
        print(f"FAIL: {error}")
        return 1

    if not SPLIT_ASSIGNMENT_PATH.exists():
        print("FAIL: Split assignment file is missing:")
        print(SPLIT_ASSIGNMENT_PATH)
        return 1

    try:
        source_dataframe = load_dataset(source_path)

        split_assignments = load_dataset(SPLIT_ASSIGNMENT_PATH)

        source_validation = validate_source_dataset(source_dataframe)

        if not source_validation["validation_passed"]:
            raise ValueError(f"Source validation failed: {source_validation}")

        merged_population = create_merged_population(source_dataframe)

        delay_dataframe = add_delay_targets(merged_population)

        delay_dataframe = attach_split_assignments(
            delay_dataframe=(delay_dataframe),
            split_assignments=(split_assignments),
        )

        threshold_summary = build_threshold_summary(delay_dataframe)

        split_summary = build_split_target_summary(
            delay_dataframe,
            target_column=(PRIMARY_DELAY_TARGET),
        )

        merge_duration_summary = build_merge_duration_summary(delay_dataframe)

        leakage_audit = audit_target_leakage_columns(delay_dataframe)

        dataset_validation = validate_delay_dataset(delay_dataframe)

        chronological_validation = validate_chronological_split_order(delay_dataframe)

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    primary_target_row = threshold_summary.loc[
        threshold_summary["target_column"] == PRIMARY_DELAY_TARGET
    ]

    if len(primary_target_row) != 1:
        print("FAIL: Primary delay target summary was not created correctly.")
        return 1

    primary_target_record = primary_target_row.iloc[0].to_dict()

    split_balance_valid = bool(split_summary["both_classes_present"].all())

    threshold_suitability_valid = bool(primary_target_record["suitable_for_modelling"])

    OUTPUT_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    delay_dataframe.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    threshold_summary.to_csv(
        THRESHOLD_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    split_summary.to_csv(
        SPLIT_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    leakage_audit.to_csv(
        LEAKAGE_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "output_dataset_exists": (OUTPUT_DATASET_PATH.exists()),
        "threshold_report_exists": (THRESHOLD_REPORT_PATH.exists()),
        "split_report_exists": (SPLIT_REPORT_PATH.exists()),
        "leakage_report_exists": (LEAKAGE_REPORT_PATH.exists()),
    }

    artifact_validation["validation_passed"] = all(artifact_validation.values())

    overall_passed = (
        source_validation["validation_passed"]
        and dataset_validation["validation_passed"]
        and chronological_validation["validation_passed"]
        and split_balance_valid
        and threshold_suitability_valid
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6A",
        "model": ("Model 2 — Merge-delay prediction"),
        "source_dataset": str(source_path),
        "output_dataset": str(OUTPUT_DATASET_PATH),
        "prediction_design": {
            "population": ("Merged pull requests only"),
            "primary_target": (PRIMARY_DELAY_TARGET),
            "target_definition": (
                "1 when merge_hours is greater than 48 hours; otherwise 0"
            ),
            "primary_threshold_hours": (PRIMARY_DELAY_THRESHOLD_HOURS),
            "model_timing": (
                "Historical PR snapshot benchmark. "
                "Features must exclude target, merge "
                "duration and post-outcome fields."
            ),
            "intended_use": (
                "Review prioritisation and delay-risk "
                "decision support, not automatic action."
            ),
        },
        "source_validation": (source_validation),
        "merge_duration_summary": (merge_duration_summary),
        "primary_target_summary": (primary_target_record),
        "split_target_summary": (split_summary.to_dict(orient="records")),
        "split_balance_valid": (split_balance_valid),
        "threshold_suitability_valid": (threshold_suitability_valid),
        "dataset_validation": (dataset_validation),
        "chronological_validation": (chronological_validation),
        "artifact_validation": (artifact_validation),
        "target_leakage_policy": {
            "merge_hours_used_for": ("Target construction and reporting only"),
            "merge_hours_allowed_as_feature": (False),
            "outcome_timestamps_allowed_as_features": (False),
            "target_columns_allowed_as_features": (False),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 6A merge-delay target and population audit")
    print("=" * 88)

    print()
    print("Source validation:")

    print(
        json.dumps(
            source_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Merge-duration summary:")

    print(
        json.dumps(
            merge_duration_summary,
            indent=2,
            default=str,
        )
    )

    print()
    print("Delay-threshold comparison:")

    print(threshold_summary.to_string(index=False))

    print()
    print("Primary 48-hour target by chronological split:")

    print(split_summary.to_string(index=False))

    print()
    print("Chronological validation:")

    print(
        json.dumps(
            chronological_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Dataset validation:")

    print(
        json.dumps(
            dataset_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Artifact validation:")

    print(
        json.dumps(
            artifact_validation,
            indent=2,
        )
    )

    print()
    print(
        "Overall Stage 6A verification passed:",
        overall_passed,
    )

    print()
    print("Model 2 dataset:")
    print(OUTPUT_DATASET_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
