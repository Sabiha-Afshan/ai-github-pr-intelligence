"""Validate Stage 5A model features, target and splits."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.models.feature_schema import (
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from src.models.feature_validation import (
    build_modelling_dataset,
    calculate_feature_missing_summary,
    validate_feature_matrix,
    validate_splits,
    validate_target,
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
    """Run Stage 5A model-input validation."""

    feature_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_feature_engineered.csv"
    )

    split_path = (
        DATA_EVALUATION_DIRECTORY / "corrected_model1_time_based_split_assignments.csv"
    )

    missing_paths = [
        path
        for path in (
            feature_path,
            split_path,
        )
        if not path.exists()
    ]

    if missing_paths:
        print("FAIL: Required Stage 5A files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    feature_dataframe = load_dataset(feature_path)

    split_dataframe = load_dataset(split_path)

    (
        modelling_dataframe,
        feature_matrix,
        feature_audit,
    ) = build_modelling_dataset(
        feature_dataframe=(feature_dataframe),
        split_dataframe=(split_dataframe),
    )

    target_validation = validate_target(modelling_dataframe)

    split_validation = validate_splits(modelling_dataframe)

    feature_validation = validate_feature_matrix(
        feature_matrix=(feature_matrix),
        source_dataframe=(modelling_dataframe),
    )

    missing_summary = calculate_feature_missing_summary(feature_matrix)

    identifier_validation = {
        "pr_number_present": (PR_IDENTIFIER_COLUMN in modelling_dataframe.columns),
        "unique_pr_count": int(modelling_dataframe[PR_IDENTIFIER_COLUMN].nunique()),
        "duplicate_pr_count": int(
            modelling_dataframe.duplicated(subset=[PR_IDENTIFIER_COLUMN]).sum()
        ),
    }

    identifier_validation["validation_passed"] = (
        identifier_validation["pr_number_present"]
        and identifier_validation["unique_pr_count"] == 600
        and identifier_validation["duplicate_pr_count"] == 0
    )

    required_columns_validation = {
        "target_column_present": (TARGET_COLUMN in modelling_dataframe.columns),
        "split_column_present": (SPLIT_COLUMN in modelling_dataframe.columns),
        "identifier_column_present": (
            PR_IDENTIFIER_COLUMN in modelling_dataframe.columns
        ),
    }

    required_columns_validation["validation_passed"] = all(
        required_columns_validation.values()
    )

    overall_passed = all(
        [
            identifier_validation["validation_passed"],
            required_columns_validation["validation_passed"],
            target_validation["validation_passed"],
            split_validation["validation_passed"],
            feature_validation["validation_passed"],
        ]
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    modelling_output = (
        PROCESSED_DATA_DIRECTORY / "model1_merge_outcome_modelling_dataset.csv"
    )

    audit_output = REPORTS_DIRECTORY / "stage_5a_feature_audit.csv"

    missing_output = REPORTS_DIRECTORY / "stage_5a_feature_missing_summary.csv"

    completion_output = REPORTS_DIRECTORY / "stage_5a_completion_report.json"

    modelling_dataframe.to_csv(
        modelling_output,
        index=False,
        encoding="utf-8",
    )

    feature_audit.to_csv(
        audit_output,
        index=False,
        encoding="utf-8",
    )

    missing_summary.to_csv(
        missing_output,
        index=False,
        encoding="utf-8",
    )

    selected_features = feature_audit[feature_audit["status"] == "selected"][
        "column"
    ].tolist()

    excluded_features = feature_audit[feature_audit["status"] == "excluded"][
        [
            "column",
            "reason",
        ]
    ].to_dict(orient="records")

    completion_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 5A",
        "source_feature_dataset": str(feature_path),
        "source_split_dataset": str(split_path),
        "generated_modelling_dataset": str(modelling_output),
        "identifier_validation": (identifier_validation),
        "required_columns_validation": (required_columns_validation),
        "target_validation": (target_validation),
        "split_validation": (split_validation),
        "feature_validation": (feature_validation),
        "selected_features": (selected_features),
        "excluded_features": (excluded_features),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_output,
    )

    print("Stage 5A model feature validation")

    print("=" * 76)

    print()
    print("Identifier validation:")
    print(
        json.dumps(
            identifier_validation,
            indent=2,
        )
    )

    print()
    print("Target validation:")
    print(
        json.dumps(
            target_validation,
            indent=2,
            default=str,
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
    print("Feature validation:")
    print(
        json.dumps(
            {
                key: value
                for key, value in feature_validation.items()
                if key != "feature_names"
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("Selected model features:")

    print(pd.DataFrame({"feature": (selected_features)}).to_string(index=False))

    print()
    print("Excluded-column summary:")

    print(
        feature_audit[feature_audit["status"] == "excluded"][
            [
                "column",
                "reason",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Overall Stage 5A verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_output)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
