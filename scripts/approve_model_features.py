"""Create the final approved merge-outcome feature set."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.discovery import load_dataset
from src.models.feature_approval import (
    build_feature_group_summary,
    get_approved_features,
    review_candidate_features,
    validate_approved_features,
)
from src.models.feature_schema import (
    PR_IDENTIFIER_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)


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


def main() -> int:
    """Run Stage 5B manual feature approval."""

    modelling_path = (
        PROCESSED_DATA_DIRECTORY / "model1_merge_outcome_modelling_dataset.csv"
    )

    stage_5a_report_path = REPORTS_DIRECTORY / "stage_5a_completion_report.json"

    if not modelling_path.exists():
        print("FAIL: Stage 5A modelling dataset is missing:")
        print(modelling_path)
        return 1

    if not stage_5a_report_path.exists():
        print("FAIL: Stage 5A completion report is missing:")
        print(stage_5a_report_path)
        return 1

    modelling_dataframe = load_dataset(modelling_path)

    required_columns = {
        PR_IDENTIFIER_COLUMN,
        TARGET_COLUMN,
        SPLIT_COLUMN,
    }

    missing_required_columns = sorted(
        required_columns - set(modelling_dataframe.columns)
    )

    if missing_required_columns:
        print("FAIL: Modelling dataset is missing columns:")
        print(missing_required_columns)
        return 1

    candidate_features = [
        column
        for column in modelling_dataframe.columns
        if column not in required_columns
    ]

    feature_review = review_candidate_features(candidate_features)

    approved_features = get_approved_features(feature_review)

    approval_validation = validate_approved_features(approved_features)

    group_summary = build_feature_group_summary(feature_review)

    approved_dataset = modelling_dataframe[
        [
            PR_IDENTIFIER_COLUMN,
            TARGET_COLUMN,
            SPLIT_COLUMN,
            *approved_features,
        ]
    ].copy()

    excluded_features = feature_review.loc[
        ~feature_review["approved"],
        [
            "feature",
            "reason",
        ],
    ].to_dict(orient="records")

    output_dataset_path = (
        PROCESSED_DATA_DIRECTORY / "model1_merge_outcome_approved_features.csv"
    )

    review_output_path = REPORTS_DIRECTORY / "stage_5b_feature_review.csv"

    group_output_path = REPORTS_DIRECTORY / "stage_5b_feature_groups.csv"

    completion_output_path = REPORTS_DIRECTORY / "stage_5b_completion_report.json"

    approved_dataset.to_csv(
        output_dataset_path,
        index=False,
        encoding="utf-8",
    )

    feature_review.to_csv(
        review_output_path,
        index=False,
        encoding="utf-8",
    )

    group_summary.to_csv(
        group_output_path,
        index=False,
        encoding="utf-8",
    )

    dataset_validation = {
        "row_count": len(approved_dataset),
        "unique_pr_count": int(approved_dataset[PR_IDENTIFIER_COLUMN].nunique()),
        "approved_feature_count": len(approved_features),
        "expected_non_feature_columns": 3,
        "actual_column_count": len(approved_dataset.columns),
    }

    dataset_validation["validation_passed"] = (
        dataset_validation["row_count"] == 600
        and dataset_validation["unique_pr_count"] == 600
        and dataset_validation["actual_column_count"]
        == (dataset_validation["approved_feature_count"] + 3)
    )

    overall_passed = (
        approval_validation["validation_passed"]
        and dataset_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 5B",
        "source_modelling_dataset": str(modelling_path),
        "approved_dataset": str(output_dataset_path),
        "candidate_feature_count": len(candidate_features),
        "approved_feature_count": len(approved_features),
        "excluded_feature_count": len(excluded_features),
        "approved_features": (approved_features),
        "excluded_features": (excluded_features),
        "approval_validation": (approval_validation),
        "dataset_validation": (dataset_validation),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_output_path,
    )

    print("Stage 5B manual feature approval")
    print("=" * 76)

    print()
    print("Excluded after domain review:")

    excluded_table = feature_review[~feature_review["approved"]][
        [
            "feature",
            "reason",
        ]
    ]

    if excluded_table.empty:
        print("No features were excluded.")
    else:
        print(excluded_table.to_string(index=False))

    print()
    print("Approved feature groups:")

    print(group_summary.to_string(index=False))

    print()
    print("Approval validation:")

    print(
        json.dumps(
            approval_validation,
            indent=2,
        )
    )

    print()
    print("Approved dataset validation:")

    print(
        json.dumps(
            dataset_validation,
            indent=2,
        )
    )

    print()
    print(
        "Overall Stage 5B verification passed:",
        overall_passed,
    )

    print()
    print("Approved modelling dataset:")
    print(output_dataset_path)

    print()
    print("Completion report:")
    print(completion_output_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
