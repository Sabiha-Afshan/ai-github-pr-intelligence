"""Approve leakage-safe features for Model 2 merge-delay prediction."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.discovery import load_dataset
from src.models.merge_delay_features import (
    build_feature_group_summary,
    build_feature_review,
    build_modelling_dataset,
    get_approved_feature_names,
    validate_delay_population_source,
    validate_modelling_dataset,
)
from src.models.merge_delay_target import (
    PRIMARY_DELAY_TARGET,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    REPORTS_DIRECTORY,
)

SOURCE_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "model2_merge_delay_population.csv"

OUTPUT_DATASET_PATH = (
    PROCESSED_DATA_DIRECTORY / "model2_merge_delay_approved_features.csv"
)

FEATURE_REVIEW_PATH = REPORTS_DIRECTORY / "stage_6b_merge_delay_feature_review.csv"

FEATURE_GROUP_PATH = REPORTS_DIRECTORY / "stage_6b_merge_delay_feature_groups.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_6b_completion_report.json"


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
    """Run Stage 6B feature approval."""

    stage_6a_path = REPORTS_DIRECTORY / "stage_6a_completion_report.json"

    required_paths = [
        SOURCE_DATASET_PATH,
        stage_6a_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 6B files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    with stage_6a_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        stage_6a_report = json.load(report_file)

    if not stage_6a_report.get(
        "overall_verification_passed",
        False,
    ):
        print("FAIL: Stage 6A did not pass verification.")
        return 1

    try:
        delay_population = load_dataset(SOURCE_DATASET_PATH)

        source_validation = validate_delay_population_source(delay_population)

        if not source_validation["validation_passed"]:
            raise ValueError(
                f"Model 2 population failed validation: {source_validation}"
            )

        feature_review = build_feature_review(delay_population)

        approved_features = get_approved_feature_names(feature_review)

        modelling_dataset = build_modelling_dataset(
            delay_population=(delay_population),
            approved_features=(approved_features),
        )

        feature_group_summary = build_feature_group_summary(feature_review)

        dataset_validation = validate_modelling_dataset(
            dataframe=(modelling_dataset),
            approved_features=(approved_features),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    modelling_dataset.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_review.to_csv(
        FEATURE_REVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    feature_group_summary.to_csv(
        FEATURE_GROUP_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "approved_dataset_exists": (OUTPUT_DATASET_PATH.exists()),
        "feature_review_exists": (FEATURE_REVIEW_PATH.exists()),
        "feature_group_report_exists": (FEATURE_GROUP_PATH.exists()),
    }

    artifact_validation["validation_passed"] = all(artifact_validation.values())

    split_summary = (
        modelling_dataset.groupby(
            "split",
            observed=False,
        )
        .agg(
            row_count=(
                PRIMARY_DELAY_TARGET,
                "size",
            ),
            delayed_count=(
                PRIMARY_DELAY_TARGET,
                "sum",
            ),
        )
        .reset_index()
    )

    split_summary["non_delayed_count"] = (
        split_summary["row_count"] - split_summary["delayed_count"]
    )

    split_summary["delayed_rate"] = (
        split_summary["delayed_count"] / split_summary["row_count"]
    )

    overall_passed = (
        source_validation["validation_passed"]
        and dataset_validation["validation_passed"]
        and artifact_validation["validation_passed"]
        and len(approved_features) > 0
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 6B",
        "model": ("Model 2 — Merge-delay prediction"),
        "source_dataset": str(SOURCE_DATASET_PATH),
        "output_dataset": str(OUTPUT_DATASET_PATH),
        "target": (PRIMARY_DELAY_TARGET),
        "source_validation": (source_validation),
        "approved_feature_count": len(approved_features),
        "approved_features": (approved_features),
        "split_target_summary": (split_summary.to_dict(orient="records")),
        "dataset_validation": (dataset_validation),
        "artifact_validation": (artifact_validation),
        "prediction_design": {
            "population": ("Merged pull requests only"),
            "target": ("Merge duration greater than 48 hours"),
            "feature_policy": ("Contributor-neutral numeric snapshot features only"),
            "blocked_information": [
                "merge duration",
                "outcome timestamps",
                "target-derived columns",
                "author association",
                "raw identity fields",
                "chronological period shortcuts",
                "extraction-quality metadata",
            ],
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    approved_review = feature_review.loc[
        feature_review["approved_as_feature"].astype(bool)
    ]

    blocked_review = feature_review.loc[feature_review["decision"] == "blocked"]

    print("Stage 6B merge-delay feature approval")
    print("=" * 94)

    print()
    print("Source validation:")

    print(
        json.dumps(
            source_validation,
            indent=2,
        )
    )

    print()
    print(
        "Approved feature count:",
        len(approved_features),
    )

    print()
    print("Approved features:")

    print(
        approved_review[
            [
                "column",
                "source_dtype",
                "reason",
            ]
        ].to_string(index=False)
    )

    print()
    print("Blocked feature summary:")

    print(feature_group_summary.to_string(index=False))

    print()
    print(
        "Blocked columns:",
        len(blocked_review),
    )

    print()
    print("Model 2 split distribution:")

    print(split_summary.to_string(index=False))

    print()
    print("Dataset validation:")

    print(
        json.dumps(
            dataset_validation,
            indent=2,
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
        "Overall Stage 6B verification passed:",
        overall_passed,
    )

    print()
    print("Approved Model 2 dataset:")
    print(OUTPUT_DATASET_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
