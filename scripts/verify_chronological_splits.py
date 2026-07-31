"""Verify chronological train-validation-test splits."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.chronological_validation import (
    compare_split_feature_membership,
    validate_basic_split_integrity,
    validate_chronological_order,
    validate_complete_quarter_assignment,
    validate_timestamp_ordering,
)
from src.data.discovery import load_dataset
from src.utils.paths import (
    DATA_EVALUATION_DIRECTORY,
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
    """Run all Stage 3C checks."""

    split_path = (
        DATA_EVALUATION_DIRECTORY / "corrected_model1_time_based_split_assignments.csv"
    )

    feature_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_corrected_600_feature_engineered.csv"
    )

    required_paths = [
        split_path,
        feature_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 3C files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    split_dataframe = load_dataset(split_path)

    feature_dataframe = load_dataset(feature_path)

    basic_integrity = validate_basic_split_integrity(
        split_dataframe,
        expected_rows=600,
    )

    quarter_assignment = validate_complete_quarter_assignment(split_dataframe)

    chronological_order = validate_chronological_order(split_dataframe)

    timestamp_ordering = validate_timestamp_ordering(split_dataframe)

    membership_alignment = compare_split_feature_membership(
        split_dataframe,
        feature_dataframe,
    )

    overall_passed = all(
        [
            basic_integrity.get(
                "validation_passed",
                False,
            ),
            quarter_assignment.get(
                "validation_passed",
                False,
            ),
            chronological_order.get(
                "validation_passed",
                False,
            ),
            timestamp_ordering.get(
                "validation_passed",
                False,
            ),
            membership_alignment.get(
                "validation_passed",
                False,
            ),
        ]
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    quarter_output = REPORTS_DIRECTORY / "stage_3c_quarter_assignments.csv"

    completion_output = REPORTS_DIRECTORY / "stage_3c_completion_report.json"

    pd.DataFrame(
        quarter_assignment.get(
            "quarter_assignments",
            [],
        )
    ).to_csv(
        quarter_output,
        index=False,
        encoding="utf-8",
    )

    final_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 3C",
        "basic_integrity": (basic_integrity),
        "quarter_assignment": {
            key: value
            for key, value in quarter_assignment.items()
            if key != "quarter_assignments"
        },
        "chronological_order": (chronological_order),
        "timestamp_ordering": (timestamp_ordering),
        "membership_alignment": (membership_alignment),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        final_report,
        completion_output,
    )

    print("Stage 3C chronological split verification")
    print("=" * 76)

    print()
    print("Basic split integrity:")
    print(
        json.dumps(
            basic_integrity,
            indent=2,
            default=str,
        )
    )

    print()
    print("Complete-quarter assignment:")
    print(
        json.dumps(
            {
                key: value
                for key, value in quarter_assignment.items()
                if key != "quarter_assignments"
            },
            indent=2,
            default=str,
        )
    )

    print()
    print("Chronological period boundaries:")
    print(
        json.dumps(
            chronological_order,
            indent=2,
            default=str,
        )
    )

    print()
    print("Timestamp boundaries:")
    print(
        json.dumps(
            timestamp_ordering,
            indent=2,
            default=str,
        )
    )

    print()
    print("Feature dataset alignment:")
    print(
        json.dumps(
            membership_alignment,
            indent=2,
            default=str,
        )
    )

    print()
    print(
        "Overall Stage 3C verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_output)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
