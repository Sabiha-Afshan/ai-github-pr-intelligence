"""Verify the time-matched 600-PR sampling design."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.discovery import load_dataset
from src.data.quarter_validation import (
    build_quarterly_outcome_summary,
    validate_assigned_periods,
)
from src.data.sampling_validation import (
    compare_population_membership,
    validate_sampling_plan,
    validate_selected_population,
)
from src.utils.paths import (
    CACHE_DIRECTORY,
    DATA_EVALUATION_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
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
    """Run all Stage 3B sampling checks."""

    population_path = (
        RAW_DATA_DIRECTORY / "pallets_flask_time_matched_600_pr_population.csv"
    )

    detailed_path = (
        PROCESSED_DATA_DIRECTORY / "pallets_flask_time_matched_600_detailed.csv"
    )

    selected_checkpoint_path = (
        CACHE_DIRECTORY / "time_matched_selected_population_checkpoint.csv"
    )

    sampling_plan_path = DATA_EVALUATION_DIRECTORY / "time_matched_sampling_plan.csv"

    required_paths = [
        population_path,
        detailed_path,
        sampling_plan_path,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 3B files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    population_dataframe = load_dataset(population_path)

    detailed_dataframe = load_dataset(detailed_path)

    sampling_plan_dataframe = load_dataset(sampling_plan_path)

    selected_population_validation = validate_selected_population(
        detailed_dataframe,
        expected_records=600,
    )

    sampling_plan_validation = validate_sampling_plan(sampling_plan_dataframe)

    period_validation = validate_assigned_periods(detailed_dataframe)

    population_detail_alignment = compare_population_membership(
        population_dataframe,
        detailed_dataframe,
    )

    checkpoint_alignment: dict[str, Any]

    if selected_checkpoint_path.exists():
        checkpoint_dataframe = load_dataset(selected_checkpoint_path)

        checkpoint_alignment = compare_population_membership(
            population_dataframe,
            checkpoint_dataframe,
        )
    else:
        checkpoint_alignment = {
            "validation_passed": False,
            "reason": ("Selected-population checkpoint is missing."),
        }

    quarterly_summary = build_quarterly_outcome_summary(detailed_dataframe)

    unmatched_quarter_count = int((~quarterly_summary["is_time_matched"]).sum())

    quarterly_balance_validation = {
        "quarter_count": len(quarterly_summary),
        "unmatched_quarter_count": (unmatched_quarter_count),
        "merged_total": int(quarterly_summary["merged_count"].sum()),
        "unmerged_total": int(quarterly_summary["unmerged_count"].sum()),
        "validation_passed": (
            unmatched_quarter_count == 0
            and int(quarterly_summary["merged_count"].sum()) == 300
            and int(quarterly_summary["unmerged_count"].sum()) == 300
        ),
    }

    overall_passed = all(
        [
            selected_population_validation.get(
                "validation_passed",
                False,
            ),
            sampling_plan_validation.get(
                "validation_passed",
                False,
            ),
            period_validation.get(
                "validation_passed",
                False,
            ),
            population_detail_alignment.get(
                "validation_passed",
                False,
            ),
            checkpoint_alignment.get(
                "validation_passed",
                False,
            ),
            quarterly_balance_validation.get(
                "validation_passed",
                False,
            ),
        ]
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    quarterly_summary_path = (
        REPORTS_DIRECTORY / "stage_3b_quarterly_outcome_summary.csv"
    )

    completion_report_path = REPORTS_DIRECTORY / "stage_3b_completion_report.json"

    period_mismatch_path = REPORTS_DIRECTORY / "stage_3b_period_mismatches.csv"

    quarterly_summary.to_csv(
        quarterly_summary_path,
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame(
        period_validation.get(
            "mismatch_records",
            [],
        )
    ).to_csv(
        period_mismatch_path,
        index=False,
        encoding="utf-8",
    )

    final_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 3B",
        "selected_population_validation": (selected_population_validation),
        "sampling_plan_validation": (sampling_plan_validation),
        "period_validation": {
            key: value
            for key, value in period_validation.items()
            if key != "mismatch_records"
        },
        "population_detail_alignment": (population_detail_alignment),
        "checkpoint_alignment": (checkpoint_alignment),
        "quarterly_balance_validation": (quarterly_balance_validation),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        final_report,
        completion_report_path,
    )

    print("Stage 3B sampling verification")
    print("=" * 72)

    print()
    print("Selected population:")
    print(
        json.dumps(
            selected_population_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Sampling plan:")
    print(
        json.dumps(
            sampling_plan_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Quarterly outcome summary:")
    print(quarterly_summary.to_string(index=False))

    print()
    print("Period validation:")
    print(
        json.dumps(
            {
                key: value
                for key, value in period_validation.items()
                if key != "mismatch_records"
            },
            indent=2,
            default=str,
        )
    )

    print()
    print()
    print("Checkpoint alignment:")
    print(
        json.dumps(
            checkpoint_alignment,
            indent=2,
            default=str,
        )
    )

    print()
    print("Completion report:")
    print(completion_report_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
