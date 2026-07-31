"""Generate reusable EDA summary reports."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analytics.eda_metrics import (
    calculate_missing_value_summary,
    calculate_outcome_metrics,
    calculate_quarterly_metrics,
    calculate_size_band_metrics,
    calculate_summary_metrics,
)
from src.data.eda_loader import (
    describe_eda_dataset,
    load_eda_dataset,
)
from src.utils.paths import (
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
    """Generate Stage 4A EDA reports."""

    try:
        dataframe = load_eda_dataset()
    except FileNotFoundError as error:
        print(f"FAIL: {error}")
        return 1

    dataset_description = describe_eda_dataset(dataframe)

    summary_metrics = calculate_summary_metrics(dataframe)

    outcome_metrics = calculate_outcome_metrics(dataframe)

    quarterly_metrics = calculate_quarterly_metrics(dataframe)

    size_band_metrics = calculate_size_band_metrics(dataframe)

    missing_value_summary = calculate_missing_value_summary(dataframe)

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcome_path = REPORTS_DIRECTORY / "stage_4a_outcome_metrics.csv"

    quarterly_path = REPORTS_DIRECTORY / "stage_4a_quarterly_metrics.csv"

    size_band_path = REPORTS_DIRECTORY / "stage_4a_size_band_metrics.csv"

    missing_values_path = REPORTS_DIRECTORY / "stage_4a_missing_value_summary.csv"

    completion_path = REPORTS_DIRECTORY / "stage_4a_completion_report.json"

    outcome_metrics.to_csv(
        outcome_path,
        index=False,
        encoding="utf-8",
    )

    quarterly_metrics.to_csv(
        quarterly_path,
        index=False,
        encoding="utf-8",
    )

    size_band_metrics.to_csv(
        size_band_path,
        index=False,
        encoding="utf-8",
    )

    missing_value_summary.to_csv(
        missing_values_path,
        index=False,
        encoding="utf-8",
    )

    validation_checks = {
        "row_count_is_600": (len(dataframe) == 600),
        "unique_pr_count_is_600": (dataset_description["unique_pr_count"] == 600),
        "merged_count_is_300": (summary_metrics["merged_prs"] == 300),
        "unmerged_count_is_300": (summary_metrics["unmerged_prs"] == 300),
        "quarterly_total_is_600": (int(quarterly_metrics["total_prs"].sum()) == 600),
        "outcome_total_is_600": (int(outcome_metrics["pr_count"].sum()) == 600),
    }

    overall_passed = all(validation_checks.values())

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 4A",
        "dataset_description": (dataset_description),
        "summary_metrics": (summary_metrics),
        "validation_checks": (validation_checks),
        "generated_reports": {
            "outcome_metrics": str(outcome_path),
            "quarterly_metrics": str(quarterly_path),
            "size_band_metrics": str(size_band_path),
            "missing_value_summary": str(missing_values_path),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    print("Stage 4A EDA summary generation")
    print("=" * 72)

    print()
    print("Dataset description:")
    print(
        json.dumps(
            dataset_description,
            indent=2,
            default=str,
        )
    )

    print()
    print("Summary metrics:")
    print(
        json.dumps(
            summary_metrics,
            indent=2,
            default=str,
        )
    )

    print()
    print("Outcome metrics:")
    print(outcome_metrics.to_string(index=False))

    print()
    print("Quarterly metrics:")
    print(quarterly_metrics.to_string(index=False))

    print()
    print("Size-band metrics:")
    print(size_band_metrics.to_string(index=False))

    print()
    print("Validation checks:")
    print(
        json.dumps(
            validation_checks,
            indent=2,
        )
    )

    print()
    print(
        "Overall Stage 4A verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
