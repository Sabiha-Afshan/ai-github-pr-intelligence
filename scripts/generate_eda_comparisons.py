"""Generate Stage 4B comparative EDA reports."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analytics.eda_comparisons import (
    calculate_complexity_summary,
    calculate_file_category_metrics,
    calculate_lifecycle_bands,
    calculate_outcome_comparisons,
    calculate_outlier_summary,
)
from src.data.eda_loader import (
    load_eda_dataset,
)
from src.utils.paths import (
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
    """Generate Stage 4B reports."""

    try:
        dataframe = load_eda_dataset()
    except FileNotFoundError as error:
        print(f"FAIL: {error}")
        return 1

    outcome_comparisons = calculate_outcome_comparisons(dataframe)

    outlier_summary = calculate_outlier_summary(dataframe)

    file_category_metrics = calculate_file_category_metrics(dataframe)

    lifecycle_bands = calculate_lifecycle_bands(dataframe)

    complexity_summary = calculate_complexity_summary(dataframe)

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcome_path = REPORTS_DIRECTORY / "stage_4b_outcome_comparisons.csv"

    outlier_path = REPORTS_DIRECTORY / "stage_4b_outlier_summary.csv"

    file_category_path = REPORTS_DIRECTORY / "stage_4b_file_category_metrics.csv"

    lifecycle_path = REPORTS_DIRECTORY / "stage_4b_lifecycle_bands.csv"

    complexity_path = REPORTS_DIRECTORY / "stage_4b_complexity_summary.csv"

    completion_path = REPORTS_DIRECTORY / "stage_4b_completion_report.json"

    outcome_comparisons.to_csv(
        outcome_path,
        index=False,
        encoding="utf-8",
    )

    outlier_summary.to_csv(
        outlier_path,
        index=False,
        encoding="utf-8",
    )

    file_category_metrics.to_csv(
        file_category_path,
        index=False,
        encoding="utf-8",
    )

    lifecycle_bands.to_csv(
        lifecycle_path,
        index=False,
        encoding="utf-8",
    )

    complexity_summary.to_csv(
        complexity_path,
        index=False,
        encoding="utf-8",
    )

    validation_checks = {
        "dataset_contains_600_prs": (len(dataframe) == 600),
        "outcome_comparisons_created": (not outcome_comparisons.empty),
        "outlier_summary_created": (not outlier_summary.empty),
        "file_category_metrics_created": (not file_category_metrics.empty),
        "lifecycle_total_is_600": (int(lifecycle_bands["total_prs"].sum()) == 600),
        "complexity_summary_created": (not complexity_summary.empty),
    }

    overall_passed = all(validation_checks.values())

    strongest_effects = (
        outcome_comparisons.head(5)[
            [
                "metric",
                "standardized_mean_difference",
                "effect_size_label",
                "merged_median",
                "unmerged_median",
            ]
        ].to_dict(orient="records")
        if not outcome_comparisons.empty
        else []
    )

    completion_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 4B",
        "row_count": len(dataframe),
        "validation_checks": (validation_checks),
        "strongest_outcome_effects": (strongest_effects),
        "generated_reports": {
            "outcome_comparisons": str(outcome_path),
            "outlier_summary": str(outlier_path),
            "file_category_metrics": str(file_category_path),
            "lifecycle_bands": str(lifecycle_path),
            "complexity_summary": str(complexity_path),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    print("Stage 4B comparative EDA")

    print("=" * 76)

    print()
    print("Merged versus unmerged comparisons:")

    print(
        outcome_comparisons[
            [
                "metric",
                "merged_median",
                "unmerged_median",
                "merged_winsorized_mean",
                "unmerged_winsorized_mean",
                "standardized_mean_difference",
                "effect_size_label",
            ]
        ].to_string(index=False)
    )

    print()
    print("Outlier summary:")

    print(
        outlier_summary[
            [
                "column",
                "median",
                "maximum",
                "upper_bound",
                "outlier_count",
                "outlier_percent",
            ]
        ].to_string(index=False)
    )

    print()
    print("File-category metrics:")

    print(file_category_metrics.to_string(index=False))

    print()
    print("Lifecycle bands:")

    print(lifecycle_bands.to_string(index=False))

    print()
    print("Complexity summary:")

    print(complexity_summary.to_string(index=False))

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
        "Overall Stage 4B verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
