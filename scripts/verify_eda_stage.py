"""Verify all Stage 4 EDA outputs and complete the EDA stage."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.eda_stage_validation import (
    validate_chart_directory,
    validate_completion_report,
    validate_csv_report,
    validate_python_file,
)
from src.data.eda_loader import (
    load_eda_dataset,
)
from src.utils.paths import (
    PAGES_DIRECTORY,
    REPORTS_DIRECTORY,
    SRC_DIRECTORY,
)

EXPECTED_CHART_NAMES = {
    "outcome_distribution",
    "quarterly_volume",
    "size_band_merge_rate",
    "lifecycle_merge_rate",
    "file_category_merge_rate",
    "outcome_effect_sizes",
    "total_changes_boxplot",
    "commit_count_boxplot",
}


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
    """Run the complete Stage 4 verification."""

    try:
        dataframe = load_eda_dataset()
    except FileNotFoundError as error:
        print(f"FAIL: {error}")
        return 1

    completion_reports = [
        validate_completion_report(
            stage_name="Stage 4A",
            report_path=(REPORTS_DIRECTORY / "stage_4a_completion_report.json"),
        ),
        validate_completion_report(
            stage_name="Stage 4B",
            report_path=(REPORTS_DIRECTORY / "stage_4b_completion_report.json"),
        ),
        validate_completion_report(
            stage_name="Stage 4C",
            report_path=(REPORTS_DIRECTORY / "stage_4c_completion_report.json"),
        ),
    ]

    csv_reports = [
        validate_csv_report(
            report_name="Outcome metrics",
            report_path=(REPORTS_DIRECTORY / "stage_4a_outcome_metrics.csv"),
        ),
        validate_csv_report(
            report_name="Quarterly metrics",
            report_path=(REPORTS_DIRECTORY / "stage_4a_quarterly_metrics.csv"),
        ),
        validate_csv_report(
            report_name="Size-band metrics",
            report_path=(REPORTS_DIRECTORY / "stage_4a_size_band_metrics.csv"),
        ),
        validate_csv_report(
            report_name="Missing-value summary",
            report_path=(REPORTS_DIRECTORY / "stage_4a_missing_value_summary.csv"),
        ),
        validate_csv_report(
            report_name="Outcome comparisons",
            report_path=(REPORTS_DIRECTORY / "stage_4b_outcome_comparisons.csv"),
        ),
        validate_csv_report(
            report_name="Outlier summary",
            report_path=(REPORTS_DIRECTORY / "stage_4b_outlier_summary.csv"),
        ),
        validate_csv_report(
            report_name="File-category metrics",
            report_path=(REPORTS_DIRECTORY / "stage_4b_file_category_metrics.csv"),
        ),
        validate_csv_report(
            report_name="Lifecycle metrics",
            report_path=(REPORTS_DIRECTORY / "stage_4b_lifecycle_bands.csv"),
        ),
        validate_csv_report(
            report_name="Complexity summary",
            report_path=(REPORTS_DIRECTORY / "stage_4b_complexity_summary.csv"),
        ),
    ]

    chart_validation = validate_chart_directory(
        chart_directory=(REPORTS_DIRECTORY / "stage_4c_charts"),
        expected_chart_names=(EXPECTED_CHART_NAMES),
    )

    python_files = [
        validate_python_file(
            file_name="EDA loader",
            file_path=(SRC_DIRECTORY / "data" / "eda_loader.py"),
            required_text=(
                "load_eda_dataset",
                "prepare_eda_dataframe",
            ),
        ),
        validate_python_file(
            file_name="EDA metrics",
            file_path=(SRC_DIRECTORY / "analytics" / "eda_metrics.py"),
            required_text=(
                "calculate_summary_metrics",
                "calculate_quarterly_metrics",
            ),
        ),
        validate_python_file(
            file_name="EDA comparisons",
            file_path=(SRC_DIRECTORY / "analytics" / "eda_comparisons.py"),
            required_text=(
                "calculate_outcome_comparisons",
                "calculate_outlier_summary",
            ),
        ),
        validate_python_file(
            file_name="EDA charts",
            file_path=(SRC_DIRECTORY / "analytics" / "eda_charts.py"),
            required_text=(
                "build_all_eda_charts",
                "create_quarterly_volume_chart",
            ),
        ),
        validate_python_file(
            file_name="EDA Streamlit page",
            file_path=(SRC_DIRECTORY / "ui" / "eda_page.py"),
            required_text=(
                "render_eda_page",
                "render_filter_sidebar",
            ),
        ),
        validate_python_file(
            file_name="EDA page entry point",
            file_path=(PAGES_DIRECTORY / "1_Exploratory_Analysis.py"),
            required_text=(
                "render_eda_page",
                "set_page_config",
            ),
        ),
    ]

    dataset_validation = {
        "row_count": len(dataframe),
        "unique_pr_count": int(dataframe["pr_number"].nunique()),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "merged_pr_count": int((dataframe["was_merged_numeric"] == 1).sum()),
        "unmerged_pr_count": int((dataframe["was_merged_numeric"] == 0).sum()),
    }

    dataset_validation["validation_passed"] = (
        dataset_validation["row_count"] == 600
        and dataset_validation["unique_pr_count"] == 600
        and dataset_validation["duplicate_row_count"] == 0
        and dataset_validation["merged_pr_count"] == 300
        and dataset_validation["unmerged_pr_count"] == 300
    )

    completion_reports_passed = all(
        report["validation_passed"] for report in completion_reports
    )

    csv_reports_passed = all(report["validation_passed"] for report in csv_reports)

    python_files_passed = all(
        file_result["validation_passed"] for file_result in python_files
    )

    overall_passed = all(
        [
            dataset_validation["validation_passed"],
            completion_reports_passed,
            csv_reports_passed,
            chart_validation["validation_passed"],
            python_files_passed,
        ]
    )

    completion_path = REPORTS_DIRECTORY / "stage_4_completion_report.json"

    final_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 4",
        "stage_status": ("complete" if overall_passed else "review_required"),
        "dataset_validation": (dataset_validation),
        "completion_reports": (completion_reports),
        "csv_reports": (csv_reports),
        "chart_validation": (chart_validation),
        "python_file_validation": (python_files),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        final_report,
        completion_path,
    )

    print("Stage 4 final EDA verification")

    print("=" * 76)

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
    print("Completion reports:")
    print(pd.DataFrame(completion_reports).to_string(index=False))

    print()
    print("CSV reports:")
    print(pd.DataFrame(csv_reports).to_string(index=False))

    print()
    print("Interactive charts:")
    print(
        json.dumps(
            chart_validation,
            indent=2,
            default=str,
        )
    )

    print()
    print("Python application files:")
    print(pd.DataFrame(python_files).to_string(index=False))

    print()
    print(
        "Overall Stage 4 verification passed:",
        overall_passed,
    )

    print()
    print("Stage 4 completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
