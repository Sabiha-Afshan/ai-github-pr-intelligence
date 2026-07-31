"""Generate Stage 4C interactive EDA charts."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.analytics.eda_charts import (
    build_all_eda_charts,
    describe_chart_collection,
)
from src.analytics.eda_comparisons import (
    calculate_file_category_metrics,
    calculate_lifecycle_bands,
    calculate_outcome_comparisons,
)
from src.analytics.eda_metrics import (
    calculate_quarterly_metrics,
    calculate_size_band_metrics,
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


def save_chart_html(
    figure: go.Figure,
    output_path: Path,
) -> None:
    """Save one interactive Plotly chart."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )


def main() -> int:
    """Generate all Stage 4C charts."""

    try:
        dataframe = load_eda_dataset()
    except FileNotFoundError as error:
        print(f"FAIL: {error}")
        return 1

    quarterly_metrics = calculate_quarterly_metrics(dataframe)

    size_band_metrics = calculate_size_band_metrics(dataframe)

    outcome_comparisons = calculate_outcome_comparisons(dataframe)

    file_category_metrics = calculate_file_category_metrics(dataframe)

    lifecycle_metrics = calculate_lifecycle_bands(dataframe)

    figures = build_all_eda_charts(
        dataframe=dataframe,
        quarterly_metrics=quarterly_metrics,
        size_band_metrics=size_band_metrics,
        outcome_comparisons=outcome_comparisons,
        file_category_metrics=(file_category_metrics),
        lifecycle_metrics=(lifecycle_metrics),
    )

    chart_directory = REPORTS_DIRECTORY / "stage_4c_charts"

    generated_files: dict[
        str,
        str,
    ] = {}

    for chart_name, figure in figures.items():
        output_path = chart_directory / f"{chart_name}.html"

        save_chart_html(
            figure,
            output_path,
        )

        generated_files[chart_name] = str(output_path)

    collection_description = describe_chart_collection(figures)

    files_exist = all(
        Path(file_path).exists() for file_path in generated_files.values()
    )

    files_non_empty = all(
        Path(file_path).stat().st_size > 0 for file_path in generated_files.values()
    )

    validation_checks = {
        "dataset_contains_600_prs": (len(dataframe) == 600),
        "expected_chart_count_created": (collection_description["chart_count"] == 8),
        "all_figures_valid": (collection_description["all_figures_valid"]),
        "all_chart_files_exist": (files_exist),
        "all_chart_files_non_empty": (files_non_empty),
        "all_charts_have_traces": all(
            trace_count > 0
            for trace_count in collection_description["trace_counts"].values()
        ),
    }

    overall_passed = all(validation_checks.values())

    completion_path = REPORTS_DIRECTORY / "stage_4c_completion_report.json"

    completion_report = {
        "generated_at": (datetime.now(UTC)),
        "stage": "Stage 4C",
        "chart_directory": str(chart_directory),
        "chart_collection": (collection_description),
        "generated_files": (generated_files),
        "validation_checks": (validation_checks),
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        completion_path,
    )

    chart_summary = pd.DataFrame(
        [
            {
                "chart_name": (chart_name),
                "trace_count": (collection_description["trace_counts"][chart_name]),
                "output_path": (generated_files[chart_name]),
            }
            for chart_name in sorted(figures)
        ]
    )

    print("Stage 4C interactive chart generation")

    print("=" * 76)

    print()
    print("Generated charts:")

    print(chart_summary.to_string(index=False))

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
        "Overall Stage 4C verification passed:",
        overall_passed,
    )

    print()
    print("Completion report:")
    print(completion_path)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
