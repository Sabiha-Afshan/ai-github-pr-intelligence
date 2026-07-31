"""Tests for final EDA-stage validation."""

import json

import pandas as pd

from src.analytics.eda_stage_validation import (
    validate_chart_directory,
    validate_completion_report,
    validate_csv_report,
    validate_python_file,
)


def test_completion_report_passes(
    tmp_path,
) -> None:
    """Confirm a passing completion report is accepted."""

    report_path = tmp_path / "completion.json"

    report_path.write_text(
        json.dumps({"overall_verification_passed": True}),
        encoding="utf-8",
    )

    result = validate_completion_report(
        stage_name="Stage X",
        report_path=report_path,
    )

    assert result["exists"] is True

    assert result["validation_passed"] is True


def test_failed_completion_report_fails(
    tmp_path,
) -> None:
    """Confirm a failed completion report is rejected."""

    report_path = tmp_path / "completion.json"

    report_path.write_text(
        json.dumps({"overall_verification_passed": False}),
        encoding="utf-8",
    )

    result = validate_completion_report(
        stage_name="Stage X",
        report_path=report_path,
    )

    assert result["validation_passed"] is False


def test_csv_report_validation(
    tmp_path,
) -> None:
    """Confirm a populated CSV report passes."""

    report_path = tmp_path / "report.csv"

    pd.DataFrame(
        {
            "metric": [
                "A",
                "B",
            ],
            "value": [
                1,
                2,
            ],
        }
    ).to_csv(
        report_path,
        index=False,
    )

    result = validate_csv_report(
        report_name="Test report",
        report_path=report_path,
    )

    assert result["row_count"] == 2

    assert result["validation_passed"] is True


def test_chart_directory_validation(
    tmp_path,
) -> None:
    """Confirm expected HTML charts are detected."""

    chart_directory = tmp_path / "charts"

    chart_directory.mkdir()

    (chart_directory / "chart_one.html").write_text(
        "<html>chart one</html>",
        encoding="utf-8",
    )

    (chart_directory / "chart_two.html").write_text(
        "<html>chart two</html>",
        encoding="utf-8",
    )

    result = validate_chart_directory(
        chart_directory=(chart_directory),
        expected_chart_names={
            "chart_one",
            "chart_two",
        },
    )

    assert result["actual_chart_count"] == 2

    assert result["missing_charts"] == []

    assert result["validation_passed"] is True


def test_python_file_validation(
    tmp_path,
) -> None:
    """Confirm required Python text is detected."""

    file_path = tmp_path / "example.py"

    file_path.write_text(
        ("def render_page():\n    return True\n"),
        encoding="utf-8",
    )

    result = validate_python_file(
        file_name="Example",
        file_path=file_path,
        required_text=("render_page",),
    )

    assert result["exists"] is True

    assert result["validation_passed"] is True


def test_missing_python_text_fails(
    tmp_path,
) -> None:
    """Confirm missing required text fails."""

    file_path = tmp_path / "example.py"

    file_path.write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    result = validate_python_file(
        file_name="Example",
        file_path=file_path,
        required_text=("render_page",),
    )

    assert result["validation_passed"] is False

    assert result["missing_required_text"] == ["render_page"]
