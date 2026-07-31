"""Validation utilities for the completed EDA stage."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """Load a JSON file safely."""

    if not file_path.exists():
        return {
            "exists": False,
            "file_path": str(file_path),
            "payload": None,
        }

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            payload = json.load(input_file)
    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        return {
            "exists": True,
            "file_path": str(file_path),
            "payload": None,
            "error": str(error),
        }

    return {
        "exists": True,
        "file_path": str(file_path),
        "payload": payload,
    }


def validate_completion_report(
    stage_name: str,
    report_path: Path,
) -> dict[str, Any]:
    """Validate one EDA-stage completion report."""

    loaded = load_json_file(report_path)

    if not loaded["exists"]:
        return {
            "stage": stage_name,
            "report_path": str(report_path),
            "exists": False,
            "overall_passed": False,
            "validation_passed": False,
            "reason": ("Completion report is missing."),
        }

    payload = loaded.get("payload")

    if payload is None:
        return {
            "stage": stage_name,
            "report_path": str(report_path),
            "exists": True,
            "overall_passed": False,
            "validation_passed": False,
            "reason": ("Completion report could not be read."),
        }

    overall_passed = bool(
        payload.get(
            "overall_verification_passed",
            False,
        )
    )

    return {
        "stage": stage_name,
        "report_path": str(report_path),
        "exists": True,
        "overall_passed": (overall_passed),
        "validation_passed": (overall_passed),
    }


def validate_csv_report(
    report_name: str,
    report_path: Path,
    minimum_rows: int = 1,
) -> dict[str, Any]:
    """Validate that one CSV report exists and has data."""

    if not report_path.exists():
        return {
            "report_name": report_name,
            "report_path": str(report_path),
            "exists": False,
            "row_count": 0,
            "column_count": 0,
            "validation_passed": False,
        }

    try:
        dataframe = pd.read_csv(report_path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
    ) as error:
        return {
            "report_name": report_name,
            "report_path": str(report_path),
            "exists": True,
            "row_count": 0,
            "column_count": 0,
            "validation_passed": False,
            "error": str(error),
        }

    validation_passed = len(dataframe) >= minimum_rows and len(dataframe.columns) > 0

    return {
        "report_name": report_name,
        "report_path": str(report_path),
        "exists": True,
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "validation_passed": (validation_passed),
    }


def validate_chart_directory(
    chart_directory: Path,
    expected_chart_names: set[str],
) -> dict[str, Any]:
    """Validate generated interactive Plotly charts."""

    if not chart_directory.exists():
        return {
            "chart_directory": str(chart_directory),
            "exists": False,
            "expected_chart_count": len(expected_chart_names),
            "actual_chart_count": 0,
            "missing_charts": sorted(expected_chart_names),
            "empty_charts": [],
            "validation_passed": False,
        }

    chart_files = {
        file_path.stem: file_path for file_path in (chart_directory.glob("*.html"))
    }

    actual_chart_names = set(chart_files)

    missing_charts = expected_chart_names - actual_chart_names

    empty_charts = sorted(
        chart_name
        for chart_name, file_path in chart_files.items()
        if file_path.stat().st_size == 0
    )

    unexpected_charts = actual_chart_names - expected_chart_names

    validation_passed = not missing_charts and not empty_charts

    return {
        "chart_directory": str(chart_directory),
        "exists": True,
        "expected_chart_count": len(expected_chart_names),
        "actual_chart_count": len(actual_chart_names),
        "missing_charts": sorted(missing_charts),
        "unexpected_charts": sorted(unexpected_charts),
        "empty_charts": empty_charts,
        "validation_passed": (validation_passed),
    }


def validate_python_file(
    file_name: str,
    file_path: Path,
    required_text: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate a required Python application file."""

    if not file_path.exists():
        return {
            "file_name": file_name,
            "file_path": str(file_path),
            "exists": False,
            "missing_required_text": list(required_text),
            "validation_passed": False,
        }

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as error:
        return {
            "file_name": file_name,
            "file_path": str(file_path),
            "exists": True,
            "missing_required_text": list(required_text),
            "validation_passed": False,
            "error": str(error),
        }

    missing_required_text = [
        required_value
        for required_value in required_text
        if required_value not in content
    ]

    validation_passed = len(content.strip()) > 0 and not missing_required_text

    return {
        "file_name": file_name,
        "file_path": str(file_path),
        "exists": True,
        "line_count": len(content.splitlines()),
        "missing_required_text": (missing_required_text),
        "validation_passed": (validation_passed),
    }
