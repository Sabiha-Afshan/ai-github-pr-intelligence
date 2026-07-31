"""Validate completion reports from earlier stages."""

import json
from pathlib import Path
from typing import Any


def load_json_report(
    report_path: Path,
) -> dict[str, Any]:
    """Load one JSON report."""

    if not report_path.exists():
        return {
            "exists": False,
            "report_path": str(report_path),
        }

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        payload = json.load(report_file)

    return {
        "exists": True,
        "report_path": str(report_path),
        "payload": payload,
    }


def identify_pass_value(
    payload: dict[str, Any],
) -> bool | None:
    """Find the main pass field in a report."""

    candidates = (
        "overall_verification_passed",
        "overall_passed",
        "validation_passed",
        "quality_checks_passed",
    )

    for candidate in candidates:
        if candidate in payload:
            return bool(payload[candidate])

    return None


def validate_stage_report(
    stage_name: str,
    report_path: Path,
) -> dict[str, Any]:
    """Validate one prior-stage completion report."""

    result = load_json_report(
        report_path
    )

    if not result["exists"]:
        return {
            "stage": stage_name,
            "report_path": str(report_path),
            "exists": False,
            "pass_value": None,
            "validation_passed": False,
        }

    payload = result["payload"]

    pass_value = identify_pass_value(
        payload
    )

    return {
        "stage": stage_name,
        "report_path": str(report_path),
        "exists": True,
        "pass_value": pass_value,
        "validation_passed": (
            pass_value is True
        ),
    }