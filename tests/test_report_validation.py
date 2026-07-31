"""Tests for completion-report validation."""

import json

from src.data.report_validation import (
    validate_stage_report,
)


def test_passing_stage_report(
    tmp_path,
) -> None:
    """Confirm a passing report is recognized."""

    report_path = tmp_path / "report.json"

    report_path.write_text(
        json.dumps({"overall_verification_passed": True}),
        encoding="utf-8",
    )

    result = validate_stage_report(
        stage_name="Test stage",
        report_path=report_path,
    )

    assert result["exists"] is True

    assert result["validation_passed"] is True


def test_missing_stage_report_fails(
    tmp_path,
) -> None:
    """Confirm missing reports fail validation."""

    report_path = tmp_path / "missing.json"

    result = validate_stage_report(
        stage_name="Test stage",
        report_path=report_path,
    )

    assert result["exists"] is False

    assert result["validation_passed"] is False
