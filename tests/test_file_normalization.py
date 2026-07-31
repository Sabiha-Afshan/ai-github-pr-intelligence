"""Tests for changed-file normalization."""

from src.github.file_normalization import (
    normalize_changed_file,
)


def test_changed_file_is_normalized() -> None:
    """Confirm a GitHub file response is normalized."""

    raw_file = {
        "filename": "tests/test_app.py",
        "status": "modified",
        "additions": 10,
        "deletions": 2,
        "changes": 12,
        "raw_url": "https://example.test/raw",
        "blob_url": "https://example.test/blob",
        "patch": "@@ -1 +1 @@",
    }

    result = normalize_changed_file(
        raw_file=raw_file,
        repository="pallets/flask",
        pr_number=123,
    )

    assert result.file_path == "tests/test_app.py"
    assert result.total_changes == 12
    assert result.is_test_file is True
    assert result.file_category == "test"
    assert result.patch_available is True


def test_missing_changes_are_calculated() -> None:
    """Confirm additions and deletions can form total changes."""

    raw_file = {
        "filename": "src/app.py",
        "status": "modified",
        "additions": 5,
        "deletions": 3,
    }

    result = normalize_changed_file(
        raw_file=raw_file,
        repository="pallets/flask",
        pr_number=123,
    )

    assert result.total_changes == 8
