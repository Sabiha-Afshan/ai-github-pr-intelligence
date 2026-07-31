"""Tests for GitHub PR normalization."""

from src.github.normalization import (
    extract_label_names,
    normalize_non_negative_integer,
    normalize_pull_request,
)


def create_raw_pull_request() -> dict:
    """Return a realistic GitHub PR response."""

    return {
        "number": 123,
        "title": "Improve error handling",
        "body": "Adds clearer error messages.",
        "state": "closed",
        "draft": False,
        "user": {
            "login": "example-user",
        },
        "author_association": "CONTRIBUTOR",
        "created_at": "2025-01-01T10:00:00Z",
        "updated_at": "2025-01-02T10:00:00Z",
        "closed_at": "2025-01-03T10:00:00Z",
        "merged_at": "2025-01-03T10:00:00Z",
        "base": {
            "ref": "main",
        },
        "head": {
            "ref": "feature/error-handling",
        },
        "html_url": ("https://github.com/pallets/flask/pull/123"),
        "labels": [
            {
                "name": "documentation",
            },
            {
                "name": "quality",
            },
        ],
        "additions": 10,
        "deletions": 2,
        "changed_files": 3,
        "commits": 1,
    }


def test_normalize_merged_pull_request() -> None:
    """Confirm merged records are normalized."""

    result = normalize_pull_request(
        raw_pull_request=(create_raw_pull_request()),
        repository="pallets/flask",
    )

    assert result.pr_number == 123
    assert result.was_merged is True
    assert result.merge_target == 1
    assert result.outcome_label == "Merged"
    assert result.target_branch == "main"
    assert result.author_login == "example-user"
    assert result.labels == [
        "documentation",
        "quality",
    ]


def test_normalize_unmerged_pull_request() -> None:
    """Confirm closed-unmerged records are normalized."""

    raw_record = create_raw_pull_request()
    raw_record["merged_at"] = None

    result = normalize_pull_request(
        raw_pull_request=raw_record,
        repository="pallets/flask",
    )

    assert result.was_merged is False
    assert result.merge_target == 0
    assert result.outcome_label == "Closed without merge"


def test_extract_label_names_ignores_invalid_values() -> None:
    """Confirm invalid labels are ignored."""

    result = extract_label_names(
        [
            {
                "name": "bug",
            },
            {
                "missing": "name",
            },
            "invalid",
        ]
    )

    assert result == ["bug"]


def test_negative_integer_returns_none() -> None:
    """Confirm invalid negative counts are rejected."""

    assert normalize_non_negative_integer(-5) is None
