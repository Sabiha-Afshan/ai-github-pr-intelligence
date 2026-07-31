"""Normalize raw GitHub pull-request responses."""

from typing import Any

from src.github.schemas import NormalizedPullRequest


def safely_get_nested_value(
    data: dict[str, Any],
    *keys: str,
) -> Any:
    """Safely retrieve a value from nested dictionaries."""

    current_value: Any = data

    for key in keys:
        if not isinstance(
            current_value,
            dict,
        ):
            return None

        current_value = current_value.get(key)

        if current_value is None:
            return None

    return current_value


def extract_label_names(
    raw_labels: Any,
) -> list[str]:
    """Extract readable label names."""

    if not isinstance(raw_labels, list):
        return []

    label_names: list[str] = []

    for label in raw_labels:
        if not isinstance(label, dict):
            continue

        name = label.get("name")

        if isinstance(name, str) and name.strip():
            label_names.append(name.strip())

    return label_names


def normalize_optional_text(
    value: Any,
) -> str | None:
    """Normalize optional text fields."""

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def normalize_non_negative_integer(
    value: Any,
) -> int | None:
    """Normalize optional non-negative integers."""

    if value is None:
        return None

    try:
        normalized_value = int(value)
    except (TypeError, ValueError):
        return None

    if normalized_value < 0:
        return None

    return normalized_value


def normalize_pull_request(
    raw_pull_request: dict[str, Any],
    repository: str,
) -> NormalizedPullRequest:
    """Convert one raw GitHub PR into a clean record."""

    merged_at = raw_pull_request.get("merged_at")
    was_merged = merged_at is not None

    merge_target = int(was_merged)

    outcome_label = "Merged" if was_merged else "Closed without merge"

    raw_author = raw_pull_request.get("user")

    author_login = raw_author.get("login") if isinstance(raw_author, dict) else None

    normalized_record = NormalizedPullRequest(
        repository=repository,
        pr_number=int(raw_pull_request["number"]),
        title=str(raw_pull_request.get("title") or "Untitled pull request").strip(),
        body=normalize_optional_text(raw_pull_request.get("body")),
        state=str(raw_pull_request.get("state") or "unknown").strip(),
        draft=bool(raw_pull_request.get("draft", False)),
        author_login=normalize_optional_text(author_login),
        author_association=normalize_optional_text(
            raw_pull_request.get("author_association")
        ),
        created_at=raw_pull_request["created_at"],
        updated_at=raw_pull_request.get("updated_at"),
        closed_at=raw_pull_request.get("closed_at"),
        merged_at=merged_at,
        target_branch=normalize_optional_text(
            safely_get_nested_value(
                raw_pull_request,
                "base",
                "ref",
            )
        ),
        source_branch=normalize_optional_text(
            safely_get_nested_value(
                raw_pull_request,
                "head",
                "ref",
            )
        ),
        html_url=str(raw_pull_request.get("html_url") or "").strip(),
        labels=extract_label_names(raw_pull_request.get("labels")),
        was_merged=was_merged,
        merge_target=merge_target,
        outcome_label=outcome_label,
        additions=normalize_non_negative_integer(raw_pull_request.get("additions")),
        deletions=normalize_non_negative_integer(raw_pull_request.get("deletions")),
        changed_files=normalize_non_negative_integer(
            raw_pull_request.get("changed_files")
        ),
        commit_count=normalize_non_negative_integer(raw_pull_request.get("commits")),
        raw_data_available=True,
    )

    return normalized_record
