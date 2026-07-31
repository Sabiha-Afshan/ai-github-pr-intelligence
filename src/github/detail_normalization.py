"""Create detailed pull-request records."""

from collections import Counter
from typing import Any

from src.github.file_normalization import (
    normalize_changed_file,
)
from src.github.normalization import (
    normalize_pull_request,
)
from src.github.schemas import (
    DetailedPullRequestRecord,
    NormalizedChangedFile,
)


def build_detailed_pull_request_record(
    raw_pull_request: dict[str, Any],
    raw_files: list[dict[str, Any]],
    repository: str,
) -> tuple[
    DetailedPullRequestRecord,
    list[NormalizedChangedFile],
]:
    """Normalize one detailed PR and all changed files."""

    normalized_pr = normalize_pull_request(
        raw_pull_request=raw_pull_request,
        repository=repository,
    )

    normalized_files = [
        normalize_changed_file(
            raw_file=raw_file,
            repository=repository,
            pr_number=normalized_pr.pr_number,
        )
        for raw_file in raw_files
    ]

    category_counts = Counter(file.file_category for file in normalized_files)

    changed_file_paths = [file.file_path for file in normalized_files]

    test_file_count = sum(file.is_test_file for file in normalized_files)

    documentation_file_count = sum(
        file.is_documentation_file for file in normalized_files
    )

    configuration_file_count = sum(
        file.is_configuration_file for file in normalized_files
    )

    security_sensitive_file_count = sum(
        file.is_security_sensitive_file for file in normalized_files
    )

    generated_file_count = sum(file.is_generated_file for file in normalized_files)

    additions = normalized_pr.additions
    deletions = normalized_pr.deletions

    total_changes = None

    if additions is not None and deletions is not None:
        total_changes = additions + deletions

    detailed_record = DetailedPullRequestRecord(
        **normalized_pr.model_dump(),
        total_changes=total_changes,
        changed_file_paths=changed_file_paths,
        test_file_count=test_file_count,
        documentation_file_count=(documentation_file_count),
        configuration_file_count=(configuration_file_count),
        security_sensitive_file_count=(security_sensitive_file_count),
        generated_file_count=(generated_file_count),
        has_test_changes=(test_file_count > 0),
        has_documentation_changes=(documentation_file_count > 0),
        has_configuration_changes=(configuration_file_count > 0),
        has_security_sensitive_changes=(security_sensitive_file_count > 0),
        file_category_counts=dict(category_counts),
    )

    return detailed_record, normalized_files
