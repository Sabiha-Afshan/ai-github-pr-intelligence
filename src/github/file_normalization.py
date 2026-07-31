"""Normalize GitHub changed-file responses."""

from typing import Any

from src.features.file_categories import (
    categorize_file,
    get_file_extension,
    is_configuration_file,
    is_documentation_file,
    is_generated_file,
    is_security_sensitive_file,
    is_test_file,
)
from src.github.normalization import (
    normalize_non_negative_integer,
    normalize_optional_text,
)
from src.github.schemas import (
    NormalizedChangedFile,
)


def normalize_changed_file(
    raw_file: dict[str, Any],
    repository: str,
    pr_number: int,
) -> NormalizedChangedFile:
    """Convert one GitHub file record into a clean record."""

    file_path = str(raw_file.get("filename") or "").strip()

    if not file_path:
        raise ValueError("Changed-file response is missing filename.")

    additions = normalize_non_negative_integer(raw_file.get("additions"))

    deletions = normalize_non_negative_integer(raw_file.get("deletions"))

    total_changes = normalize_non_negative_integer(raw_file.get("changes"))

    if total_changes is None:
        if additions is not None and deletions is not None:
            total_changes = additions + deletions

    patch = raw_file.get("patch")

    return NormalizedChangedFile(
        repository=repository,
        pr_number=pr_number,
        file_path=file_path,
        file_status=normalize_optional_text(raw_file.get("status")),
        additions=additions,
        deletions=deletions,
        total_changes=total_changes,
        previous_file_path=normalize_optional_text(raw_file.get("previous_filename")),
        raw_url=normalize_optional_text(raw_file.get("raw_url")),
        blob_url=normalize_optional_text(raw_file.get("blob_url")),
        patch_available=(isinstance(patch, str) and bool(patch.strip())),
        file_extension=get_file_extension(file_path),
        file_category=categorize_file(file_path),
        is_test_file=is_test_file(file_path),
        is_documentation_file=(is_documentation_file(file_path)),
        is_configuration_file=(is_configuration_file(file_path)),
        is_security_sensitive_file=(is_security_sensitive_file(file_path)),
        is_generated_file=is_generated_file(file_path),
    )
