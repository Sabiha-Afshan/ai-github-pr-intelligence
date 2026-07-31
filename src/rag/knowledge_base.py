"""Knowledge-document preparation for the local PR intelligence RAG system."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PR_IDENTIFIER_COLUMN = "pr_number"

REQUIRED_UNIFIED_COLUMNS = {
    PR_IDENTIFIER_COLUMN,
    "merge_probability",
    "merge_prediction",
    "policy_risk_score",
    "policy_risk_band",
    "triggered_rule_count",
    "triggered_rules",
    "recommended_actions",
    "manual_review_required",
    "review_priority_score",
    "review_priority",
    "recommended_next_action",
}


@dataclass(frozen=True)
class KnowledgeDocument:
    """One PR-level knowledge document."""

    document_id: str
    pr_number: str
    repository: str
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class KnowledgeChunk:
    """One searchable text chunk derived from a PR document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    section_name: str
    content: str
    character_count: int
    word_count: int
    metadata: dict[str, Any]


def is_missing(value: Any) -> bool:
    """Return True for missing scalar values."""

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(
            result,
            bool | np.bool_,
        ):
            return bool(result)

    except (
        TypeError,
        ValueError,
    ):
        return False

    return False


def safe_text(
    value: Any,
    default: str = "",
) -> str:
    """Convert a scalar value into normalized text."""

    if is_missing(value):
        return default

    normalized = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    return normalized or default


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Convert one value to a finite float."""

    if is_missing(value):
        return default

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(result):
        return default

    return result


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Convert one value to an integer."""

    numeric_value = safe_float(
        value,
        default=None,
    )

    if numeric_value is None:
        return default

    return int(round(numeric_value))


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """Convert common Boolean-like values."""

    if is_missing(value):
        return default

    if isinstance(
        value,
        bool | np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        int | float | np.integer | np.floating,
    ):
        return bool(int(value))

    normalized = safe_text(value).lower()

    if normalized in {
        "true",
        "yes",
        "y",
        "1",
    }:
        return True

    if normalized in {
        "false",
        "no",
        "n",
        "0",
        "",
    }:
        return False

    return default


def format_probability(
    value: Any,
) -> str:
    """Format a probability as a percentage."""

    probability = safe_float(
        value,
        default=None,
    )

    if probability is None:
        return "Not available"

    clipped_probability = float(
        np.clip(
            probability,
            0,
            1,
        )
    )

    return f"{clipped_probability * 100:.2f}%"


def format_number(
    value: Any,
    decimals: int = 0,
) -> str:
    """Format a numeric value for readable evidence text."""

    numeric_value = safe_float(
        value,
        default=None,
    )

    if numeric_value is None:
        return "Not available"

    return f"{numeric_value:,.{decimals}f}"


def first_available_text(
    row: pd.Series,
    columns: Iterable[str],
    default: str = "",
) -> str:
    """Return the first available non-empty text value."""

    for column in columns:
        if column not in row.index:
            continue

        value = safe_text(row[column])

        if value:
            return value

    return default


def first_available_value(
    row: pd.Series,
    columns: Iterable[str],
    default: Any = None,
) -> Any:
    """Return the first available non-missing scalar value."""

    for column in columns:
        if column not in row.index:
            continue

        value = row[column]

        if not is_missing(value):
            return value

    return default


def stable_hash(
    *parts: Any,
    prefix: str,
    length: int = 20,
) -> str:
    """Create a deterministic identifier from normalized values."""

    normalized_parts = [
        safe_text(
            part,
            default="<missing>",
        )
        for part in parts
    ]

    payload = "||".join(normalized_parts)

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

    return f"{prefix}_{digest}"


def split_pipe_values(
    value: Any,
) -> list[str]:
    """Split pipe-delimited policy values into a clean list."""

    text = safe_text(value)

    if not text:
        return []

    values = [item.strip() for item in text.split("|") if item.strip()]

    return list(dict.fromkeys(values))


def validate_unified_source(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the unified intelligence source."""

    missing_columns = sorted(REQUIRED_UNIFIED_COLUMNS - set(dataframe.columns))

    duplicate_pr_count = (
        int(
            dataframe.duplicated(
                subset=[
                    PR_IDENTIFIER_COLUMN,
                ]
            ).sum()
        )
        if PR_IDENTIFIER_COLUMN in dataframe.columns
        else None
    )

    blank_pr_identifier_count = 0

    if PR_IDENTIFIER_COLUMN in dataframe.columns:
        blank_pr_identifier_count = int(
            dataframe[PR_IDENTIFIER_COLUMN]
            .apply(lambda value: safe_text(value) == "")
            .sum()
        )

    expected_row_count_valid = bool(len(dataframe) == 600)

    validation_passed = bool(
        not missing_columns
        and duplicate_pr_count == 0
        and blank_pr_identifier_count == 0
        and expected_row_count_valid
    )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "expected_row_count": 600,
        "expected_row_count_valid": (expected_row_count_valid),
        "missing_required_columns": (missing_columns),
        "duplicate_pr_count": (duplicate_pr_count),
        "blank_pr_identifier_count": (blank_pr_identifier_count),
        "validation_passed": (validation_passed),
    }


def build_identity_section(
    row: pd.Series,
) -> str:
    """Build the PR identity and lifecycle section."""

    repository = first_available_text(
        row,
        [
            "repository",
            "repo_name",
            "repository_name",
            "full_name",
        ],
        default="pallets/flask",
    )

    pr_number = safe_text(row.get(PR_IDENTIFIER_COLUMN))

    title = first_available_text(
        row,
        [
            "title",
            "pr_title",
        ],
        default=(f"Pull request {pr_number}"),
    )

    state = first_available_text(
        row,
        [
            "state",
            "pr_state",
        ],
        default="Unknown",
    )

    author = first_available_text(
        row,
        [
            "author_login",
            "user_login",
            "author",
            "creator_login",
        ],
        default="Unknown",
    )

    created_at = first_available_text(
        row,
        [
            "created_at",
            "pr_created_at",
        ],
        default="Unknown",
    )

    merged_at = first_available_text(
        row,
        [
            "merged_at",
            "pr_merged_at",
        ],
        default="Not merged or unavailable",
    )

    html_url = first_available_text(
        row,
        [
            "html_url",
            "pr_url",
            "url",
        ],
        default="Unavailable",
    )

    return "\n".join(
        [
            "PR identity",
            f"- Repository: {repository}",
            f"- PR number: {pr_number}",
            f"- Title: {title}",
            f"- Author: {author}",
            f"- State: {state}",
            f"- Created at: {created_at}",
            f"- Merged at: {merged_at}",
            f"- Source URL: {html_url}",
        ]
    )


def build_description_section(
    row: pd.Series,
) -> str:
    """Build the title and description section."""

    title = first_available_text(
        row,
        [
            "title",
            "pr_title",
        ],
        default="No title available",
    )

    body = first_available_text(
        row,
        [
            "body",
            "description",
            "pr_body",
        ],
        default="No PR description was available.",
    )

    has_description = safe_bool(
        row.get("has_description"),
        default=bool(body and body != "No PR description was available."),
    )

    has_detailed_description = safe_bool(row.get("has_detailed_description"))

    body_word_count = safe_int(row.get("body_word_count"))

    return "\n".join(
        [
            "PR description",
            f"- Title: {title}",
            f"- Description present: {has_description}",
            (f"- Detailed description detected: {has_detailed_description}"),
            f"- Description word count: {body_word_count}",
            f"- Description text: {body}",
        ]
    )


def build_change_section(
    row: pd.Series,
) -> str:
    """Build code-change and review-context evidence."""

    return "\n".join(
        [
            "Change evidence",
            (f"- Total changed lines: {format_number(row.get('total_changes'))}"),
            (f"- Additions: {format_number(row.get('additions'))}"),
            (f"- Deletions: {format_number(row.get('deletions'))}"),
            (f"- Changed files: {format_number(row.get('changed_files'))}"),
            (f"- Files added: {format_number(row.get('files_added'))}"),
            (f"- Files modified: {format_number(row.get('files_modified'))}"),
            (f"- Files removed: {format_number(row.get('files_removed'))}"),
            (f"- Commit count: {format_number(row.get('commit_count'))}"),
            (
                "- Requested reviewer count: "
                f"{format_number(row.get('requested_reviewer_count'))}"
            ),
            (f"- Label count: {format_number(row.get('label_count'))}"),
            (f"- Test changes detected: {safe_bool(row.get('has_test_changes'))}"),
            (
                "- Documentation changes detected: "
                f"{safe_bool(row.get('has_documentation_changes'))}"
            ),
            (
                "- Configuration changes detected: "
                f"{safe_bool(row.get('has_configuration_changes'))}"
            ),
            (
                "- Security-sensitive changes detected: "
                f"{safe_bool(row.get('has_security_sensitive_changes'))}"
            ),
            (
                "- Historical outlier feature count: "
                f"{format_number(row.get('iqr_outlier_feature_count'))}"
            ),
        ]
    )


def build_model_section(
    row: pd.Series,
) -> str:
    """Build machine-learning prediction evidence."""

    merge_prediction = safe_int(row.get("merge_prediction"))

    merge_label = (
        "Predicted to merge" if merge_prediction == 1 else "Predicted not to merge"
    )

    delay_available = safe_bool(row.get("delay_score_available"))

    if delay_available:
        delay_prediction = safe_int(row.get("delay_prediction"))

        delay_label = (
            "Predicted merge delay greater than 48 hours"
            if delay_prediction == 1
            else "Predicted merge within 48 hours"
        )

        delay_probability_text = format_probability(row.get("delay_probability"))

        delay_confidence_text = format_probability(
            row.get("delay_prediction_confidence")
        )

        delay_threshold_text = format_probability(row.get("delay_prediction_threshold"))

    else:
        delay_label = (
            "Not applicable because the delay model is restricted to merged PRs"
        )
        delay_probability_text = "Not available"
        delay_confidence_text = "Not available"
        delay_threshold_text = "Not available"

    return "\n".join(
        [
            "Predictive intelligence",
            f"- Merge outcome: {merge_label}",
            (
                "- Merge probability: "
                f"{format_probability(row.get('merge_probability'))}"
            ),
            (
                "- Merge prediction confidence: "
                f"{format_probability(row.get('merge_prediction_confidence'))}"
            ),
            (
                "- Merge decision threshold: "
                f"{format_probability(row.get('merge_prediction_threshold'))}"
            ),
            f"- Delay outcome: {delay_label}",
            (f"- Delay probability: {delay_probability_text}"),
            (f"- Delay prediction confidence: {delay_confidence_text}"),
            (f"- Delay decision threshold: {delay_threshold_text}"),
            (
                "- Important limitation: predictions are decision-support "
                "signals and are not automatic merge decisions."
            ),
        ]
    )


def build_policy_section(
    row: pd.Series,
) -> str:
    """Build deterministic rule and policy evidence."""

    triggered_rules = split_pipe_values(row.get("triggered_rules"))

    triggered_categories = split_pipe_values(row.get("triggered_categories"))

    recommended_actions = split_pipe_values(row.get("recommended_actions"))

    rule_text = (
        ", ".join(triggered_rules)
        if triggered_rules
        else "No deterministic policy rules triggered"
    )

    category_text = (
        ", ".join(triggered_categories)
        if triggered_categories
        else "No triggered categories"
    )

    action_text = (
        " ".join(recommended_actions)
        if recommended_actions
        else "Continue through the standard review workflow."
    )

    return "\n".join(
        [
            "Deterministic policy intelligence",
            (f"- Policy risk score: {format_number(row.get('policy_risk_score'))}/100"),
            (
                "- Policy risk band: "
                f"{safe_text(row.get('policy_risk_band'), 'Unknown')}"
            ),
            (f"- Triggered rule count: {safe_int(row.get('triggered_rule_count'))}"),
            f"- Triggered rules: {rule_text}",
            f"- Triggered categories: {category_text}",
            (
                "- Manual review required: "
                f"{safe_bool(row.get('manual_review_required'))}"
            ),
            f"- Rule recommendations: {action_text}",
        ]
    )


def build_priority_section(
    row: pd.Series,
) -> str:
    """Build unified review-priority evidence."""

    return "\n".join(
        [
            "Unified review priority",
            (f"- Review priority: {safe_text(row.get('review_priority'), 'Unknown')}"),
            (
                "- Review priority score: "
                f"{format_number(row.get('review_priority_score'), 2)}/100"
            ),
            (
                "- Recommended next action: "
                f"{safe_text(row.get('recommended_next_action'), 'Unavailable')}"
            ),
            (
                "- Governance note: human review remains responsible "
                "for final approval, rejection and escalation decisions."
            ),
        ]
    )


def build_document_metadata(
    row: pd.Series,
    document_id: str,
) -> dict[str, Any]:
    """Build JSON-safe metadata for one document."""

    repository = first_available_text(
        row,
        [
            "repository",
            "repo_name",
            "repository_name",
            "full_name",
        ],
        default="pallets/flask",
    )

    pr_number = safe_text(row.get(PR_IDENTIFIER_COLUMN))

    metadata = {
        "document_id": document_id,
        "source_type": "github_pull_request",
        "repository": repository,
        "pr_number": pr_number,
        "title": first_available_text(
            row,
            [
                "title",
                "pr_title",
            ],
            default=(f"Pull request {pr_number}"),
        ),
        "author": first_available_text(
            row,
            [
                "author_login",
                "user_login",
                "author",
                "creator_login",
            ],
            default="Unknown",
        ),
        "created_at": first_available_text(
            row,
            [
                "created_at",
                "pr_created_at",
            ],
            default="Unknown",
        ),
        "state": first_available_text(
            row,
            [
                "state",
                "pr_state",
            ],
            default="Unknown",
        ),
        "review_priority": safe_text(
            row.get("review_priority"),
            default="Unknown",
        ),
        "review_priority_score": safe_float(
            row.get("review_priority_score"),
        ),
        "policy_risk_band": safe_text(
            row.get("policy_risk_band"),
            default="Unknown",
        ),
        "policy_risk_score": safe_float(
            row.get("policy_risk_score"),
        ),
        "manual_review_required": safe_bool(row.get("manual_review_required")),
        "merge_probability": safe_float(
            row.get("merge_probability"),
        ),
        "merge_prediction": safe_int(row.get("merge_prediction")),
        "delay_score_available": safe_bool(row.get("delay_score_available")),
        "delay_probability": safe_float(
            row.get("delay_probability"),
        ),
        "delay_prediction": (
            safe_int(row.get("delay_prediction"))
            if safe_bool(row.get("delay_score_available"))
            else None
        ),
        "triggered_rule_count": safe_int(row.get("triggered_rule_count")),
        "triggered_rules": split_pipe_values(row.get("triggered_rules")),
        "triggered_categories": split_pipe_values(row.get("triggered_categories")),
    }

    return metadata


def build_knowledge_document(
    row: pd.Series,
) -> KnowledgeDocument:
    """Build one PR-level RAG knowledge document."""

    repository = first_available_text(
        row,
        [
            "repository",
            "repo_name",
            "repository_name",
            "full_name",
        ],
        default="pallets/flask",
    )

    pr_number = safe_text(row.get(PR_IDENTIFIER_COLUMN))

    title = first_available_text(
        row,
        [
            "title",
            "pr_title",
        ],
        default=(f"Pull request {pr_number}"),
    )

    document_id = stable_hash(
        repository,
        pr_number,
        prefix="prdoc",
    )

    sections = [
        build_identity_section(row),
        build_description_section(row),
        build_change_section(row),
        build_model_section(row),
        build_policy_section(row),
        build_priority_section(row),
    ]

    content = "\n\n".join(sections).strip()

    metadata = build_document_metadata(
        row=row,
        document_id=document_id,
    )

    return KnowledgeDocument(
        document_id=document_id,
        pr_number=pr_number,
        repository=repository,
        title=title,
        content=content,
        metadata=metadata,
    )


def build_knowledge_documents(
    dataframe: pd.DataFrame,
) -> list[KnowledgeDocument]:
    """Build one knowledge document for every unified PR record."""

    source_validation = validate_unified_source(dataframe)

    if not source_validation["validation_passed"]:
        raise ValueError(
            f"Unified knowledge source failed validation: {source_validation}"
        )

    documents = [build_knowledge_document(row) for _, row in dataframe.iterrows()]

    document_ids = [document.document_id for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Generated document IDs are not unique.")

    return documents


def split_document_sections(
    content: str,
) -> list[tuple[str, str]]:
    """Split a knowledge document into its named sections."""

    normalized_content = content.strip()

    if not normalized_content:
        return []

    raw_sections = [
        section.strip()
        for section in re.split(
            r"\n\s*\n",
            normalized_content,
        )
        if section.strip()
    ]

    sections: list[tuple[str, str]] = []

    for section in raw_sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]

        if not lines:
            continue

        section_name = lines[0]

        sections.append(
            (
                section_name,
                "\n".join(lines),
            )
        )

    return sections


def split_text_by_words(
    text: str,
    maximum_words: int,
    overlap_words: int,
) -> list[str]:
    """Split oversized section text into overlapping word windows."""

    if maximum_words <= 0:
        raise ValueError("maximum_words must be positive.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= maximum_words:
        raise ValueError("overlap_words must be smaller than maximum_words.")

    words = text.split()

    if len(words) <= maximum_words:
        return [text.strip()]

    step = maximum_words - overlap_words

    chunks = []

    for start in range(
        0,
        len(words),
        step,
    ):
        end = start + maximum_words

        window = words[start:end]

        if not window:
            break

        chunks.append(" ".join(window))

        if end >= len(words):
            break

    return chunks


def chunk_knowledge_document(
    document: KnowledgeDocument,
    maximum_words: int = 180,
    overlap_words: int = 25,
) -> list[KnowledgeChunk]:
    """Convert one PR document into searchable section-aware chunks."""

    sections = split_document_sections(document.content)

    chunks: list[KnowledgeChunk] = []

    chunk_index = 0

    for section_name, section_content in sections:
        section_chunks = split_text_by_words(
            text=section_content,
            maximum_words=maximum_words,
            overlap_words=overlap_words,
        )

        for section_part_index, chunk_text in enumerate(section_chunks):
            chunk_id = stable_hash(
                document.document_id,
                section_name,
                section_part_index,
                chunk_text,
                prefix="prchunk",
            )

            metadata = {
                **document.metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "section_name": (section_name),
                "section_part_index": (section_part_index),
            }

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id=(document.document_id),
                    chunk_index=(chunk_index),
                    section_name=(section_name),
                    content=chunk_text,
                    character_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    metadata=metadata,
                )
            )

            chunk_index += 1

    if not chunks:
        raise ValueError(f"Document {document.document_id} produced no chunks.")

    return chunks


def chunk_knowledge_documents(
    documents: list[KnowledgeDocument],
    maximum_words: int = 180,
    overlap_words: int = 25,
) -> list[KnowledgeChunk]:
    """Chunk all PR knowledge documents."""

    chunks = []

    for document in documents:
        chunks.extend(
            chunk_knowledge_document(
                document=document,
                maximum_words=maximum_words,
                overlap_words=overlap_words,
            )
        )

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Generated chunk IDs are not unique.")

    return chunks


def knowledge_document_to_dict(
    document: KnowledgeDocument,
) -> dict[str, Any]:
    """Convert one document to a JSON-safe dictionary."""

    return {
        "document_id": (document.document_id),
        "pr_number": (document.pr_number),
        "repository": (document.repository),
        "title": (document.title),
        "content": (document.content),
        "metadata": (document.metadata),
    }


def knowledge_chunk_to_dict(
    chunk: KnowledgeChunk,
) -> dict[str, Any]:
    """Convert one chunk to a JSON-safe dictionary."""

    return {
        "chunk_id": (chunk.chunk_id),
        "document_id": (chunk.document_id),
        "chunk_index": (chunk.chunk_index),
        "section_name": (chunk.section_name),
        "content": (chunk.content),
        "character_count": (chunk.character_count),
        "word_count": (chunk.word_count),
        "metadata": (chunk.metadata),
    }


def build_document_manifest(
    documents: list[KnowledgeDocument],
    chunks: list[KnowledgeChunk],
) -> pd.DataFrame:
    """Build a PR-level knowledge-base manifest."""

    chunk_counts = (
        pd.Series(
            [chunk.document_id for chunk in chunks],
            dtype="object",
        )
        .value_counts()
        .to_dict()
    )

    records = []

    for document in documents:
        metadata = document.metadata

        records.append(
            {
                "document_id": (document.document_id),
                "repository": (document.repository),
                "pr_number": (document.pr_number),
                "title": (document.title),
                "document_character_count": len(document.content),
                "document_word_count": len(document.content.split()),
                "chunk_count": int(
                    chunk_counts.get(
                        document.document_id,
                        0,
                    )
                ),
                "review_priority": (metadata.get("review_priority")),
                "review_priority_score": (metadata.get("review_priority_score")),
                "policy_risk_band": (metadata.get("policy_risk_band")),
                "policy_risk_score": (metadata.get("policy_risk_score")),
                "manual_review_required": (metadata.get("manual_review_required")),
                "merge_probability": (metadata.get("merge_probability")),
                "delay_score_available": (metadata.get("delay_score_available")),
                "delay_probability": (metadata.get("delay_probability")),
                "triggered_rule_count": (metadata.get("triggered_rule_count")),
            }
        )

    return pd.DataFrame(records)


def build_chunk_manifest(
    chunks: list[KnowledgeChunk],
) -> pd.DataFrame:
    """Build a chunk-level knowledge-base manifest."""

    return pd.DataFrame(
        [
            {
                "chunk_id": (chunk.chunk_id),
                "document_id": (chunk.document_id),
                "chunk_index": (chunk.chunk_index),
                "section_name": (chunk.section_name),
                "character_count": (chunk.character_count),
                "word_count": (chunk.word_count),
                "repository": (chunk.metadata.get("repository")),
                "pr_number": (chunk.metadata.get("pr_number")),
                "review_priority": (chunk.metadata.get("review_priority")),
                "policy_risk_band": (chunk.metadata.get("policy_risk_band")),
                "manual_review_required": (
                    chunk.metadata.get("manual_review_required")
                ),
            }
            for chunk in chunks
        ]
    )


def validate_knowledge_outputs(
    source_dataframe: pd.DataFrame,
    documents: list[KnowledgeDocument],
    chunks: list[KnowledgeChunk],
    document_manifest: pd.DataFrame,
    chunk_manifest: pd.DataFrame,
    maximum_words: int,
) -> dict[str, Any]:
    """Validate the prepared RAG knowledge-base outputs."""

    expected_document_count = len(source_dataframe)

    document_count_valid = bool(len(documents) == expected_document_count)

    unique_document_ids_valid = bool(
        len({document.document_id for document in documents}) == expected_document_count
    )

    document_content_complete = bool(
        all(document.content.strip() for document in documents)
    )

    document_metadata_complete = bool(
        all(
            document.metadata.get("pr_number") and document.metadata.get("document_id")
            for document in documents
        )
    )

    chunk_count_valid = bool(len(chunks) >= expected_document_count)

    unique_chunk_ids_valid = bool(
        len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    )

    every_document_has_chunk = bool(
        {document.document_id for document in documents}
        == {chunk.document_id for chunk in chunks}
    )

    chunk_word_limits_valid = bool(
        all(chunk.word_count <= maximum_words for chunk in chunks)
    )

    manifest_document_count_valid = bool(
        len(document_manifest) == expected_document_count
    )

    manifest_chunk_count_valid = bool(len(chunk_manifest) == len(chunks))

    priority_metadata_complete = bool(
        document_manifest["review_priority"].astype(str).str.strip().ne("").all()
    )

    validation_passed = bool(
        document_count_valid
        and unique_document_ids_valid
        and document_content_complete
        and document_metadata_complete
        and chunk_count_valid
        and unique_chunk_ids_valid
        and every_document_has_chunk
        and chunk_word_limits_valid
        and manifest_document_count_valid
        and manifest_chunk_count_valid
        and priority_metadata_complete
    )

    return {
        "expected_document_count": (expected_document_count),
        "actual_document_count": len(documents),
        "document_count_valid": (document_count_valid),
        "unique_document_ids_valid": (unique_document_ids_valid),
        "document_content_complete": (document_content_complete),
        "document_metadata_complete": (document_metadata_complete),
        "actual_chunk_count": len(chunks),
        "chunk_count_valid": (chunk_count_valid),
        "unique_chunk_ids_valid": (unique_chunk_ids_valid),
        "every_document_has_chunk": (every_document_has_chunk),
        "maximum_chunk_words": (maximum_words),
        "chunk_word_limits_valid": (chunk_word_limits_valid),
        "manifest_document_count_valid": (manifest_document_count_valid),
        "manifest_chunk_count_valid": (manifest_chunk_count_valid),
        "priority_metadata_complete": (priority_metadata_complete),
        "validation_passed": (validation_passed),
    }


def serialize_json_line(
    payload: dict[str, Any],
) -> str:
    """Serialize one compact JSONL record."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
