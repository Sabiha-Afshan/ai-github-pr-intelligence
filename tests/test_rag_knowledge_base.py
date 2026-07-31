"""Tests for structured RAG knowledge-base preparation."""

import pandas as pd

from src.rag.knowledge_base import (
    build_chunk_manifest,
    build_document_manifest,
    build_knowledge_document,
    build_knowledge_documents,
    chunk_knowledge_document,
    chunk_knowledge_documents,
    split_pipe_values,
    stable_hash,
    validate_knowledge_outputs,
    validate_unified_source,
)


def create_unified_dataset(
    row_count: int = 600,
) -> pd.DataFrame:
    """Create a representative unified intelligence dataset."""

    records = []

    for index in range(row_count):
        pr_number = index + 1

        delay_available = bool(index % 2 == 0)

        priority = (
            "Critical"
            if index < 20
            else ("High" if index < 60 else ("Moderate" if index < 120 else "Routine"))
        )

        policy_band = (
            "Critical"
            if priority == "Critical"
            else (
                "High"
                if priority == "High"
                else ("Moderate" if priority == "Moderate" else "Low")
            )
        )

        records.append(
            {
                "pr_number": (pr_number),
                "repository": ("pallets/flask"),
                "title": (f"Example pull request {pr_number}"),
                "body": (
                    "This pull request updates application "
                    "behaviour and includes implementation details."
                ),
                "author_login": (f"author_{index % 5}"),
                "state": ("closed"),
                "created_at": ("2026-01-01T00:00:00Z"),
                "merged_at": ("2026-01-02T00:00:00Z" if delay_available else None),
                "html_url": (f"https://example.com/pr/{pr_number}"),
                "has_description": (True),
                "has_detailed_description": (True),
                "body_word_count": (30),
                "total_changes": (100 + index),
                "additions": (80),
                "deletions": (20),
                "changed_files": (5),
                "files_added": (1),
                "files_modified": (4),
                "files_removed": (0),
                "commit_count": (3),
                "requested_reviewer_count": (1),
                "label_count": (2),
                "has_test_changes": (True),
                "has_documentation_changes": (False),
                "has_configuration_changes": (False),
                "has_security_sensitive_changes": (False),
                "iqr_outlier_feature_count": (0),
                "merge_probability": (0.75),
                "merge_prediction": (1),
                "merge_prediction_confidence": (0.75),
                "merge_prediction_threshold": (0.50),
                "delay_score_available": (delay_available),
                "delay_probability": (0.80 if delay_available else None),
                "delay_prediction": (1 if delay_available else None),
                "delay_prediction_confidence": (0.80 if delay_available else None),
                "delay_prediction_threshold": (0.75 if delay_available else None),
                "policy_risk_score": (80 if priority == "Critical" else 20),
                "policy_risk_band": (policy_band),
                "triggered_rule_count": (2),
                "triggered_rules": ("PR003 | PR011"),
                "triggered_categories": ("Testing | Governance"),
                "recommended_actions": ("Add tests | Assign a reviewer"),
                "manual_review_required": (
                    priority
                    in {
                        "Critical",
                        "High",
                    }
                ),
                "review_priority_score": (80.0 if priority == "Critical" else 30.0),
                "review_priority": (priority),
                "recommended_next_action": ("Complete policy checks before approval."),
            }
        )

    return pd.DataFrame(records)


def test_stable_hash_is_deterministic() -> None:
    """Confirm identifiers remain stable."""

    first_value = stable_hash(
        "pallets/flask",
        100,
        prefix="test",
    )

    second_value = stable_hash(
        "pallets/flask",
        100,
        prefix="test",
    )

    assert first_value == second_value


def test_pipe_value_splitting() -> None:
    """Confirm policy lists split and deduplicate."""

    result = split_pipe_values("PR003 | PR011 | PR003")

    assert result == [
        "PR003",
        "PR011",
    ]


def test_unified_source_validation() -> None:
    """Confirm the 600-row source passes validation."""

    validation = validate_unified_source(create_unified_dataset())

    assert validation["validation_passed"] is True


def test_single_document_creation() -> None:
    """Confirm one unified record becomes a document."""

    dataframe = create_unified_dataset()

    document = build_knowledge_document(dataframe.iloc[0])

    assert document.document_id.startswith("prdoc_")

    assert "Predictive intelligence" in document.content

    assert "Deterministic policy intelligence" in document.content

    assert "Unified review priority" in document.content


def test_all_documents_are_created() -> None:
    """Confirm one document is produced per PR."""

    documents = build_knowledge_documents(create_unified_dataset())

    assert len(documents) == 600

    assert len({document.document_id for document in documents}) == 600


def test_document_chunking() -> None:
    """Confirm section-aware chunks are created."""

    dataframe = create_unified_dataset()

    document = build_knowledge_document(dataframe.iloc[0])

    chunks = chunk_knowledge_document(
        document=document,
        maximum_words=80,
        overlap_words=10,
    )

    assert len(chunks) >= 6

    assert all(chunk.word_count <= 80 for chunk in chunks)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_manifests_are_created() -> None:
    """Confirm document and chunk manifests are complete."""

    documents = build_knowledge_documents(create_unified_dataset())

    chunks = chunk_knowledge_documents(
        documents=documents,
        maximum_words=180,
        overlap_words=25,
    )

    document_manifest = build_document_manifest(
        documents=documents,
        chunks=chunks,
    )

    chunk_manifest = build_chunk_manifest(chunks)

    assert len(document_manifest) == 600

    assert len(chunk_manifest) == len(chunks)

    assert document_manifest["chunk_count"].ge(1).all()


def test_complete_knowledge_output_validation() -> None:
    """Confirm complete Stage 8A outputs pass validation."""

    dataframe = create_unified_dataset()

    documents = build_knowledge_documents(dataframe)

    chunks = chunk_knowledge_documents(
        documents=documents,
        maximum_words=180,
        overlap_words=25,
    )

    document_manifest = build_document_manifest(
        documents=documents,
        chunks=chunks,
    )

    chunk_manifest = build_chunk_manifest(chunks)

    validation = validate_knowledge_outputs(
        source_dataframe=dataframe,
        documents=documents,
        chunks=chunks,
        document_manifest=(document_manifest),
        chunk_manifest=(chunk_manifest),
        maximum_words=180,
    )

    assert validation["validation_passed"] is True
