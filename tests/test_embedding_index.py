"""Tests for local embeddings and FAISS retrieval."""

from __future__ import annotations

import hashlib

import numpy as np

from src.rag.embedding_index import (
    build_faiss_cosine_index,
    encode_query,
    encode_texts,
    normalize_embedding_text,
    prepare_embedding_records,
    retrieve_by_pr_number,
    semantic_search,
    validate_index_outputs,
)


class FakeEmbeddingModel:
    """Small deterministic embedding model for tests."""

    dimension = 16

    def encode(
        self,
        sentences,
        **kwargs,
    ):
        """Create deterministic normalized vectors."""

        if isinstance(
            sentences,
            str,
        ):
            sentences = [sentences]

        vectors = []

        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()

            values = np.array(
                [digest[index] for index in range(self.dimension)],
                dtype=np.float32,
            )

            values = values - values.mean()

            norm = np.linalg.norm(values)

            if norm == 0:
                values[0] = 1
                norm = 1

            vectors.append(values / norm)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def create_chunk_records():
    """Create representative chunk records."""

    return [
        {
            "chunk_id": "chunk_1",
            "document_id": "document_1",
            "chunk_index": 0,
            "section_name": "PR description",
            "content": (
                "PR description. Adds authentication tests. "
                "<!-- Before opening a PR, add issue details. -->"
            ),
            "metadata": {
                "pr_number": "101",
                "repository": "pallets/flask",
                "title": "Add authentication tests",
                "review_priority": "High",
                "policy_risk_band": "High",
                "manual_review_required": True,
            },
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "document_2",
            "chunk_index": 0,
            "section_name": "Predictive intelligence",
            "content": (
                "Security-sensitive configuration change without automated tests."
            ),
            "metadata": {
                "pr_number": "202",
                "repository": "pallets/flask",
                "title": "Update security configuration",
                "review_priority": "Critical",
                "policy_risk_band": "Critical",
                "manual_review_required": True,
            },
        },
        {
            "chunk_id": "chunk_3",
            "document_id": "document_1",
            "chunk_index": 1,
            "section_name": "Change evidence",
            "content": (
                "The pull request changes five files and includes three commits."
            ),
            "metadata": {
                "pr_number": "101",
                "repository": "pallets/flask",
                "title": "Add authentication tests",
                "review_priority": "High",
                "policy_risk_band": "High",
                "manual_review_required": True,
            },
        },
    ]


def test_template_comments_are_removed() -> None:
    """Confirm GitHub comments are not embedded."""

    text = (
        "Useful contributor content. "
        "<!-- Repeated pull request template instructions. -->"
    )

    result = normalize_embedding_text(text)

    assert "Useful contributor content" in result

    assert "<!--" not in result

    assert "Repeated pull request template" not in result


def test_embedding_records_are_prepared() -> None:
    """Confirm chunk metadata and cleaned text are retained."""

    records = prepare_embedding_records(create_chunk_records())

    assert len(records) == 3

    assert records[0]["vector_position"] == 0

    assert records[0]["pr_number"] == "101"

    assert records[0]["original_content"] != records[0]["embedding_content"]


def test_embeddings_are_normalized() -> None:
    """Confirm generated vectors have unit length."""

    model = FakeEmbeddingModel()

    embeddings = encode_texts(
        model=model,
        texts=[
            "first text",
            "second text",
        ],
        batch_size=2,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    assert embeddings.shape == (
        2,
        16,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-4,
    )


def test_faiss_index_creation() -> None:
    """Confirm vectors are added to the exact index."""

    model = FakeEmbeddingModel()

    embeddings = encode_texts(
        model=model,
        texts=[
            "first",
            "second",
            "third",
        ],
    )

    index = build_faiss_cosine_index(embeddings)

    assert index.ntotal == 3

    assert index.d == 16


def test_query_encoding() -> None:
    """Confirm query vectors are normalized."""

    query = encode_query(
        model=FakeEmbeddingModel(),
        query="security testing",
    )

    assert query.shape == (
        1,
        16,
    )

    assert np.allclose(
        np.linalg.norm(
            query,
            axis=1,
        ),
        1.0,
        atol=1e-4,
    )


def test_exact_pr_number_lookup() -> None:
    """Confirm exact PR retrieval returns all matching chunks."""

    prepared_records = prepare_embedding_records(create_chunk_records())

    results = retrieve_by_pr_number(
        pr_number=101,
        metadata_records=(prepared_records),
    )

    assert len(results) == 2

    assert all(result["pr_number"] == "101" for result in results)


def test_semantic_search() -> None:
    """Confirm semantic search returns ranked records."""

    prepared_records = prepare_embedding_records(create_chunk_records())

    model = FakeEmbeddingModel()

    embeddings = encode_texts(
        model=model,
        texts=[record["embedding_content"] for record in prepared_records],
    )

    index = build_faiss_cosine_index(embeddings)

    results = semantic_search(
        query=("security configuration without tests"),
        model=model,
        index=index,
        metadata_records=(prepared_records),
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].rank == 1

    assert np.isfinite(results[0].score)


def test_complete_index_validation() -> None:
    """Confirm complete embedding outputs pass validation."""

    prepared_records = prepare_embedding_records(create_chunk_records())

    embeddings = encode_texts(
        model=FakeEmbeddingModel(),
        texts=[record["embedding_content"] for record in prepared_records],
    )

    index = build_faiss_cosine_index(embeddings)

    validation = validate_index_outputs(
        embeddings=embeddings,
        index=index,
        metadata_records=(prepared_records),
        expected_chunk_count=3,
    )

    assert validation["validation_passed"] is True
