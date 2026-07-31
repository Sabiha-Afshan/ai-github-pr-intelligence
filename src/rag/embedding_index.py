"""Local embedding generation and FAISS retrieval utilities."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelProtocol(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(
        self,
        sentences: list[str] | str,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text into vectors."""


@dataclass(frozen=True)
class SearchResult:
    """One retrieved knowledge-base chunk."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    pr_number: str
    repository: str
    section_name: str
    title: str
    original_content: str
    embedding_content: str
    metadata: dict[str, Any]


def load_jsonl(
    input_path: Path,
) -> list[dict[str, Any]]:
    """Load JSONL records."""

    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            normalized_line = line.strip()

            if not normalized_line:
                continue

            try:
                record = json.loads(normalized_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}: {error}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Every JSONL record must be an object. Line: {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No JSONL records were found in {input_path}.")

    return records


def save_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save JSONL records atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            output_file.write("\n")

    temporary_path.replace(output_path)


def remove_html_comments(
    text: str,
) -> str:
    """Remove complete HTML comment blocks."""

    return re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )


def remove_common_template_phrases(
    text: str,
) -> str:
    """Remove recurring GitHub PR-template instructions."""

    template_patterns = [
        (
            r"before opening a pr,?\s*"
            r"open a ticket describing the issue.*?"
            r"replace this comment with a description of the change\."
        ),
        (
            r"link to relevant issues or previous prs.*?"
            r"fixes\s*#?<issue number>"
        ),
        (
            r"ensure each step in contributing\.rst is complete.*?"
            r"versionchanged::.*?code docs\."
        ),
    ]

    result = text

    for pattern in template_patterns:
        result = re.sub(
            pattern,
            " ",
            result,
            flags=(re.IGNORECASE | re.DOTALL),
        )

    return result


def normalize_embedding_text(
    text: Any,
) -> str:
    """Create clean semantic text while retaining useful evidence."""

    if text is None:
        return ""

    normalized = html.unescape(str(text))

    normalized = remove_html_comments(normalized)

    normalized = remove_common_template_phrases(normalized)

    normalized = re.sub(
        r"```.*?```",
        lambda match: (
            " "
            + re.sub(
                r"\s+",
                " ",
                match.group(0),
            )
            + " "
        ),
        normalized,
        flags=re.DOTALL,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def prepare_embedding_records(
    chunk_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Clean chunk text and prepare index metadata."""

    prepared_records: list[dict[str, Any]] = []

    seen_chunk_ids: set[str] = set()

    for vector_position, record in enumerate(chunk_records):
        required_fields = {
            "chunk_id",
            "document_id",
            "section_name",
            "content",
            "metadata",
        }

        missing_fields = sorted(required_fields - set(record))

        if missing_fields:
            raise ValueError(f"Chunk record is missing fields: {missing_fields}")

        chunk_id = str(record["chunk_id"])

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk ID detected: {chunk_id}")

        seen_chunk_ids.add(chunk_id)

        metadata = record["metadata"]

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(f"Chunk {chunk_id} has invalid metadata.")

        original_content = str(record["content"]).strip()

        embedding_content = normalize_embedding_text(original_content)

        if not embedding_content:
            embedding_content = original_content

        pr_number = str(
            metadata.get(
                "pr_number",
                "",
            )
        ).strip()

        if not pr_number:
            raise ValueError(f"Chunk {chunk_id} has no PR number.")

        prepared_records.append(
            {
                "vector_position": (vector_position),
                "chunk_id": chunk_id,
                "document_id": str(record["document_id"]),
                "chunk_index": int(
                    record.get(
                        "chunk_index",
                        0,
                    )
                ),
                "section_name": str(record["section_name"]),
                "repository": str(
                    metadata.get(
                        "repository",
                        "unknown",
                    )
                ),
                "pr_number": (pr_number),
                "title": str(
                    metadata.get(
                        "title",
                        f"Pull request {pr_number}",
                    )
                ),
                "review_priority": str(
                    metadata.get(
                        "review_priority",
                        "Unknown",
                    )
                ),
                "policy_risk_band": str(
                    metadata.get(
                        "policy_risk_band",
                        "Unknown",
                    )
                ),
                "manual_review_required": bool(
                    metadata.get(
                        "manual_review_required",
                        False,
                    )
                ),
                "original_content": (original_content),
                "embedding_content": (embedding_content),
                "original_character_count": len(original_content),
                "embedding_character_count": len(embedding_content),
                "template_text_removed": bool(
                    embedding_content
                    != re.sub(
                        r"\s+",
                        " ",
                        original_content,
                    ).strip()
                ),
                "metadata": metadata,
            }
        )

    return prepared_records


def create_embedding_model(
    model_name: str = (DEFAULT_EMBEDDING_MODEL),
    device: str = "cpu",
) -> Any:
    """Load the local Sentence Transformer model."""

    from sentence_transformers import (
        SentenceTransformer,
    )

    return SentenceTransformer(
        model_name,
        device=device,
    )


def encode_texts(
    model: EmbeddingModelProtocol,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode and normalize a collection of texts."""

    if not texts:
        raise ValueError("No texts were supplied for embedding.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embedding_array.ndim != 2:
        raise ValueError("Embedding output must be a two-dimensional array.")

    if embedding_array.shape[0] != len(texts):
        raise ValueError("Embedding row count does not match the text count.")

    if not np.isfinite(embedding_array).all():
        raise ValueError("Embeddings contain missing or infinite values.")

    vector_norms = np.linalg.norm(
        embedding_array,
        axis=1,
    )

    if not np.allclose(
        vector_norms,
        1.0,
        atol=1e-4,
    ):
        faiss.normalize_L2(embedding_array)

    return np.ascontiguousarray(
        embedding_array,
        dtype=np.float32,
    )


def build_faiss_cosine_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    """Build an exact cosine-similarity FAISS index."""

    embedding_array = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    if embedding_array.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if embedding_array.shape[0] == 0:
        raise ValueError("Cannot build an index with zero vectors.")

    if not np.isfinite(embedding_array).all():
        raise ValueError("Embeddings contain missing or infinite values.")

    vector_norms = np.linalg.norm(
        embedding_array,
        axis=1,
    )

    if not np.allclose(
        vector_norms,
        1.0,
        atol=1e-4,
    ):
        faiss.normalize_L2(embedding_array)

    dimension = int(embedding_array.shape[1])

    index = faiss.IndexFlatIP(dimension)

    index.add(embedding_array)

    if index.ntotal != len(embedding_array):
        raise ValueError("FAISS vector count does not match embedding count.")

    return index


def save_faiss_index(
    index: faiss.Index,
    output_path: Path,
) -> None:
    """Save a FAISS index atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    faiss.write_index(
        index,
        str(temporary_path),
    )

    temporary_path.replace(output_path)


def load_faiss_index(
    input_path: Path,
) -> faiss.Index:
    """Load a saved FAISS index."""

    if not input_path.exists():
        raise FileNotFoundError(f"FAISS index does not exist: {input_path}")

    return faiss.read_index(str(input_path))


def encode_query(
    model: EmbeddingModelProtocol,
    query: str,
) -> np.ndarray:
    """Encode and normalize a single search query."""

    normalized_query = normalize_embedding_text(query)

    if not normalized_query:
        raise ValueError("Search query cannot be empty.")

    query_embedding = model.encode(
        [normalized_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_array = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_array.ndim != 2:
        raise ValueError("Query embedding must be two-dimensional.")

    if query_array.shape[0] != 1:
        raise ValueError("Expected exactly one query embedding.")

    if not np.isfinite(query_array).all():
        raise ValueError("Query embedding contains invalid values.")

    vector_norm = np.linalg.norm(
        query_array,
        axis=1,
    )

    if not np.allclose(
        vector_norm,
        1.0,
        atol=1e-4,
    ):
        faiss.normalize_L2(query_array)

    return np.ascontiguousarray(
        query_array,
        dtype=np.float32,
    )


def semantic_search(
    query: str,
    model: EmbeddingModelProtocol,
    index: faiss.Index,
    metadata_records: list[dict[str, Any]],
    top_k: int = 10,
    repository: str | None = None,
    section_name: str | None = None,
    review_priority: str | None = None,
) -> list[SearchResult]:
    """Run semantic retrieval with optional metadata filters."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if index.ntotal != len(metadata_records):
        raise ValueError("FAISS index size and metadata count do not match.")

    query_embedding = encode_query(
        model=model,
        query=query,
    )

    candidate_count = min(
        max(
            top_k * 10,
            top_k,
        ),
        int(index.ntotal),
    )

    scores, positions = index.search(
        query_embedding,
        candidate_count,
    )

    results: list[SearchResult] = []

    for score, position in zip(
        scores[0],
        positions[0],
        strict=True,
    ):
        if position < 0:
            continue

        record = metadata_records[int(position)]

        if (
            repository is not None
            and str(record.get("repository")).lower() != repository.lower()
        ):
            continue

        if (
            section_name is not None
            and str(record.get("section_name")).lower() != section_name.lower()
        ):
            continue

        if (
            review_priority is not None
            and str(record.get("review_priority")).lower() != review_priority.lower()
        ):
            continue

        results.append(
            SearchResult(
                rank=len(results) + 1,
                score=float(score),
                chunk_id=str(record["chunk_id"]),
                document_id=str(record["document_id"]),
                pr_number=str(record["pr_number"]),
                repository=str(record["repository"]),
                section_name=str(record["section_name"]),
                title=str(record["title"]),
                original_content=str(record["original_content"]),
                embedding_content=str(record["embedding_content"]),
                metadata=record,
            )
        )

        if len(results) >= top_k:
            break

    return results


def retrieve_by_pr_number(
    pr_number: str | int,
    metadata_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retrieve every chunk belonging to one exact PR number."""

    normalized_pr_number = str(pr_number).strip()

    results = [
        record
        for record in metadata_records
        if str(
            record.get(
                "pr_number",
                "",
            )
        ).strip()
        == normalized_pr_number
    ]

    return sorted(
        results,
        key=lambda record: int(
            record.get(
                "chunk_index",
                0,
            )
        ),
    )


def validate_index_outputs(
    embeddings: np.ndarray,
    index: faiss.Index,
    metadata_records: list[dict[str, Any]],
    expected_chunk_count: int,
) -> dict[str, Any]:
    """Validate embedding and FAISS index outputs."""

    embedding_array = np.asarray(embeddings)

    embedding_count_valid = bool(embedding_array.shape[0] == expected_chunk_count)

    embedding_dimension_valid = bool(
        embedding_array.ndim == 2 and embedding_array.shape[1] > 0
    )

    embedding_values_valid = bool(np.isfinite(embedding_array).all())

    embedding_norms = np.linalg.norm(
        embedding_array,
        axis=1,
    )

    embeddings_normalized = bool(
        np.allclose(
            embedding_norms,
            1.0,
            atol=1e-4,
        )
    )

    index_count_valid = bool(index.ntotal == expected_chunk_count)

    metadata_count_valid = bool(len(metadata_records) == expected_chunk_count)

    vector_positions_valid = bool(
        [int(record["vector_position"]) for record in metadata_records]
        == list(range(expected_chunk_count))
    )

    unique_chunk_ids_valid = bool(
        len({str(record["chunk_id"]) for record in metadata_records})
        == expected_chunk_count
    )

    validation_passed = bool(
        embedding_count_valid
        and embedding_dimension_valid
        and embedding_values_valid
        and embeddings_normalized
        and index_count_valid
        and metadata_count_valid
        and vector_positions_valid
        and unique_chunk_ids_valid
    )

    return {
        "expected_chunk_count": int(expected_chunk_count),
        "embedding_count": int(embedding_array.shape[0]),
        "embedding_count_valid": (embedding_count_valid),
        "embedding_dimension": int(embedding_array.shape[1]),
        "embedding_dimension_valid": (embedding_dimension_valid),
        "embedding_values_valid": (embedding_values_valid),
        "embeddings_normalized": (embeddings_normalized),
        "index_vector_count": int(index.ntotal),
        "index_count_valid": (index_count_valid),
        "metadata_count": int(len(metadata_records)),
        "metadata_count_valid": (metadata_count_valid),
        "vector_positions_valid": (vector_positions_valid),
        "unique_chunk_ids_valid": (unique_chunk_ids_valid),
        "validation_passed": (validation_passed),
    }
