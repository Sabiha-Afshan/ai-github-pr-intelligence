"""Generate local embeddings and build the FAISS vector index."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from src.rag.embedding_index import (
    DEFAULT_EMBEDDING_MODEL,
    build_faiss_cosine_index,
    create_embedding_model,
    encode_texts,
    load_jsonl,
    prepare_embedding_records,
    retrieve_by_pr_number,
    save_faiss_index,
    save_jsonl,
    semantic_search,
    validate_index_outputs,
)
from src.utils.paths import (
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

KNOWLEDGE_BASE_DIRECTORY = PROJECT_ROOT / "data" / "knowledge_base"

VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "data" / "vector_store"

CHUNKS_JSONL_PATH = KNOWLEDGE_BASE_DIRECTORY / "pr_chunks.jsonl"

STAGE_8A_COMPLETION_PATH = REPORTS_DIRECTORY / "stage_8a_completion_report.json"

FAISS_INDEX_PATH = VECTOR_STORE_DIRECTORY / "pr_chunks.faiss"

EMBEDDINGS_PATH = VECTOR_STORE_DIRECTORY / "pr_chunk_embeddings.npy"

METADATA_JSONL_PATH = VECTOR_STORE_DIRECTORY / "pr_chunk_metadata.jsonl"

INDEX_CONFIGURATION_PATH = VECTOR_STORE_DIRECTORY / "index_configuration.json"

EMBEDDING_MANIFEST_PATH = REPORTS_DIRECTORY / "stage_8b_embedding_manifest.csv"

RETRIEVAL_DIAGNOSTICS_PATH = REPORTS_DIRECTORY / "stage_8b_retrieval_diagnostics.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_8b_completion_report.json"

EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL

EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 64


def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        result = json.load(input_file)

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(f"Expected JSON object in {input_path}.")

    return result


def save_json(
    payload: Any,
    output_path: Path,
) -> None:
    """Save JSON atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )

    temporary_path.replace(output_path)


def build_retrieval_diagnostics(
    model: Any,
    index: faiss.Index,
    metadata_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Run exact and semantic retrieval checks."""

    diagnostic_records: list[dict[str, Any]] = []

    exact_results = retrieve_by_pr_number(
        pr_number="5844",
        metadata_records=(metadata_records),
    )

    diagnostic_records.append(
        {
            "diagnostic_type": ("exact_pr_lookup"),
            "query": "5844",
            "result_rank": 1,
            "result_count": len(exact_results),
            "pr_number": (exact_results[0]["pr_number"] if exact_results else None),
            "section_name": (
                exact_results[0]["section_name"] if exact_results else None
            ),
            "similarity_score": None,
            "passed": bool(
                exact_results
                and all(str(result["pr_number"]) == "5844" for result in exact_results)
            ),
        }
    )

    semantic_queries = [
        (
            "security-sensitive changes without tests that require mandatory review",
            None,
        ),
        (
            "pull requests with weak or missing descriptions",
            "PR description",
        ),
        (
            "high delay risk and urgent reviewer assignment",
            "Predictive intelligence",
        ),
    ]

    for query, section_filter in semantic_queries:
        results = semantic_search(
            query=query,
            model=model,
            index=index,
            metadata_records=(metadata_records),
            top_k=5,
            section_name=(section_filter),
        )

        for result in results:
            diagnostic_records.append(
                {
                    "diagnostic_type": ("semantic_search"),
                    "query": query,
                    "result_rank": (result.rank),
                    "result_count": len(results),
                    "pr_number": (result.pr_number),
                    "section_name": (result.section_name),
                    "similarity_score": (result.score),
                    "passed": bool(np.isfinite(result.score)),
                }
            )

    return pd.DataFrame(diagnostic_records)


def main() -> int:
    """Run Stage 8B embedding and index creation."""

    required_paths = [
        CHUNKS_JSONL_PATH,
        STAGE_8A_COMPLETION_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 8B files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    try:
        stage_8a_report = load_json(STAGE_8A_COMPLETION_PATH)

        if not stage_8a_report.get(
            "overall_verification_passed",
            False,
        ):
            raise ValueError("Stage 8A did not pass verification.")

        chunk_records = load_jsonl(CHUNKS_JSONL_PATH)

        prepared_records = prepare_embedding_records(chunk_records)

        embedding_texts = [record["embedding_content"] for record in prepared_records]

        print(
            "Loading embedding model:",
            EMBEDDING_MODEL_NAME,
        )

        embedding_model = create_embedding_model(
            model_name=(EMBEDDING_MODEL_NAME),
            device=(EMBEDDING_DEVICE),
        )

        print()
        print(
            "Encoding",
            len(embedding_texts),
            "knowledge chunks...",
        )

        embeddings = encode_texts(
            model=embedding_model,
            texts=embedding_texts,
            batch_size=(EMBEDDING_BATCH_SIZE),
        )

        index = build_faiss_cosine_index(embeddings)

        output_validation = validate_index_outputs(
            embeddings=embeddings,
            index=index,
            metadata_records=(prepared_records),
            expected_chunk_count=len(chunk_records),
        )

        if not output_validation["validation_passed"]:
            raise ValueError(
                f"Embedding and index outputs failed validation: {output_validation}"
            )

        retrieval_diagnostics = build_retrieval_diagnostics(
            model=embedding_model,
            index=index,
            metadata_records=(prepared_records),
        )

    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    VECTOR_STORE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_faiss_index(
        index=index,
        output_path=(FAISS_INDEX_PATH),
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
        allow_pickle=False,
    )

    save_jsonl(
        records=prepared_records,
        output_path=(METADATA_JSONL_PATH),
    )

    embedding_manifest = pd.DataFrame(
        [
            {
                "vector_position": (record["vector_position"]),
                "chunk_id": (record["chunk_id"]),
                "document_id": (record["document_id"]),
                "repository": (record["repository"]),
                "pr_number": (record["pr_number"]),
                "section_name": (record["section_name"]),
                "review_priority": (record["review_priority"]),
                "original_character_count": (record["original_character_count"]),
                "embedding_character_count": (record["embedding_character_count"]),
                "template_text_removed": (record["template_text_removed"]),
            }
            for record in prepared_records
        ]
    )

    embedding_manifest.to_csv(
        EMBEDDING_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    retrieval_diagnostics.to_csv(
        RETRIEVAL_DIAGNOSTICS_PATH,
        index=False,
        encoding="utf-8",
    )

    index_configuration = {
        "generated_at": datetime.now(UTC),
        "embedding_model": (EMBEDDING_MODEL_NAME),
        "embedding_device": (EMBEDDING_DEVICE),
        "embedding_batch_size": (EMBEDDING_BATCH_SIZE),
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "faiss_index_type": ("IndexFlatIP"),
        "similarity_metric": ("cosine_similarity"),
        "document_vectors_normalized": True,
        "query_vectors_normalized": True,
        "exact_search": True,
        "ollama_called": False,
    }

    save_json(
        index_configuration,
        INDEX_CONFIGURATION_PATH,
    )

    exact_lookup_passed = bool(
        retrieval_diagnostics.loc[
            retrieval_diagnostics["diagnostic_type"] == "exact_pr_lookup",
            "passed",
        ].all()
    )

    semantic_search_rows = retrieval_diagnostics.loc[
        retrieval_diagnostics["diagnostic_type"] == "semantic_search"
    ]

    semantic_retrieval_passed = bool(
        not semantic_search_rows.empty
        and semantic_search_rows["passed"].all()
        and semantic_search_rows["similarity_score"].notna().all()
    )

    artifact_validation = {
        "faiss_index_exists": (FAISS_INDEX_PATH.exists()),
        "embeddings_exist": (EMBEDDINGS_PATH.exists()),
        "metadata_exists": (METADATA_JSONL_PATH.exists()),
        "index_configuration_exists": (INDEX_CONFIGURATION_PATH.exists()),
        "embedding_manifest_exists": (EMBEDDING_MANIFEST_PATH.exists()),
        "retrieval_diagnostics_exist": (RETRIEVAL_DIAGNOSTICS_PATH.exists()),
        "exact_pr_lookup_passed": (exact_lookup_passed),
        "semantic_retrieval_passed": (semantic_retrieval_passed),
    }

    artifact_validation["validation_passed"] = bool(all(artifact_validation.values()))

    boilerplate_cleaned_count = int(
        embedding_manifest["template_text_removed"].astype(bool).sum()
    )

    overall_passed = bool(
        output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 8B",
        "component": ("Local embeddings and FAISS vector index"),
        "embedding_model": (EMBEDDING_MODEL_NAME),
        "embedding_device": (EMBEDDING_DEVICE),
        "source_chunk_count": len(chunk_records),
        "indexed_vector_count": int(index.ntotal),
        "embedding_dimension": int(embeddings.shape[1]),
        "boilerplate_cleaned_chunk_count": (boilerplate_cleaned_count),
        "index_type": ("IndexFlatIP"),
        "similarity_metric": ("Cosine similarity using normalized vectors"),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "retrieval_diagnostic_count": len(retrieval_diagnostics),
        "retrieval_policy": {
            "exact_pr_number_lookup": True,
            "semantic_vector_search": True,
            "metadata_filtering": True,
            "original_evidence_preserved": True,
            "cleaned_text_used_for_embedding": True,
            "llm_called": False,
            "ollama_called": False,
        },
        "outputs": {
            "faiss_index": str(FAISS_INDEX_PATH),
            "embeddings": str(EMBEDDINGS_PATH),
            "metadata": str(METADATA_JSONL_PATH),
            "index_configuration": str(INDEX_CONFIGURATION_PATH),
            "embedding_manifest": str(EMBEDDING_MANIFEST_PATH),
            "retrieval_diagnostics": str(RETRIEVAL_DIAGNOSTICS_PATH),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print()
    print("Stage 8B local embeddings and FAISS vector index")
    print("=" * 108)

    print()
    print("Embedding and index summary:")

    print(
        json.dumps(
            {
                "embedding_model": (EMBEDDING_MODEL_NAME),
                "embedding_device": (EMBEDDING_DEVICE),
                "chunk_count": len(chunk_records),
                "embedding_dimension": int(embeddings.shape[1]),
                "faiss_vector_count": int(index.ntotal),
                "index_type": ("IndexFlatIP"),
                "boilerplate_cleaned_chunks": (boilerplate_cleaned_count),
            },
            indent=2,
        )
    )

    print()
    print("Retrieval diagnostics:")

    print(retrieval_diagnostics.to_string(index=False))

    print()
    print("Output validation:")

    print(
        json.dumps(
            output_validation,
            indent=2,
        )
    )

    print()
    print("Artifact validation:")

    print(
        json.dumps(
            artifact_validation,
            indent=2,
        )
    )

    print()
    print(
        "Ollama called:",
        False,
    )

    print()
    print(
        "Overall Stage 8B verification passed:",
        overall_passed,
    )

    print()
    print("FAISS index:")
    print(FAISS_INDEX_PATH)

    print()
    print("Completion report:")
    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
