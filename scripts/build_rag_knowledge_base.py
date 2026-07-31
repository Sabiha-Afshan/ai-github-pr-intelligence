"""Prepare the structured local RAG knowledge base."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.discovery import load_dataset
from src.rag.knowledge_base import (
    build_chunk_manifest,
    build_document_manifest,
    build_knowledge_documents,
    chunk_knowledge_documents,
    knowledge_chunk_to_dict,
    knowledge_document_to_dict,
    serialize_json_line,
    validate_knowledge_outputs,
    validate_unified_source,
)
from src.utils.paths import (
    PROCESSED_DATA_DIRECTORY,
    PROJECT_ROOT,
    REPORTS_DIRECTORY,
)

UNIFIED_DATASET_PATH = PROCESSED_DATA_DIRECTORY / "unified_pr_intelligence.csv"

STAGE_7B_COMPLETION_PATH = REPORTS_DIRECTORY / "stage_7b_completion_report.json"

KNOWLEDGE_BASE_DIRECTORY = PROJECT_ROOT / "data" / "knowledge_base"

DOCUMENTS_JSONL_PATH = KNOWLEDGE_BASE_DIRECTORY / "pr_documents.jsonl"

CHUNKS_JSONL_PATH = KNOWLEDGE_BASE_DIRECTORY / "pr_chunks.jsonl"

DOCUMENT_MANIFEST_PATH = REPORTS_DIRECTORY / "stage_8a_document_manifest.csv"

CHUNK_MANIFEST_PATH = REPORTS_DIRECTORY / "stage_8a_chunk_manifest.csv"

COMPLETION_REPORT_PATH = REPORTS_DIRECTORY / "stage_8a_completion_report.json"

MAXIMUM_CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 25


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
        raise ValueError(f"Expected a JSON object in {input_path}.")

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


def save_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save records to JSONL atomically."""

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
            output_file.write(serialize_json_line(record))

            output_file.write("\n")

    temporary_path.replace(output_path)


def main() -> int:
    """Run Stage 8A knowledge-base preparation."""

    required_paths = [
        UNIFIED_DATASET_PATH,
        STAGE_7B_COMPLETION_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        print("FAIL: Required Stage 8A files are missing:")

        for path in missing_paths:
            print(path)

        return 1

    try:
        stage_7b_report = load_json(STAGE_7B_COMPLETION_PATH)

        if not stage_7b_report.get(
            "overall_verification_passed",
            False,
        ):
            raise ValueError("Stage 7B did not pass verification.")

        unified_dataframe = load_dataset(UNIFIED_DATASET_PATH)

        source_validation = validate_unified_source(unified_dataframe)

        if not source_validation["validation_passed"]:
            raise ValueError(f"Unified source failed validation: {source_validation}")

        documents = build_knowledge_documents(unified_dataframe)

        chunks = chunk_knowledge_documents(
            documents=documents,
            maximum_words=(MAXIMUM_CHUNK_WORDS),
            overlap_words=(CHUNK_OVERLAP_WORDS),
        )

        document_manifest = build_document_manifest(
            documents=documents,
            chunks=chunks,
        )

        chunk_manifest = build_chunk_manifest(chunks)

        output_validation = validate_knowledge_outputs(
            source_dataframe=(unified_dataframe),
            documents=documents,
            chunks=chunks,
            document_manifest=(document_manifest),
            chunk_manifest=(chunk_manifest),
            maximum_words=(MAXIMUM_CHUNK_WORDS),
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"FAIL: {error}")
        return 1

    document_records = [knowledge_document_to_dict(document) for document in documents]

    chunk_records = [knowledge_chunk_to_dict(chunk) for chunk in chunks]

    KNOWLEDGE_BASE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(
        document_records,
        DOCUMENTS_JSONL_PATH,
    )

    save_jsonl(
        chunk_records,
        CHUNKS_JSONL_PATH,
    )

    document_manifest.to_csv(
        DOCUMENT_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    chunk_manifest.to_csv(
        CHUNK_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    artifact_validation = {
        "documents_jsonl_exists": (DOCUMENTS_JSONL_PATH.exists()),
        "chunks_jsonl_exists": (CHUNKS_JSONL_PATH.exists()),
        "document_manifest_exists": (DOCUMENT_MANIFEST_PATH.exists()),
        "chunk_manifest_exists": (CHUNK_MANIFEST_PATH.exists()),
        "documents_jsonl_non_empty": bool(
            DOCUMENTS_JSONL_PATH.exists() and DOCUMENTS_JSONL_PATH.stat().st_size > 0
        ),
        "chunks_jsonl_non_empty": bool(
            CHUNKS_JSONL_PATH.exists() and CHUNKS_JSONL_PATH.stat().st_size > 0
        ),
    }

    artifact_validation["validation_passed"] = bool(all(artifact_validation.values()))

    section_distribution = (
        chunk_manifest["section_name"].value_counts().sort_index().to_dict()
    )

    priority_distribution = (
        document_manifest["review_priority"].value_counts().to_dict()
    )

    overall_passed = bool(
        source_validation["validation_passed"]
        and output_validation["validation_passed"]
        and artifact_validation["validation_passed"]
    )

    completion_report = {
        "generated_at": datetime.now(UTC),
        "stage": "Stage 8A",
        "component": ("Structured RAG knowledge-base preparation"),
        "source_dataset": str(UNIFIED_DATASET_PATH),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "maximum_chunk_words": (MAXIMUM_CHUNK_WORDS),
        "chunk_overlap_words": (CHUNK_OVERLAP_WORDS),
        "average_chunks_per_document": (
            round(
                len(chunks) / len(documents),
                4,
            )
        ),
        "section_distribution": (section_distribution),
        "priority_distribution": (priority_distribution),
        "source_validation": (source_validation),
        "output_validation": (output_validation),
        "artifact_validation": (artifact_validation),
        "knowledge_policy": {
            "one_document_per_pr": True,
            "stable_document_ids": True,
            "stable_chunk_ids": True,
            "section_aware_chunking": True,
            "human_decision_authority_retained": True,
            "model_predictions_presented_as_facts": False,
            "causal_claims_allowed": False,
            "embeddings_created": False,
            "vector_index_created": False,
            "llm_called": False,
        },
        "outputs": {
            "documents_jsonl": str(DOCUMENTS_JSONL_PATH),
            "chunks_jsonl": str(CHUNKS_JSONL_PATH),
            "document_manifest": str(DOCUMENT_MANIFEST_PATH),
            "chunk_manifest": str(CHUNK_MANIFEST_PATH),
        },
        "overall_verification_passed": (overall_passed),
    }

    save_json(
        completion_report,
        COMPLETION_REPORT_PATH,
    )

    print("Stage 8A structured RAG knowledge-base preparation")
    print("=" * 108)

    print()
    print("Source validation:")

    print(
        json.dumps(
            source_validation,
            indent=2,
        )
    )

    print()
    print("Knowledge-base size:")

    print(
        json.dumps(
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "average_chunks_per_document": (
                    round(
                        len(chunks) / len(documents),
                        4,
                    )
                ),
                "maximum_chunk_words": (MAXIMUM_CHUNK_WORDS),
                "chunk_overlap_words": (CHUNK_OVERLAP_WORDS),
            },
            indent=2,
        )
    )

    print()
    print("Chunk section distribution:")

    print(
        json.dumps(
            section_distribution,
            indent=2,
        )
    )

    print()
    print("Document priority distribution:")

    print(
        json.dumps(
            priority_distribution,
            indent=2,
        )
    )

    print()
    print("Sample document:")

    print(
        json.dumps(
            document_records[0],
            indent=2,
            ensure_ascii=False,
            default=str,
        )[:5000]
    )

    print()
    print("Sample chunks:")

    for sample_chunk in chunk_records[:3]:
        print(
            json.dumps(
                sample_chunk,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("-" * 80)

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
        "Embeddings created:",
        False,
    )

    print(
        "Vector index created:",
        False,
    )

    print(
        "LLM called:",
        False,
    )

    print()
    print(
        "Overall Stage 8A verification passed:",
        overall_passed,
    )

    print()
    print("Documents JSONL:")

    print(DOCUMENTS_JSONL_PATH)

    print()
    print("Chunks JSONL:")

    print(CHUNKS_JSONL_PATH)

    print()
    print("Completion report:")

    print(COMPLETION_REPORT_PATH)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
