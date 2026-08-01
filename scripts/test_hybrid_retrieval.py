from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag.hybrid_retrieval import HybridRetriever
from src.utils.paths import PROJECT_ROOT


REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "reports"
OUTPUT_PATH = REPORTS_DIRECTORY / "stage_8c_retrieval_diagnostics.json"


def serialise_results(results: list[Any]) -> list[dict[str, Any]]:
    return [result.to_dict() for result in results]


def print_results(
    test_name: str,
    query: str,
    results: list[Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"TEST: {test_name}")
    print(f"QUERY: {query}")
    print("=" * 90)

    if not results:
        print("No results returned.")
        return

    for rank, result in enumerate(results, start=1):
        preview = result.text.replace("\n", " ").strip()
        preview = preview[:220]

        print(
            f"\nRank: {rank}"
            f"\nPR Number: {result.pr_number}"
            f"\nSection: {result.section}"
            f"\nHybrid Score: {result.hybrid_score:.6f}"
            f"\nSemantic Score: {result.semantic_score:.6f}"
            f"\nKeyword Score: {result.keyword_score:.6f}"
            f"\nExact Match Score: {result.exact_match_score:.6f}"
            f"\nChunk ID: {result.chunk_id}"
            f"\nPreview: {preview}"
        )


def validate_score_ordering(results: list[Any]) -> bool:
    scores = [result.hybrid_score for result in results]

    return all(
        scores[index] >= scores[index + 1]
        for index in range(len(scores) - 1)
    )


def validate_unique_chunks(results: list[Any]) -> bool:
    chunk_ids = [result.chunk_id for result in results]

    return len(chunk_ids) == len(set(chunk_ids))


def validate_exact_pr_results(
    expected_pr_number: int,
    results: list[Any],
) -> bool:
    if not results:
        return False

    top_result = results[0]

    return (
        top_result.pr_number == expected_pr_number
        and top_result.exact_match_score == 1.0
    )


def run_test_case(
    retriever: HybridRetriever,
    test_name: str,
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        filters=filters,
    )

    print_results(
        test_name=test_name,
        query=query,
        results=results,
    )

    score_ordering_valid = validate_score_ordering(results)
    unique_chunks_valid = validate_unique_chunks(results)

    diagnostics = {
        "test_name": test_name,
        "query": query,
        "filters": filters,
        "result_count": len(results),
        "score_ordering_valid": score_ordering_valid,
        "unique_chunks_valid": unique_chunks_valid,
        "results": serialise_results(results),
    }

    return diagnostics


def main() -> None:
    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading hybrid retriever...")

    retriever = HybridRetriever(
        semantic_weight=0.60,
        keyword_weight=0.25,
        exact_match_weight=0.15,
    )

    print("Hybrid retriever loaded successfully.")
    print(f"Embedding model: {retriever.model_name}")
    print(f"FAISS vectors: {retriever.index.ntotal}")
    print(f"Metadata records: {len(retriever.metadata)}")

    diagnostics: list[dict[str, Any]] = []

    diagnostics.append(
        run_test_case(
            retriever=retriever,
            test_name="Semantic retrieval",
            query=(
                "pull requests with large code changes, "
                "many modified files and higher review complexity"
            ),
        )
    )

    diagnostics.append(
        run_test_case(
            retriever=retriever,
            test_name="Merge-delay retrieval",
            query=(
                "pull requests likely to experience delayed merge "
                "because of review workload and complexity"
            ),
        )
    )

    diagnostics.append(
        run_test_case(
            retriever=retriever,
            test_name="Policy-risk retrieval",
            query=(
                "high-risk pull requests requiring manual review "
                "because policy rules were triggered"
            ),
        )
    )

    available_pr_numbers = sorted(
        retriever.pr_number_to_indices.keys()
    )

    exact_match_diagnostics: dict[str, Any]

    if available_pr_numbers:
        exact_pr_number = available_pr_numbers[0]
        exact_query = f"PR #{exact_pr_number}"

        exact_results = retriever.retrieve(
            query=exact_query,
            top_k=5,
        )

        print_results(
            test_name="Exact PR-number retrieval",
            query=exact_query,
            results=exact_results,
        )

        exact_match_valid = validate_exact_pr_results(
            expected_pr_number=exact_pr_number,
            results=exact_results,
        )

        exact_match_diagnostics = {
            "test_name": "Exact PR-number retrieval",
            "query": exact_query,
            "expected_pr_number": exact_pr_number,
            "result_count": len(exact_results),
            "exact_match_valid": exact_match_valid,
            "score_ordering_valid": validate_score_ordering(
                exact_results
            ),
            "unique_chunks_valid": validate_unique_chunks(
                exact_results
            ),
            "results": serialise_results(exact_results),
        }
    else:
        exact_match_diagnostics = {
            "test_name": "Exact PR-number retrieval",
            "query": None,
            "expected_pr_number": None,
            "result_count": 0,
            "exact_match_valid": False,
            "score_ordering_valid": False,
            "unique_chunks_valid": False,
            "results": [],
            "reason": "No PR numbers were available in metadata.",
        }

    diagnostics.append(exact_match_diagnostics)

    all_score_ordering_valid = all(
        item["score_ordering_valid"]
        for item in diagnostics
    )

    all_unique_chunks_valid = all(
        item["unique_chunks_valid"]
        for item in diagnostics
    )

    exact_match_test = next(
        item
        for item in diagnostics
        if item["test_name"] == "Exact PR-number retrieval"
    )

    overall_status = (
        "passed"
        if (
            all_score_ordering_valid
            and all_unique_chunks_valid
            and exact_match_test.get("exact_match_valid", False)
        )
        else "failed"
    )

    report = {
        "stage": "8C",
        "stage_name": "Hybrid Retrieval",
        "status": overall_status,
        "embedding_model": retriever.model_name,
        "faiss_vector_count": retriever.index.ntotal,
        "metadata_record_count": len(retriever.metadata),
        "semantic_weight": retriever.semantic_weight,
        "keyword_weight": retriever.keyword_weight,
        "exact_match_weight": retriever.exact_match_weight,
        "all_score_ordering_valid": all_score_ordering_valid,
        "all_unique_chunks_valid": all_unique_chunks_valid,
        "exact_match_valid": exact_match_test.get(
            "exact_match_valid",
            False,
        ),
        "tests": diagnostics,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print("STAGE 8C SUMMARY")
    print("=" * 90)
    print(f"Status: {overall_status.upper()}")
    print(
        "Score ordering valid: "
        f"{all_score_ordering_valid}"
    )
    print(
        "Unique chunk results valid: "
        f"{all_unique_chunks_valid}"
    )
    print(
        "Exact PR lookup valid: "
        f"{exact_match_test.get('exact_match_valid', False)}"
    )
    print(f"Report saved to: {OUTPUT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 8C hybrid retrieval validation failed."
        )


if __name__ == "__main__":
    main()