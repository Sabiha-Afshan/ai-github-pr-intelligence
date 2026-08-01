from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.rag.hybrid_retrieval import (
    HybridRetriever,
    RetrievalResult,
)
from src.utils.paths import PROJECT_ROOT


REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "reports"

EVALUATION_RESULTS_PATH = (
    REPORTS_DIRECTORY
    / "stage_8d_retrieval_evaluation_results.json"
)

COMPLETION_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8d_completion_report.json"
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    query: str
    relevance_description: str
    relevance_function: Callable[[RetrievalResult], bool]
    top_k: int = 5


def normalise_text(value: Any) -> str:
    return str(value or "").strip().lower()


def reciprocal_rank(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
) -> float:
    for rank, result in enumerate(results, start=1):
        if relevance_function(result):
            return 1.0 / rank

    return 0.0


def precision_at_k(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    top_results = results[:k]

    relevant_count = sum(
        1
        for result in top_results
        if relevance_function(result)
    )

    return relevant_count / k


def hit_rate_at_k(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
    k: int,
) -> float:
    return float(
        any(
            relevance_function(result)
            for result in results[:k]
        )
    )


def relevant_result_count(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
) -> int:
    return sum(
        1
        for result in results
        if relevance_function(result)
    )


def build_exact_pr_cases(
    retriever: HybridRetriever,
    sample_size: int = 10,
) -> list[EvaluationCase]:
    available_pr_numbers = sorted(
        retriever.pr_number_to_indices.keys()
    )

    if not available_pr_numbers:
        return []

    if len(available_pr_numbers) <= sample_size:
        selected_pr_numbers = available_pr_numbers
    else:
        step = max(
            1,
            len(available_pr_numbers) // sample_size,
        )

        selected_pr_numbers = (
            available_pr_numbers[::step][:sample_size]
        )

    evaluation_cases: list[EvaluationCase] = []

    for pr_number in selected_pr_numbers:
        evaluation_cases.append(
            EvaluationCase(
                case_id=f"exact_pr_{pr_number}",
                category="exact_pr_lookup",
                query=f"PR #{pr_number}",
                relevance_description=(
                    f"Result belongs to PR #{pr_number}"
                ),
                relevance_function=(
                    lambda result, expected=pr_number:
                    result.pr_number == expected
                ),
                top_k=5,
            )
        )

    return evaluation_cases


def build_section_cases() -> list[EvaluationCase]:
    section_queries = [
        {
            "case_id": "section_change_evidence",
            "query": (
                "pull requests with large code changes, "
                "many modified files, additions and deletions"
            ),
            "expected_section": "Change evidence",
        },
        {
            "case_id": "section_policy_intelligence",
            "query": (
                "high-risk pull requests requiring manual review "
                "because deterministic policy rules were triggered"
            ),
            "expected_section": (
                "Deterministic policy intelligence"
            ),
        },
        {
            "case_id": "section_predictive_intelligence",
            "query": (
                "merge probability, predicted merge outcome "
                "and merge-delay prediction"
            ),
            "expected_section": "Predictive intelligence",
        },
        {
            "case_id": "section_review_priority",
            "query": (
                "review priority score and recommended next action "
                "for pull request reviewers"
            ),
            "expected_section": "Unified review priority",
        },
        {
            "case_id": "section_pr_description",
            "query": (
                "pull request description quality, word count "
                "and detailed description"
            ),
            "expected_section": "PR description",
        },
        {
            "case_id": "section_pr_identity",
            "query": (
                "pull request repository, number, title, author, "
                "state and creation date"
            ),
            "expected_section": "PR identity",
        },
    ]

    evaluation_cases: list[EvaluationCase] = []

    for item in section_queries:
        expected_section = item["expected_section"]

        evaluation_cases.append(
            EvaluationCase(
                case_id=item["case_id"],
                category="section_retrieval",
                query=item["query"],
                relevance_description=(
                    f"Section equals '{expected_section}'"
                ),
                relevance_function=(
                    lambda result, expected=expected_section:
                    normalise_text(result.section)
                    == normalise_text(expected)
                ),
                top_k=5,
            )
        )

    return evaluation_cases


def evaluate_case(
    retriever: HybridRetriever,
    evaluation_case: EvaluationCase,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    results = retriever.retrieve(
        query=evaluation_case.query,
        top_k=evaluation_case.top_k,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    reciprocal_rank_value = reciprocal_rank(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
    )

    precision_value = precision_at_k(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
        k=evaluation_case.top_k,
    )

    hit_rate_value = hit_rate_at_k(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
        k=evaluation_case.top_k,
    )

    relevant_count = relevant_result_count(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
    )

    ranked_results: list[dict[str, Any]] = []

    for rank, result in enumerate(results, start=1):
        ranked_results.append(
            {
                "rank": rank,
                "is_relevant": (
                    evaluation_case.relevance_function(
                        result
                    )
                ),
                "pr_number": result.pr_number,
                "section": result.section,
                "chunk_id": result.chunk_id,
                "hybrid_score": result.hybrid_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "exact_match_score": (
                    result.exact_match_score
                ),
                "text_preview": (
                    result.text
                    .replace("\n", " ")
                    .strip()[:250]
                ),
            }
        )

    return {
        "case_id": evaluation_case.case_id,
        "category": evaluation_case.category,
        "query": evaluation_case.query,
        "relevance_description": (
            evaluation_case.relevance_description
        ),
        "top_k": evaluation_case.top_k,
        "result_count": len(results),
        "relevant_result_count": relevant_count,
        "hit_rate_at_k": round(hit_rate_value, 6),
        "precision_at_k": round(precision_value, 6),
        "reciprocal_rank": round(
            reciprocal_rank_value,
            6,
        ),
        "latency_ms": round(latency_ms, 3),
        "results": ranked_results,
    }


def calculate_category_summary(
    evaluation_results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    matching_results = [
        result
        for result in evaluation_results
        if result["category"] == category
    ]

    if not matching_results:
        return {
            "category": category,
            "case_count": 0,
            "mean_hit_rate_at_k": 0.0,
            "mean_precision_at_k": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_latency_ms": 0.0,
        }

    return {
        "category": category,
        "case_count": len(matching_results),
        "mean_hit_rate_at_k": round(
            statistics.mean(
                result["hit_rate_at_k"]
                for result in matching_results
            ),
            6,
        ),
        "mean_precision_at_k": round(
            statistics.mean(
                result["precision_at_k"]
                for result in matching_results
            ),
            6,
        ),
        "mean_reciprocal_rank": round(
            statistics.mean(
                result["reciprocal_rank"]
                for result in matching_results
            ),
            6,
        ),
        "mean_latency_ms": round(
            statistics.mean(
                result["latency_ms"]
                for result in matching_results
            ),
            3,
        ),
    }


def print_case_result(
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"CATEGORY: {result['category']}")
    print(f"QUERY: {result['query']}")
    print(
        "RELEVANCE: "
        f"{result['relevance_description']}"
    )
    print("-" * 90)
    print(
        f"Hit Rate@{result['top_k']}: "
        f"{result['hit_rate_at_k']:.4f}"
    )
    print(
        f"Precision@{result['top_k']}: "
        f"{result['precision_at_k']:.4f}"
    )
    print(
        "Reciprocal Rank: "
        f"{result['reciprocal_rank']:.4f}"
    )
    print(
        f"Latency: {result['latency_ms']:.2f} ms"
    )

    for ranked_result in result["results"]:
        marker = (
            "RELEVANT"
            if ranked_result["is_relevant"]
            else "not relevant"
        )

        print(
            f"\nRank {ranked_result['rank']} "
            f"[{marker}]"
            f"\nPR: {ranked_result['pr_number']}"
            f"\nSection: {ranked_result['section']}"
            f"\nHybrid score: "
            f"{ranked_result['hybrid_score']:.6f}"
        )


def main() -> None:
    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Stage 8D hybrid retriever...")

    retriever = HybridRetriever(
        semantic_weight=0.60,
        keyword_weight=0.25,
        exact_match_weight=0.15,
    )

    exact_pr_cases = build_exact_pr_cases(
        retriever=retriever,
        sample_size=10,
    )

    section_cases = build_section_cases()

    evaluation_cases = (
        exact_pr_cases + section_cases
    )

    print(
        f"Evaluation cases prepared: "
        f"{len(evaluation_cases)}"
    )
    print(
        f"Exact PR cases: {len(exact_pr_cases)}"
    )
    print(
        f"Section retrieval cases: "
        f"{len(section_cases)}"
    )

    evaluation_results: list[dict[str, Any]] = []

    for evaluation_case in evaluation_cases:
        case_result = evaluate_case(
            retriever=retriever,
            evaluation_case=evaluation_case,
        )

        evaluation_results.append(case_result)
        print_case_result(case_result)

    overall_hit_rate = statistics.mean(
        result["hit_rate_at_k"]
        for result in evaluation_results
    )

    overall_precision = statistics.mean(
        result["precision_at_k"]
        for result in evaluation_results
    )

    overall_mrr = statistics.mean(
        result["reciprocal_rank"]
        for result in evaluation_results
    )

    overall_mean_latency = statistics.mean(
        result["latency_ms"]
        for result in evaluation_results
    )

    exact_pr_summary = calculate_category_summary(
        evaluation_results=evaluation_results,
        category="exact_pr_lookup",
    )

    section_summary = calculate_category_summary(
        evaluation_results=evaluation_results,
        category="section_retrieval",
    )

    thresholds = {
        "minimum_overall_hit_rate_at_5": 0.85,
        "minimum_overall_mrr": 0.75,
        "minimum_exact_pr_hit_rate_at_5": 1.00,
        "minimum_section_hit_rate_at_5": 0.80,
    }

    threshold_checks = {
        "overall_hit_rate_passed": (
            overall_hit_rate
            >= thresholds[
                "minimum_overall_hit_rate_at_5"
            ]
        ),
        "overall_mrr_passed": (
            overall_mrr
            >= thresholds["minimum_overall_mrr"]
        ),
        "exact_pr_hit_rate_passed": (
            exact_pr_summary["mean_hit_rate_at_k"]
            >= thresholds[
                "minimum_exact_pr_hit_rate_at_5"
            ]
        ),
        "section_hit_rate_passed": (
            section_summary["mean_hit_rate_at_k"]
            >= thresholds[
                "minimum_section_hit_rate_at_5"
            ]
        ),
    }

    overall_status = (
        "passed"
        if all(threshold_checks.values())
        else "failed"
    )

    detailed_report = {
        "stage": "8D",
        "stage_name": (
            "Hybrid Retrieval Quality Evaluation"
        ),
        "status": overall_status,
        "embedding_model": retriever.model_name,
        "evaluation_case_count": len(
            evaluation_results
        ),
        "overall_metrics": {
            "hit_rate_at_5": round(
                overall_hit_rate,
                6,
            ),
            "precision_at_5": round(
                overall_precision,
                6,
            ),
            "mean_reciprocal_rank": round(
                overall_mrr,
                6,
            ),
            "mean_latency_ms": round(
                overall_mean_latency,
                3,
            ),
        },
        "category_summaries": {
            "exact_pr_lookup": exact_pr_summary,
            "section_retrieval": section_summary,
        },
        "thresholds": thresholds,
        "threshold_checks": threshold_checks,
        "evaluation_results": evaluation_results,
    }

    with EVALUATION_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detailed_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    completion_report = {
        "stage": "8D",
        "stage_name": (
            "Hybrid Retrieval Quality Evaluation"
        ),
        "status": overall_status,
        "evaluation_case_count": len(
            evaluation_results
        ),
        "exact_pr_case_count": len(
            exact_pr_cases
        ),
        "section_case_count": len(
            section_cases
        ),
        "overall_hit_rate_at_5": round(
            overall_hit_rate,
            6,
        ),
        "overall_precision_at_5": round(
            overall_precision,
            6,
        ),
        "mean_reciprocal_rank": round(
            overall_mrr,
            6,
        ),
        "mean_latency_ms": round(
            overall_mean_latency,
            3,
        ),
        "exact_pr_hit_rate_at_5": (
            exact_pr_summary[
                "mean_hit_rate_at_k"
            ]
        ),
        "section_hit_rate_at_5": (
            section_summary[
                "mean_hit_rate_at_k"
            ]
        ),
        "threshold_checks": threshold_checks,
        "detailed_results_path": str(
            EVALUATION_RESULTS_PATH
        ),
    }

    with COMPLETION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            completion_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print("STAGE 8D SUMMARY")
    print("=" * 90)
    print(f"Status: {overall_status.upper()}")
    print(
        f"Evaluation cases: "
        f"{len(evaluation_results)}"
    )
    print(
        f"Overall Hit Rate@5: "
        f"{overall_hit_rate:.4f}"
    )
    print(
        f"Overall Precision@5: "
        f"{overall_precision:.4f}"
    )
    print(
        f"Mean Reciprocal Rank: "
        f"{overall_mrr:.4f}"
    )
    print(
        f"Mean latency: "
        f"{overall_mean_latency:.2f} ms"
    )
    print(
        "Exact PR Hit Rate@5: "
        f"{exact_pr_summary['mean_hit_rate_at_k']:.4f}"
    )
    print(
        "Section Hit Rate@5: "
        f"{section_summary['mean_hit_rate_at_k']:.4f}"
    )
    print(
        f"Detailed report: "
        f"{EVALUATION_RESULTS_PATH}"
    )
    print(
        f"Completion report: "
        f"{COMPLETION_REPORT_PATH}"
    )

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 8D retrieval quality evaluation "
            "did not meet the required thresholds."
        )


if __name__ == "__main__":
    main()