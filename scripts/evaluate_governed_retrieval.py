from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from scripts.evaluate_realistic_retrieval import (
    EvaluationCase,
    build_metadata_condition_cases,
    build_negative_cases,
    build_noisy_identifier_cases,
    build_paraphrased_section_cases,
)
from src.rag.governed_retrieval import (
    GovernedRetrievalResult,
    GovernedRetriever,
)
from src.utils.paths import PROJECT_ROOT


REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "reports"

BASELINE_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8e_realistic_retrieval_results.json"
)

DETAILED_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8g_governed_retrieval_evaluation.json"
)

COMPLETION_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8g_completion_report.json"
)


def reciprocal_rank(
    results: list[GovernedRetrievalResult],
    relevance_function: Callable[
        [GovernedRetrievalResult],
        bool,
    ],
) -> float:
    for rank, result in enumerate(results, start=1):
        if relevance_function(result):
            return 1.0 / rank

    return 0.0


def precision_at_k(
    results: list[GovernedRetrievalResult],
    relevance_function: Callable[
        [GovernedRetrievalResult],
        bool,
    ],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    relevant_count = sum(
        1
        for result in results[:k]
        if relevance_function(result)
    )

    return relevant_count / k


def hit_rate_at_k(
    results: list[GovernedRetrievalResult],
    relevance_function: Callable[
        [GovernedRetrievalResult],
        bool,
    ],
    k: int,
) -> float:
    return float(
        any(
            relevance_function(result)
            for result in results[:k]
        )
    )


def first_relevant_rank(
    results: list[GovernedRetrievalResult],
    relevance_function: Callable[
        [GovernedRetrievalResult],
        bool,
    ],
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if relevance_function(result):
            return rank

    return None


def evaluate_positive_case(
    retriever: GovernedRetriever,
    evaluation_case: EvaluationCase,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    response = retriever.retrieve(
        query=evaluation_case.query,
        top_k=evaluation_case.top_k,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    results = response.results

    hit_rate = hit_rate_at_k(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
        k=evaluation_case.top_k,
    )

    precision = precision_at_k(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
        k=evaluation_case.top_k,
    )

    rr = reciprocal_rank(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
    )

    relevant_rank = first_relevant_rank(
        results=results,
        relevance_function=(
            evaluation_case.relevance_function
        ),
    )

    ranked_results: list[dict[str, Any]] = []

    for result in results:
        ranked_results.append(
            {
                "rank": result.rank,
                "relevant": (
                    evaluation_case.relevance_function(
                        result
                    )
                ),
                "pr_number": result.pr_number,
                "section": result.section,
                "base_hybrid_score": (
                    result.base_hybrid_score
                ),
                "section_boost": result.section_boost,
                "metadata_boost": (
                    result.metadata_boost
                ),
                "governed_score": (
                    result.governed_score
                ),
                "metadata_condition_matches": (
                    result.metadata_condition_matches
                ),
                "metadata_condition_total": (
                    result.metadata_condition_total
                ),
                "chunk_id": result.chunk_id,
                "text_preview": (
                    result.text
                    .replace("\n", " ")
                    .strip()[:220]
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
        "action": response.action,
        "retrieval_executed": (
            response.retrieval_executed
        ),
        "result_count": len(results),
        "hit_rate_at_5": round(hit_rate, 6),
        "precision_at_5": round(precision, 6),
        "reciprocal_rank": round(rr, 6),
        "first_relevant_rank": relevant_rank,
        "latency_ms": round(latency_ms, 3),
        "query_understanding": (
            response.query_understanding.to_dict()
        ),
        "trace": response.trace,
        "results": ranked_results,
    }


def evaluate_negative_case(
    retriever: GovernedRetriever,
    evaluation_case: EvaluationCase,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    response = retriever.retrieve(
        query=evaluation_case.query,
        top_k=evaluation_case.top_k,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    correctly_abstained = (
        response.action
        in {
            "abstain_out_of_domain",
            "abstain_no_evidence",
        }
        and len(response.results) == 0
    )

    return {
        "case_id": evaluation_case.case_id,
        "category": evaluation_case.category,
        "query": evaluation_case.query,
        "action": response.action,
        "message": response.message,
        "retrieval_executed": (
            response.retrieval_executed
        ),
        "result_count": len(response.results),
        "correctly_abstained": (
            correctly_abstained
        ),
        "latency_ms": round(latency_ms, 3),
        "query_understanding": (
            response.query_understanding.to_dict()
        ),
        "trace": response.trace,
        "results": [
            result.to_dict()
            for result in response.results
        ],
    }


def category_summary(
    positive_results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    matching_results = [
        result
        for result in positive_results
        if result["category"] == category
    ]

    if not matching_results:
        return {
            "category": category,
            "case_count": 0,
            "hit_rate_at_5": 0.0,
            "precision_at_5": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_latency_ms": 0.0,
        }

    return {
        "category": category,
        "case_count": len(matching_results),
        "hit_rate_at_5": round(
            statistics.mean(
                result["hit_rate_at_5"]
                for result in matching_results
            ),
            6,
        ),
        "precision_at_5": round(
            statistics.mean(
                result["precision_at_5"]
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


def load_baseline_report() -> dict[str, Any]:
    if not BASELINE_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Stage 8E baseline report was not found: "
            f"{BASELINE_REPORT_PATH}"
        )

    with BASELINE_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def metric_change(
    current_value: float,
    baseline_value: float,
) -> dict[str, float]:
    absolute_change = (
        current_value - baseline_value
    )

    if baseline_value == 0:
        relative_change_percent = (
            100.0
            if current_value > 0
            else 0.0
        )
    else:
        relative_change_percent = (
            absolute_change
            / baseline_value
            * 100
        )

    return {
        "baseline": round(
            baseline_value,
            6,
        ),
        "current": round(
            current_value,
            6,
        ),
        "absolute_change": round(
            absolute_change,
            6,
        ),
        "relative_change_percent": round(
            relative_change_percent,
            2,
        ),
    }


def print_positive_result(
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"CATEGORY: {result['category']}")
    print(f"QUERY: {result['query']}")
    print("-" * 90)
    print(f"ACTION: {result['action']}")
    print(
        f"Hit Rate@5: "
        f"{result['hit_rate_at_5']:.4f}"
    )
    print(
        f"Precision@5: "
        f"{result['precision_at_5']:.4f}"
    )
    print(
        f"Reciprocal Rank: "
        f"{result['reciprocal_rank']:.4f}"
    )
    print(
        f"First relevant rank: "
        f"{result['first_relevant_rank']}"
    )
    print(
        f"Latency: "
        f"{result['latency_ms']:.2f} ms"
    )

    for ranked_result in result["results"]:
        marker = (
            "RELEVANT"
            if ranked_result["relevant"]
            else "not relevant"
        )

        print(
            f"\nRank {ranked_result['rank']} "
            f"[{marker}]"
            f"\nPR: {ranked_result['pr_number']}"
            f"\nSection: {ranked_result['section']}"
            f"\nGoverned score: "
            f"{ranked_result['governed_score']:.6f}"
            f"\nMetadata matches: "
            f"{ranked_result['metadata_condition_matches']}"
            f"/"
            f"{ranked_result['metadata_condition_total']}"
        )


def print_negative_result(
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print("CATEGORY: negative_out_of_domain")
    print(f"QUERY: {result['query']}")
    print("-" * 90)
    print(f"ACTION: {result['action']}")
    print(
        "RETRIEVAL EXECUTED: "
        f"{result['retrieval_executed']}"
    )
    print(
        "CORRECTLY ABSTAINED: "
        f"{result['correctly_abstained']}"
    )
    print(
        f"MESSAGE: {result['message']}"
    )
    print(
        f"Latency: "
        f"{result['latency_ms']:.2f} ms"
    )


def main() -> None:
    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Stage 8G governed retrieval evaluator..."
    )

    governed_retriever = GovernedRetriever(
        candidate_pool_size=200,
        minimum_governed_score=0.20,
    )

    hybrid_retriever = (
        governed_retriever.hybrid_retriever
    )

    positive_cases = (
        build_noisy_identifier_cases(
            hybrid_retriever
        )
        + build_paraphrased_section_cases()
        + build_metadata_condition_cases()
    )

    negative_cases = build_negative_cases()

    print(
        f"Positive evaluation cases: "
        f"{len(positive_cases)}"
    )
    print(
        f"Negative evaluation cases: "
        f"{len(negative_cases)}"
    )
    print(
        f"Total evaluation cases: "
        f"{len(positive_cases) + len(negative_cases)}"
    )

    positive_results: list[dict[str, Any]] = []

    for evaluation_case in positive_cases:
        result = evaluate_positive_case(
            retriever=governed_retriever,
            evaluation_case=evaluation_case,
        )

        positive_results.append(result)
        print_positive_result(result)

    negative_results: list[dict[str, Any]] = []

    for evaluation_case in negative_cases:
        result = evaluate_negative_case(
            retriever=governed_retriever,
            evaluation_case=evaluation_case,
        )

        negative_results.append(result)
        print_negative_result(result)

    positive_hit_rate = statistics.mean(
        result["hit_rate_at_5"]
        for result in positive_results
    )

    positive_precision = statistics.mean(
        result["precision_at_5"]
        for result in positive_results
    )

    positive_mrr = statistics.mean(
        result["reciprocal_rank"]
        for result in positive_results
    )

    positive_mean_latency = statistics.mean(
        result["latency_ms"]
        for result in positive_results
    )

    negative_abstention_rate = statistics.mean(
        float(result["correctly_abstained"])
        for result in negative_results
    )

    categories = [
        "noisy_identifier",
        "paraphrased_section",
        "metadata_condition",
    ]

    category_summaries = {
        category: category_summary(
            positive_results=positive_results,
            category=category,
        )
        for category in categories
    }

    baseline_report = load_baseline_report()

    baseline_positive_metrics = (
        baseline_report[
            "overall_positive_metrics"
        ]
    )

    baseline_negative_metrics = (
        baseline_report[
            "negative_case_metrics"
        ]
    )

    baseline_category_summaries = (
        baseline_report[
            "category_summaries"
        ]
    )

    comparisons = {
        "positive_hit_rate_at_5": metric_change(
            current_value=positive_hit_rate,
            baseline_value=(
                baseline_positive_metrics[
                    "hit_rate_at_5"
                ]
            ),
        ),
        "positive_precision_at_5": metric_change(
            current_value=positive_precision,
            baseline_value=(
                baseline_positive_metrics[
                    "precision_at_5"
                ]
            ),
        ),
        "positive_mean_reciprocal_rank": metric_change(
            current_value=positive_mrr,
            baseline_value=(
                baseline_positive_metrics[
                    "mean_reciprocal_rank"
                ]
            ),
        ),
        "negative_abstention_rate": metric_change(
            current_value=negative_abstention_rate,
            baseline_value=(
                baseline_negative_metrics[
                    "correct_abstention_rate"
                ]
            ),
        ),
    }

    category_comparisons: dict[
        str,
        dict[str, Any]
    ] = {}

    for category in categories:
        category_comparisons[category] = {
            "hit_rate_at_5": metric_change(
                current_value=(
                    category_summaries[
                        category
                    ]["hit_rate_at_5"]
                ),
                baseline_value=(
                    baseline_category_summaries[
                        category
                    ]["hit_rate_at_5"]
                ),
            ),
            "precision_at_5": metric_change(
                current_value=(
                    category_summaries[
                        category
                    ]["precision_at_5"]
                ),
                baseline_value=(
                    baseline_category_summaries[
                        category
                    ]["precision_at_5"]
                ),
            ),
            "mean_reciprocal_rank": metric_change(
                current_value=(
                    category_summaries[
                        category
                    ]["mean_reciprocal_rank"]
                ),
                baseline_value=(
                    baseline_category_summaries[
                        category
                    ]["mean_reciprocal_rank"]
                ),
            ),
        }

    thresholds = {
        "minimum_positive_hit_rate_at_5": 0.80,
        "minimum_positive_precision_at_5": 0.70,
        "minimum_positive_mrr": 0.75,
        "minimum_negative_abstention_rate": 1.00,
        "minimum_noisy_identifier_hit_rate": 1.00,
        "minimum_paraphrased_section_hit_rate": 0.70,
        "minimum_metadata_condition_hit_rate": 0.80,
    }

    threshold_checks = {
        "positive_hit_rate_passed": (
            positive_hit_rate
            >= thresholds[
                "minimum_positive_hit_rate_at_5"
            ]
        ),
        "positive_precision_passed": (
            positive_precision
            >= thresholds[
                "minimum_positive_precision_at_5"
            ]
        ),
        "positive_mrr_passed": (
            positive_mrr
            >= thresholds[
                "minimum_positive_mrr"
            ]
        ),
        "negative_abstention_passed": (
            negative_abstention_rate
            >= thresholds[
                "minimum_negative_abstention_rate"
            ]
        ),
        "noisy_identifier_passed": (
            category_summaries[
                "noisy_identifier"
            ]["hit_rate_at_5"]
            >= thresholds[
                "minimum_noisy_identifier_hit_rate"
            ]
        ),
        "paraphrased_section_passed": (
            category_summaries[
                "paraphrased_section"
            ]["hit_rate_at_5"]
            >= thresholds[
                "minimum_paraphrased_section_hit_rate"
            ]
        ),
        "metadata_condition_passed": (
            category_summaries[
                "metadata_condition"
            ]["hit_rate_at_5"]
            >= thresholds[
                "minimum_metadata_condition_hit_rate"
            ]
        ),
    }

    overall_status = (
        "passed"
        if all(threshold_checks.values())
        else "needs_improvement"
    )

    detailed_report = {
        "stage": "8G",
        "stage_name": (
            "Governed Retrieval Before-and-After Evaluation"
        ),
        "status": overall_status,
        "positive_case_count": len(
            positive_results
        ),
        "negative_case_count": len(
            negative_results
        ),
        "current_metrics": {
            "positive_hit_rate_at_5": round(
                positive_hit_rate,
                6,
            ),
            "positive_precision_at_5": round(
                positive_precision,
                6,
            ),
            "positive_mean_reciprocal_rank": round(
                positive_mrr,
                6,
            ),
            "negative_abstention_rate": round(
                negative_abstention_rate,
                6,
            ),
            "positive_mean_latency_ms": round(
                positive_mean_latency,
                3,
            ),
        },
        "category_summaries": (
            category_summaries
        ),
        "baseline_comparisons": comparisons,
        "category_comparisons": (
            category_comparisons
        ),
        "thresholds": thresholds,
        "threshold_checks": threshold_checks,
        "positive_results": positive_results,
        "negative_results": negative_results,
    }

    with DETAILED_REPORT_PATH.open(
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
        "stage": "8G",
        "stage_name": (
            "Governed Retrieval Before-and-After Evaluation"
        ),
        "status": overall_status,
        "positive_case_count": len(
            positive_results
        ),
        "negative_case_count": len(
            negative_results
        ),
        "positive_hit_rate_at_5": round(
            positive_hit_rate,
            6,
        ),
        "positive_precision_at_5": round(
            positive_precision,
            6,
        ),
        "positive_mean_reciprocal_rank": round(
            positive_mrr,
            6,
        ),
        "negative_abstention_rate": round(
            negative_abstention_rate,
            6,
        ),
        "positive_mean_latency_ms": round(
            positive_mean_latency,
            3,
        ),
        "category_summaries": (
            category_summaries
        ),
        "baseline_comparisons": comparisons,
        "threshold_checks": threshold_checks,
        "detailed_report_path": str(
            DETAILED_REPORT_PATH
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
    print("STAGE 8G SUMMARY")
    print("=" * 90)
    print(f"Status: {overall_status.upper()}")
    print(
        f"Positive cases: "
        f"{len(positive_results)}"
    )
    print(
        f"Negative cases: "
        f"{len(negative_results)}"
    )
    print(
        f"Positive Hit Rate@5: "
        f"{positive_hit_rate:.4f}"
    )
    print(
        f"Positive Precision@5: "
        f"{positive_precision:.4f}"
    )
    print(
        f"Positive MRR: "
        f"{positive_mrr:.4f}"
    )
    print(
        "Correct negative-query abstention rate: "
        f"{negative_abstention_rate:.4f}"
    )
    print(
        f"Mean positive-query latency: "
        f"{positive_mean_latency:.2f} ms"
    )

    print("\nCATEGORY RESULTS")

    for category, summary in (
        category_summaries.items()
    ):
        print(
            f"{category}: "
            f"Hit Rate@5="
            f"{summary['hit_rate_at_5']:.4f}, "
            f"Precision@5="
            f"{summary['precision_at_5']:.4f}, "
            f"MRR="
            f"{summary['mean_reciprocal_rank']:.4f}"
        )

    print("\nBEFORE-AND-AFTER CHANGES")

    for metric_name, comparison in (
        comparisons.items()
    ):
        print(
            f"{metric_name}: "
            f"{comparison['baseline']:.4f} "
            f"-> "
            f"{comparison['current']:.4f} "
            f"(change "
            f"{comparison['absolute_change']:+.4f})"
        )

    print(
        f"\nDetailed report: "
        f"{DETAILED_REPORT_PATH}"
    )
    print(
        f"Completion report: "
        f"{COMPLETION_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()