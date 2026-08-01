from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.rag.hybrid_retrieval import (
    HybridRetriever,
    RetrievalResult,
)
from src.utils.paths import PROJECT_ROOT


REPORTS_DIRECTORY = PROJECT_ROOT / "data" / "reports"

DETAILED_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8e_realistic_retrieval_results.json"
)

COMPLETION_REPORT_PATH = (
    REPORTS_DIRECTORY
    / "stage_8e_completion_report.json"
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    query: str
    relevance_description: str
    relevance_function: Callable[[RetrievalResult], bool]
    top_k: int = 5
    is_negative_case: bool = False


def normalise_text(value: Any) -> str:
    return str(value or "").strip().lower()


def metadata_value(
    result: RetrievalResult,
    key: str,
    default: Any = None,
) -> Any:
    return result.metadata.get(key, default)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def reciprocal_rank(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
) -> float:
    for rank, result in enumerate(results, start=1):
        if relevance_function(result):
            return 1.0 / rank

    return 0.0


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


def precision_at_k(
    results: list[RetrievalResult],
    relevance_function: Callable[[RetrievalResult], bool],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    selected_results = results[:k]

    relevant_count = sum(
        1
        for result in selected_results
        if relevance_function(result)
    )

    return relevant_count / k


def build_noisy_identifier_cases(
    retriever: HybridRetriever,
) -> list[EvaluationCase]:
    available_pr_numbers = sorted(
        retriever.pr_number_to_indices.keys()
    )

    if len(available_pr_numbers) < 5:
        return []

    selected_positions = [
        0,
        len(available_pr_numbers) // 4,
        len(available_pr_numbers) // 2,
        (3 * len(available_pr_numbers)) // 4,
        len(available_pr_numbers) - 1,
    ]

    selected_pr_numbers = [
        available_pr_numbers[position]
        for position in selected_positions
    ]

    query_templates = [
        "Can you show me what happened with pull request {pr_number}?",
        "Find the review information for PR#{pr_number}, please.",
        "I need the risk and prediction details for pr {pr_number}.",
        "Open the evidence associated with pull request number {pr_number}.",
        "What do we know about #{pr_number} in the Flask repository?",
    ]

    cases: list[EvaluationCase] = []

    for index, pr_number in enumerate(
        selected_pr_numbers
    ):
        query = query_templates[index].format(
            pr_number=pr_number
        )

        cases.append(
            EvaluationCase(
                case_id=f"noisy_identifier_{pr_number}",
                category="noisy_identifier",
                query=query,
                relevance_description=(
                    f"Any result belonging to PR #{pr_number}"
                ),
                relevance_function=(
                    lambda result, expected=pr_number:
                    result.pr_number == expected
                ),
            )
        )

    return cases


def build_paraphrased_section_cases() -> list[EvaluationCase]:
    specifications = [
        {
            "case_id": "paraphrase_change_evidence_1",
            "query": (
                "Which contributions look unusually large and "
                "could be difficult for a reviewer to inspect?"
            ),
            "section": "Change evidence",
        },
        {
            "case_id": "paraphrase_change_evidence_2",
            "query": (
                "Find code submissions touching lots of files "
                "or containing a heavy volume of edits."
            ),
            "section": "Change evidence",
        },
        {
            "case_id": "paraphrase_policy_1",
            "query": (
                "Which changes should be escalated for a human "
                "governance check?"
            ),
            "section": "Deterministic policy intelligence",
        },
        {
            "case_id": "paraphrase_policy_2",
            "query": (
                "Show changes that violated several repository "
                "controls or review policies."
            ),
            "section": "Deterministic policy intelligence",
        },
        {
            "case_id": "paraphrase_prediction_1",
            "query": (
                "Which submissions appear unlikely to get accepted?"
            ),
            "section": "Predictive intelligence",
        },
        {
            "case_id": "paraphrase_prediction_2",
            "query": (
                "Find work items that may remain open for a long "
                "time before being integrated."
            ),
            "section": "Predictive intelligence",
        },
        {
            "case_id": "paraphrase_priority_1",
            "query": (
                "What should the maintainers look at first?"
            ),
            "section": "Unified review priority",
        },
        {
            "case_id": "paraphrase_priority_2",
            "query": (
                "Show the recommended reviewer action and urgency."
            ),
            "section": "Unified review priority",
        },
        {
            "case_id": "paraphrase_description_1",
            "query": (
                "Find submissions where the author gave a useful "
                "explanation of the proposed change."
            ),
            "section": "PR description",
        },
        {
            "case_id": "paraphrase_identity_1",
            "query": (
                "Who opened the change, when was it opened, and "
                "what was its title?"
            ),
            "section": "PR identity",
        },
    ]

    cases: list[EvaluationCase] = []

    for specification in specifications:
        expected_section = specification["section"]

        cases.append(
            EvaluationCase(
                case_id=specification["case_id"],
                category="paraphrased_section",
                query=specification["query"],
                relevance_description=(
                    f"Section equals '{expected_section}'"
                ),
                relevance_function=(
                    lambda result, expected=expected_section:
                    normalise_text(result.section)
                    == normalise_text(expected)
                ),
            )
        )

    return cases


def build_metadata_condition_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="condition_critical_manual_review",
            category="metadata_condition",
            query=(
                "Show critical pull requests that must be "
                "reviewed manually."
            ),
            relevance_description=(
                "Critical policy risk and manual review required"
            ),
            relevance_function=lambda result: (
                normalise_text(
                    metadata_value(
                        result,
                        "policy_risk_band",
                    )
                )
                == "critical"
                and bool(
                    metadata_value(
                        result,
                        "manual_review_required",
                        False,
                    )
                )
            ),
        ),
        EvaluationCase(
            case_id="condition_high_review_priority",
            category="metadata_condition",
            query=(
                "Find changes with high or critical review urgency."
            ),
            relevance_description=(
                "Review priority is High or Critical"
            ),
            relevance_function=lambda result: (
                normalise_text(
                    metadata_value(
                        result,
                        "review_priority",
                    )
                )
                in {"high", "critical"}
            ),
        ),
        EvaluationCase(
            case_id="condition_low_merge_probability",
            category="metadata_condition",
            query=(
                "Which pull requests have a low chance of being "
                "merged?"
            ),
            relevance_description=(
                "Merge probability is below 40 percent"
            ),
            relevance_function=lambda result: (
                safe_float(
                    metadata_value(
                        result,
                        "merge_probability",
                        1.0,
                    )
                )
                < 0.40
            ),
        ),
        EvaluationCase(
            case_id="condition_delay_risk",
            category="metadata_condition",
            query=(
                "Find merged changes expected to take longer than "
                "normal to complete."
            ),
            relevance_description=(
                "Delay score exists and delay prediction equals 1"
            ),
            relevance_function=lambda result: (
                bool(
                    metadata_value(
                        result,
                        "delay_score_available",
                        False,
                    )
                )
                and safe_int(
                    metadata_value(
                        result,
                        "delay_prediction",
                        0,
                    )
                )
                == 1
            ),
        ),
        EvaluationCase(
            case_id="condition_multiple_rules",
            category="metadata_condition",
            query=(
                "Show changes that triggered several different "
                "repository rules."
            ),
            relevance_description=(
                "Triggered rule count is at least 3"
            ),
            relevance_function=lambda result: (
                safe_int(
                    metadata_value(
                        result,
                        "triggered_rule_count",
                        0,
                    )
                )
                >= 3
            ),
        ),
        EvaluationCase(
            case_id="condition_security_governance",
            category="metadata_condition",
            query=(
                "Find changes associated with security or "
                "governance concerns."
            ),
            relevance_description=(
                "Triggered categories contain Security or Governance"
            ),
            relevance_function=lambda result: (
                bool(
                    {
                        normalise_text(category)
                        for category in metadata_value(
                            result,
                            "triggered_categories",
                            [],
                        )
                    }
                    & {"security", "governance"}
                )
            ),
        ),
    ]


def build_negative_cases() -> list[EvaluationCase]:
    unrelated_queries = [
        (
            "negative_weather",
            "What will the weather be in Dubai tomorrow?",
        ),
        (
            "negative_salary",
            "Calculate the monthly salary after UAE tax.",
        ),
        (
            "negative_recipe",
            "Give me a recipe for chocolate cake.",
        ),
        (
            "negative_flight",
            "Find the cheapest flight from India to London.",
        ),
    ]

    cases: list[EvaluationCase] = []

    for case_id, query in unrelated_queries:
        cases.append(
            EvaluationCase(
                case_id=case_id,
                category="negative_out_of_domain",
                query=query,
                relevance_description=(
                    "No repository chunk should be treated as relevant"
                ),
                relevance_function=lambda result: False,
                is_negative_case=True,
            )
        )

    return cases


def evaluate_positive_case(
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

    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):
        if evaluation_case.relevance_function(result):
            first_relevant_rank = rank
            break

    ranked_results = []

    for rank, result in enumerate(results, start=1):
        ranked_results.append(
            {
                "rank": rank,
                "relevant": (
                    evaluation_case.relevance_function(
                        result
                    )
                ),
                "pr_number": result.pr_number,
                "section": result.section,
                "hybrid_score": result.hybrid_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "exact_match_score": (
                    result.exact_match_score
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
        "is_negative_case": False,
        "result_count": len(results),
        "hit_rate_at_5": round(hit_rate, 6),
        "precision_at_5": round(precision, 6),
        "reciprocal_rank": round(rr, 6),
        "first_relevant_rank": first_relevant_rank,
        "latency_ms": round(latency_ms, 3),
        "results": ranked_results,
    }


def evaluate_negative_case(
    retriever: HybridRetriever,
    evaluation_case: EvaluationCase,
    abstention_threshold: float,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    results = retriever.retrieve(
        query=evaluation_case.query,
        top_k=evaluation_case.top_k,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    top_score = (
        results[0].hybrid_score
        if results
        else 0.0
    )

    abstained = (
        not results
        or top_score < abstention_threshold
    )

    ranked_results = [
        {
            "rank": rank,
            "pr_number": result.pr_number,
            "section": result.section,
            "hybrid_score": result.hybrid_score,
            "chunk_id": result.chunk_id,
            "text_preview": (
                result.text
                .replace("\n", " ")
                .strip()[:220]
            ),
        }
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]

    return {
        "case_id": evaluation_case.case_id,
        "category": evaluation_case.category,
        "query": evaluation_case.query,
        "relevance_description": (
            evaluation_case.relevance_description
        ),
        "is_negative_case": True,
        "result_count": len(results),
        "top_hybrid_score": round(top_score, 6),
        "abstention_threshold": abstention_threshold,
        "abstained": abstained,
        "latency_ms": round(latency_ms, 3),
        "results": ranked_results,
    }


def category_summary(
    results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    matching_results = [
        result
        for result in results
        if result["category"] == category
        and not result["is_negative_case"]
    ]

    if not matching_results:
        return {
            "category": category,
            "case_count": 0,
            "hit_rate_at_5": 0.0,
            "precision_at_5": 0.0,
            "mean_reciprocal_rank": 0.0,
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
    }


def print_positive_result(
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"CATEGORY: {result['category']}")
    print(f"QUERY: {result['query']}")
    print("-" * 90)
    print(
        f"Hit Rate@5: {result['hit_rate_at_5']:.4f}"
    )
    print(
        f"Precision@5: {result['precision_at_5']:.4f}"
    )
    print(
        "Reciprocal Rank: "
        f"{result['reciprocal_rank']:.4f}"
    )
    print(
        f"First relevant rank: "
        f"{result['first_relevant_rank']}"
    )
    print(
        f"Latency: {result['latency_ms']:.2f} ms"
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
            f"\nHybrid score: "
            f"{ranked_result['hybrid_score']:.6f}"
        )


def print_negative_result(
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print("CATEGORY: negative_out_of_domain")
    print(f"QUERY: {result['query']}")
    print("-" * 90)
    print(
        f"Top hybrid score: "
        f"{result['top_hybrid_score']:.6f}"
    )
    print(
        f"Abstention threshold: "
        f"{result['abstention_threshold']:.6f}"
    )
    print(
        f"Abstained correctly: "
        f"{result['abstained']}"
    )
    print(
        f"Latency: {result['latency_ms']:.2f} ms"
    )


def main() -> None:
    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Stage 8E realistic retrieval evaluator..."
    )

    retriever = HybridRetriever(
        semantic_weight=0.60,
        keyword_weight=0.25,
        exact_match_weight=0.15,
    )

    abstention_threshold = 0.45

    positive_cases = (
        build_noisy_identifier_cases(retriever)
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
            retriever=retriever,
            evaluation_case=evaluation_case,
        )

        positive_results.append(result)
        print_positive_result(result)

    negative_results: list[dict[str, Any]] = []

    for evaluation_case in negative_cases:
        result = evaluate_negative_case(
            retriever=retriever,
            evaluation_case=evaluation_case,
            abstention_threshold=abstention_threshold,
        )

        negative_results.append(result)
        print_negative_result(result)

    overall_hit_rate = statistics.mean(
        result["hit_rate_at_5"]
        for result in positive_results
    )

    overall_precision = statistics.mean(
        result["precision_at_5"]
        for result in positive_results
    )

    overall_mrr = statistics.mean(
        result["reciprocal_rank"]
        for result in positive_results
    )

    positive_latency = statistics.mean(
        result["latency_ms"]
        for result in positive_results
    )

    negative_abstention_rate = statistics.mean(
        float(result["abstained"])
        for result in negative_results
    )

    category_summaries = {
        category: category_summary(
            results=positive_results,
            category=category,
        )
        for category in [
            "noisy_identifier",
            "paraphrased_section",
            "metadata_condition",
        ]
    }

    thresholds = {
        "minimum_positive_hit_rate_at_5": 0.70,
        "minimum_positive_mrr": 0.55,
        "minimum_noisy_identifier_hit_rate": 0.80,
        "minimum_paraphrased_section_hit_rate": 0.60,
        "minimum_metadata_condition_hit_rate": 0.50,
    }

    threshold_checks = {
        "positive_hit_rate_passed": (
            overall_hit_rate
            >= thresholds[
                "minimum_positive_hit_rate_at_5"
            ]
        ),
        "positive_mrr_passed": (
            overall_mrr
            >= thresholds[
                "minimum_positive_mrr"
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

    positive_status = (
        "passed"
        if all(threshold_checks.values())
        else "failed"
    )

    overall_status = (
        "passed"
        if positive_status == "passed"
        else "needs_improvement"
    )

    detailed_report = {
        "stage": "8E",
        "stage_name": (
            "Realistic Hybrid Retrieval Evaluation"
        ),
        "status": overall_status,
        "positive_case_count": len(
            positive_results
        ),
        "negative_case_count": len(
            negative_results
        ),
        "abstention_threshold": (
            abstention_threshold
        ),
        "overall_positive_metrics": {
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
                positive_latency,
                3,
            ),
        },
        "negative_case_metrics": {
            "correct_abstention_rate": round(
                negative_abstention_rate,
                6,
            ),
        },
        "category_summaries": (
            category_summaries
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
        "stage": "8E",
        "stage_name": (
            "Realistic Hybrid Retrieval Evaluation"
        ),
        "status": overall_status,
        "positive_case_count": len(
            positive_results
        ),
        "negative_case_count": len(
            negative_results
        ),
        "positive_hit_rate_at_5": round(
            overall_hit_rate,
            6,
        ),
        "positive_precision_at_5": round(
            overall_precision,
            6,
        ),
        "positive_mean_reciprocal_rank": round(
            overall_mrr,
            6,
        ),
        "negative_correct_abstention_rate": round(
            negative_abstention_rate,
            6,
        ),
        "mean_latency_ms": round(
            positive_latency,
            3,
        ),
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
    print("STAGE 8E SUMMARY")
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
        f"{overall_hit_rate:.4f}"
    )
    print(
        f"Positive Precision@5: "
        f"{overall_precision:.4f}"
    )
    print(
        f"Positive MRR: "
        f"{overall_mrr:.4f}"
    )
    print(
        "Correct negative-query abstention rate: "
        f"{negative_abstention_rate:.4f}"
    )
    print(
        f"Mean latency: "
        f"{positive_latency:.2f} ms"
    )

    for category, summary in (
        category_summaries.items()
    ):
        print(
            f"{category} Hit Rate@5: "
            f"{summary['hit_rate_at_5']:.4f}"
        )

    print(
        f"Detailed report: "
        f"{DETAILED_REPORT_PATH}"
    )
    print(
        f"Completion report: "
        f"{COMPLETION_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()