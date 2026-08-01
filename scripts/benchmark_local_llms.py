from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

from src.rag.deterministic_citation_repair import (
    DeterministicProductionResponse,
    DeterministicRepairProductionGenerator,
)
from src.rag.grounded_generation import (
    GroundedResponseGenerator,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_11a_multi_llm_benchmark.json"
)


MODELS = [
    "qwen2.5-coder:3b",
    "qwen3:4b",
    "phi4-mini",
    "gemma3:4b",
]


TEST_CASES = [
    {
        "case_id": "exact_pr_summary",
        "category": "exact_pr",
        "query": (
            "What do we know about pull request 5336?"
        ),
    },
    {
        "case_id": "low_merge_probability",
        "category": "prediction",
        "query": (
            "Which pull requests have a low "
            "chance of being merged?"
        ),
    },
    {
        "case_id": "human_governance_review",
        "category": "policy",
        "query": (
            "Which changes require a human "
            "governance review?"
        ),
    },
    {
        "case_id": "review_priority",
        "category": "priority",
        "query": (
            "Which changes should maintainers "
            "review first?"
        ),
    },
]


SAFE_ACTIONS = {
    "answer",
    "abstain_citation_validation",
    "abstain_groundedness_validation",
    "abstain_no_evidence",
}


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    model: str
    status: str
    case_count: int
    completed_case_count: int
    exception_count: int
    safe_case_count: int
    safe_pipeline_rate: float
    released_answer_count: int
    withheld_answer_count: int
    answer_release_rate: float
    citation_block_count: int
    groundedness_block_count: int
    repair_attempt_count: int
    repair_success_count: int
    repair_success_rate: float
    mean_generation_latency_ms: float
    median_generation_latency_ms: float
    mean_total_latency_ms: float
    median_total_latency_ms: float
    mean_output_tokens: float
    score: float
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "case_count": self.case_count,
            "completed_case_count": (
                self.completed_case_count
            ),
            "exception_count": self.exception_count,
            "safe_case_count": self.safe_case_count,
            "safe_pipeline_rate": (
                self.safe_pipeline_rate
            ),
            "released_answer_count": (
                self.released_answer_count
            ),
            "withheld_answer_count": (
                self.withheld_answer_count
            ),
            "answer_release_rate": (
                self.answer_release_rate
            ),
            "citation_block_count": (
                self.citation_block_count
            ),
            "groundedness_block_count": (
                self.groundedness_block_count
            ),
            "repair_attempt_count": (
                self.repair_attempt_count
            ),
            "repair_success_count": (
                self.repair_success_count
            ),
            "repair_success_rate": (
                self.repair_success_rate
            ),
            "mean_generation_latency_ms": (
                self.mean_generation_latency_ms
            ),
            "median_generation_latency_ms": (
                self.median_generation_latency_ms
            ),
            "mean_total_latency_ms": (
                self.mean_total_latency_ms
            ),
            "median_total_latency_ms": (
                self.median_total_latency_ms
            ),
            "mean_output_tokens": (
                self.mean_output_tokens
            ),
            "score": self.score,
            "results": self.results,
        }


def calculate_rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def released_answer_is_safe(
    response: DeterministicProductionResponse,
) -> bool:
    final_response = response.final_response

    return bool(
        response.action == "answer"
        and response.answer_released
        and final_response
        .sentence_validation
        .passed
        and final_response
        .claim_validation
        .passed
        and final_response
        .sentence_validation
        .citation_coverage
        == 1.0
        and final_response
        .claim_validation
        .groundedness_rate
        == 1.0
        and final_response
        .claim_validation
        .unsupported_claim_count
        == 0
    )


def withheld_answer_is_safe(
    response: DeterministicProductionResponse,
) -> bool:
    return bool(
        not response.answer_released
        and response.action
        in {
            "abstain_citation_validation",
            "abstain_groundedness_validation",
            "abstain_no_evidence",
        }
    )


def determine_safety(
    response: DeterministicProductionResponse,
) -> tuple[bool, str]:
    if released_answer_is_safe(response):
        if response.repair_succeeded:
            return (
                True,
                "safe_repaired_answer_released",
            )

        return (
            True,
            "safe_original_answer_released",
        )

    if withheld_answer_is_safe(response):
        if response.action == (
            "abstain_citation_validation"
        ):
            return (
                True,
                "citation_failure_safely_withheld",
            )

        if response.action == (
            "abstain_groundedness_validation"
        ):
            return (
                True,
                "groundedness_failure_safely_withheld",
            )

        return (
            True,
            "no_evidence_safely_withheld",
        )

    return False, "unsafe_pipeline_outcome"


def evaluate_case(
    generator: DeterministicRepairProductionGenerator,
    model: str,
    test_case: dict[str, str],
) -> dict[str, Any]:
    start_time = time.perf_counter()

    response = generator.generate(
        query=test_case["query"]
    )

    total_latency_ms = round(
        (
            time.perf_counter()
            - start_time
        )
        * 1000,
        3,
    )

    safety_passed, safety_outcome = (
        determine_safety(response)
    )

    initial_base_response = (
        response
        .initial_response
        .base_response
    )

    generation_latency_ms = (
        initial_base_response
        .generation_latency_ms
        if initial_base_response
        .generation_executed
        else 0.0
    )

    prompt_tokens = (
        initial_base_response
        .prompt_eval_count
    )

    output_tokens = (
        initial_base_response
        .eval_count
    )

    final_response = (
        response.final_response
    )

    checks: dict[str, bool] = {
        "safe_pipeline_outcome": (
            safety_passed
        ),
        "valid_action": (
            response.action in SAFE_ACTIONS
        ),
    }

    if response.answer_released:
        checks[
            "citation_validation"
        ] = (
            final_response
            .sentence_validation
            .passed
        )

        checks[
            "claim_validation"
        ] = (
            final_response
            .claim_validation
            .passed
        )

        checks[
            "citation_coverage"
        ] = (
            final_response
            .sentence_validation
            .citation_coverage
            == 1.0
        )

        checks[
            "groundedness_rate"
        ] = (
            final_response
            .claim_validation
            .groundedness_rate
            == 1.0
        )

        checks[
            "unsupported_claims"
        ] = (
            final_response
            .claim_validation
            .unsupported_claim_count
            == 0
        )

    else:
        checks["unsafe_answer_hidden"] = (
            response.action
            in {
                "abstain_citation_validation",
                "abstain_groundedness_validation",
                "abstain_no_evidence",
            }
        )

    return {
        "model": model,
        "case_id": test_case["case_id"],
        "category": test_case["category"],
        "query": test_case["query"],
        "completed": True,
        "passed": all(
            checks.values()
        ),
        "safety_outcome": safety_outcome,
        "checks": checks,
        "initial_action": (
            response
            .initial_response
            .action
        ),
        "final_action": response.action,
        "generation_executed": (
            response.generation_executed
        ),
        "answer_released": (
            response.answer_released
        ),
        "repair_attempted": (
            response.repair_attempted
        ),
        "repair_succeeded": (
            response.repair_succeeded
        ),
        "citation_validation_passed": (
            final_response
            .sentence_validation
            .passed
        ),
        "citation_coverage": (
            final_response
            .sentence_validation
            .citation_coverage
        ),
        "claim_validation_passed": (
            final_response
            .claim_validation
            .passed
        ),
        "groundedness_rate": (
            final_response
            .claim_validation
            .groundedness_rate
        ),
        "unsupported_claim_count": (
            final_response
            .claim_validation
            .unsupported_claim_count
        ),
        "generation_latency_ms": (
            generation_latency_ms
        ),
        "repair_latency_ms": (
            response
            .repair_result
            .latency_ms
        ),
        "total_latency_ms": (
            total_latency_ms
        ),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "visible_answer": response.answer,
        "original_model_answer": (
            initial_base_response.answer
        ),
        "response": response.to_dict(),
    }


def evaluate_model(
    model: str,
) -> ModelBenchmarkSummary:
    print("\n" + "#" * 100)
    print(f"BENCHMARKING MODEL: {model}")
    print("#" * 100)

    base_generator = GroundedResponseGenerator(
        model=model,
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
        request_timeout_seconds=300,
    )

    generator = (
        DeterministicRepairProductionGenerator(
            base_generator=base_generator,
            minimum_support_score=0.60,
            minimum_token_coverage=0.45,
        )
    )

    results: list[dict[str, Any]] = []

    for case_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"MODEL: {model}"
        )
        print(
            f"CASE {case_number}/{len(TEST_CASES)}: "
            f"{test_case['case_id']}"
        )
        print(
            f"QUERY: {test_case['query']}"
        )
        print("-" * 100)

        try:
            result = evaluate_case(
                generator=generator,
                model=model,
                test_case=test_case,
            )

        except Exception as error:
            result = {
                "model": model,
                "case_id": (
                    test_case["case_id"]
                ),
                "category": (
                    test_case["category"]
                ),
                "query": (
                    test_case["query"]
                ),
                "completed": False,
                "passed": False,
                "safety_outcome": (
                    "pipeline_exception"
                ),
                "checks": {
                    "execution": False,
                },
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(error),
            }

        results.append(result)

        if result["completed"]:
            print(
                "STATUS: "
                f"{'PASSED' if result['passed'] else 'FAILED'}"
            )
            print(
                "SAFETY OUTCOME: "
                f"{result['safety_outcome']}"
            )
            print(
                "INITIAL ACTION: "
                f"{result['initial_action']}"
            )
            print(
                "FINAL ACTION: "
                f"{result['final_action']}"
            )
            print(
                "ANSWER RELEASED: "
                f"{result['answer_released']}"
            )
            print(
                "REPAIR ATTEMPTED: "
                f"{result['repair_attempted']}"
            )
            print(
                "REPAIR SUCCEEDED: "
                f"{result['repair_succeeded']}"
            )
            print(
                "CITATION COVERAGE: "
                f"{result['citation_coverage']:.4f}"
            )
            print(
                "GROUNDEDNESS RATE: "
                f"{result['groundedness_rate']:.4f}"
            )
            print(
                "UNSUPPORTED CLAIMS: "
                f"{result['unsupported_claim_count']}"
            )
            print(
                "GENERATION LATENCY: "
                f"{result['generation_latency_ms']:.2f} ms"
            )
            print(
                "TOTAL LATENCY: "
                f"{result['total_latency_ms']:.2f} ms"
            )
            print(
                "OUTPUT TOKENS: "
                f"{result['output_tokens']}"
            )

            print("\nVISIBLE ANSWER:")
            print(result["visible_answer"])

            if (
                result["visible_answer"]
                != result[
                    "original_model_answer"
                ]
            ):
                print(
                    "\nORIGINAL MODEL ANSWER:"
                )
                print(
                    result[
                        "original_model_answer"
                    ]
                )

        else:
            print("STATUS: FAILED")
            print(
                "ERROR TYPE: "
                f"{result['error_type']}"
            )
            print(
                "ERROR: "
                f"{result['error_message']}"
            )

    completed_results = [
        result
        for result in results
        if result["completed"]
    ]

    exception_results = [
        result
        for result in results
        if not result["completed"]
    ]

    safe_results = [
        result
        for result in completed_results
        if result["passed"]
    ]

    released_results = [
        result
        for result in completed_results
        if result["answer_released"]
    ]

    withheld_results = [
        result
        for result in completed_results
        if not result["answer_released"]
    ]

    citation_blocks = [
        result
        for result in completed_results
        if result["final_action"]
        == "abstain_citation_validation"
    ]

    groundedness_blocks = [
        result
        for result in completed_results
        if result["final_action"]
        == "abstain_groundedness_validation"
    ]

    repair_attempts = [
        result
        for result in completed_results
        if result["repair_attempted"]
    ]

    repair_successes = [
        result
        for result in repair_attempts
        if result["repair_succeeded"]
    ]

    generation_latencies = [
        result["generation_latency_ms"]
        for result in completed_results
    ]

    total_latencies = [
        result["total_latency_ms"]
        for result in completed_results
    ]

    output_token_values = [
        result["output_tokens"]
        for result in completed_results
        if result["output_tokens"]
        is not None
    ]

    safe_pipeline_rate = calculate_rate(
        numerator=len(safe_results),
        denominator=len(TEST_CASES),
    )

    answer_release_rate = calculate_rate(
        numerator=len(released_results),
        denominator=len(
            completed_results
        ),
    )

    repair_success_rate = calculate_rate(
        numerator=len(repair_successes),
        denominator=len(repair_attempts),
    )

    mean_generation_latency_ms = (
        statistics.mean(
            generation_latencies
        )
        if generation_latencies
        else 0.0
    )

    median_generation_latency_ms = (
        statistics.median(
            generation_latencies
        )
        if generation_latencies
        else 0.0
    )

    mean_total_latency_ms = (
        statistics.mean(
            total_latencies
        )
        if total_latencies
        else 0.0
    )

    median_total_latency_ms = (
        statistics.median(
            total_latencies
        )
        if total_latencies
        else 0.0
    )

    mean_output_tokens = (
        statistics.mean(
            output_token_values
        )
        if output_token_values
        else 0.0
    )

    latency_penalty = min(
        mean_generation_latency_ms
        / 300000.0,
        1.0,
    )

    score = (
        safe_pipeline_rate * 50.0
        + answer_release_rate * 40.0
        + repair_success_rate * 5.0
        + (1.0 - latency_penalty) * 5.0
    )

    status = (
        "passed"
        if len(exception_results) == 0
        else "completed_with_errors"
    )

    summary = ModelBenchmarkSummary(
        model=model,
        status=status,
        case_count=len(TEST_CASES),
        completed_case_count=len(
            completed_results
        ),
        exception_count=len(
            exception_results
        ),
        safe_case_count=len(
            safe_results
        ),
        safe_pipeline_rate=round(
            safe_pipeline_rate,
            6,
        ),
        released_answer_count=len(
            released_results
        ),
        withheld_answer_count=len(
            withheld_results
        ),
        answer_release_rate=round(
            answer_release_rate,
            6,
        ),
        citation_block_count=len(
            citation_blocks
        ),
        groundedness_block_count=len(
            groundedness_blocks
        ),
        repair_attempt_count=len(
            repair_attempts
        ),
        repair_success_count=len(
            repair_successes
        ),
        repair_success_rate=round(
            repair_success_rate,
            6,
        ),
        mean_generation_latency_ms=round(
            mean_generation_latency_ms,
            3,
        ),
        median_generation_latency_ms=round(
            median_generation_latency_ms,
            3,
        ),
        mean_total_latency_ms=round(
            mean_total_latency_ms,
            3,
        ),
        median_total_latency_ms=round(
            median_total_latency_ms,
            3,
        ),
        mean_output_tokens=round(
            mean_output_tokens,
            3,
        ),
        score=round(
            score,
            6,
        ),
        results=results,
    )

    print("\n" + "-" * 100)
    print(f"MODEL SUMMARY: {model}")
    print("-" * 100)
    print(
        f"Safe pipeline rate: "
        f"{summary.safe_pipeline_rate:.4f}"
    )
    print(
        f"Answer release rate: "
        f"{summary.answer_release_rate:.4f}"
    )
    print(
        f"Released answers: "
        f"{summary.released_answer_count}"
    )
    print(
        f"Withheld answers: "
        f"{summary.withheld_answer_count}"
    )
    print(
        f"Repair success rate: "
        f"{summary.repair_success_rate:.4f}"
    )
    print(
        f"Mean generation latency: "
        f"{summary.mean_generation_latency_ms:.2f} ms"
    )
    print(
        f"Exceptions: "
        f"{summary.exception_count}"
    )
    print(
        f"Benchmark score: "
        f"{summary.score:.4f}"
    )

    return summary


def build_category_comparison(
    summaries: list[ModelBenchmarkSummary],
) -> list[dict[str, Any]]:
    comparison: list[
        dict[str, Any]
    ] = []

    for test_case in TEST_CASES:
        row: dict[str, Any] = {
            "case_id": test_case["case_id"],
            "category": test_case["category"],
            "query": test_case["query"],
            "models": {},
        }

        for summary in summaries:
            matching_results = [
                result
                for result in summary.results
                if (
                    result["case_id"]
                    == test_case["case_id"]
                )
            ]

            if not matching_results:
                continue

            result = matching_results[0]

            if result["completed"]:
                row["models"][
                    summary.model
                ] = {
                    "completed": True,
                    "passed": result["passed"],
                    "answer_released": (
                        result[
                            "answer_released"
                        ]
                    ),
                    "final_action": (
                        result["final_action"]
                    ),
                    "repair_succeeded": (
                        result[
                            "repair_succeeded"
                        ]
                    ),
                    "generation_latency_ms": (
                        result[
                            "generation_latency_ms"
                        ]
                    ),
                    "visible_answer": (
                        result[
                            "visible_answer"
                        ]
                    ),
                }
            else:
                row["models"][
                    summary.model
                ] = {
                    "completed": False,
                    "passed": False,
                    "error_type": (
                        result["error_type"]
                    ),
                    "error_message": (
                        result["error_message"]
                    ),
                }

        comparison.append(row)

    return comparison


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Starting Stage 11A multi-LLM benchmark."
    )
    print(
        "Models: "
        + ", ".join(MODELS)
    )
    print(
        "Queries per model: "
        f"{len(TEST_CASES)}"
    )
    print(
        "Total possible generation calls: "
        f"{len(MODELS) * len(TEST_CASES)}"
    )

    summaries: list[
        ModelBenchmarkSummary
    ] = []

    for model in MODELS:
        summary = evaluate_model(model)
        summaries.append(summary)

    ranked_summaries = sorted(
        summaries,
        key=lambda item: (
            item.score,
            item.safe_pipeline_rate,
            item.answer_release_rate,
            -item.mean_generation_latency_ms,
        ),
        reverse=True,
    )

    ranking = [
        {
            "rank": rank,
            "model": summary.model,
            "score": summary.score,
            "safe_pipeline_rate": (
                summary.safe_pipeline_rate
            ),
            "answer_release_rate": (
                summary.answer_release_rate
            ),
            "repair_success_rate": (
                summary.repair_success_rate
            ),
            "mean_generation_latency_ms": (
                summary
                .mean_generation_latency_ms
            ),
            "exception_count": (
                summary.exception_count
            ),
        }
        for rank, summary in enumerate(
            ranked_summaries,
            start=1,
        )
    ]

    winner = (
        ranked_summaries[0]
        if ranked_summaries
        else None
    )

    category_comparison = (
        build_category_comparison(
            summaries
        )
    )

    overall_status = (
        "passed"
        if all(
            summary.exception_count == 0
            for summary in summaries
        )
        else "completed_with_errors"
    )

    report = {
        "stage": "11A",
        "stage_name": (
            "Multi-LLM Governed RAG Benchmark"
        ),
        "status": overall_status,
        "benchmark_method": {
            "models": MODELS,
            "query_count_per_model": len(
                TEST_CASES
            ),
            "deterministic_repair_enabled": True,
            "temperature": 0.0,
            "evidence_top_k": 5,
            "maximum_evidence_characters": 2400,
            "minimum_support_score": 0.60,
            "minimum_token_coverage": 0.45,
            "score_weights": {
                "safe_pipeline_rate": 50,
                "answer_release_rate": 40,
                "repair_success_rate": 5,
                "latency": 5,
            },
        },
        "winner": (
            {
                "model": winner.model,
                "score": winner.score,
                "safe_pipeline_rate": (
                    winner.safe_pipeline_rate
                ),
                "answer_release_rate": (
                    winner.answer_release_rate
                ),
                "mean_generation_latency_ms": (
                    winner
                    .mean_generation_latency_ms
                ),
            }
            if winner is not None
            else None
        ),
        "ranking": ranking,
        "model_summaries": [
            summary.to_dict()
            for summary in summaries
        ],
        "category_comparison": (
            category_comparison
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print(
        "STAGE 11A MULTI-LLM BENCHMARK SUMMARY"
    )
    print("=" * 100)
    print(
        f"Status: {overall_status.upper()}"
    )

    for ranking_row in ranking:
        print(
            f"{ranking_row['rank']}. "
            f"{ranking_row['model']}"
        )
        print(
            f"   Score: "
            f"{ranking_row['score']:.4f}"
        )
        print(
            f"   Safe pipeline rate: "
            f"{ranking_row['safe_pipeline_rate']:.4f}"
        )
        print(
            f"   Answer release rate: "
            f"{ranking_row['answer_release_rate']:.4f}"
        )
        print(
            f"   Repair success rate: "
            f"{ranking_row['repair_success_rate']:.4f}"
        )
        print(
            f"   Mean generation latency: "
            f"{ranking_row['mean_generation_latency_ms']:.2f} ms"
        )
        print(
            f"   Exceptions: "
            f"{ranking_row['exception_count']}"
        )

    if winner is not None:
        print("\nSELECTED WINNER:")
        print(winner.model)
        print(
            f"Winner score: "
            f"{winner.score:.4f}"
        )

    print(
        f"\nReport: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()