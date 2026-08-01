from __future__ import annotations

import json
import statistics
import time
from typing import Any

from src.rag.grounded_generation import (
    GroundedResponseGenerator,
)
from src.rag.retrying_production_generation import (
    RetryingProductionGroundedGenerator,
    RetryingProductionResponse,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_10d_real_retrying_production_generation.json"
)


TEST_CASES = [
    {
        "case_id": "real_exact_pr_with_retry",
        "category": "exact_pr",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "expected_generation": True,
        "expected_out_of_domain": False,
        "repair_expected": True,
    },
    {
        "case_id": "real_low_merge_probability",
        "category": "prediction",
        "query": (
            "Which pull requests have a low "
            "chance of being merged?"
        ),
        "expected_generation": True,
        "expected_out_of_domain": False,
        "repair_expected": False,
    },
    {
        "case_id": "real_human_governance_review",
        "category": "policy",
        "query": (
            "Which changes require a human "
            "governance review?"
        ),
        "expected_generation": True,
        "expected_out_of_domain": False,
        "repair_expected": False,
    },
    {
        "case_id": "real_review_priority",
        "category": "priority",
        "query": (
            "Which changes should maintainers "
            "review first?"
        ),
        "expected_generation": True,
        "expected_out_of_domain": False,
        "repair_expected": False,
    },
    {
        "case_id": "real_out_of_domain",
        "category": "out_of_domain",
        "query": (
            "What will the weather be in "
            "Dubai tomorrow?"
        ),
        "expected_generation": False,
        "expected_out_of_domain": True,
        "repair_expected": False,
    },
]


SAFE_FINAL_ACTIONS = {
    "answer",
    "abstain_citation_validation",
    "abstain_groundedness_validation",
    "abstain_out_of_domain",
    "abstain_no_evidence",
}


def released_answer_is_safe(
    response: RetryingProductionResponse,
) -> bool:
    final_response = response.final_response

    return bool(
        response.action == "answer"
        and response.answer_released
        and final_response.sentence_validation.passed
        and final_response.claim_validation.passed
        and (
            final_response
            .sentence_validation
            .citation_coverage
            == 1.0
        )
        and (
            final_response
            .claim_validation
            .groundedness_rate
            == 1.0
        )
        and (
            final_response
            .claim_validation
            .unsupported_claim_count
            == 0
        )
    )


def withheld_answer_is_safe(
    response: RetryingProductionResponse,
) -> bool:
    return bool(
        not response.answer_released
        and response.action
        in {
            "abstain_citation_validation",
            "abstain_groundedness_validation",
        }
    )


def out_of_domain_response_is_safe(
    response: RetryingProductionResponse,
) -> bool:
    final_response = response.final_response

    return bool(
        response.action
        == "abstain_out_of_domain"
        and response.answer_released
        and not response.generation_executed
        and not response.repair_attempted
        and final_response.model is None
        and final_response.evidence_count == 0
    )


def determine_safety_outcome(
    response: RetryingProductionResponse,
    expected_out_of_domain: bool,
) -> tuple[bool, str]:
    if expected_out_of_domain:
        safe = out_of_domain_response_is_safe(
            response
        )

        return (
            safe,
            (
                "safe_out_of_domain_abstention"
                if safe
                else "unsafe_out_of_domain_handling"
            ),
        )

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
        if response.repair_attempted:
            return (
                True,
                "failed_repair_safely_withheld",
            )

        return (
            True,
            "unsafe_answer_safely_withheld",
        )

    return (
        False,
        "unsafe_pipeline_outcome",
    )


def evaluate_case(
    generator: RetryingProductionGroundedGenerator,
    test_case: dict[str, Any],
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

    (
        safety_passed,
        safety_outcome,
    ) = determine_safety_outcome(
        response=response,
        expected_out_of_domain=(
            test_case["expected_out_of_domain"]
        ),
    )

    initial_response = (
        response.initial_response
    )

    final_response = (
        response.final_response
    )

    checks: dict[str, bool] = {
        "safety_passed": safety_passed,
        "valid_final_action": (
            response.action
            in SAFE_FINAL_ACTIONS
        ),
    }

    if test_case["expected_out_of_domain"]:
        checks["out_of_domain_action"] = (
            response.action
            == "abstain_out_of_domain"
        )

        checks["generation_skipped"] = (
            not response.generation_executed
        )

        checks["repair_skipped"] = (
            not response.repair_attempted
        )

        checks["no_model"] = (
            final_response.model is None
        )

        checks["no_evidence"] = (
            final_response.evidence_count == 0
        )

    else:
        checks["generation_executed"] = (
            response.generation_executed
            == test_case["expected_generation"]
        )

        if response.answer_released:
            checks["final_citation_validation"] = (
                final_response
                .sentence_validation
                .passed
            )

            checks["final_claim_validation"] = (
                final_response
                .claim_validation
                .passed
            )

            checks["full_citation_coverage"] = (
                final_response
                .sentence_validation
                .citation_coverage
                == 1.0
            )

            checks["full_groundedness"] = (
                final_response
                .claim_validation
                .groundedness_rate
                == 1.0
            )

            checks["no_unsupported_claims"] = (
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
                }
            )

    if (
        test_case["repair_expected"]
        and initial_response.action
        == "abstain_citation_validation"
    ):
        checks["repair_attempted"] = (
            response.repair_attempted
        )

    passed = all(
        checks.values()
    )

    base_response = (
        initial_response.base_response
    )

    generation_latency_ms = (
        base_response.generation_latency_ms
        if base_response.generation_executed
        else 0.0
    )

    return {
        "case_id": test_case["case_id"],
        "category": test_case["category"],
        "query": test_case["query"],
        "passed": passed,
        "safety_outcome": safety_outcome,
        "checks": checks,
        "total_latency_ms": (
            total_latency_ms
        ),
        "generation_latency_ms": (
            generation_latency_ms
        ),
        "repair_latency_ms": (
            response
            .repair_result
            .latency_ms
        ),
        "response": response.to_dict(),
    }


def print_case_result(
    result: dict[str, Any],
) -> None:
    response = result["response"]

    initial_response = (
        response["initial_response"]
    )

    final_response = (
        response["final_response"]
    )

    initial_sentence_validation = (
        initial_response[
            "sentence_validation"
        ]
    )

    final_sentence_validation = (
        final_response[
            "sentence_validation"
        ]
    )

    final_claim_validation = (
        final_response[
            "claim_validation"
        ]
    )

    print("\n" + "=" * 90)
    print(
        f"CASE: {result['case_id']}"
    )
    print(
        f"CATEGORY: {result['category']}"
    )
    print(f"QUERY: {result['query']}")
    print("-" * 90)

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
        f"{initial_response['action']}"
    )

    print(
        "FINAL ACTION: "
        f"{response['action']}"
    )

    print(
        "GENERATION EXECUTED: "
        f"{response['generation_executed']}"
    )

    print(
        "REPAIR ATTEMPTED: "
        f"{response['repair_attempted']}"
    )

    print(
        "REPAIR SUCCEEDED: "
        f"{response['repair_succeeded']}"
    )

    print(
        "ANSWER RELEASED: "
        f"{response['answer_released']}"
    )

    print(
        "INITIAL CITATION VALIDATION: "
        f"{initial_sentence_validation['passed']}"
    )

    print(
        "INITIAL CITATION COVERAGE: "
        f"{initial_sentence_validation['citation_coverage']:.4f}"
    )

    print(
        "FINAL CITATION VALIDATION: "
        f"{final_sentence_validation['passed']}"
    )

    print(
        "FINAL CITATION COVERAGE: "
        f"{final_sentence_validation['citation_coverage']:.4f}"
    )

    print(
        "FINAL CLAIM VALIDATION: "
        f"{final_claim_validation['passed']}"
    )

    print(
        "FINAL GROUNDEDNESS RATE: "
        f"{final_claim_validation['groundedness_rate']:.4f}"
    )

    print(
        "FINAL UNSUPPORTED CLAIMS: "
        f"{final_claim_validation['unsupported_claim_count']}"
    )

    print(
        "GENERATION LATENCY: "
        f"{result['generation_latency_ms']:.2f} ms"
    )

    print(
        "REPAIR LATENCY: "
        f"{result['repair_latency_ms']:.2f} ms"
    )

    print(
        "TOTAL PIPELINE LATENCY: "
        f"{result['total_latency_ms']:.2f} ms"
    )

    print(
        f"CHECKS: {result['checks']}"
    )

    print("\nVISIBLE ANSWER:")
    print(response["answer"])

    if response["repair_attempted"]:
        print("\nORIGINAL MODEL ANSWER:")
        print(
            response[
                "repair_result"
            ][
                "original_answer"
            ]
        )

        print("\nREPAIR MODEL OUTPUT:")
        print(
            response[
                "repair_result"
            ][
                "repaired_answer"
            ]
        )

        if response[
            "repair_result"
        ][
            "error"
        ]:
            print("\nREPAIR ERROR:")
            print(
                response[
                    "repair_result"
                ][
                    "error"
                ]
            )

    if not response["answer_released"]:
        print("\nWITHHELD ANSWER:")
        print(
            final_response[
                "base_response"
            ][
                "answer"
            ]
        )

    if (
        final_claim_validation[
            "unsupported_claim_count"
        ]
        > 0
    ):
        print(
            "\nUNSUPPORTED CLAIM DETAILS:"
        )

        for claim_result in (
            final_claim_validation[
                "claim_results"
            ]
        ):
            if claim_result["passed"]:
                continue

            print(
                f"- Claim: "
                f"{claim_result['claim']}"
            )

            for reason in (
                claim_result[
                    "failure_reasons"
                ]
            ):
                print(
                    f"  Reason: {reason}"
                )


def calculate_rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Stage 10D real retrying "
        "production evaluator..."
    )

    base_generator = GroundedResponseGenerator(
        model="qwen2.5-coder:3b",
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
        request_timeout_seconds=240,
    )

    generator = (
        RetryingProductionGroundedGenerator(
            base_generator=base_generator,
            minimum_support_score=0.60,
            minimum_token_coverage=0.45,
            repair_model="qwen2.5-coder:3b",
            request_timeout_seconds=240,
        )
    )

    results: list[
        dict[str, Any]
    ] = []

    for test_case in TEST_CASES:
        try:
            result = evaluate_case(
                generator=generator,
                test_case=test_case,
            )

        except Exception as error:
            result = {
                "case_id": (
                    test_case["case_id"]
                ),
                "category": (
                    test_case["category"]
                ),
                "query": (
                    test_case["query"]
                ),
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

        if "response" in result:
            print_case_result(result)

        else:
            print("\n" + "=" * 90)
            print(
                f"CASE: "
                f"{result['case_id']}"
            )
            print(
                f"CATEGORY: "
                f"{result['category']}"
            )
            print(
                f"QUERY: "
                f"{result['query']}"
            )
            print("STATUS: FAILED")
            print(
                "ERROR TYPE: "
                f"{result['error_type']}"
            )
            print(
                "ERROR: "
                f"{result['error_message']}"
            )

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = (
        len(results)
        - passed_count
    )

    generated_results = [
        result
        for result in results
        if (
            "response" in result
            and result[
                "response"
            ][
                "generation_executed"
            ]
        )
    ]

    released_results = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "answer_released"
        ]
    ]

    withheld_results = [
        result
        for result in generated_results
        if not result[
            "response"
        ][
            "answer_released"
        ]
    ]

    repair_attempted_results = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "repair_attempted"
        ]
    ]

    repair_succeeded_results = [
        result
        for result in repair_attempted_results
        if result[
            "response"
        ][
            "repair_succeeded"
        ]
    ]

    safe_repaired_results = [
        result
        for result in results
        if result.get(
            "safety_outcome"
        )
        == "safe_repaired_answer_released"
    ]

    safe_original_results = [
        result
        for result in results
        if result.get(
            "safety_outcome"
        )
        == "safe_original_answer_released"
    ]

    safe_out_of_domain_results = [
        result
        for result in results
        if result.get(
            "safety_outcome"
        )
        == "safe_out_of_domain_abstention"
    ]

    citation_blocked_results = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "action"
        ]
        == "abstain_citation_validation"
    ]

    groundedness_blocked_results = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "action"
        ]
        == "abstain_groundedness_validation"
    ]

    generation_latencies = [
        result[
            "generation_latency_ms"
        ]
        for result in generated_results
    ]

    repair_latencies = [
        result[
            "repair_latency_ms"
        ]
        for result in repair_attempted_results
    ]

    total_latencies = [
        result[
            "total_latency_ms"
        ]
        for result in results
        if "total_latency_ms" in result
    ]

    safe_pipeline_rate = calculate_rate(
        numerator=passed_count,
        denominator=len(results),
    )

    answer_release_rate = calculate_rate(
        numerator=len(
            released_results
        ),
        denominator=len(
            generated_results
        ),
    )

    answer_withholding_rate = (
        calculate_rate(
            numerator=len(
                withheld_results
            ),
            denominator=len(
                generated_results
            ),
        )
    )

    repair_success_rate = calculate_rate(
        numerator=len(
            repair_succeeded_results
        ),
        denominator=len(
            repair_attempted_results
        ),
    )

    mean_generation_latency_ms = (
        statistics.mean(
            generation_latencies
        )
        if generation_latencies
        else 0.0
    )

    mean_repair_latency_ms = (
        statistics.mean(
            repair_latencies
        )
        if repair_latencies
        else 0.0
    )

    mean_total_latency_ms = (
        statistics.mean(
            total_latencies
        )
        if total_latencies
        else 0.0
    )

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    summary = {
        "stage": "10D",
        "stage_name": (
            "Real Ollama Citation-Repair "
            "and Retry Evaluation"
        ),
        "status": overall_status,
        "model": "qwen2.5-coder:3b",
        "case_count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "safe_pipeline_rate": round(
            safe_pipeline_rate,
            6,
        ),
        "generated_case_count": len(
            generated_results
        ),
        "released_generated_answer_count": len(
            released_results
        ),
        "withheld_generated_answer_count": len(
            withheld_results
        ),
        "answer_release_rate": round(
            answer_release_rate,
            6,
        ),
        "answer_withholding_rate": round(
            answer_withholding_rate,
            6,
        ),
        "repair_attempt_count": len(
            repair_attempted_results
        ),
        "repair_success_count": len(
            repair_succeeded_results
        ),
        "repair_success_rate": round(
            repair_success_rate,
            6,
        ),
        "safe_repaired_answer_count": len(
            safe_repaired_results
        ),
        "safe_original_answer_count": len(
            safe_original_results
        ),
        "citation_block_count": len(
            citation_blocked_results
        ),
        "groundedness_block_count": len(
            groundedness_blocked_results
        ),
        "safe_out_of_domain_count": len(
            safe_out_of_domain_results
        ),
        "mean_generation_latency_ms": round(
            mean_generation_latency_ms,
            3,
        ),
        "mean_repair_latency_ms": round(
            mean_repair_latency_ms,
            3,
        ),
        "mean_total_pipeline_latency_ms": round(
            mean_total_latency_ms,
            3,
        ),
        "results": results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print(
        "STAGE 10D REAL RETRYING "
        "PRODUCTION SUMMARY"
    )
    print("=" * 90)

    print(
        f"Status: "
        f"{overall_status.upper()}"
    )

    print(
        f"Cases: {len(results)}"
    )

    print(
        f"Passed: {passed_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        "Safe pipeline rate: "
        f"{safe_pipeline_rate:.4f}"
    )

    print(
        "Generated cases: "
        f"{len(generated_results)}"
    )

    print(
        "Released generated answers: "
        f"{len(released_results)}"
    )

    print(
        "Withheld generated answers: "
        f"{len(withheld_results)}"
    )

    print(
        "Answer release rate: "
        f"{answer_release_rate:.4f}"
    )

    print(
        "Answer withholding rate: "
        f"{answer_withholding_rate:.4f}"
    )

    print(
        "Repair attempts: "
        f"{len(repair_attempted_results)}"
    )

    print(
        "Successful repairs: "
        f"{len(repair_succeeded_results)}"
    )

    print(
        "Repair success rate: "
        f"{repair_success_rate:.4f}"
    )

    print(
        "Safe repaired answers: "
        f"{len(safe_repaired_results)}"
    )

    print(
        "Safe original answers: "
        f"{len(safe_original_results)}"
    )

    print(
        "Citation validation blocks: "
        f"{len(citation_blocked_results)}"
    )

    print(
        "Groundedness validation blocks: "
        f"{len(groundedness_blocked_results)}"
    )

    print(
        "Safe out-of-domain cases: "
        f"{len(safe_out_of_domain_results)}"
    )

    print(
        "Mean generation latency: "
        f"{mean_generation_latency_ms:.2f} ms"
    )

    print(
        "Mean repair latency: "
        f"{mean_repair_latency_ms:.2f} ms"
    )

    print(
        "Mean total pipeline latency: "
        f"{mean_total_latency_ms:.2f} ms"
    )

    print(
        "Model: qwen2.5-coder:3b"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 10D real retrying "
            "production evaluation failed."
        )


if __name__ == "__main__":
    main()