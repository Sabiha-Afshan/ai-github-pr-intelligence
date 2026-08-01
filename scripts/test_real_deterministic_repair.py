from __future__ import annotations

import json
import statistics
import time
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
    / "stage_10f_real_deterministic_repair.json"
)


TEST_CASES = [
    {
        "case_id": "real_exact_pr",
        "category": "exact_pr",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "expected_out_of_domain": False,
    },
    {
        "case_id": "real_low_merge_probability",
        "category": "prediction",
        "query": (
            "Which pull requests have a low "
            "chance of being merged?"
        ),
        "expected_out_of_domain": False,
    },
    {
        "case_id": "real_human_governance_review",
        "category": "policy",
        "query": (
            "Which changes require a human "
            "governance review?"
        ),
        "expected_out_of_domain": False,
    },
    {
        "case_id": "real_review_priority",
        "category": "priority",
        "query": (
            "Which changes should maintainers "
            "review first?"
        ),
        "expected_out_of_domain": False,
    },
    {
        "case_id": "real_out_of_domain",
        "category": "out_of_domain",
        "query": (
            "What will the weather be in "
            "Dubai tomorrow?"
        ),
        "expected_out_of_domain": True,
    },
]


SAFE_ACTIONS = {
    "answer",
    "abstain_citation_validation",
    "abstain_groundedness_validation",
    "abstain_no_evidence",
    "abstain_out_of_domain",
}


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


def out_of_domain_is_safe(
    response: DeterministicProductionResponse,
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
    response: DeterministicProductionResponse,
    expected_out_of_domain: bool,
) -> tuple[bool, str]:
    if expected_out_of_domain:
        safe = out_of_domain_is_safe(
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
                "safe_deterministically_repaired_answer",
            )

        return (
            True,
            "safe_original_answer_released",
        )

    if withheld_answer_is_safe(response):
        return (
            True,
            "unsafe_answer_safely_withheld",
        )

    return (
        False,
        "unsafe_pipeline_outcome",
    )


def evaluate_case(
    generator: DeterministicRepairProductionGenerator,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    started_at = time.perf_counter()

    response = generator.generate(
        query=test_case["query"]
    )

    total_latency_ms = round(
        (
            time.perf_counter()
            - started_at
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

    base_response = (
        initial_response.base_response
    )

    generation_latency_ms = (
        base_response.generation_latency_ms
        if base_response.generation_executed
        else 0.0
    )

    checks: dict[str, bool] = {
        "safety_passed": safety_passed,
        "valid_action": (
            response.action in SAFE_ACTIONS
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

    elif response.answer_released:
        checks["citation_validation"] = (
            final_response
            .sentence_validation
            .passed
        )

        checks["claim_validation"] = (
            final_response
            .claim_validation
            .passed
        )

        checks["citation_coverage"] = (
            final_response
            .sentence_validation
            .citation_coverage
            == 1.0
        )

        checks["groundedness_rate"] = (
            final_response
            .claim_validation
            .groundedness_rate
            == 1.0
        )

        checks["unsupported_claims"] = (
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
        "case_id": test_case["case_id"],
        "category": test_case["category"],
        "query": test_case["query"],
        "passed": all(
            checks.values()
        ),
        "safety_outcome": safety_outcome,
        "checks": checks,
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

    print("\n" + "=" * 90)
    print(
        f"CASE: {result['case_id']}"
    )
    print(
        f"CATEGORY: {result['category']}"
    )
    print(
        f"QUERY: {result['query']}"
    )
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
        f"{initial_response['sentence_validation']['passed']}"
    )

    print(
        "INITIAL CITATION COVERAGE: "
        f"{initial_response['sentence_validation']['citation_coverage']:.4f}"
    )

    print(
        "FINAL CITATION VALIDATION: "
        f"{final_response['sentence_validation']['passed']}"
    )

    print(
        "FINAL CITATION COVERAGE: "
        f"{final_response['sentence_validation']['citation_coverage']:.4f}"
    )

    print(
        "FINAL CLAIM VALIDATION: "
        f"{final_response['claim_validation']['passed']}"
    )

    print(
        "FINAL GROUNDEDNESS RATE: "
        f"{final_response['claim_validation']['groundedness_rate']:.4f}"
    )

    print(
        "FINAL UNSUPPORTED CLAIMS: "
        f"{final_response['claim_validation']['unsupported_claim_count']}"
    )

    print(
        "GENERATION LATENCY: "
        f"{result['generation_latency_ms']:.2f} ms"
    )

    print(
        "DETERMINISTIC REPAIR LATENCY: "
        f"{result['repair_latency_ms']:.3f} ms"
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

        print("\nREPAIRED ANSWER:")
        print(
            response[
                "repair_result"
            ][
                "repaired_answer"
            ]
        )

        print("\nREPAIR DECISIONS:")

        for decision in (
            response[
                "repair_result"
            ][
                "decisions"
            ]
        ):
            if not decision[
                "repair_required"
            ]:
                continue

            print(
                f"- Sentence: "
                f"{decision['original_sentence']}"
            )

            print(
                "  Candidates: "
                f"{decision['candidate_evidence_ids']}"
            )

            print(
                "  Selected: "
                f"{decision['selected_evidence_id']}"
            )

            print(
                "  Reason: "
                f"{decision['reason']}"
            )

    if not response["answer_released"]:
        print("\nWITHHELD MODEL ANSWER:")
        print(
            final_response[
                "base_response"
            ][
                "answer"
            ]
        )

    if (
        final_response[
            "claim_validation"
        ][
            "unsupported_claim_count"
        ]
        > 0
    ):
        print(
            "\nUNSUPPORTED CLAIM DETAILS:"
        )

        for claim_result in (
            final_response[
                "claim_validation"
            ][
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
        "Loading Stage 10F live deterministic "
        "repair evaluator..."
    )

    base_generator = GroundedResponseGenerator(
        model="qwen2.5-coder:3b",
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
        request_timeout_seconds=240,
    )

    generator = (
        DeterministicRepairProductionGenerator(
            base_generator=base_generator,
            minimum_support_score=0.60,
            minimum_token_coverage=0.45,
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

    repair_attempt_results = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "repair_attempted"
        ]
    ]

    repair_success_results = [
        result
        for result in repair_attempt_results
        if result[
            "response"
        ][
            "repair_succeeded"
        ]
    ]

    citation_blocks = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "action"
        ]
        == "abstain_citation_validation"
    ]

    groundedness_blocks = [
        result
        for result in generated_results
        if result[
            "response"
        ][
            "action"
        ]
        == "abstain_groundedness_validation"
    ]

    out_of_domain_results = [
        result
        for result in results
        if result.get(
            "safety_outcome"
        )
        == "safe_out_of_domain_abstention"
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
        for result in repair_attempt_results
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

    answer_withholding_rate = calculate_rate(
        numerator=len(
            withheld_results
        ),
        denominator=len(
            generated_results
        ),
    )

    repair_success_rate = calculate_rate(
        numerator=len(
            repair_success_results
        ),
        denominator=len(
            repair_attempt_results
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

    report = {
        "stage": "10F",
        "stage_name": (
            "Live Deterministic Citation Repair "
            "Evaluation"
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
            repair_attempt_results
        ),
        "repair_success_count": len(
            repair_success_results
        ),
        "repair_success_rate": round(
            repair_success_rate,
            6,
        ),
        "citation_block_count": len(
            citation_blocks
        ),
        "groundedness_block_count": len(
            groundedness_blocks
        ),
        "safe_out_of_domain_count": len(
            out_of_domain_results
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
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print(
        "STAGE 10F LIVE DETERMINISTIC "
        "REPAIR SUMMARY"
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
        f"{len(repair_attempt_results)}"
    )

    print(
        "Successful repairs: "
        f"{len(repair_success_results)}"
    )

    print(
        "Repair success rate: "
        f"{repair_success_rate:.4f}"
    )

    print(
        "Citation validation blocks: "
        f"{len(citation_blocks)}"
    )

    print(
        "Groundedness validation blocks: "
        f"{len(groundedness_blocks)}"
    )

    print(
        "Safe out-of-domain cases: "
        f"{len(out_of_domain_results)}"
    )

    print(
        "Mean generation latency: "
        f"{mean_generation_latency_ms:.2f} ms"
    )

    print(
        "Mean deterministic repair latency: "
        f"{mean_repair_latency_ms:.3f} ms"
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
            "Stage 10F live deterministic "
            "repair evaluation failed."
        )


if __name__ == "__main__":
    main()