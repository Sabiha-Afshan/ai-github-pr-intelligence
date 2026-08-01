from __future__ import annotations

import json
import statistics
import time
from typing import Any

from src.rag.grounded_generation import (
    GroundedResponseGenerator,
)
from src.rag.production_grounded_generation import (
    ProductionGroundedResponse,
    ProductionGroundedResponseGenerator,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_10a_real_production_grounded_generation.json"
)


TEST_CASES = [
    {
        "case_id": "real_exact_pr_answer",
        "category": "exact_pr",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "expected_generation": True,
        "expected_out_of_domain": False,
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
    },
]


SAFE_GENERATED_ACTIONS = {
    "answer",
    "abstain_citation_validation",
    "abstain_groundedness_validation",
}

SAFE_ABSTENTION_ACTIONS = {
    "abstain_out_of_domain",
    "abstain_no_evidence",
}


def released_answer_is_safe(
    response: ProductionGroundedResponse,
) -> bool:
    if not response.answer_released:
        return False

    if response.action != "answer":
        return False

    return bool(
        response.sentence_validation.passed
        and response.claim_validation.passed
        and (
            response
            .sentence_validation
            .citation_coverage
            == 1.0
        )
        and (
            response
            .claim_validation
            .groundedness_rate
            == 1.0
        )
        and (
            response
            .claim_validation
            .unsupported_claim_count
            == 0
        )
    )


def withheld_answer_is_safe(
    response: ProductionGroundedResponse,
) -> bool:
    if response.answer_released:
        return False

    return response.action in {
        "abstain_citation_validation",
        "abstain_groundedness_validation",
    }


def out_of_domain_response_is_safe(
    response: ProductionGroundedResponse,
) -> bool:
    return bool(
        response.action
        == "abstain_out_of_domain"
        and response.generation_executed is False
        and response.answer_released is True
        and response.evidence_count == 0
        and response.model is None
    )


def determine_safety_status(
    response: ProductionGroundedResponse,
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
        return True, "safe_answer_released"

    if withheld_answer_is_safe(response):
        return True, "unsafe_answer_withheld"

    return False, "unsafe_pipeline_outcome"


def evaluate_case(
    generator: ProductionGroundedResponseGenerator,
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
    ) = determine_safety_status(
        response=response,
        expected_out_of_domain=(
            test_case["expected_out_of_domain"]
        ),
    )

    checks: dict[str, bool] = {
        "safety_passed": safety_passed,
    }

    if test_case["expected_out_of_domain"]:
        checks["action"] = (
            response.action
            == "abstain_out_of_domain"
        )

        checks["generation_skipped"] = (
            response.generation_executed
            is False
        )

        checks["no_model"] = (
            response.model is None
        )

        checks["no_evidence"] = (
            response.evidence_count == 0
        )
    else:
        checks["generation_executed"] = (
            response.generation_executed
            == test_case["expected_generation"]
        )

        checks["valid_action"] = (
            response.action
            in SAFE_GENERATED_ACTIONS
        )

        if response.answer_released:
            checks["released_answer_citations"] = (
                response
                .sentence_validation
                .passed
            )

            checks["released_answer_grounding"] = (
                response
                .claim_validation
                .passed
            )

            checks["no_unsupported_claims"] = (
                response
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

    passed = all(checks.values())

    base_response = response.base_response

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
        "total_latency_ms": total_latency_ms,
        "generation_latency_ms": (
            generation_latency_ms
        ),
        "response": response.to_dict(),
    }


def print_case_result(
    result: dict[str, Any],
) -> None:
    response = result["response"]

    sentence_validation = (
        response["sentence_validation"]
    )

    claim_validation = (
        response["claim_validation"]
    )

    base_response = response["base_response"]

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"CATEGORY: {result['category']}")
    print(f"QUERY: {result['query']}")
    print("-" * 90)
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(
        f"SAFETY OUTCOME: "
        f"{result['safety_outcome']}"
    )
    print(
        f"ACTION: {response['action']}"
    )
    print(
        "GENERATION EXECUTED: "
        f"{response['generation_executed']}"
    )
    print(
        "ANSWER RELEASED: "
        f"{response['answer_released']}"
    )
    print(
        f"MODEL: {response['model']}"
    )
    print(
        f"EVIDENCE COUNT: "
        f"{response['evidence_count']}"
    )
    print(
        "SENTENCE VALIDATION: "
        f"{sentence_validation['passed']}"
    )
    print(
        "CITATION COVERAGE: "
        f"{sentence_validation['citation_coverage']:.4f}"
    )
    print(
        "UNCITED FACTUAL SENTENCES: "
        f"{sentence_validation['uncited_factual_sentence_count']}"
    )
    print(
        "CLAIM VALIDATION: "
        f"{claim_validation['passed']}"
    )
    print(
        "GROUNDEDNESS RATE: "
        f"{claim_validation['groundedness_rate']:.4f}"
    )
    print(
        "UNSUPPORTED CLAIMS: "
        f"{claim_validation['unsupported_claim_count']}"
    )
    print(
        "GENERATION LATENCY: "
        f"{result['generation_latency_ms']:.2f} ms"
    )
    print(
        "TOTAL PIPELINE LATENCY: "
        f"{result['total_latency_ms']:.2f} ms"
    )
    print(
        "PROMPT TOKENS: "
        f"{base_response['prompt_eval_count']}"
    )
    print(
        "OUTPUT TOKENS: "
        f"{base_response['eval_count']}"
    )
    print(
        f"CHECKS: {result['checks']}"
    )

    print("\nVISIBLE ANSWER:")
    print(response["answer"])

    if not response["answer_released"]:
        print("\nWITHHELD MODEL ANSWER:")
        print(
            base_response["answer"]
        )

    if (
        claim_validation[
            "unsupported_claim_count"
        ]
        > 0
    ):
        print("\nUNSUPPORTED CLAIM DETAILS:")

        for claim_result in (
            claim_validation[
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
        "Loading Stage 10A real production "
        "grounded-response evaluator..."
    )

    base_generator = GroundedResponseGenerator(
        model="qwen2.5-coder:3b",
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
        request_timeout_seconds=240,
    )

    production_generator = (
        ProductionGroundedResponseGenerator(
            base_generator=base_generator,
            minimum_support_score=0.60,
            minimum_token_coverage=0.45,
        )
    )

    results: list[dict[str, Any]] = []

    for test_case in TEST_CASES:
        try:
            result = evaluate_case(
                generator=production_generator,
                test_case=test_case,
            )
        except Exception as error:
            result = {
                "case_id": test_case["case_id"],
                "category": test_case["category"],
                "query": test_case["query"],
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
                f"CASE: {result['case_id']}"
            )
            print(
                f"CATEGORY: "
                f"{result['category']}"
            )
            print(
                f"QUERY: {result['query']}"
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
        len(results) - passed_count
    )

    generated_results = [
        result
        for result in results
        if (
            "response" in result
            and result["response"][
                "generation_executed"
            ]
        )
    ]

    released_generated_results = [
        result
        for result in generated_results
        if result["response"][
            "answer_released"
        ]
    ]

    withheld_generated_results = [
        result
        for result in generated_results
        if not result["response"][
            "answer_released"
        ]
    ]

    citation_blocked_results = [
        result
        for result in generated_results
        if result["response"]["action"]
        == "abstain_citation_validation"
    ]

    groundedness_blocked_results = [
        result
        for result in generated_results
        if result["response"]["action"]
        == "abstain_groundedness_validation"
    ]

    safe_out_of_domain_results = [
        result
        for result in results
        if result.get("safety_outcome")
        == "safe_out_of_domain_abstention"
    ]

    generation_latencies = [
        result["generation_latency_ms"]
        for result in generated_results
    ]

    total_latencies = [
        result["total_latency_ms"]
        for result in results
        if "total_latency_ms" in result
    ]

    safe_pipeline_rate = calculate_rate(
        numerator=passed_count,
        denominator=len(results),
    )

    answer_release_rate = calculate_rate(
        numerator=len(
            released_generated_results
        ),
        denominator=len(
            generated_results
        ),
    )

    answer_withholding_rate = calculate_rate(
        numerator=len(
            withheld_generated_results
        ),
        denominator=len(
            generated_results
        ),
    )

    mean_generation_latency_ms = (
        statistics.mean(
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

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    summary = {
        "stage": "10A",
        "stage_name": (
            "Real Ollama Production-Pipeline "
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
            released_generated_results
        ),
        "withheld_generated_answer_count": len(
            withheld_generated_results
        ),
        "answer_release_rate": round(
            answer_release_rate,
            6,
        ),
        "answer_withholding_rate": round(
            answer_withholding_rate,
            6,
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
        "STAGE 10A REAL PRODUCTION "
        "PIPELINE SUMMARY"
    )
    print("=" * 90)
    print(
        f"Status: {overall_status.upper()}"
    )
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(
        f"Safe pipeline rate: "
        f"{safe_pipeline_rate:.4f}"
    )
    print(
        f"Generated cases: "
        f"{len(generated_results)}"
    )
    print(
        "Released generated answers: "
        f"{len(released_generated_results)}"
    )
    print(
        "Withheld generated answers: "
        f"{len(withheld_generated_results)}"
    )
    print(
        f"Answer release rate: "
        f"{answer_release_rate:.4f}"
    )
    print(
        f"Answer withholding rate: "
        f"{answer_withholding_rate:.4f}"
    )
    print(
        f"Citation validation blocks: "
        f"{len(citation_blocked_results)}"
    )
    print(
        f"Groundedness validation blocks: "
        f"{len(groundedness_blocked_results)}"
    )
    print(
        f"Safe out-of-domain cases: "
        f"{len(safe_out_of_domain_results)}"
    )
    print(
        "Mean generation latency: "
        f"{mean_generation_latency_ms:.2f} ms"
    )
    print(
        "Mean total pipeline latency: "
        f"{mean_total_latency_ms:.2f} ms"
    )
    print(f"Model: qwen2.5-coder:3b")
    print(f"Report: {REPORT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 10A real production-pipeline "
            "evaluation failed."
        )


if __name__ == "__main__":
    main()