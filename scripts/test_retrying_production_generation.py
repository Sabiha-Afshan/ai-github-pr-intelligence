from __future__ import annotations

import json
from typing import Any

from src.rag.grounded_generation import (
    CitationValidation,
    EvidenceItem,
    GroundedGenerationResponse,
)
from src.rag.governed_retrieval import (
    GovernedRetrievalResponse,
)
from src.rag.query_understanding import (
    understand_query,
)
from src.rag.retrying_production_generation import (
    CitationRepairResult,
    RetryingProductionGroundedGenerator,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_10c_retrying_production_generation.json"
)


class StubBaseGenerator:
    def __init__(
        self,
        response: GroundedGenerationResponse,
    ) -> None:
        self.response = response

    def generate(
        self,
        query: str,
    ) -> GroundedGenerationResponse:
        return self.response


def build_retrieval_response(
    query: str,
) -> GovernedRetrievalResponse:
    return GovernedRetrievalResponse(
        query=query,
        action="retrieve",
        message=(
            "Relevant pull-request evidence "
            "was retrieved successfully."
        ),
        retrieval_executed=True,
        query_understanding=(
            understand_query(query)
        ),
        results=[],
        trace={
            "action": "retrieve",
        },
    )


def build_base_response(
    answer: str,
) -> GroundedGenerationResponse:
    query = (
        "What do we know about "
        "pull request 5336?"
    )

    evidence = [
        EvidenceItem(
            evidence_id="E1",
            rank=1,
            pr_number=5336,
            section="PR identity",
            text=(
                "PR #5336 was opened by dannyi96. "
                "The PR title is type hint fix for "
                "flask.send_file. It was closed and "
                "merged on 2023-11-15."
            ),
            governed_score=0.90,
            metadata={
                "author": "dannyi96",
                "merged": True,
                "merged_at": "2023-11-15",
            },
        ),
        EvidenceItem(
            evidence_id="E4",
            rank=4,
            pr_number=5336,
            section="Predictive intelligence",
            text=(
                "The predicted merge probability "
                "for PR #5336 is 47.01%. This is a "
                "model prediction and not a confirmed "
                "outcome."
            ),
            governed_score=0.75,
            metadata={
                "merge_probability": 0.4701,
            },
        ),
    ]

    return GroundedGenerationResponse(
        query=query,
        action="answer",
        message=(
            "A grounded response was generated."
        ),
        answer=answer,
        evidence_used=[
            "E1",
            "E4",
        ],
        insufficient_evidence=False,
        limitations=[],
        retrieval_response=(
            build_retrieval_response(query)
        ),
        evidence=evidence,
        citation_validation=(
            CitationValidation(
                citations_found=[
                    "E1",
                    "E4",
                ],
                valid_citations=[
                    "E1",
                    "E4",
                ],
                invalid_citations=[],
                uncited_factual_answer=False,
                passed=True,
            )
        ),
        model="stub-model",
        generation_executed=True,
        generation_latency_ms=10.0,
        prompt_eval_count=100,
        eval_count=50,
        total_duration_ms=20.0,
        raw_model_response={},
        trace={
            "generation_action": "completed",
        },
    )


def successful_repair(
    query: str,
    base_response: GroundedGenerationResponse,
    model: str,
    ollama_url: str,
    request_timeout_seconds: int,
) -> CitationRepairResult:
    repaired_answer = (
        "PR #5336 was opened by dannyi96 [E1]. "
        "The PR title is 'type hint fix for "
        "flask.send_file' [E1]. "
        "It was closed and merged on "
        "2023-11-15 [E1]. "
        "The predicted merge probability is "
        "47.01%, but this is not a confirmed "
        "outcome [E4]."
    )

    return CitationRepairResult(
        attempted=True,
        succeeded=True,
        original_answer=(
            base_response.answer
        ),
        repaired_answer=repaired_answer,
        model=model,
        latency_ms=5.0,
        error=None,
        raw_response={
            "message": {
                "content": repaired_answer,
            }
        },
    )


def unsupported_repair(
    query: str,
    base_response: GroundedGenerationResponse,
    model: str,
    ollama_url: str,
    request_timeout_seconds: int,
) -> CitationRepairResult:
    repaired_answer = (
        "PR #5336 was closed and merged on "
        "2024-01-01 [E1]."
    )

    return CitationRepairResult(
        attempted=True,
        succeeded=True,
        original_answer=(
            base_response.answer
        ),
        repaired_answer=repaired_answer,
        model=model,
        latency_ms=5.0,
        error=None,
        raw_response={
            "message": {
                "content": repaired_answer,
            }
        },
    )


def failed_repair(
    query: str,
    base_response: GroundedGenerationResponse,
    model: str,
    ollama_url: str,
    request_timeout_seconds: int,
) -> CitationRepairResult:
    return CitationRepairResult(
        attempted=True,
        succeeded=False,
        original_answer=(
            base_response.answer
        ),
        repaired_answer=None,
        model=model,
        latency_ms=5.0,
        error="Simulated repair failure.",
        raw_response=None,
    )


TEST_CASES = [
    {
        "case_id": (
            "valid_repair_released"
        ),
        "answer": (
            "PR #5336 was opened by dannyi96 "
            "[E1]. The PR title is 'type hint "
            "fix for flask.send_file' [E1]. "
            "It was closed and merged on "
            "2023-11-15. The predicted merge "
            "probability is 47.01%, but this is "
            "not a confirmed outcome [E4]."
        ),
        "repair_function": successful_repair,
        "expected_action": "answer",
        "expected_released": True,
        "expected_repair_succeeded": True,
    },
    {
        "case_id": (
            "unsupported_repair_blocked"
        ),
        "answer": (
            "PR #5336 was closed and merged "
            "on 2023-11-15."
        ),
        "repair_function": unsupported_repair,
        "expected_action": (
            "abstain_groundedness_validation"
        ),
        "expected_released": False,
        "expected_repair_succeeded": False,
    },
    {
        "case_id": (
            "failed_repair_remains_blocked"
        ),
        "answer": (
            "PR #5336 was closed and merged "
            "on 2023-11-15."
        ),
        "repair_function": failed_repair,
        "expected_action": (
            "abstain_citation_validation"
        ),
        "expected_released": False,
        "expected_repair_succeeded": False,
    },
    {
        "case_id": (
            "already_valid_answer_skips_repair"
        ),
        "answer": (
            "PR #5336 was closed and merged "
            "on 2023-11-15 [E1]."
        ),
        "repair_function": failed_repair,
        "expected_action": "answer",
        "expected_released": True,
        "expected_repair_succeeded": False,
        "expected_repair_attempted": False,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    base_response = build_base_response(
        answer=test_case["answer"]
    )

    generator = (
        RetryingProductionGroundedGenerator(
            base_generator=(
                StubBaseGenerator(
                    base_response
                )
            ),
            repair_model="stub-model",
            repair_function=(
                test_case[
                    "repair_function"
                ]
            ),
        )
    )

    response = generator.generate(
        query=base_response.query
    )

    expected_repair_attempted = (
        test_case.get(
            "expected_repair_attempted",
            True,
        )
    )

    checks = {
        "action": (
            response.action
            == test_case["expected_action"]
        ),
        "answer_released": (
            response.answer_released
            == test_case[
                "expected_released"
            ]
        ),
        "repair_attempted": (
            response.repair_attempted
            == expected_repair_attempted
        ),
        "repair_succeeded": (
            response.repair_succeeded
            == test_case[
                "expected_repair_succeeded"
            ]
        ),
    }

    if response.answer_released:
        checks["final_citation_validation"] = (
            response
            .final_response
            .sentence_validation
            .passed
        )

        checks["final_groundedness"] = (
            response
            .final_response
            .claim_validation
            .passed
        )

    return {
        "case_id": test_case["case_id"],
        "passed": all(
            checks.values()
        ),
        "checks": checks,
        "response": response.to_dict(),
    }


def print_result(
    result: dict[str, Any],
) -> None:
    response = result["response"]

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(
        f"INITIAL ACTION: "
        f"{response['initial_response']['action']}"
    )
    print(
        f"FINAL ACTION: "
        f"{response['action']}"
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
        "FINAL CITATION VALIDATION: "
        f"{response['final_response']['sentence_validation']['passed']}"
    )
    print(
        "FINAL GROUNDEDNESS VALIDATION: "
        f"{response['final_response']['claim_validation']['passed']}"
    )
    print(f"CHECKS: {result['checks']}")

    print("\nVISIBLE ANSWER:")
    print(response["answer"])


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = [
        evaluate_case(test_case)
        for test_case in TEST_CASES
    ]

    for result in results:
        print_result(result)

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = (
        len(results) - passed_count
    )

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    report = {
        "stage": "10C",
        "stage_name": (
            "Citation-Aware Repair and "
            "Fail-Closed Retry"
        ),
        "status": overall_status,
        "case_count": len(results),
        "passed_count": passed_count,
        "failed_count": failed_count,
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
        "STAGE 10C CITATION-AWARE "
        "RETRY SUMMARY"
    )
    print("=" * 90)
    print(
        f"Status: {overall_status.upper()}"
    )
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {REPORT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 10C citation-aware "
            "retry validation failed."
        )


if __name__ == "__main__":
    main()