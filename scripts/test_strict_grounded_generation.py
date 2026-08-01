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
from src.rag.strict_grounded_generation import (
    StrictGroundedResponseGenerator,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_9c_strict_grounded_generation.json"
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
    action: str = "retrieve",
) -> GovernedRetrievalResponse:
    understanding = understand_query(query)

    return GovernedRetrievalResponse(
        query=query,
        action=action,
        message=(
            "Relevant pull-request evidence "
            "was retrieved successfully."
            if action == "retrieve"
            else (
                "This question is outside the scope "
                "of the GitHub PR Intelligence system."
            )
        ),
        retrieval_executed=(
            action == "retrieve"
        ),
        query_understanding=understanding,
        results=[],
        trace={
            "action": action,
        },
    )


def build_generated_response(
    query: str,
    answer: str,
    evidence_ids: list[str],
) -> GroundedGenerationResponse:
    evidence = [
        EvidenceItem(
            evidence_id=evidence_id,
            rank=index,
            pr_number=5336,
            section="PR identity",
            text=(
                "PR #5336 was opened by dannyi96. "
                "The title is type hint fix for "
                "flask.send_file. The PR was merged "
                "on 2023-11-15."
            ),
            governed_score=0.80,
            metadata={},
        )
        for index, evidence_id in enumerate(
            evidence_ids,
            start=1,
        )
    ]

    return GroundedGenerationResponse(
        query=query,
        action="answer",
        message=(
            "A grounded response was generated."
        ),
        answer=answer,
        evidence_used=evidence_ids,
        insufficient_evidence=False,
        limitations=[],
        retrieval_response=(
            build_retrieval_response(query)
        ),
        evidence=evidence,
        citation_validation=(
            CitationValidation(
                citations_found=evidence_ids,
                valid_citations=evidence_ids,
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


def build_abstention_response(
    query: str,
) -> GroundedGenerationResponse:
    message = (
        "This question is outside the scope of "
        "the GitHub PR Intelligence system."
    )

    return GroundedGenerationResponse(
        query=query,
        action="abstain_out_of_domain",
        message=message,
        answer=message,
        evidence_used=[],
        insufficient_evidence=True,
        limitations=[
            (
                "LLM generation was not executed "
                "because governed retrieval abstained."
            )
        ],
        retrieval_response=(
            build_retrieval_response(
                query=query,
                action="abstain_out_of_domain",
            )
        ),
        evidence=[],
        citation_validation=(
            CitationValidation(
                citations_found=[],
                valid_citations=[],
                invalid_citations=[],
                uncited_factual_answer=False,
                passed=True,
            )
        ),
        model=None,
        generation_executed=False,
        generation_latency_ms=0.0,
        prompt_eval_count=None,
        eval_count=None,
        total_duration_ms=None,
        raw_model_response=None,
        trace={
            "generation_action": "skipped",
        },
    )


TEST_CASES = [
    {
        "case_id": "fully_cited_answer_released",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "base_response": build_generated_response(
            query=(
                "What do we know about "
                "pull request 5336?"
            ),
            answer=(
                "PR #5336 was opened by dannyi96 "
                "[E1]. The PR was merged on "
                "2023-11-15 [E1]."
            ),
            evidence_ids=["E1"],
        ),
        "expected_action": "answer",
        "expected_answer_released": True,
        "expected_validation_passed": True,
        "expected_uncited_count": 0,
    },
    {
        "case_id": "uncited_sentence_blocked",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "base_response": build_generated_response(
            query=(
                "What do we know about "
                "pull request 5336?"
            ),
            answer=(
                "PR #5336 was opened by dannyi96 "
                "[E1]. The PR was merged on "
                "2023-11-15."
            ),
            evidence_ids=["E1"],
        ),
        "expected_action": (
            "abstain_citation_validation"
        ),
        "expected_answer_released": False,
        "expected_validation_passed": False,
        "expected_uncited_count": 1,
    },
    {
        "case_id": "invalid_citation_blocked",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "base_response": build_generated_response(
            query=(
                "What do we know about "
                "pull request 5336?"
            ),
            answer=(
                "PR #5336 was opened by "
                "dannyi96 [E9]."
            ),
            evidence_ids=["E1"],
        ),
        "expected_action": (
            "abstain_citation_validation"
        ),
        "expected_answer_released": False,
        "expected_validation_passed": False,
        "expected_uncited_count": 1,
    },
    {
        "case_id": "out_of_domain_message_released",
        "query": (
            "What will the weather be "
            "in Dubai tomorrow?"
        ),
        "base_response": build_abstention_response(
            query=(
                "What will the weather be "
                "in Dubai tomorrow?"
            )
        ),
        "expected_action": (
            "abstain_out_of_domain"
        ),
        "expected_answer_released": True,
        "expected_validation_passed": True,
        "expected_uncited_count": 0,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    generator = StrictGroundedResponseGenerator(
        base_generator=StubBaseGenerator(
            test_case["base_response"]
        )
    )

    response = generator.generate(
        query=test_case["query"]
    )

    checks = {
        "action": (
            response.action
            == test_case["expected_action"]
        ),
        "answer_released": (
            response.answer_released
            == test_case[
                "expected_answer_released"
            ]
        ),
        "validation_passed": (
            response
            .sentence_validation
            .passed
            == test_case[
                "expected_validation_passed"
            ]
        ),
        "uncited_count": (
            response
            .sentence_validation
            .uncited_factual_sentence_count
            == test_case[
                "expected_uncited_count"
            ]
        ),
    }

    if not test_case[
        "expected_answer_released"
    ]:
        checks["unsafe_answer_hidden"] = (
            response.answer
            != test_case[
                "base_response"
            ].answer
        )

    passed = all(checks.values())

    return {
        "case_id": test_case["case_id"],
        "passed": passed,
        "checks": checks,
        "response": response.to_dict(),
    }


def print_result(
    result: dict[str, Any],
) -> None:
    response = result["response"]
    validation = (
        response["sentence_validation"]
    )

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(f"ACTION: {response['action']}")
    print(
        "ANSWER RELEASED: "
        f"{response['answer_released']}"
    )
    print(
        "SENTENCE VALIDATION: "
        f"{validation['passed']}"
    )
    print(
        "CITATION COVERAGE: "
        f"{validation['citation_coverage']:.4f}"
    )
    print(
        "UNCITED FACTUAL SENTENCES: "
        f"{validation['uncited_factual_sentence_count']}"
    )
    print(
        "INVALID CITATIONS: "
        f"{validation['invalid_citations']}"
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
        "stage": "9C",
        "stage_name": (
            "Strict Sentence-Level Citation "
            "Governance Integration"
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
        "STAGE 9C STRICT GROUNDED "
        "GENERATION SUMMARY"
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
            "Stage 9C strict grounded "
            "generation validation failed."
        )


if __name__ == "__main__":
    main()