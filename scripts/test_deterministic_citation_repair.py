from __future__ import annotations

import json
from typing import Any

from src.rag.deterministic_citation_repair import (
    DeterministicRepairProductionGenerator,
)
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
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_10e_deterministic_citation_repair.json"
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


def build_response(
    answer: str,
    evidence: list[EvidenceItem],
) -> GroundedGenerationResponse:
    query = (
        "What do we know about "
        "pull request 5336?"
    )

    return GroundedGenerationResponse(
        query=query,
        action="answer",
        message=(
            "A grounded response was generated."
        ),
        answer=answer,
        evidence_used=[
            item.evidence_id
            for item in evidence
        ],
        insufficient_evidence=False,
        limitations=[],
        retrieval_response=(
            build_retrieval_response(query)
        ),
        evidence=evidence,
        citation_validation=(
            CitationValidation(
                citations_found=[],
                valid_citations=[],
                invalid_citations=[],
                uncited_factual_answer=True,
                passed=False,
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


IDENTITY_EVIDENCE = EvidenceItem(
    evidence_id="E1",
    rank=1,
    pr_number=5336,
    section="PR identity",
    text=(
        "PR #5336 was opened by dannyi96. "
        "The PR title is type hint fix for "
        "flask.send_file. The PR was closed "
        "and merged on 2023-11-15."
    ),
    governed_score=0.90,
    metadata={
        "author": "dannyi96",
        "state": "closed",
        "merged": True,
        "merged_at": "2023-11-15",
    },
)


PREDICTION_EVIDENCE = EvidenceItem(
    evidence_id="E4",
    rank=4,
    pr_number=5336,
    section="Predictive intelligence",
    text=(
        "The predicted merge probability "
        "for PR #5336 is 47.01%. This is a "
        "model prediction and is not a "
        "confirmed outcome."
    ),
    governed_score=0.75,
    metadata={
        "merge_probability": 0.4701,
    },
)


DUPLICATE_IDENTITY_EVIDENCE = EvidenceItem(
    evidence_id="E2",
    rank=2,
    pr_number=5336,
    section="Historical PR identity",
    text=(
        "PR #5336 was closed and merged "
        "on 2023-11-15."
    ),
    governed_score=0.80,
    metadata={
        "merged": True,
        "merged_at": "2023-11-15",
    },
)


TEST_CASES = [
    {
        "case_id": (
            "unique_supported_sentence_repaired"
        ),
        "answer": (
            "PR #5336 was opened by "
            "dannyi96 [E1]. "
            "It was closed and merged on "
            "2023-11-15. "
            "The predicted merge probability "
            "is 47.01% [E4]."
        ),
        "evidence": [
            IDENTITY_EVIDENCE,
            PREDICTION_EVIDENCE,
        ],
        "expected_action": "answer",
        "expected_repair_attempted": True,
        "expected_repair_succeeded": True,
        "expected_released": True,
        "expected_selected_evidence": "E1",
    },
    {
        "case_id": (
            "ambiguous_support_remains_blocked"
        ),
        "answer": (
            "PR #5336 was closed and "
            "merged on 2023-11-15."
        ),
        "evidence": [
            IDENTITY_EVIDENCE,
            DUPLICATE_IDENTITY_EVIDENCE,
        ],
        "expected_action": (
            "abstain_citation_validation"
        ),
        "expected_repair_attempted": True,
        "expected_repair_succeeded": False,
        "expected_released": False,
        "expected_candidate_count": 2,
    },
    {
        "case_id": (
            "unsupported_sentence_remains_blocked"
        ),
        "answer": (
            "PR #5336 was merged on "
            "2024-01-01."
        ),
        "evidence": [
            IDENTITY_EVIDENCE,
            PREDICTION_EVIDENCE,
        ],
        "expected_action": (
            "abstain_citation_validation"
        ),
        "expected_repair_attempted": True,
        "expected_repair_succeeded": False,
        "expected_released": False,
        "expected_candidate_count": 0,
    },
    {
        "case_id": (
            "groundedness_failure_skips_repair"
        ),
        "answer": (
            "PR #5336 was merged on "
            "2024-01-01 [E1]."
        ),
        "evidence": [
            IDENTITY_EVIDENCE,
        ],
        "expected_action": (
            "abstain_groundedness_validation"
        ),
        "expected_repair_attempted": False,
        "expected_repair_succeeded": False,
        "expected_released": False,
    },
    {
        "case_id": (
            "valid_answer_skips_repair"
        ),
        "answer": (
            "PR #5336 was closed and "
            "merged on 2023-11-15 [E1]."
        ),
        "evidence": [
            IDENTITY_EVIDENCE,
        ],
        "expected_action": "answer",
        "expected_repair_attempted": False,
        "expected_repair_succeeded": False,
        "expected_released": True,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    base_response = build_response(
        answer=test_case["answer"],
        evidence=test_case["evidence"],
    )

    generator = (
        DeterministicRepairProductionGenerator(
            base_generator=(
                StubBaseGenerator(
                    base_response
                )
            ),
            minimum_support_score=0.60,
            minimum_token_coverage=0.45,
        )
    )

    response = generator.generate(
        query=base_response.query
    )

    checks: dict[str, bool] = {
        "action": (
            response.action
            == test_case["expected_action"]
        ),
        "repair_attempted": (
            response.repair_attempted
            == test_case[
                "expected_repair_attempted"
            ]
        ),
        "repair_succeeded": (
            response.repair_succeeded
            == test_case[
                "expected_repair_succeeded"
            ]
        ),
        "answer_released": (
            response.answer_released
            == test_case[
                "expected_released"
            ]
        ),
    }

    expected_selected_evidence = (
        test_case.get(
            "expected_selected_evidence"
        )
    )

    if expected_selected_evidence:
        selected_ids = [
            decision.selected_evidence_id
            for decision in (
                response
                .repair_result
                .decisions
            )
            if decision.selected_evidence_id
        ]

        checks[
            "selected_evidence"
        ] = (
            expected_selected_evidence
            in selected_ids
        )

    expected_candidate_count = (
        test_case.get(
            "expected_candidate_count"
        )
    )

    if expected_candidate_count is not None:
        candidate_counts = [
            len(
                decision
                .candidate_evidence_ids
            )
            for decision in (
                response
                .repair_result
                .decisions
            )
            if decision.repair_required
        ]

        checks[
            "candidate_count"
        ] = bool(
            candidate_counts
            and candidate_counts[0]
            == expected_candidate_count
        )

    if response.answer_released:
        checks[
            "final_citation_validation"
        ] = (
            response
            .final_response
            .sentence_validation
            .passed
        )

        checks[
            "final_groundedness_validation"
        ] = (
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
    print(
        f"CASE: {result['case_id']}"
    )

    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )

    print(
        "INITIAL ACTION: "
        f"{response['initial_response']['action']}"
    )

    print(
        "FINAL ACTION: "
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
        "REPAIR LATENCY: "
        f"{response['repair_result']['latency_ms']:.3f} ms"
    )

    print(
        "SENTENCES REQUIRING REPAIR: "
        f"{response['repair_result']['sentences_requiring_repair']}"
    )

    print(
        "SENTENCES REPAIRED: "
        f"{response['repair_result']['sentences_repaired']}"
    )

    print(
        "UNRESOLVED SENTENCES: "
        f"{response['repair_result']['unresolved_sentence_count']}"
    )

    print(
        "FINAL CITATION VALIDATION: "
        f"{response['final_response']['sentence_validation']['passed']}"
    )

    print(
        "FINAL GROUNDEDNESS VALIDATION: "
        f"{response['final_response']['claim_validation']['passed']}"
    )

    print(
        f"CHECKS: {result['checks']}"
    )

    print("\nVISIBLE ANSWER:")
    print(response["answer"])

    if response["repair_attempted"]:
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
                f"  Reason: "
                f"{decision['reason']}"
            )


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
        len(results)
        - passed_count
    )

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    report = {
        "stage": "10E",
        "stage_name": (
            "Deterministic Citation Repair"
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
        "STAGE 10E DETERMINISTIC "
        "CITATION REPAIR SUMMARY"
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
        f"Report: {REPORT_PATH}"
    )

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 10E deterministic citation "
            "repair validation failed."
        )


if __name__ == "__main__":
    main()