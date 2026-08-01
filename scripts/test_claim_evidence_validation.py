from __future__ import annotations

import json
from typing import Any

from src.rag.claim_evidence_validation import (
    EvidenceRecord,
    validate_claim_evidence,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_9d_claim_evidence_validation.json"
)


EVIDENCE = [
    EvidenceRecord(
        evidence_id="E1",
        pr_number=5336,
        section="PR identity",
        text=(
            "PR #5336 was opened by dannyi96. "
            "The pull request title is type hint fix for "
            "flask.send_file. The PR was closed and merged "
            "on 2023-11-15."
        ),
        metadata={
            "author": "dannyi96",
            "state": "closed",
            "merged": True,
            "merged_at": "2023-11-15",
        },
    ),
    EvidenceRecord(
        evidence_id="E2",
        pr_number=5336,
        section="Predictive intelligence",
        text=(
            "The predicted merge probability for PR #5336 "
            "is 47.01%. This value is a model prediction and "
            "is not a confirmed outcome."
        ),
        metadata={
            "merge_probability": 0.4701,
            "prediction_type": "model prediction",
        },
    ),
    EvidenceRecord(
        evidence_id="E3",
        pr_number=5736,
        section="Deterministic policy intelligence",
        text=(
            "PR #5736 requires manual governance review. "
            "The manual_review_required field is True and "
            "the policy risk band is Critical."
        ),
        metadata={
            "manual_review_required": True,
            "policy_risk_band": "Critical",
        },
    ),
]


TEST_CASES = [
    {
        "case_id": "supported_identity_claim",
        "answer": (
            "PR #5336 was opened by dannyi96 [E1]."
        ),
        "expected_passed": True,
        "expected_groundedness": 1.0,
        "expected_unsupported": 0,
    },
    {
        "case_id": "supported_merge_date",
        "answer": (
            "PR #5336 was merged on 2023-11-15 [E1]."
        ),
        "expected_passed": True,
        "expected_groundedness": 1.0,
        "expected_unsupported": 0,
    },
    {
        "case_id": "wrong_merge_date",
        "answer": (
            "PR #5336 was merged on 2024-01-01 [E1]."
        ),
        "expected_passed": False,
        "expected_groundedness": 0.0,
        "expected_unsupported": 1,
        "expected_failed_entity": "dates",
    },
    {
        "case_id": "wrong_pr_number",
        "answer": (
            "PR #9999 was opened by dannyi96 [E1]."
        ),
        "expected_passed": False,
        "expected_groundedness": 0.0,
        "expected_unsupported": 1,
        "expected_failed_entity": "pr_numbers",
    },
    {
        "case_id": "supported_prediction",
        "answer": (
            "The predicted merge probability for PR #5336 "
            "is 47.01% [E2]."
        ),
        "expected_passed": True,
        "expected_groundedness": 1.0,
        "expected_unsupported": 0,
    },
    {
        "case_id": "wrong_prediction_percentage",
        "answer": (
            "The predicted merge probability for PR #5336 "
            "is 87.50% [E2]."
        ),
        "expected_passed": False,
        "expected_groundedness": 0.0,
        "expected_unsupported": 1,
        "expected_failed_entity": "percentages",
    },
    {
        "case_id": "supported_manual_review",
        "answer": (
            "PR #5736 requires manual governance review "
            "because manual_review_required is True [E3]."
        ),
        "expected_passed": True,
        "expected_groundedness": 1.0,
        "expected_unsupported": 0,
    },
    {
        "case_id": "wrong_manual_review_boolean",
        "answer": (
            "PR #5736 does not require manual governance "
            "review because manual_review_required is False "
            "[E3]."
        ),
        "expected_passed": False,
        "expected_groundedness": 0.0,
        "expected_unsupported": 1,
        "expected_failed_entity": "booleans",
    },
    {
        "case_id": "invalid_evidence_identifier",
        "answer": (
            "PR #5336 was opened by dannyi96 [E9]."
        ),
        "expected_passed": False,
        "expected_groundedness": 0.0,
        "expected_unsupported": 1,
        "expected_invalid_citation": "E9",
    },
    {
        "case_id": "out_of_domain_abstention",
        "answer": (
            "This question is outside the scope of the "
            "GitHub PR Intelligence system."
        ),
        "insufficient_evidence": True,
        "expected_passed": True,
        "expected_groundedness": 1.0,
        "expected_unsupported": 0,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_claim_evidence(
        answer=test_case["answer"],
        evidence=EVIDENCE,
        insufficient_evidence=(
            test_case.get(
                "insufficient_evidence",
                False,
            )
        ),
        minimum_support_score=0.60,
        minimum_token_coverage=0.45,
    )

    checks: dict[str, bool] = {
        "passed": (
            validation.passed
            == test_case["expected_passed"]
        ),
        "groundedness": (
            validation.groundedness_rate
            == test_case[
                "expected_groundedness"
            ]
        ),
        "unsupported_count": (
            validation.unsupported_claim_count
            == test_case[
                "expected_unsupported"
            ]
        ),
    }

    expected_failed_entity = (
        test_case.get(
            "expected_failed_entity"
        )
    )

    if expected_failed_entity:
        failed_entity_found = False

        for result in validation.claim_results:
            entity_check = (
                result.entity_checks.get(
                    expected_failed_entity
                )
            )

            if (
                entity_check
                and entity_check["applicable"]
                and not entity_check["passed"]
            ):
                failed_entity_found = True
                break

        checks["failed_entity_detected"] = (
            failed_entity_found
        )

    expected_invalid_citation = (
        test_case.get(
            "expected_invalid_citation"
        )
    )

    if expected_invalid_citation:
        checks["invalid_citation_detected"] = (
            expected_invalid_citation
            in validation.invalid_citations
        )

    return {
        "case_id": test_case["case_id"],
        "passed": all(
            checks.values()
        ),
        "checks": checks,
        "answer": test_case["answer"],
        "validation": validation.to_dict(),
    }


def print_result(
    result: dict[str, Any],
) -> None:
    validation = result["validation"]

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(
        "GROUNDEDNESS VALIDATION: "
        f"{validation['passed']}"
    )
    print(
        "FACTUAL CLAIMS: "
        f"{validation['factual_claim_count']}"
    )
    print(
        "SUPPORTED CLAIMS: "
        f"{validation['supported_claim_count']}"
    )
    print(
        "UNSUPPORTED CLAIMS: "
        f"{validation['unsupported_claim_count']}"
    )
    print(
        "GROUNDEDNESS RATE: "
        f"{validation['groundedness_rate']:.4f}"
    )
    print(
        "MEAN SUPPORT SCORE: "
        f"{validation['mean_support_score']:.4f}"
    )
    print(
        "INVALID CITATIONS: "
        f"{validation['invalid_citations']}"
    )
    print(f"CHECKS: {result['checks']}")

    for claim_result in (
        validation["claim_results"]
    ):
        print(
            f"\nClaim "
            f"{claim_result['sentence_index']}: "
            f"{claim_result['claim']}"
        )
        print(
            "  Requires validation: "
            f"{claim_result['requires_validation']}"
        )
        print(
            "  Citations: "
            f"{claim_result['citations']}"
        )
        print(
            "  Token coverage: "
            f"{claim_result['token_coverage']:.4f}"
        )
        print(
            "  Entity coverage: "
            f"{claim_result['entity_coverage']:.4f}"
        )
        print(
            "  Support score: "
            f"{claim_result['support_score']:.4f}"
        )
        print(
            "  Passed: "
            f"{claim_result['passed']}"
        )

        if claim_result["failure_reasons"]:
            print("  Failure reasons:")

            for reason in (
                claim_result[
                    "failure_reasons"
                ]
            ):
                print(f"    - {reason}")


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
        "stage": "9D",
        "stage_name": (
            "Claim-to-Evidence Groundedness "
            "Validation"
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
        "STAGE 9D CLAIM-TO-EVIDENCE "
        "VALIDATION SUMMARY"
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
            "Stage 9D claim-to-evidence "
            "validation failed."
        )


if __name__ == "__main__":
    main()