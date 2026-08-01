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
    / "stage_10b_semantic_groundedness.json"
)


EVIDENCE = [
    EvidenceRecord(
        evidence_id="E1",
        pr_number=4897,
        section="Predictive intelligence",
        text=(
            "PR #4897 has a low predicted merge probability. "
            "The merge probability is 22.40%."
        ),
        metadata={
            "merge_probability": 0.224,
        },
    ),
    EvidenceRecord(
        evidence_id="E2",
        pr_number=5736,
        section="Deterministic policy intelligence",
        text=(
            "PR #5736 requires manual governance review. "
            "The manual_review_required field is True."
        ),
        metadata={
            "manual_review_required": True,
        },
    ),
    EvidenceRecord(
        evidence_id="E3",
        pr_number=3073,
        section="Unified review priority",
        text=(
            "PR #3073 has High review priority. "
            "Recommended action: review this pull request promptly."
        ),
        metadata={
            "review_priority": "High",
        },
    ),
]


TEST_CASES = [
    {
        "case_id": "chance_equals_probability",
        "answer": (
            "PR #4897 has a low chance of being merged [E1]."
        ),
        "expected_passed": True,
    },
    {
        "case_id": "accepted_equals_merged",
        "answer": (
            "PR #4897 is unlikely to be accepted [E1]."
        ),
        "expected_passed": True,
    },
    {
        "case_id": "human_review_equals_manual_review",
        "answer": (
            "PR #5736 requires a human governance review [E2]."
        ),
        "expected_passed": True,
    },
    {
        "case_id": "review_first_equals_priority",
        "answer": (
            "Maintainers should review PR #3073 first [E3]."
        ),
        "expected_passed": True,
    },
    {
        "case_id": "wrong_percentage_still_blocked",
        "answer": (
            "PR #4897 has a 78.50% chance of being merged [E1]."
        ),
        "expected_passed": False,
        "expected_failed_entity": "percentages",
    },
    {
        "case_id": "unsupported_ranking_score_still_blocked",
        "answer": (
            "PR #3073 has a governed ranking score of "
            "0.677534 [E3]."
        ),
        "expected_passed": False,
        "expected_failed_entity": "decimals",
    },
    {
        "case_id": "wrong_pr_number_still_blocked",
        "answer": (
            "PR #9999 requires a human governance review [E2]."
        ),
        "expected_passed": False,
        "expected_failed_entity": "pr_numbers",
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_claim_evidence(
        answer=test_case["answer"],
        evidence=EVIDENCE,
        minimum_support_score=0.60,
        minimum_token_coverage=0.45,
    )

    checks: dict[str, bool] = {
        "passed": (
            validation.passed
            == test_case["expected_passed"]
        )
    }

    expected_failed_entity = (
        test_case.get(
            "expected_failed_entity"
        )
    )

    if expected_failed_entity:
        detected = False

        for claim_result in (
            validation.claim_results
        ):
            entity_check = (
                claim_result.entity_checks.get(
                    expected_failed_entity
                )
            )

            if (
                entity_check
                and entity_check["applicable"]
                and not entity_check["passed"]
            ):
                detected = True
                break

        checks["failed_entity_detected"] = (
            detected
        )

    return {
        "case_id": test_case["case_id"],
        "answer": test_case["answer"],
        "passed": all(checks.values()),
        "checks": checks,
        "validation": validation.to_dict(),
    }


def print_result(
    result: dict[str, Any],
) -> None:
    validation = result["validation"]

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"ANSWER: {result['answer']}")
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(
        "GROUNDEDNESS VALIDATION: "
        f"{validation['passed']}"
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
        "UNSUPPORTED CLAIMS: "
        f"{validation['unsupported_claim_count']}"
    )
    print(f"CHECKS: {result['checks']}")

    for claim in validation["claim_results"]:
        print(
            f"\nClaim tokens: "
            f"{claim['claim_tokens']}"
        )
        print(
            f"Supported tokens: "
            f"{claim['supported_tokens']}"
        )
        print(
            f"Unsupported tokens: "
            f"{claim['unsupported_tokens']}"
        )
        print(
            f"Token coverage: "
            f"{claim['token_coverage']:.4f}"
        )
        print(
            f"Entity coverage: "
            f"{claim['entity_coverage']:.4f}"
        )

        if claim["failure_reasons"]:
            print("Failure reasons:")

            for reason in claim[
                "failure_reasons"
            ]:
                print(f"- {reason}")


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
        "stage": "10B",
        "stage_name": (
            "Controlled Semantic Groundedness"
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
        "STAGE 10B SEMANTIC GROUNDEDNESS SUMMARY"
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
            "Stage 10B semantic groundedness "
            "validation failed."
        )


if __name__ == "__main__":
    main()