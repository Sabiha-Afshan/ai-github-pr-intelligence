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
    / "stage_10g_pr_identity_semantic_normalisation.json"
)


TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "opened_matches_created",
        "answer": "PR #5336 was opened by dannyi96 [E1].",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5336,
                section="Pull request identity",
                text="PR #5336 was created by dannyi96.",
                metadata={"author": "dannyi96"},
            )
        ],
        "expected_passed": True,
    },
    {
        "case_id": "authored_matches_created",
        "answer": "PR #5336 was authored by dannyi96 [E1].",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5336,
                section="Pull request identity",
                text="PR #5336 was created by dannyi96.",
                metadata={"author": "dannyi96"},
            )
        ],
        "expected_passed": True,
    },
    {
        "case_id": "submitted_matches_created",
        "answer": "PR #5336 was submitted by dannyi96 [E1].",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5336,
                section="Pull request identity",
                text="PR #5336 was created by dannyi96.",
                metadata={"author": "dannyi96"},
            )
        ],
        "expected_passed": True,
    },
    {
        "case_id": "wrong_author_remains_blocked",
        "answer": "PR #5336 was opened by incorrect_user [E1].",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5336,
                section="Pull request identity",
                text="PR #5336 was created by dannyi96.",
                metadata={"author": "dannyi96"},
            )
        ],
        "expected_passed": False,
    },
    {
        "case_id": "wrong_pr_number_remains_blocked",
        "answer": "PR #9999 was opened by dannyi96 [E1].",
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5336,
                section="Pull request identity",
                text="PR #5336 was created by dannyi96.",
                metadata={"author": "dannyi96"},
            )
        ],
        "expected_passed": False,
    },
    {
        "case_id": "probability_alias_regression",
        "answer": (
            "PR #4897 has a low chance of being merged "
            "with a predicted probability of 12.68% [E1]."
        ),
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=4897,
                section="Predictive intelligence",
                text=(
                    "PR #4897 has a low merge probability "
                    "of 12.68%."
                ),
                metadata={"merge_probability": "12.68%"},
            )
        ],
        "expected_passed": True,
    },
    {
        "case_id": "manual_review_alias_regression",
        "answer": (
            "PR #5736 requires a human governance review [E1]."
        ),
        "evidence": [
            EvidenceRecord(
                evidence_id="E1",
                pr_number=5736,
                section="Policy intelligence",
                text="PR #5736 has manual review required.",
                metadata={"manual_review_required": True},
            )
        ],
        "expected_passed": True,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_claim_evidence(
        answer=test_case["answer"],
        evidence=test_case["evidence"],
        insufficient_evidence=False,
        minimum_support_score=0.60,
        minimum_token_coverage=0.45,
    )

    return {
        "case_id": test_case["case_id"],
        "passed": (
            validation.passed
            == test_case["expected_passed"]
        ),
        "expected_validation_passed": (
            test_case["expected_passed"]
        ),
        "actual_validation_passed": (
            validation.passed
        ),
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
        "EXPECTED VALIDATION: "
        f"{result['expected_validation_passed']}"
    )
    print(
        "ACTUAL VALIDATION: "
        f"{result['actual_validation_passed']}"
    )
    print(
        "GROUNDEDNESS RATE: "
        f"{validation['groundedness_rate']:.4f}"
    )
    print(
        "UNSUPPORTED CLAIMS: "
        f"{validation['unsupported_claim_count']}"
    )

    for claim_result in validation["claim_results"]:
        print(
            "CLAIM TOKENS: "
            f"{claim_result['claim_tokens']}"
        )
        print(
            "SUPPORTED TOKENS: "
            f"{claim_result['supported_tokens']}"
        )
        print(
            "UNSUPPORTED TOKENS: "
            f"{claim_result['unsupported_tokens']}"
        )
        print(
            "FAILURE REASONS: "
            f"{claim_result['failure_reasons']}"
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

    failed_count = len(results) - passed_count

    status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    report = {
        "stage": "10G",
        "stage_name": (
            "PR Identity Semantic Normalisation"
        ),
        "status": status,
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
        "STAGE 10G PR IDENTITY SEMANTIC "
        "NORMALISATION SUMMARY"
    )
    print("=" * 90)
    print(f"Status: {status.upper()}")
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {REPORT_PATH}")

    if status != "passed":
        raise RuntimeError(
            "Stage 10G PR identity semantic "
            "normalisation validation failed."
        )


if __name__ == "__main__":
    main()