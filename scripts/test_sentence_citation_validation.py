from __future__ import annotations

import json
from typing import Any

from src.rag.sentence_citation_validation import (
    validate_sentence_citations,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_9b_sentence_citation_validation.json"
)


TEST_CASES = [
    {
        "case_id": "fully_cited_answer",
        "answer": (
            "PR #5336 was opened by dannyi96 [E1]. "
            "The PR title is 'type hint fix for "
            "flask.send_file' [E1]. "
            "It was merged on 2023-11-15 [E2]."
        ),
        "available_evidence_ids": {
            "E1",
            "E2",
        },
        "insufficient_evidence": False,
        "expected_passed": True,
        "expected_coverage": 1.0,
        "expected_uncited_count": 0,
    },
    {
        "case_id": "one_uncited_sentence",
        "answer": (
            "PR #5336 was opened by dannyi96 [E1]. "
            "The PR title is 'type hint fix for "
            "flask.send_file' [E1]. "
            "It has been closed and merged on "
            "2023-11-15."
        ),
        "available_evidence_ids": {
            "E1",
            "E2",
        },
        "insufficient_evidence": False,
        "expected_passed": False,
        "expected_coverage": 0.666667,
        "expected_uncited_count": 1,
    },
    {
        "case_id": "prediction_with_citation",
        "answer": (
            "The predicted merge probability is "
            "47.01% [E4]. "
            "This prediction is not a confirmed "
            "outcome [E4]."
        ),
        "available_evidence_ids": {
            "E4",
        },
        "insufficient_evidence": False,
        "expected_passed": True,
        "expected_coverage": 1.0,
        "expected_uncited_count": 0,
    },
    {
        "case_id": "invalid_citation",
        "answer": (
            "PR #5336 was opened by dannyi96 [E9]."
        ),
        "available_evidence_ids": {
            "E1",
            "E2",
        },
        "insufficient_evidence": False,
        "expected_passed": False,
        "expected_coverage": 0.0,
        "expected_uncited_count": 1,
        "expected_invalid_citation": "E9",
    },
    {
        "case_id": "out_of_domain_abstention",
        "answer": (
            "This question is outside the scope of "
            "the GitHub PR Intelligence system. "
            "Please ask about pull requests, merge "
            "outcomes, review priority, policy risk, "
            "repository activity or PR evidence."
        ),
        "available_evidence_ids": set(),
        "insufficient_evidence": True,
        "expected_passed": True,
        "expected_coverage": 1.0,
        "expected_uncited_count": 0,
    },
    {
        "case_id": "insufficient_evidence_answer",
        "answer": (
            "The available evidence is insufficient "
            "to determine whether the PR will be "
            "approved."
        ),
        "available_evidence_ids": {
            "E1",
        },
        "insufficient_evidence": True,
        "expected_passed": True,
        "expected_coverage": 1.0,
        "expected_uncited_count": 0,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_sentence_citations(
        answer=test_case["answer"],
        available_evidence_ids=(
            test_case[
                "available_evidence_ids"
            ]
        ),
        insufficient_evidence=(
            test_case[
                "insufficient_evidence"
            ]
        ),
    )

    checks = {
        "passed": (
            validation.passed
            == test_case["expected_passed"]
        ),
        "coverage": (
            validation.citation_coverage
            == test_case["expected_coverage"]
        ),
        "uncited_count": (
            validation
            .uncited_factual_sentence_count
            == test_case[
                "expected_uncited_count"
            ]
        ),
    }

    expected_invalid_citation = (
        test_case.get(
            "expected_invalid_citation"
        )
    )

    if expected_invalid_citation:
        checks["invalid_citation"] = (
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
        "VALIDATION PASSED: "
        f"{validation['passed']}"
    )
    print(
        "SENTENCE COUNT: "
        f"{validation['sentence_count']}"
    )
    print(
        "FACTUAL SENTENCES: "
        f"{validation['factual_sentence_count']}"
    )
    print(
        "CITED FACTUAL SENTENCES: "
        f"{validation['cited_factual_sentence_count']}"
    )
    print(
        "UNCITED FACTUAL SENTENCES: "
        f"{validation['uncited_factual_sentence_count']}"
    )
    print(
        "CITATION COVERAGE: "
        f"{validation['citation_coverage']:.4f}"
    )
    print(
        "INVALID CITATIONS: "
        f"{validation['invalid_citations']}"
    )
    print(f"CHECKS: {result['checks']}")

    for sentence_result in (
        validation["sentence_results"]
    ):
        print(
            f"\nSentence "
            f"{sentence_result['sentence_index']}: "
            f"{sentence_result['sentence']}"
        )
        print(
            "  Requires citation: "
            f"{sentence_result['requires_citation']}"
        )
        print(
            "  Valid citations: "
            f"{sentence_result['valid_citations']}"
        )
        print(
            "  Invalid citations: "
            f"{sentence_result['invalid_citations']}"
        )
        print(
            "  Passed: "
            f"{sentence_result['passed']}"
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
        len(results) - passed_count
    )

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    report = {
        "stage": "9B",
        "stage_name": (
            "Sentence-Level Citation Validation"
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
        "STAGE 9B SENTENCE CITATION "
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
            "Stage 9B sentence citation "
            "validation failed."
        )


if __name__ == "__main__":
    main()