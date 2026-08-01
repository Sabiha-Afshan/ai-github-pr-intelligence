from __future__ import annotations

import json
from typing import Any

from src.rag.governed_retrieval import (
    GovernedRetriever,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_8f_governed_retrieval_diagnostics.json"
)


TEST_CASES = [
    {
        "case_id": "exact_pr_number",
        "query": (
            "Open the evidence associated with "
            "pull request number 5336."
        ),
        "expected_action": "retrieve",
        "expected_pr_number": 5336,
    },
    {
        "case_id": "prediction_paraphrase",
        "query": (
            "Which submissions appear unlikely "
            "to get accepted?"
        ),
        "expected_action": "retrieve",
        "expected_section": (
            "Predictive intelligence"
        ),
    },
    {
        "case_id": "policy_paraphrase",
        "query": (
            "Which changes should be escalated "
            "for a human governance check?"
        ),
        "expected_action": "retrieve",
        "expected_section": (
            "Deterministic policy intelligence"
        ),
    },
    {
        "case_id": "priority_paraphrase",
        "query": (
            "What should the maintainers "
            "look at first?"
        ),
        "expected_action": "retrieve",
        "expected_section": (
            "Unified review priority"
        ),
    },
    {
        "case_id": "identity_paraphrase",
        "query": (
            "Who opened the change, when was it "
            "opened, and what was its title?"
        ),
        "expected_action": "retrieve",
        "expected_section": "PR identity",
    },
    {
        "case_id": "critical_manual_review",
        "query": (
            "Show critical pull requests that "
            "must be reviewed manually."
        ),
        "expected_action": "retrieve",
        "expected_metadata_matches": 2,
    },
    {
        "case_id": "delay_condition",
        "query": (
            "Find merged changes expected to "
            "take longer than normal to complete."
        ),
        "expected_action": "retrieve",
        "expected_section": (
            "Predictive intelligence"
        ),
        "expected_metadata_matches": 1,
    },
    {
        "case_id": "weather_abstention",
        "query": (
            "What will the weather be in "
            "Dubai tomorrow?"
        ),
        "expected_action": (
            "abstain_out_of_domain"
        ),
        "expected_no_results": True,
    },
    {
        "case_id": "recipe_abstention",
        "query": (
            "Give me a recipe for chocolate cake."
        ),
        "expected_action": (
            "abstain_out_of_domain"
        ),
        "expected_no_results": True,
    },
    {
        "case_id": "flight_abstention",
        "query": (
            "Find the cheapest flight from "
            "India to London."
        ),
        "expected_action": (
            "abstain_out_of_domain"
        ),
        "expected_no_results": True,
    },
]


def evaluate_case(
    retriever: GovernedRetriever,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    response = retriever.retrieve(
        query=test_case["query"],
        top_k=5,
    )

    checks: dict[str, bool] = {
        "action": (
            response.action
            == test_case["expected_action"]
        )
    }

    if "expected_pr_number" in test_case:
        checks["pr_number"] = bool(
            response.results
            and response.results[0].pr_number
            == test_case["expected_pr_number"]
        )

    if "expected_section" in test_case:
        expected_section = (
            test_case["expected_section"]
        )

        checks["section"] = bool(
            response.results
            and response.results[0].section
            == expected_section
        )

    if "expected_metadata_matches" in test_case:
        required_matches = (
            test_case[
                "expected_metadata_matches"
            ]
        )

        checks["metadata_matches"] = bool(
            response.results
            and (
                response.results[0]
                .metadata_condition_matches
                >= required_matches
            )
        )

    if test_case.get(
        "expected_no_results",
        False,
    ):
        checks["no_results"] = (
            len(response.results) == 0
        )

        checks["retrieval_not_executed"] = (
            response.retrieval_executed
            is False
        )

    passed = all(checks.values())

    return {
        "case_id": test_case["case_id"],
        "query": test_case["query"],
        "passed": passed,
        "checks": checks,
        "response": response.to_dict(),
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Stage 8F governed retriever..."
    )

    retriever = GovernedRetriever(
        candidate_pool_size=50,
        minimum_governed_score=0.20,
    )

    results = [
        evaluate_case(
            retriever=retriever,
            test_case=test_case,
        )
        for test_case in TEST_CASES
    ]

    for result in results:
        response = result["response"]

        print("\n" + "=" * 90)
        print(f"CASE: {result['case_id']}")
        print(f"QUERY: {result['query']}")
        print(
            f"STATUS: "
            f"{'PASSED' if result['passed'] else 'FAILED'}"
        )
        print(
            f"ACTION: {response['action']}"
        )
        print(
            "RETRIEVAL EXECUTED: "
            f"{response['retrieval_executed']}"
        )
        print(
            f"MESSAGE: {response['message']}"
        )
        print(
            f"CHECKS: {result['checks']}"
        )

        if response["results"]:
            top_result = response["results"][0]

            print(
                f"TOP PR: "
                f"{top_result['pr_number']}"
            )
            print(
                f"TOP SECTION: "
                f"{top_result['section']}"
            )
            print(
                f"BASE SCORE: "
                f"{top_result['base_hybrid_score']}"
            )
            print(
                f"SECTION BOOST: "
                f"{top_result['section_boost']}"
            )
            print(
                f"METADATA BOOST: "
                f"{top_result['metadata_boost']}"
            )
            print(
                f"GOVERNED SCORE: "
                f"{top_result['governed_score']}"
            )
            print(
                "METADATA MATCHES: "
                f"{top_result['metadata_condition_matches']}"
                "/"
                f"{top_result['metadata_condition_total']}"
            )

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
        "stage": "8F",
        "component": (
            "Governed retrieval integration"
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
        "STAGE 8F GOVERNED RETRIEVAL SUMMARY"
    )
    print("=" * 90)
    print(f"Status: {overall_status.upper()}")
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {REPORT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 8F governed retrieval "
            "validation failed."
        )


if __name__ == "__main__":
    main()