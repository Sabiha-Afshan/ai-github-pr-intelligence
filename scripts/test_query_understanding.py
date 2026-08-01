from __future__ import annotations

import json
from typing import Any

from src.rag.query_understanding import (
    understand_query,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_8f_query_understanding_diagnostics.json"
)


TEST_CASES = [
    {
        "query": (
            "Open the evidence associated with "
            "pull request number 5336."
        ),
        "expected_pr_number": 5336,
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "Which submissions appear unlikely "
            "to get accepted?"
        ),
        "expected_section": (
            "Predictive intelligence"
        ),
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "Which changes should be escalated "
            "for a human governance check?"
        ),
        "expected_section": (
            "Deterministic policy intelligence"
        ),
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "What should the maintainers "
            "look at first?"
        ),
        "expected_section": (
            "Unified review priority"
        ),
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "Who opened the change, when was it "
            "opened, and what was its title?"
        ),
        "expected_section": "PR identity",
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "Find submissions where the author "
            "gave a useful explanation."
        ),
        "expected_section": "PR description",
        "expected_repository_query": True,
        "expected_out_of_domain": False,
    },
    {
        "query": (
            "Show critical pull requests that "
            "must be reviewed manually."
        ),
        "expected_repository_query": True,
        "expected_out_of_domain": False,
        "expected_condition_fields": {
            "policy_risk_band",
            "manual_review_required",
        },
    },
    {
        "query": (
            "Find merged changes expected to take "
            "longer than normal to complete."
        ),
        "expected_section": (
            "Predictive intelligence"
        ),
        "expected_repository_query": True,
        "expected_out_of_domain": False,
        "expected_condition_fields": {
            "delay_prediction",
        },
    },
    {
        "query": (
            "What will the weather be in "
            "Dubai tomorrow?"
        ),
        "expected_repository_query": False,
        "expected_out_of_domain": True,
    },
    {
        "query": (
            "Give me a recipe for chocolate cake."
        ),
        "expected_repository_query": False,
        "expected_out_of_domain": True,
    },
    {
        "query": (
            "Find the cheapest flight from "
            "India to London."
        ),
        "expected_repository_query": False,
        "expected_out_of_domain": True,
    },
]


def evaluate_case(
    test_case: dict[str, Any],
) -> dict[str, Any]:
    result = understand_query(
        test_case["query"]
    )

    checks: dict[str, bool] = {}

    if "expected_pr_number" in test_case:
        checks["pr_number"] = (
            result.pr_number
            == test_case["expected_pr_number"]
        )

    if "expected_section" in test_case:
        checks["section"] = (
            test_case["expected_section"]
            in result.detected_sections
        )

    if "expected_repository_query" in test_case:
        checks["repository_query"] = (
            result.is_repository_query
            == test_case[
                "expected_repository_query"
            ]
        )

    if "expected_out_of_domain" in test_case:
        checks["out_of_domain"] = (
            result.is_out_of_domain
            == test_case["expected_out_of_domain"]
        )

    if "expected_condition_fields" in test_case:
        actual_fields = {
            condition.field
            for condition in (
                result.metadata_conditions
            )
        }

        checks["metadata_conditions"] = (
            test_case["expected_condition_fields"]
            .issubset(actual_fields)
        )

    passed = all(checks.values())

    return {
        "query": test_case["query"],
        "passed": passed,
        "checks": checks,
        "understanding": result.to_dict(),
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = [
        evaluate_case(test_case)
        for test_case in TEST_CASES
    ]

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

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 90)
        print(f"CASE {index}")
        print(f"Query: {result['query']}")
        print(
            f"Status: "
            f"{'PASSED' if result['passed'] else 'FAILED'}"
        )
        print(
            "PR number: "
            f"{result['understanding']['pr_number']}"
        )
        print(
            "Repository query: "
            f"{result['understanding']['is_repository_query']}"
        )
        print(
            "Out of domain: "
            f"{result['understanding']['is_out_of_domain']}"
        )
        print(
            "Domain confidence: "
            f"{result['understanding']['domain_confidence']}"
        )
        print(
            "Detected sections: "
            f"{result['understanding']['detected_sections']}"
        )
        print(
            "Metadata conditions: "
            f"{result['understanding']['metadata_conditions']}"
        )
        print(
            "Expanded query: "
            f"{result['understanding']['expanded_query']}"
        )
        print(
            f"Checks: {result['checks']}"
        )

    report = {
        "stage": "8F",
        "component": "Query understanding",
        "status": overall_status,
        "test_case_count": len(results),
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
    print("STAGE 8F QUERY UNDERSTANDING SUMMARY")
    print("=" * 90)
    print(f"Status: {overall_status.upper()}")
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Report: {REPORT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 8F query-understanding "
            "validation failed."
        )


if __name__ == "__main__":
    main()