from __future__ import annotations

import json
from typing import Any

from src.rag.grounded_generation import (
    GroundedResponseGenerator,
)
from src.utils.paths import PROJECT_ROOT


REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "stage_9a_grounded_generation_diagnostics.json"
)


TEST_CASES = [
    {
        "case_id": "exact_pr_grounded_answer",
        "query": (
            "What do we know about pull request 5336?"
        ),
        "expected_action": "answer",
        "generation_expected": True,
        "citations_expected": True,
    },
    {
        "case_id": "prediction_grounded_answer",
        "query": (
            "Which pull requests have a low "
            "chance of being merged?"
        ),
        "expected_action": "answer",
        "generation_expected": True,
        "citations_expected": True,
    },
    {
        "case_id": "policy_grounded_answer",
        "query": (
            "Which changes require a human "
            "governance review?"
        ),
        "expected_action": "answer",
        "generation_expected": True,
        "citations_expected": True,
    },
    {
        "case_id": "out_of_domain_abstention",
        "query": (
            "What will the weather be in Dubai tomorrow?"
        ),
        "expected_action": (
            "abstain_out_of_domain"
        ),
        "generation_expected": False,
        "citations_expected": False,
    },
]


def evaluate_case(
    generator: GroundedResponseGenerator,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    response = generator.generate(
        query=test_case["query"]
    )

    checks: dict[str, bool] = {
        "action": (
            response.action
            == test_case["expected_action"]
        ),
        "generation_execution": (
            response.generation_executed
            == test_case["generation_expected"]
        ),
        "citation_validation": (
            response.citation_validation.passed
        ),
    }

    if test_case["citations_expected"]:
        checks["citations_present"] = (
            len(
                response
                .citation_validation
                .valid_citations
            )
            > 0
        )

        checks["evidence_available"] = (
            len(response.evidence) > 0
        )

        checks["model_recorded"] = (
            response.model is not None
        )

        checks["answer_not_empty"] = bool(
            response.answer.strip()
        )
    else:
        checks["no_generation"] = (
            response.generation_executed
            is False
        )

        checks["no_evidence"] = (
            len(response.evidence) == 0
        )

        checks["no_model"] = (
            response.model is None
        )

    passed = all(checks.values())

    return {
        "case_id": test_case["case_id"],
        "query": test_case["query"],
        "passed": passed,
        "checks": checks,
        "response": response.to_dict(),
    }


def print_result(
    result: dict[str, Any],
) -> None:
    response = result["response"]

    print("\n" + "=" * 90)
    print(f"CASE: {result['case_id']}")
    print(f"QUERY: {result['query']}")
    print(
        "STATUS: "
        f"{'PASSED' if result['passed'] else 'FAILED'}"
    )
    print(f"ACTION: {response['action']}")
    print(
        "GENERATION EXECUTED: "
        f"{response['generation_executed']}"
    )
    print(
        f"MODEL: {response['model']}"
    )
    print(
        "CITATION VALIDATION: "
        f"{response['citation_validation']['passed']}"
    )
    print(
        "VALID CITATIONS: "
        f"{response['citation_validation']['valid_citations']}"
    )
    print(
        "INVALID CITATIONS: "
        f"{response['citation_validation']['invalid_citations']}"
    )
    print(
        f"EVIDENCE COUNT: "
        f"{len(response['evidence'])}"
    )
    print(
        "GENERATION LATENCY: "
        f"{response['generation_latency_ms']:.2f} ms"
    )
    print(
        "PROMPT TOKENS: "
        f"{response['prompt_eval_count']}"
    )
    print(
        "OUTPUT TOKENS: "
        f"{response['eval_count']}"
    )
    print(f"CHECKS: {result['checks']}")
    print("\nANSWER:")
    print(response["answer"])

    if response["limitations"]:
        print("\nLIMITATIONS:")

        for limitation in response["limitations"]:
            print(f"- {limitation}")


def main() -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading Stage 9A grounded response generator..."
    )

    generator = GroundedResponseGenerator(
        model="qwen2.5-coder:3b",
        evidence_top_k=5,
        maximum_evidence_characters=2400,
        temperature=0.0,
    )

    results: list[dict[str, Any]] = []

    for test_case in TEST_CASES:
        try:
            result = evaluate_case(
                generator=generator,
                test_case=test_case,
            )
        except Exception as error:
            result = {
                "case_id": test_case["case_id"],
                "query": test_case["query"],
                "passed": False,
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
            print_result(result)
        else:
            print("\n" + "=" * 90)
            print(
                f"CASE: {result['case_id']}"
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

    overall_status = (
        "passed"
        if failed_count == 0
        else "failed"
    )

    report = {
        "stage": "9A",
        "stage_name": (
            "Grounded LLM Response Generation"
        ),
        "status": overall_status,
        "model": "qwen2.5-coder:3b",
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
        "STAGE 9A GROUNDED GENERATION SUMMARY"
    )
    print("=" * 90)
    print(
        f"Status: {overall_status.upper()}"
    )
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Model: qwen2.5-coder:3b")
    print(f"Report: {REPORT_PATH}")

    if overall_status != "passed":
        raise RuntimeError(
            "Stage 9A grounded generation "
            "validation failed."
        )


if __name__ == "__main__":
    main()