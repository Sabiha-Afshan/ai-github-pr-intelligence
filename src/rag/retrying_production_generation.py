from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import requests

from src.rag.grounded_generation import (
    GroundedGenerationResponse,
    GroundedResponseGenerator,
)
from src.rag.production_grounded_generation import (
    ProductionGroundedResponse,
    ProductionGroundedResponseGenerator,
)


CITATION_PATTERN = re.compile(
    r"\[(E\d+)\]",
    flags=re.IGNORECASE,
)

DEFAULT_OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)


@dataclass(frozen=True)
class CitationRepairResult:
    attempted: bool
    succeeded: bool
    original_answer: str
    repaired_answer: str | None
    model: str | None
    latency_ms: float
    error: str | None
    raw_response: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "original_answer": self.original_answer,
            "repaired_answer": self.repaired_answer,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class RetryingProductionResponse:
    query: str
    action: str
    answer: str
    answer_released: bool
    generation_executed: bool
    repair_attempted: bool
    repair_succeeded: bool
    initial_response: ProductionGroundedResponse
    final_response: ProductionGroundedResponse
    repair_result: CitationRepairResult
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "action": self.action,
            "answer": self.answer,
            "answer_released": self.answer_released,
            "generation_executed": self.generation_executed,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "initial_response": (
                self.initial_response.to_dict()
            ),
            "final_response": (
                self.final_response.to_dict()
            ),
            "repair_result": (
                self.repair_result.to_dict()
            ),
            "trace": self.trace,
        }


class FixedResponseGenerator:
    """
    Small adapter used to send an already-generated response back through
    ProductionGroundedResponseGenerator after citation repair.
    """

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


def _extract_evidence_ids(
    answer: str,
) -> list[str]:
    evidence_ids: list[str] = []

    for match in CITATION_PATTERN.findall(
        answer
    ):
        evidence_id = match.upper()

        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)

    return evidence_ids


def _build_evidence_text(
    response: GroundedGenerationResponse,
) -> str:
    sections: list[str] = []

    for evidence in response.evidence:
        sections.append(
            "\n".join(
                [
                    f"[{evidence.evidence_id}]",
                    (
                        "PR number: "
                        f"{evidence.pr_number}"
                    ),
                    (
                        "Section: "
                        f"{evidence.section}"
                    ),
                    f"Evidence: {evidence.text}",
                    (
                        "Metadata: "
                        f"{evidence.metadata}"
                    ),
                ]
            )
        )

    return "\n\n".join(sections)


def _build_repair_prompt(
    query: str,
    answer: str,
    evidence_text: str,
) -> str:
    return f"""
You are repairing citation placement in a GitHub pull-request intelligence
answer.

USER QUERY:
{query}

ORIGINAL ANSWER:
{answer}

AVAILABLE EVIDENCE:
{evidence_text}

STRICT REPAIR RULES:

1. Preserve the meaning and factual content of the original answer.
2. Do not introduce new facts, numbers, dates, PR identifiers or conclusions.
3. Every factual sentence must end with one or more applicable evidence
   citations such as [E1] or [E2].
4. Place citations before the final sentence punctuation.
5. Use only evidence identifiers shown in AVAILABLE EVIDENCE.
6. Attach a citation only when that evidence directly supports the sentence.
7. Remove a sentence when no available evidence directly supports it.
8. Do not write explanations, headings, JSON, markdown fences or notes.
9. Return only the repaired answer.
""".strip()


def repair_citations_with_ollama(
    query: str,
    base_response: GroundedGenerationResponse,
    model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    request_timeout_seconds: int = 240,
) -> CitationRepairResult:
    evidence_text = _build_evidence_text(
        base_response
    )

    prompt = _build_repair_prompt(
        query=query,
        answer=base_response.answer,
        evidence_text=evidence_text,
    )

    request_payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You perform conservative citation "
                    "repair. You never invent facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": 0.0,
        },
    }

    start_time = time.perf_counter()

    try:
        http_response = requests.post(
            ollama_url,
            json=request_payload,
            timeout=request_timeout_seconds,
        )

        http_response.raise_for_status()

        raw_response = http_response.json()

        repaired_answer = (
            raw_response
            .get("message", {})
            .get("content", "")
            .strip()
        )

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            3,
        )

        if not repaired_answer:
            return CitationRepairResult(
                attempted=True,
                succeeded=False,
                original_answer=(
                    base_response.answer
                ),
                repaired_answer=None,
                model=model,
                latency_ms=latency_ms,
                error=(
                    "Ollama returned an empty "
                    "citation-repair answer."
                ),
                raw_response=raw_response,
            )

        return CitationRepairResult(
            attempted=True,
            succeeded=True,
            original_answer=(
                base_response.answer
            ),
            repaired_answer=repaired_answer,
            model=model,
            latency_ms=latency_ms,
            error=None,
            raw_response=raw_response,
        )

    except Exception as error:
        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            3,
        )

        return CitationRepairResult(
            attempted=True,
            succeeded=False,
            original_answer=(
                base_response.answer
            ),
            repaired_answer=None,
            model=model,
            latency_ms=latency_ms,
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
            raw_response=None,
        )


def _not_attempted_repair(
    original_answer: str,
) -> CitationRepairResult:
    return CitationRepairResult(
        attempted=False,
        succeeded=False,
        original_answer=original_answer,
        repaired_answer=None,
        model=None,
        latency_ms=0.0,
        error=None,
        raw_response=None,
    )


class RetryingProductionGroundedGenerator:
    """
    Production pipeline with one conservative citation-repair retry.

    Repair is attempted only when the original output fails sentence-level
    citation validation. Groundedness failures remain blocked without repair.
    """

    def __init__(
        self,
        base_generator: GroundedResponseGenerator | None = None,
        minimum_support_score: float = 0.60,
        minimum_token_coverage: float = 0.45,
        repair_model: str = "qwen2.5-coder:3b",
        ollama_url: str = DEFAULT_OLLAMA_URL,
        request_timeout_seconds: int = 240,
        repair_function: Callable[
            [
                str,
                GroundedGenerationResponse,
                str,
                str,
                int,
            ],
            CitationRepairResult,
        ] = repair_citations_with_ollama,
    ) -> None:
        self.base_generator = (
            base_generator
            if base_generator is not None
            else GroundedResponseGenerator(
                model=repair_model
            )
        )

        self.minimum_support_score = (
            minimum_support_score
        )

        self.minimum_token_coverage = (
            minimum_token_coverage
        )

        self.repair_model = repair_model
        self.ollama_url = ollama_url

        self.request_timeout_seconds = (
            request_timeout_seconds
        )

        self.repair_function = (
            repair_function
        )

    def _build_production_generator(
        self,
        base_generator: Any,
    ) -> ProductionGroundedResponseGenerator:
        return ProductionGroundedResponseGenerator(
            base_generator=base_generator,
            minimum_support_score=(
                self.minimum_support_score
            ),
            minimum_token_coverage=(
                self.minimum_token_coverage
            ),
        )

    def generate(
        self,
        query: str,
    ) -> RetryingProductionResponse:
        initial_generator = (
            self._build_production_generator(
                base_generator=(
                    self.base_generator
                )
            )
        )

        initial_response = (
            initial_generator.generate(
                query=query
            )
        )

        if (
            initial_response.action
            != "abstain_citation_validation"
        ):
            repair_result = (
                _not_attempted_repair(
                    original_answer=(
                        initial_response
                        .base_response
                        .answer
                    )
                )
            )

            return RetryingProductionResponse(
                query=query,
                action=initial_response.action,
                answer=initial_response.answer,
                answer_released=(
                    initial_response
                    .answer_released
                ),
                generation_executed=(
                    initial_response
                    .generation_executed
                ),
                repair_attempted=False,
                repair_succeeded=False,
                initial_response=(
                    initial_response
                ),
                final_response=initial_response,
                repair_result=repair_result,
                trace={
                    "initial_action": (
                        initial_response.action
                    ),
                    "repair_attempted": False,
                    "repair_succeeded": False,
                    "final_action": (
                        initial_response.action
                    ),
                    "reason": (
                        "Citation repair was not "
                        "applicable."
                    ),
                },
            )

        repair_result = self.repair_function(
            query,
            initial_response.base_response,
            self.repair_model,
            self.ollama_url,
            self.request_timeout_seconds,
        )

        if (
            not repair_result.succeeded
            or not repair_result.repaired_answer
        ):
            return RetryingProductionResponse(
                query=query,
                action=initial_response.action,
                answer=initial_response.answer,
                answer_released=False,
                generation_executed=True,
                repair_attempted=True,
                repair_succeeded=False,
                initial_response=(
                    initial_response
                ),
                final_response=initial_response,
                repair_result=repair_result,
                trace={
                    "initial_action": (
                        initial_response.action
                    ),
                    "repair_attempted": True,
                    "repair_succeeded": False,
                    "final_action": (
                        initial_response.action
                    ),
                    "reason": (
                        "Citation repair failed. "
                        "The original answer remained "
                        "withheld."
                    ),
                },
            )

        repaired_evidence_ids = (
            _extract_evidence_ids(
                repair_result.repaired_answer
            )
        )

        repaired_base_response = replace(
            initial_response.base_response,
            answer=(
                repair_result.repaired_answer
            ),
            evidence_used=(
                repaired_evidence_ids
            ),
            trace={
                **initial_response
                .base_response
                .trace,
                "citation_repair_attempted": True,
                "citation_repair_succeeded": True,
                "original_answer": (
                    initial_response
                    .base_response
                    .answer
                ),
                "repaired_answer": (
                    repair_result.repaired_answer
                ),
            },
        )

        repaired_generator = (
            self._build_production_generator(
                base_generator=(
                    FixedResponseGenerator(
                        repaired_base_response
                    )
                )
            )
        )

        final_response = (
            repaired_generator.generate(
                query=query
            )
        )

        repair_succeeded = bool(
            final_response.action == "answer"
            and final_response.answer_released
            and final_response
            .sentence_validation
            .passed
            and final_response
            .claim_validation
            .passed
        )

        return RetryingProductionResponse(
            query=query,
            action=final_response.action,
            answer=final_response.answer,
            answer_released=(
                final_response.answer_released
            ),
            generation_executed=True,
            repair_attempted=True,
            repair_succeeded=(
                repair_succeeded
            ),
            initial_response=initial_response,
            final_response=final_response,
            repair_result=repair_result,
            trace={
                "initial_action": (
                    initial_response.action
                ),
                "repair_attempted": True,
                "repair_model_succeeded": (
                    repair_result.succeeded
                ),
                "repair_succeeded": (
                    repair_succeeded
                ),
                "final_action": (
                    final_response.action
                ),
                "final_answer_released": (
                    final_response
                    .answer_released
                ),
                "final_citation_coverage": (
                    final_response
                    .sentence_validation
                    .citation_coverage
                ),
                "final_groundedness_rate": (
                    final_response
                    .claim_validation
                    .groundedness_rate
                ),
                "reason": (
                    "The repaired answer was released "
                    "only after citation and groundedness "
                    "validation."
                    if repair_succeeded
                    else (
                        "The repair output failed one or "
                        "more governance checks and "
                        "remained withheld."
                    )
                ),
            },
        )