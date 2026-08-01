from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rag.grounded_generation import (
    GroundedGenerationResponse,
    GroundedResponseGenerator,
)
from src.rag.sentence_citation_validation import (
    SentenceCitationValidation,
    validate_sentence_citations,
)


CITATION_GOVERNANCE_MESSAGE = (
    "The system retrieved relevant pull-request evidence, but the generated "
    "answer did not meet sentence-level citation requirements. The answer "
    "was withheld to prevent unsupported or insufficiently cited claims."
)


@dataclass(frozen=True)
class StrictGroundedGenerationResponse:
    query: str
    action: str
    message: str
    answer: str
    generation_executed: bool
    answer_released: bool
    insufficient_evidence: bool
    model: str | None
    evidence_used: list[str]
    evidence_count: int
    sentence_validation: SentenceCitationValidation
    base_response: GroundedGenerationResponse
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "action": self.action,
            "message": self.message,
            "answer": self.answer,
            "generation_executed": self.generation_executed,
            "answer_released": self.answer_released,
            "insufficient_evidence": self.insufficient_evidence,
            "model": self.model,
            "evidence_used": self.evidence_used,
            "evidence_count": self.evidence_count,
            "sentence_validation": (
                self.sentence_validation.to_dict()
            ),
            "base_response": (
                self.base_response.to_dict()
            ),
            "trace": self.trace,
        }


def _empty_sentence_validation() -> SentenceCitationValidation:
    return SentenceCitationValidation(
        sentence_count=0,
        factual_sentence_count=0,
        cited_factual_sentence_count=0,
        uncited_factual_sentence_count=0,
        citation_coverage=1.0,
        invalid_citations=[],
        passed=True,
        sentence_results=[],
    )


class StrictGroundedResponseGenerator:
    """
    Production-facing wrapper around GroundedResponseGenerator.

    Generated answers are released only when every factual sentence contains
    at least one valid evidence citation and no invalid citation identifiers
    are present.
    """

    def __init__(
        self,
        base_generator: GroundedResponseGenerator | None = None,
    ) -> None:
        self.base_generator = (
            base_generator
            if base_generator is not None
            else GroundedResponseGenerator()
        )

    def generate(
        self,
        query: str,
    ) -> StrictGroundedGenerationResponse:
        base_response = self.base_generator.generate(
            query=query
        )

        if not base_response.generation_executed:
            return StrictGroundedGenerationResponse(
                query=query,
                action=base_response.action,
                message=base_response.message,
                answer=base_response.answer,
                generation_executed=False,
                answer_released=True,
                insufficient_evidence=(
                    base_response.insufficient_evidence
                ),
                model=base_response.model,
                evidence_used=base_response.evidence_used,
                evidence_count=len(
                    base_response.evidence
                ),
                sentence_validation=(
                    _empty_sentence_validation()
                ),
                base_response=base_response,
                trace={
                    "base_action": base_response.action,
                    "strict_action": (
                        base_response.action
                    ),
                    "generation_executed": False,
                    "sentence_validation_executed": False,
                    "answer_released": True,
                    "reason": (
                        "The governed retrieval layer "
                        "abstained before LLM generation."
                    ),
                },
            )

        available_evidence_ids = {
            evidence.evidence_id
            for evidence in base_response.evidence
        }

        sentence_validation = (
            validate_sentence_citations(
                answer=base_response.answer,
                available_evidence_ids=(
                    available_evidence_ids
                ),
                insufficient_evidence=(
                    base_response
                    .insufficient_evidence
                ),
            )
        )

        if not sentence_validation.passed:
            return StrictGroundedGenerationResponse(
                query=query,
                action=(
                    "abstain_citation_validation"
                ),
                message=CITATION_GOVERNANCE_MESSAGE,
                answer=CITATION_GOVERNANCE_MESSAGE,
                generation_executed=True,
                answer_released=False,
                insufficient_evidence=True,
                model=base_response.model,
                evidence_used=[],
                evidence_count=len(
                    base_response.evidence
                ),
                sentence_validation=(
                    sentence_validation
                ),
                base_response=base_response,
                trace={
                    "base_action": base_response.action,
                    "strict_action": (
                        "abstain_citation_validation"
                    ),
                    "generation_executed": True,
                    "sentence_validation_executed": True,
                    "answer_released": False,
                    "sentence_count": (
                        sentence_validation
                        .sentence_count
                    ),
                    "factual_sentence_count": (
                        sentence_validation
                        .factual_sentence_count
                    ),
                    "cited_factual_sentence_count": (
                        sentence_validation
                        .cited_factual_sentence_count
                    ),
                    "uncited_factual_sentence_count": (
                        sentence_validation
                        .uncited_factual_sentence_count
                    ),
                    "citation_coverage": (
                        sentence_validation
                        .citation_coverage
                    ),
                    "invalid_citations": (
                        sentence_validation
                        .invalid_citations
                    ),
                    "withheld_answer": (
                        base_response.answer
                    ),
                    "reason": (
                        "The generated answer failed "
                        "sentence-level citation "
                        "governance."
                    ),
                },
            )

        return StrictGroundedGenerationResponse(
            query=query,
            action="answer",
            message=(
                "A grounded response passed "
                "sentence-level citation governance."
            ),
            answer=base_response.answer,
            generation_executed=True,
            answer_released=True,
            insufficient_evidence=(
                base_response.insufficient_evidence
            ),
            model=base_response.model,
            evidence_used=(
                base_response.evidence_used
            ),
            evidence_count=len(
                base_response.evidence
            ),
            sentence_validation=(
                sentence_validation
            ),
            base_response=base_response,
            trace={
                "base_action": base_response.action,
                "strict_action": "answer",
                "generation_executed": True,
                "sentence_validation_executed": True,
                "answer_released": True,
                "sentence_count": (
                    sentence_validation
                    .sentence_count
                ),
                "factual_sentence_count": (
                    sentence_validation
                    .factual_sentence_count
                ),
                "cited_factual_sentence_count": (
                    sentence_validation
                    .cited_factual_sentence_count
                ),
                "uncited_factual_sentence_count": (
                    sentence_validation
                    .uncited_factual_sentence_count
                ),
                "citation_coverage": (
                    sentence_validation
                    .citation_coverage
                ),
                "invalid_citations": (
                    sentence_validation
                    .invalid_citations
                ),
                "reason": (
                    "Every factual sentence contained "
                    "at least one valid evidence "
                    "citation."
                ),
            },
        )