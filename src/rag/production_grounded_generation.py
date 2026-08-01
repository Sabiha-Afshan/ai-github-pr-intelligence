from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rag.claim_evidence_validation import (
    ClaimEvidenceValidation,
    EvidenceRecord,
    validate_claim_evidence,
)
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

GROUNDEDNESS_GOVERNANCE_MESSAGE = (
    "The system retrieved relevant pull-request evidence, but one or more "
    "generated claims were not sufficiently supported by the cited evidence. "
    "The answer was withheld to prevent unsupported claims."
)


@dataclass(frozen=True)
class ProductionGroundedResponse:
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
    claim_validation: ClaimEvidenceValidation
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
            "claim_validation": (
                self.claim_validation.to_dict()
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


def _empty_claim_validation() -> ClaimEvidenceValidation:
    return ClaimEvidenceValidation(
        sentence_count=0,
        factual_claim_count=0,
        supported_claim_count=0,
        unsupported_claim_count=0,
        groundedness_rate=1.0,
        mean_support_score=1.0,
        invalid_citations=[],
        passed=True,
        claim_results=[],
    )


def _build_evidence_records(
    base_response: GroundedGenerationResponse,
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []

    for evidence in base_response.evidence:
        records.append(
            EvidenceRecord(
                evidence_id=evidence.evidence_id,
                text=evidence.text,
                pr_number=evidence.pr_number,
                section=evidence.section,
                metadata=evidence.metadata,
            )
        )

    return records


class ProductionGroundedResponseGenerator:
    """
    Production-facing governed RAG response generator.

    An LLM answer is released only when:

    1. Every factual sentence contains a valid citation.
    2. Every factual claim is supported by the cited evidence.
    3. No invalid evidence identifiers are present.
    """

    def __init__(
        self,
        base_generator: GroundedResponseGenerator | None = None,
        minimum_support_score: float = 0.60,
        minimum_token_coverage: float = 0.45,
    ) -> None:
        self.base_generator = (
            base_generator
            if base_generator is not None
            else GroundedResponseGenerator()
        )

        self.minimum_support_score = (
            minimum_support_score
        )

        self.minimum_token_coverage = (
            minimum_token_coverage
        )

    def generate(
        self,
        query: str,
    ) -> ProductionGroundedResponse:
        base_response = self.base_generator.generate(
            query=query
        )

        if not base_response.generation_executed:
            return ProductionGroundedResponse(
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
                claim_validation=(
                    _empty_claim_validation()
                ),
                base_response=base_response,
                trace={
                    "base_action": base_response.action,
                    "production_action": (
                        base_response.action
                    ),
                    "generation_executed": False,
                    "sentence_validation_executed": False,
                    "claim_validation_executed": False,
                    "answer_released": True,
                    "reason": (
                        "Governed retrieval abstained "
                        "before LLM generation."
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
                    base_response.insufficient_evidence
                ),
            )
        )

        if not sentence_validation.passed:
            return ProductionGroundedResponse(
                query=query,
                action="abstain_citation_validation",
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
                claim_validation=(
                    _empty_claim_validation()
                ),
                base_response=base_response,
                trace={
                    "base_action": base_response.action,
                    "production_action": (
                        "abstain_citation_validation"
                    ),
                    "generation_executed": True,
                    "sentence_validation_executed": True,
                    "claim_validation_executed": False,
                    "answer_released": False,
                    "citation_coverage": (
                        sentence_validation
                        .citation_coverage
                    ),
                    "uncited_factual_sentence_count": (
                        sentence_validation
                        .uncited_factual_sentence_count
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
                        "sentence-level citation validation."
                    ),
                },
            )

        evidence_records = _build_evidence_records(
            base_response
        )

        claim_validation = validate_claim_evidence(
            answer=base_response.answer,
            evidence=evidence_records,
            insufficient_evidence=(
                base_response.insufficient_evidence
            ),
            minimum_support_score=(
                self.minimum_support_score
            ),
            minimum_token_coverage=(
                self.minimum_token_coverage
            ),
        )

        if not claim_validation.passed:
            return ProductionGroundedResponse(
                query=query,
                action="abstain_groundedness_validation",
                message=GROUNDEDNESS_GOVERNANCE_MESSAGE,
                answer=GROUNDEDNESS_GOVERNANCE_MESSAGE,
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
                claim_validation=(
                    claim_validation
                ),
                base_response=base_response,
                trace={
                    "base_action": base_response.action,
                    "production_action": (
                        "abstain_groundedness_validation"
                    ),
                    "generation_executed": True,
                    "sentence_validation_executed": True,
                    "claim_validation_executed": True,
                    "answer_released": False,
                    "citation_coverage": (
                        sentence_validation
                        .citation_coverage
                    ),
                    "groundedness_rate": (
                        claim_validation
                        .groundedness_rate
                    ),
                    "mean_support_score": (
                        claim_validation
                        .mean_support_score
                    ),
                    "unsupported_claim_count": (
                        claim_validation
                        .unsupported_claim_count
                    ),
                    "invalid_citations": (
                        claim_validation
                        .invalid_citations
                    ),
                    "withheld_answer": (
                        base_response.answer
                    ),
                    "reason": (
                        "One or more factual claims "
                        "were not supported by their "
                        "cited evidence."
                    ),
                },
            )

        return ProductionGroundedResponse(
            query=query,
            action="answer",
            message=(
                "The grounded response passed "
                "citation and claim-support governance."
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
            claim_validation=claim_validation,
            base_response=base_response,
            trace={
                "base_action": base_response.action,
                "production_action": "answer",
                "generation_executed": True,
                "sentence_validation_executed": True,
                "claim_validation_executed": True,
                "answer_released": True,
                "citation_coverage": (
                    sentence_validation
                    .citation_coverage
                ),
                "groundedness_rate": (
                    claim_validation
                    .groundedness_rate
                ),
                "mean_support_score": (
                    claim_validation
                    .mean_support_score
                ),
                "unsupported_claim_count": (
                    claim_validation
                    .unsupported_claim_count
                ),
                "invalid_citations": [],
                "reason": (
                    "Every factual sentence had a "
                    "valid citation and every factual "
                    "claim was supported by the cited "
                    "evidence."
                ),
            },
        )