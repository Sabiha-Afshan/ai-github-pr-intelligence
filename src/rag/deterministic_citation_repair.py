from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any

from src.rag.claim_evidence_validation import (
    EvidenceRecord,
    validate_claim_evidence,
)
from src.rag.grounded_generation import (
    GroundedGenerationResponse,
    GroundedResponseGenerator,
)
from src.rag.production_grounded_generation import (
    ProductionGroundedResponse,
    ProductionGroundedResponseGenerator,
)
from src.rag.sentence_citation_validation import (
    extract_citations,
    sentence_requires_citation,
    split_sentences,
)


FINAL_PUNCTUATION_PATTERN = re.compile(
    r"([.!?])$"
)


@dataclass(frozen=True)
class SentenceRepairDecision:
    sentence_index: int
    original_sentence: str
    repaired_sentence: str
    repair_required: bool
    repaired: bool
    candidate_evidence_ids: list[str]
    selected_evidence_id: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "original_sentence": self.original_sentence,
            "repaired_sentence": self.repaired_sentence,
            "repair_required": self.repair_required,
            "repaired": self.repaired,
            "candidate_evidence_ids": (
                self.candidate_evidence_ids
            ),
            "selected_evidence_id": (
                self.selected_evidence_id
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeterministicCitationRepairResult:
    attempted: bool
    succeeded: bool
    original_answer: str
    repaired_answer: str
    sentences_requiring_repair: int
    sentences_repaired: int
    unresolved_sentence_count: int
    latency_ms: float
    decisions: list[SentenceRepairDecision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "original_answer": self.original_answer,
            "repaired_answer": self.repaired_answer,
            "sentences_requiring_repair": (
                self.sentences_requiring_repair
            ),
            "sentences_repaired": (
                self.sentences_repaired
            ),
            "unresolved_sentence_count": (
                self.unresolved_sentence_count
            ),
            "latency_ms": self.latency_ms,
            "decisions": [
                decision.to_dict()
                for decision in self.decisions
            ],
        }


@dataclass(frozen=True)
class DeterministicProductionResponse:
    query: str
    action: str
    answer: str
    answer_released: bool
    generation_executed: bool
    repair_attempted: bool
    repair_succeeded: bool
    initial_response: ProductionGroundedResponse
    final_response: ProductionGroundedResponse
    repair_result: DeterministicCitationRepairResult
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


def _build_evidence_record(
    evidence: Any,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence.evidence_id,
        text=evidence.text,
        pr_number=evidence.pr_number,
        section=evidence.section,
        metadata=evidence.metadata,
    )


def _append_citation(
    sentence: str,
    evidence_id: str,
) -> str:
    stripped_sentence = sentence.strip()

    punctuation_match = (
        FINAL_PUNCTUATION_PATTERN.search(
            stripped_sentence
        )
    )

    citation = f"[{evidence_id}]"

    if punctuation_match:
        punctuation = punctuation_match.group(1)

        body = stripped_sentence[
            : punctuation_match.start()
        ].rstrip()

        return (
            f"{body} {citation}{punctuation}"
        )

    return (
        f"{stripped_sentence} {citation}"
    )


def _candidate_supports_sentence(
    sentence: str,
    evidence_record: EvidenceRecord,
    minimum_support_score: float,
    minimum_token_coverage: float,
) -> tuple[bool, float]:
    temporary_answer = _append_citation(
        sentence=sentence,
        evidence_id=(
            evidence_record.evidence_id
        ),
    )

    validation = validate_claim_evidence(
        answer=temporary_answer,
        evidence=[evidence_record],
        insufficient_evidence=False,
        minimum_support_score=(
            minimum_support_score
        ),
        minimum_token_coverage=(
            minimum_token_coverage
        ),
    )

    if not validation.claim_results:
        return False, 0.0

    claim_result = (
        validation.claim_results[0]
    )

    return (
        bool(
            validation.passed
            and claim_result.passed
        ),
        claim_result.support_score,
    )


def repair_citations_deterministically(
    base_response: GroundedGenerationResponse,
    minimum_support_score: float = 0.60,
    minimum_token_coverage: float = 0.45,
) -> DeterministicCitationRepairResult:
    start_time = time.perf_counter()

    sentences = split_sentences(
        base_response.answer
    )

    evidence_records = [
        _build_evidence_record(evidence)
        for evidence in base_response.evidence
    ]

    decisions: list[
        SentenceRepairDecision
    ] = []

    repaired_sentences: list[str] = []

    sentences_requiring_repair = 0
    sentences_repaired = 0

    for sentence_index, sentence in enumerate(
        sentences,
        start=1,
    ):
        requires_citation = (
            sentence_requires_citation(
                sentence=sentence,
                insufficient_evidence=(
                    base_response
                    .insufficient_evidence
                ),
            )
        )

        existing_citations = (
            extract_citations(sentence)
        )

        repair_required = bool(
            requires_citation
            and not existing_citations
        )

        if not repair_required:
            repaired_sentences.append(
                sentence
            )

            decisions.append(
                SentenceRepairDecision(
                    sentence_index=sentence_index,
                    original_sentence=sentence,
                    repaired_sentence=sentence,
                    repair_required=False,
                    repaired=False,
                    candidate_evidence_ids=[],
                    selected_evidence_id=None,
                    reason=(
                        "The sentence already had a "
                        "citation or did not require one."
                    ),
                )
            )

            continue

        sentences_requiring_repair += 1

        supporting_candidates: list[
            tuple[str, float]
        ] = []

        for evidence_record in evidence_records:
            (
                supported,
                support_score,
            ) = _candidate_supports_sentence(
                sentence=sentence,
                evidence_record=evidence_record,
                minimum_support_score=(
                    minimum_support_score
                ),
                minimum_token_coverage=(
                    minimum_token_coverage
                ),
            )

            if supported:
                supporting_candidates.append(
                    (
                        evidence_record.evidence_id,
                        support_score,
                    )
                )

        candidate_evidence_ids = [
            candidate[0]
            for candidate in supporting_candidates
        ]

        if len(
            supporting_candidates
        ) == 1:
            selected_evidence_id = (
                supporting_candidates[0][0]
            )

            repaired_sentence = (
                _append_citation(
                    sentence=sentence,
                    evidence_id=(
                        selected_evidence_id
                    ),
                )
            )

            repaired_sentences.append(
                repaired_sentence
            )

            sentences_repaired += 1

            decisions.append(
                SentenceRepairDecision(
                    sentence_index=sentence_index,
                    original_sentence=sentence,
                    repaired_sentence=(
                        repaired_sentence
                    ),
                    repair_required=True,
                    repaired=True,
                    candidate_evidence_ids=(
                        candidate_evidence_ids
                    ),
                    selected_evidence_id=(
                        selected_evidence_id
                    ),
                    reason=(
                        "Exactly one evidence item "
                        "fully supported the sentence."
                    ),
                )
            )

            continue

        repaired_sentences.append(
            sentence
        )

        if not supporting_candidates:
            reason = (
                "No individual evidence item fully "
                "supported the uncited sentence."
            )
        else:
            reason = (
                "Multiple evidence items independently "
                "supported the sentence, so repair was "
                "withheld to avoid ambiguous attribution."
            )

        decisions.append(
            SentenceRepairDecision(
                sentence_index=sentence_index,
                original_sentence=sentence,
                repaired_sentence=sentence,
                repair_required=True,
                repaired=False,
                candidate_evidence_ids=(
                    candidate_evidence_ids
                ),
                selected_evidence_id=None,
                reason=reason,
            )
        )

    repaired_answer = " ".join(
        repaired_sentences
    ).strip()

    unresolved_sentence_count = (
        sentences_requiring_repair
        - sentences_repaired
    )

    succeeded = bool(
        sentences_requiring_repair > 0
        and unresolved_sentence_count == 0
    )

    latency_ms = round(
        (
            time.perf_counter()
            - start_time
        )
        * 1000,
        3,
    )

    return DeterministicCitationRepairResult(
        attempted=(
            sentences_requiring_repair > 0
        ),
        succeeded=succeeded,
        original_answer=(
            base_response.answer
        ),
        repaired_answer=repaired_answer,
        sentences_requiring_repair=(
            sentences_requiring_repair
        ),
        sentences_repaired=(
            sentences_repaired
        ),
        unresolved_sentence_count=(
            unresolved_sentence_count
        ),
        latency_ms=latency_ms,
        decisions=decisions,
    )


def _not_attempted_repair(
    answer: str,
) -> DeterministicCitationRepairResult:
    return DeterministicCitationRepairResult(
        attempted=False,
        succeeded=False,
        original_answer=answer,
        repaired_answer=answer,
        sentences_requiring_repair=0,
        sentences_repaired=0,
        unresolved_sentence_count=0,
        latency_ms=0.0,
        decisions=[],
    )


class DeterministicRepairProductionGenerator:
    """
    Production generator with deterministic citation repair.

    Repair is attempted only when sentence-level citation validation fails.
    A repaired answer is released only after the complete citation and
    claim-to-evidence governance pipeline passes again.
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

    def _production_generator(
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
    ) -> DeterministicProductionResponse:
        initial_response = (
            self._production_generator(
                base_generator=(
                    self.base_generator
                )
            ).generate(
                query=query
            )
        )

        if (
            initial_response.action
            != "abstain_citation_validation"
        ):
            repair_result = (
                _not_attempted_repair(
                    answer=(
                        initial_response
                        .base_response
                        .answer
                    )
                )
            )

            return DeterministicProductionResponse(
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
                        "Deterministic citation repair "
                        "was not applicable."
                    ),
                },
            )

        repair_result = (
            repair_citations_deterministically(
                base_response=(
                    initial_response
                    .base_response
                ),
                minimum_support_score=(
                    self.minimum_support_score
                ),
                minimum_token_coverage=(
                    self.minimum_token_coverage
                ),
            )
        )

        if not repair_result.succeeded:
            return DeterministicProductionResponse(
                query=query,
                action=initial_response.action,
                answer=initial_response.answer,
                answer_released=False,
                generation_executed=True,
                repair_attempted=(
                    repair_result.attempted
                ),
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
                    "repair_attempted": (
                        repair_result.attempted
                    ),
                    "repair_succeeded": False,
                    "final_action": (
                        initial_response.action
                    ),
                    "sentences_requiring_repair": (
                        repair_result
                        .sentences_requiring_repair
                    ),
                    "sentences_repaired": (
                        repair_result
                        .sentences_repaired
                    ),
                    "unresolved_sentence_count": (
                        repair_result
                        .unresolved_sentence_count
                    ),
                    "reason": (
                        "One or more uncited sentences "
                        "could not be attributed to a "
                        "unique supporting evidence item."
                    ),
                },
            )

        repaired_base_response = replace(
            initial_response.base_response,
            answer=(
                repair_result.repaired_answer
            ),
            trace={
                **initial_response
                .base_response
                .trace,
                "deterministic_repair_attempted": True,
                "deterministic_repair_succeeded": True,
                "original_answer": (
                    repair_result.original_answer
                ),
                "repaired_answer": (
                    repair_result.repaired_answer
                ),
            },
        )

        final_response = (
            self._production_generator(
                base_generator=(
                    FixedResponseGenerator(
                        repaired_base_response
                    )
                )
            ).generate(
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

        return DeterministicProductionResponse(
            query=query,
            action=final_response.action,
            answer=final_response.answer,
            answer_released=(
                final_response.answer_released
            ),
            generation_executed=True,
            repair_attempted=True,
            repair_succeeded=repair_succeeded,
            initial_response=initial_response,
            final_response=final_response,
            repair_result=repair_result,
            trace={
                "initial_action": (
                    initial_response.action
                ),
                "repair_attempted": True,
                "repair_candidate_succeeded": (
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
                "repair_latency_ms": (
                    repair_result.latency_ms
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
                    "The deterministically repaired "
                    "answer passed complete governance."
                    if repair_succeeded
                    else (
                        "The repaired answer failed one "
                        "or more final governance checks."
                    )
                ),
            },
        )