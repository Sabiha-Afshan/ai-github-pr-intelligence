from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


CITATION_PATTERN = re.compile(
    r"\[(E\d+)\]",
    flags=re.IGNORECASE,
)

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)

NON_FACTUAL_PREFIXES = {
    "insufficient evidence",
    "the available evidence is insufficient",
    "no governed evidence was available",
    "no sufficiently relevant evidence was found",
    "this question is outside the scope",
    "i cannot determine",
    "it cannot be determined",
}

NON_FACTUAL_SENTENCES = {
    "please provide a pull request number.",
    "please ask about pull requests, merge outcomes, review priority, "
    "policy risk, repository activity or pr evidence.",
}


@dataclass(frozen=True)
class SentenceCitationResult:
    sentence_index: int
    sentence: str
    citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    requires_citation: bool
    citation_present: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "sentence": self.sentence,
            "citations": self.citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "requires_citation": self.requires_citation,
            "citation_present": self.citation_present,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SentenceCitationValidation:
    sentence_count: int
    factual_sentence_count: int
    cited_factual_sentence_count: int
    uncited_factual_sentence_count: int
    citation_coverage: float
    invalid_citations: list[str]
    passed: bool
    sentence_results: list[SentenceCitationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_count": self.sentence_count,
            "factual_sentence_count": (
                self.factual_sentence_count
            ),
            "cited_factual_sentence_count": (
                self.cited_factual_sentence_count
            ),
            "uncited_factual_sentence_count": (
                self.uncited_factual_sentence_count
            ),
            "citation_coverage": self.citation_coverage,
            "invalid_citations": self.invalid_citations,
            "passed": self.passed,
            "sentence_results": [
                result.to_dict()
                for result in self.sentence_results
            ],
        }


def _normalise_whitespace(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def _protect_decimal_points(
    text: str,
) -> str:
    return re.sub(
        r"(?<=\d)\.(?=\d)",
        "<DECIMAL_POINT>",
        text,
    )


def _restore_decimal_points(
    text: str,
) -> str:
    return text.replace(
        "<DECIMAL_POINT>",
        ".",
    )


def split_sentences(
    answer: str,
) -> list[str]:
    """
    Splits an answer into sentences while preserving decimal values such
    as 47.01 and common evidence citations such as [E1].
    """
    normalised_answer = _normalise_whitespace(
        answer
    )

    if not normalised_answer:
        return []

    protected_answer = _protect_decimal_points(
        normalised_answer
    )

    raw_sentences = SENTENCE_BOUNDARY_PATTERN.split(
        protected_answer
    )

    sentences: list[str] = []

    for raw_sentence in raw_sentences:
        sentence = _restore_decimal_points(
            raw_sentence
        ).strip()

        if sentence:
            sentences.append(sentence)

    return sentences


def extract_citations(
    sentence: str,
) -> list[str]:
    matches = CITATION_PATTERN.findall(
        sentence
    )

    citations: list[str] = []

    for match in matches:
        citation = match.upper()

        if citation not in citations:
            citations.append(citation)

    return citations


def sentence_requires_citation(
    sentence: str,
    insufficient_evidence: bool = False,
) -> bool:
    normalised = _normalise_whitespace(
        sentence
    ).lower()

    if not normalised:
        return False

    if normalised in NON_FACTUAL_SENTENCES:
        return False

    if any(
        normalised.startswith(prefix)
        for prefix in NON_FACTUAL_PREFIXES
    ):
        return False

    if insufficient_evidence:
        insufficient_markers = {
            "not enough evidence",
            "insufficient evidence",
            "cannot determine",
            "could not determine",
            "unable to determine",
            "not available in the evidence",
        }

        if any(
            marker in normalised
            for marker in insufficient_markers
        ):
            return False

    if not re.search(
        r"[a-zA-Z0-9]",
        normalised,
    ):
        return False

    return True


def validate_sentence_citations(
    answer: str,
    available_evidence_ids: set[str],
    insufficient_evidence: bool = False,
) -> SentenceCitationValidation:
    normalised_available_ids = {
        evidence_id.upper()
        for evidence_id in available_evidence_ids
    }

    sentences = split_sentences(answer)

    sentence_results: list[
        SentenceCitationResult
    ] = []

    invalid_citations_across_answer: list[str] = []

    for index, sentence in enumerate(
        sentences,
        start=1,
    ):
        citations = extract_citations(
            sentence
        )

        valid_citations = [
            citation
            for citation in citations
            if citation in normalised_available_ids
        ]

        invalid_citations = [
            citation
            for citation in citations
            if citation not in normalised_available_ids
        ]

        for citation in invalid_citations:
            if (
                citation
                not in invalid_citations_across_answer
            ):
                invalid_citations_across_answer.append(
                    citation
                )

        requires_citation = (
            sentence_requires_citation(
                sentence=sentence,
                insufficient_evidence=(
                    insufficient_evidence
                ),
            )
        )

        citation_present = bool(
            valid_citations
        )

        passed = (
            not invalid_citations
            and (
                not requires_citation
                or citation_present
            )
        )

        sentence_results.append(
            SentenceCitationResult(
                sentence_index=index,
                sentence=sentence,
                citations=citations,
                valid_citations=valid_citations,
                invalid_citations=(
                    invalid_citations
                ),
                requires_citation=(
                    requires_citation
                ),
                citation_present=(
                    citation_present
                ),
                passed=passed,
            )
        )

    factual_results = [
        result
        for result in sentence_results
        if result.requires_citation
    ]

    cited_factual_results = [
        result
        for result in factual_results
        if (
            result.citation_present
            and not result.invalid_citations
        )
    ]

    factual_sentence_count = len(
        factual_results
    )

    cited_factual_sentence_count = len(
        cited_factual_results
    )

    uncited_factual_sentence_count = (
        factual_sentence_count
        - cited_factual_sentence_count
    )

    if factual_sentence_count == 0:
        citation_coverage = 1.0
    else:
        citation_coverage = (
            cited_factual_sentence_count
            / factual_sentence_count
        )

    passed = (
        uncited_factual_sentence_count == 0
        and not invalid_citations_across_answer
    )

    return SentenceCitationValidation(
        sentence_count=len(
            sentence_results
        ),
        factual_sentence_count=(
            factual_sentence_count
        ),
        cited_factual_sentence_count=(
            cited_factual_sentence_count
        ),
        uncited_factual_sentence_count=(
            uncited_factual_sentence_count
        ),
        citation_coverage=round(
            citation_coverage,
            6,
        ),
        invalid_citations=(
            invalid_citations_across_answer
        ),
        passed=passed,
        sentence_results=sentence_results,
    )