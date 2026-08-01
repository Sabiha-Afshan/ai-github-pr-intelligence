from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.rag.sentence_citation_validation import (
    extract_citations,
    sentence_requires_citation,
    split_sentences,
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "based",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

TOKEN_CANONICAL_MAP = {
    "chance": "probability",
    "chances": "probability",
    "likelihood": "probability",
    "likelihoods": "probability",
    "probabilities": "probability",
    "likely": "probability",
    "unlikely": "probability",
    "accepted": "merged",
    "acceptance": "merged",
    "integrated": "merged",
    "integration": "merged",
    "merge": "merged",
    "merging": "merged",
    "open": "created",
    "opened": "created",
    "opening": "created",
    "opens": "created",
    "create": "created",
    "creates": "created",
    "creating": "created",
    "authored": "author",
    "authoring": "author",
    "authors": "author",
    "creator": "author",
    "creators": "author",
    "submitted": "created",
    "submitting": "created",
    "submitter": "author",
    "submitters": "author",
    "reviewed": "review",
    "reviewing": "review",
    "reviewer": "review",
    "reviewers": "review",
    "governance": "governance",
    "governed": "governance",
    "human": "manual",
    "manually": "manual",
    "require": "required",
    "requires": "required",
    "requiring": "required",
    "priority": "priority",
    "prioritise": "priority",
    "prioritize": "priority",
    "prioritised": "priority",
    "prioritized": "priority",
    "urgent": "urgency",
    "risk": "risk",
    "risks": "risk",
    "predictions": "prediction",
    "predicted": "prediction",
    "predictive": "prediction",
    "forecast": "prediction",
    "forecasted": "prediction",
    "forecasts": "prediction",
    "controls": "policy",
    "policies": "policy",
    "rules": "rule",
    "submissions": "pull_request",
    "submission": "pull_request",
    "changes": "pull_request",
    "change": "pull_request",
    "prs": "pull_request",
    "pr": "pull_request",
}

PHRASE_CANONICAL_MAP = {
    "low chance": "low probability",
    "high chance": "high probability",
    "human governance review": "manual governance review",
    "human governance check": "manual governance review",
    "manual review": "manual governance review",
    "opened by": "created by",
    "submitted by": "created by",
    "authored by": "created by",
    "created by": "created by",
    "get accepted": "be merged",
    "being accepted": "being merged",
    "remain open": "merge delay",
    "take longer": "merge delay",
    "review first": "review priority",
    "look at first": "review priority",
}

PREDICTION_TERMS = {
    "prediction",
    "probability",
    "merge",
    "merged",
    "delay",
}

POLICY_TERMS = {
    "manual",
    "review",
    "required",
    "policy",
    "governance",
    "risk",
    "rule",
    "critical",
    "high",
}

IDENTITY_TERMS = {
    "opened",
    "author",
    "title",
    "created",
    "merged",
    "closed",
    "repository",
}

DATE_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
)

PERCENTAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
)

DECIMAL_PATTERN = re.compile(
    r"\b0?\.\d+\b"
)

PR_NUMBER_PATTERN = re.compile(
    r"(?:\bPR\s*#?\s*|\bpull\s+request\s*#?\s*)(\d+)\b",
    flags=re.IGNORECASE,
)

BOOLEAN_PATTERN = re.compile(
    r"\b(?:true|false)\b",
    flags=re.IGNORECASE,
)

AUTHOR_BY_PATTERN = re.compile(
    r"\b(?:opened|created|authored|submitted)\s+by\s+"
    r"([a-zA-Z0-9_.-]+)\b",
    flags=re.IGNORECASE,
)

AUTHOR_METADATA_PATTERN = re.compile(
    r"\bauthor\s+([a-zA-Z0-9_.-]+)\b",
    flags=re.IGNORECASE,
)

CITATION_PATTERN = re.compile(
    r"\[(E\d+)\]",
    flags=re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9_.#%-]+"
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    text: str
    pr_number: int | None = None
    section: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "pr_number": self.pr_number,
            "section": self.section,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class ClaimSupportResult:
    sentence_index: int
    claim: str
    citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    requires_validation: bool
    evidence_text: str
    claim_tokens: list[str]
    supported_tokens: list[str]
    unsupported_tokens: list[str]
    token_coverage: float
    entity_checks: dict[str, Any]
    entity_coverage: float
    support_score: float
    passed: bool
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "claim": self.claim,
            "citations": self.citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "requires_validation": self.requires_validation,
            "evidence_text": self.evidence_text,
            "claim_tokens": self.claim_tokens,
            "supported_tokens": self.supported_tokens,
            "unsupported_tokens": self.unsupported_tokens,
            "token_coverage": self.token_coverage,
            "entity_checks": self.entity_checks,
            "entity_coverage": self.entity_coverage,
            "support_score": self.support_score,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }


@dataclass(frozen=True)
class ClaimEvidenceValidation:
    sentence_count: int
    factual_claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    groundedness_rate: float
    mean_support_score: float
    invalid_citations: list[str]
    passed: bool
    claim_results: list[ClaimSupportResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_count": self.sentence_count,
            "factual_claim_count": self.factual_claim_count,
            "supported_claim_count": self.supported_claim_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "groundedness_rate": self.groundedness_rate,
            "mean_support_score": self.mean_support_score,
            "invalid_citations": self.invalid_citations,
            "passed": self.passed,
            "claim_results": [
                result.to_dict()
                for result in self.claim_results
            ],
        }


def _normalise_text(
    value: Any,
) -> str:
    text = str(value or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    for phrase, replacement in (
        PHRASE_CANONICAL_MAP.items()
    ):
        text = text.replace(
            phrase,
            replacement,
        )

    return text


def _strip_citations(
    sentence: str,
) -> str:
    return CITATION_PATTERN.sub(
        "",
        sentence,
    ).strip()


def _canonicalise_token(
    token: str,
) -> str:
    normalised = token.strip(
        ".,:;!?()[]{}'\""
    ).lower()

    return TOKEN_CANONICAL_MAP.get(
        normalised,
        normalised,
    )


def _extract_tokens(
    text: str,
) -> list[str]:
    raw_tokens = TOKEN_PATTERN.findall(
        _normalise_text(text)
    )

    filtered_tokens: list[str] = []

    for raw_token in raw_tokens:
        token = _canonicalise_token(
            raw_token
        )

        if not token:
            continue

        if token in STOP_WORDS:
            continue

        if (
            token.startswith("e")
            and token[1:].isdigit()
        ):
            continue

        if token not in filtered_tokens:
            filtered_tokens.append(token)

    return filtered_tokens


def _extract_pr_numbers(
    text: str,
) -> list[str]:
    return list(
        dict.fromkeys(
            PR_NUMBER_PATTERN.findall(text)
        )
    )


def _extract_dates(
    text: str,
) -> list[str]:
    return list(
        dict.fromkeys(
            DATE_PATTERN.findall(text)
        )
    )


def _extract_percentages(
    text: str,
) -> list[str]:
    return [
        value.replace(" ", "")
        for value in dict.fromkeys(
            PERCENTAGE_PATTERN.findall(text)
        )
    ]


def _extract_decimals(
    text: str,
) -> list[str]:
    return list(
        dict.fromkeys(
            DECIMAL_PATTERN.findall(text)
        )
    )


def _extract_booleans(
    text: str,
) -> list[str]:
    return [
        value.lower()
        for value in dict.fromkeys(
            BOOLEAN_PATTERN.findall(text)
        )
    ]


def _extract_authors(
    text: str,
) -> list[str]:
    values: list[str] = []

    for pattern in (
        AUTHOR_BY_PATTERN,
        AUTHOR_METADATA_PATTERN,
    ):
        for value in pattern.findall(text):
            normalised_value = value.lower()

            if normalised_value not in values:
                values.append(normalised_value)

    return values


def _metadata_to_text(
    metadata: dict[str, Any] | None,
) -> str:
    if not metadata:
        return ""

    parts: list[str] = []

    for key, value in metadata.items():
        parts.append(
            f"{key} {value}"
        )

    return " ".join(parts)


def _build_evidence_text(
    records: list[EvidenceRecord],
) -> str:
    parts: list[str] = []

    for record in records:
        if record.pr_number is not None:
            parts.append(
                f"PR #{record.pr_number}"
            )

        if record.section:
            parts.append(record.section)

        parts.append(record.text)

        metadata_text = _metadata_to_text(
            record.metadata
        )

        if metadata_text:
            parts.append(metadata_text)

    return _normalise_text(
        " ".join(parts)
    )


def _evaluate_entity_group(
    claim_values: list[str],
    evidence_values: list[str],
) -> dict[str, Any]:
    if not claim_values:
        return {
            "applicable": False,
            "claim_values": [],
            "evidence_values": evidence_values,
            "matched_values": [],
            "unmatched_values": [],
            "passed": True,
        }

    evidence_set = {
        value.lower()
        for value in evidence_values
    }

    matched_values = [
        value
        for value in claim_values
        if value.lower() in evidence_set
    ]

    unmatched_values = [
        value
        for value in claim_values
        if value.lower() not in evidence_set
    ]

    return {
        "applicable": True,
        "claim_values": claim_values,
        "evidence_values": evidence_values,
        "matched_values": matched_values,
        "unmatched_values": unmatched_values,
        "passed": len(unmatched_values) == 0,
    }


def _calculate_entity_checks(
    claim: str,
    evidence_text: str,
) -> dict[str, Any]:
    return {
        "pr_numbers": _evaluate_entity_group(
            claim_values=_extract_pr_numbers(
                claim
            ),
            evidence_values=_extract_pr_numbers(
                evidence_text
            ),
        ),
        "dates": _evaluate_entity_group(
            claim_values=_extract_dates(
                claim
            ),
            evidence_values=_extract_dates(
                evidence_text
            ),
        ),
        "percentages": _evaluate_entity_group(
            claim_values=_extract_percentages(
                claim
            ),
            evidence_values=_extract_percentages(
                evidence_text
            ),
        ),
        "decimals": _evaluate_entity_group(
            claim_values=_extract_decimals(
                claim
            ),
            evidence_values=_extract_decimals(
                evidence_text
            ),
        ),
        "booleans": _evaluate_entity_group(
            claim_values=_extract_booleans(
                claim
            ),
            evidence_values=_extract_booleans(
                evidence_text
            ),
        ),
        "authors": _evaluate_entity_group(
            claim_values=_extract_authors(
                claim
            ),
            evidence_values=_extract_authors(
                evidence_text
            ),
        ),
    }


def _calculate_entity_coverage(
    entity_checks: dict[str, Any],
) -> float:
    applicable_checks = [
        check
        for check in entity_checks.values()
        if check["applicable"]
    ]

    if not applicable_checks:
        return 1.0

    passed_count = sum(
        1
        for check in applicable_checks
        if check["passed"]
    )

    return passed_count / len(
        applicable_checks
    )


def _contains_required_domain_terms(
    claim_tokens: list[str],
    evidence_tokens: set[str],
) -> tuple[bool, list[str]]:
    domain_terms = (
        PREDICTION_TERMS
        | POLICY_TERMS
        | IDENTITY_TERMS
    )

    required_terms = [
        token
        for token in claim_tokens
        if token in domain_terms
    ]

    missing_terms = [
        token
        for token in required_terms
        if token not in evidence_tokens
    ]

    return (
        len(missing_terms) == 0,
        missing_terms,
    )


def validate_claim_evidence(
    answer: str,
    evidence: list[EvidenceRecord],
    insufficient_evidence: bool = False,
    minimum_support_score: float = 0.60,
    minimum_token_coverage: float = 0.45,
) -> ClaimEvidenceValidation:
    evidence_lookup = {
        record.evidence_id.upper(): record
        for record in evidence
    }

    sentences = split_sentences(answer)

    claim_results: list[
        ClaimSupportResult
    ] = []

    invalid_citations_across_answer: list[
        str
    ] = []

    for sentence_index, sentence in enumerate(
        sentences,
        start=1,
    ):
        requires_validation = (
            sentence_requires_citation(
                sentence=sentence,
                insufficient_evidence=(
                    insufficient_evidence
                ),
            )
        )

        citations = extract_citations(
            sentence
        )

        valid_citations = [
            citation
            for citation in citations
            if citation in evidence_lookup
        ]

        invalid_citations = [
            citation
            for citation in citations
            if citation not in evidence_lookup
        ]

        for citation in invalid_citations:
            if (
                citation
                not in invalid_citations_across_answer
            ):
                invalid_citations_across_answer.append(
                    citation
                )

        cited_records = [
            evidence_lookup[citation]
            for citation in valid_citations
        ]

        evidence_text = _build_evidence_text(
            cited_records
        )

        claim_without_citations = (
            _strip_citations(sentence)
        )

        claim_tokens = _extract_tokens(
            claim_without_citations
        )

        evidence_tokens = set(
            _extract_tokens(evidence_text)
        )

        supported_tokens = [
            token
            for token in claim_tokens
            if token in evidence_tokens
        ]

        unsupported_tokens = [
            token
            for token in claim_tokens
            if token not in evidence_tokens
        ]

        if claim_tokens:
            token_coverage = (
                len(supported_tokens)
                / len(claim_tokens)
            )
        else:
            token_coverage = 1.0

        entity_checks = (
            _calculate_entity_checks(
                claim=claim_without_citations,
                evidence_text=evidence_text,
            )
        )

        entity_coverage = (
            _calculate_entity_coverage(
                entity_checks
            )
        )

        (
            domain_terms_passed,
            missing_domain_terms,
        ) = _contains_required_domain_terms(
            claim_tokens=claim_tokens,
            evidence_tokens=evidence_tokens,
        )

        support_score = (
            0.60 * token_coverage
            + 0.40 * entity_coverage
        )

        failure_reasons: list[str] = []

        if requires_validation and not citations:
            failure_reasons.append(
                "The factual claim contains no evidence citation."
            )

        if invalid_citations:
            failure_reasons.append(
                "The claim contains invalid evidence identifiers: "
                + ", ".join(invalid_citations)
            )

        if (
            requires_validation
            and valid_citations
            and token_coverage
            < minimum_token_coverage
        ):
            failure_reasons.append(
                "The cited evidence has insufficient lexical or "
                "approved semantic support for the claim."
            )

        failed_entity_groups = [
            name
            for name, check in entity_checks.items()
            if (
                check["applicable"]
                and not check["passed"]
            )
        ]

        if failed_entity_groups:
            failure_reasons.append(
                "The cited evidence does not match claim entities: "
                + ", ".join(
                    failed_entity_groups
                )
            )

        if (
            requires_validation
            and not domain_terms_passed
        ):
            failure_reasons.append(
                "Important domain concepts are absent from the cited "
                "evidence: "
                + ", ".join(
                    missing_domain_terms
                )
            )

        if (
            requires_validation
            and support_score
            < minimum_support_score
        ):
            failure_reasons.append(
                "The overall claim-to-evidence support score is below "
                f"{minimum_support_score:.2f}."
            )

        if requires_validation:
            passed = (
                bool(valid_citations)
                and not invalid_citations
                and token_coverage
                >= minimum_token_coverage
                and entity_coverage == 1.0
                and domain_terms_passed
                and support_score
                >= minimum_support_score
            )
        else:
            passed = not invalid_citations

        claim_results.append(
            ClaimSupportResult(
                sentence_index=sentence_index,
                claim=sentence,
                citations=citations,
                valid_citations=valid_citations,
                invalid_citations=(
                    invalid_citations
                ),
                requires_validation=(
                    requires_validation
                ),
                evidence_text=evidence_text,
                claim_tokens=claim_tokens,
                supported_tokens=supported_tokens,
                unsupported_tokens=(
                    unsupported_tokens
                ),
                token_coverage=round(
                    token_coverage,
                    6,
                ),
                entity_checks=entity_checks,
                entity_coverage=round(
                    entity_coverage,
                    6,
                ),
                support_score=round(
                    support_score,
                    6,
                ),
                passed=passed,
                failure_reasons=failure_reasons,
            )
        )

    factual_results = [
        result
        for result in claim_results
        if result.requires_validation
    ]

    supported_results = [
        result
        for result in factual_results
        if result.passed
    ]

    factual_claim_count = len(
        factual_results
    )

    supported_claim_count = len(
        supported_results
    )

    unsupported_claim_count = (
        factual_claim_count
        - supported_claim_count
    )

    if factual_claim_count == 0:
        groundedness_rate = 1.0
        mean_support_score = 1.0
    else:
        groundedness_rate = (
            supported_claim_count
            / factual_claim_count
        )

        mean_support_score = sum(
            result.support_score
            for result in factual_results
        ) / factual_claim_count

    passed = (
        unsupported_claim_count == 0
        and not invalid_citations_across_answer
    )

    return ClaimEvidenceValidation(
        sentence_count=len(
            claim_results
        ),
        factual_claim_count=(
            factual_claim_count
        ),
        supported_claim_count=(
            supported_claim_count
        ),
        unsupported_claim_count=(
            unsupported_claim_count
        ),
        groundedness_rate=round(
            groundedness_rate,
            6,
        ),
        mean_support_score=round(
            mean_support_score,
            6,
        ),
        invalid_citations=(
            invalid_citations_across_answer
        ),
        passed=passed,
        claim_results=claim_results,
    )