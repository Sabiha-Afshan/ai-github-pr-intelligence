from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


PR_NUMBER_PATTERNS = [
    r"\bPR\s*#?\s*(\d+)\b",
    r"\bpull\s+request\s+number\s*#?\s*(\d+)\b",
    r"\bpull\s+request\s*#?\s*(\d+)\b",
    r"\brequest\s+number\s*#?\s*(\d+)\b",
    r"#(\d+)\b",
]

REPOSITORY_TERMS = {
    "pull request",
    "pull requests",
    "pr",
    "prs",
    "repository",
    "repo",
    "github",
    "merge",
    "merged",
    "merging",
    "review",
    "reviewer",
    "reviewers",
    "maintainer",
    "maintainers",
    "commit",
    "commits",
    "code change",
    "code changes",
    "changed files",
    "additions",
    "deletions",
    "policy",
    "policies",
    "governance",
    "risk",
    "rules",
    "security",
    "testing",
    "description",
    "author",
    "title",
    "opened",
    "accepted",
    "submissions",
    "changes",
    "flask",
    "pallets",
}

OUT_OF_DOMAIN_TERMS = {
    "weather",
    "temperature",
    "forecast",
    "rain",
    "salary",
    "tax",
    "recipe",
    "cake",
    "food",
    "flight",
    "hotel",
    "airport",
    "currency",
    "stock price",
    "football",
    "movie",
    "medical",
    "medicine",
}

SECTION_SYNONYMS = {
    "PR identity": {
        "who opened",
        "opened by",
        "author",
        "title",
        "created",
        "creation date",
        "opened",
        "state",
        "repository",
        "pr identity",
        "pull request identity",
        "basic details",
        "request details",
    },
    "PR description": {
        "description",
        "explanation",
        "explained",
        "summary",
        "context",
        "proposal",
        "proposed change",
        "detailed description",
        "word count",
        "useful explanation",
        "author gave",
    },
    "Change evidence": {
        "large change",
        "large changes",
        "lots of files",
        "many files",
        "changed files",
        "heavy volume",
        "volume of edits",
        "additions",
        "deletions",
        "code size",
        "change size",
        "complexity",
        "difficult to inspect",
        "review complexity",
        "code evidence",
    },
    "Predictive intelligence": {
        "merge probability",
        "chance of being merged",
        "unlikely to merge",
        "unlikely to get accepted",
        "not accepted",
        "merge prediction",
        "merge outcome",
        "delay prediction",
        "delayed merge",
        "remain open",
        "long time",
        "take longer",
        "integration delay",
        "before being integrated",
    },
    "Deterministic policy intelligence": {
        "policy",
        "policies",
        "governance",
        "controls",
        "violated",
        "rules triggered",
        "triggered rules",
        "manual review",
        "human governance",
        "security concern",
        "governance concern",
        "repository controls",
        "escalated",
    },
    "Unified review priority": {
        "review priority",
        "review urgency",
        "urgency",
        "what should",
        "look at first",
        "recommended action",
        "reviewer action",
        "next action",
        "prioritize",
        "priority score",
        "maintainers look at",
    },
}

QUERY_EXPANSIONS = {
    "unlikely to get accepted": [
        "predicted not to merge",
        "low merge probability",
        "merge outcome",
    ],
    "unlikely to merge": [
        "predicted not to merge",
        "low merge probability",
    ],
    "remain open for a long time": [
        "merge delay",
        "delay prediction",
        "long merge duration",
    ],
    "take longer than normal": [
        "merge delay",
        "delay prediction",
        "delayed merge",
    ],
    "human governance check": [
        "manual review required",
        "policy risk",
        "governance rules",
    ],
    "repository controls": [
        "policy rules",
        "triggered rules",
        "governance",
    ],
    "look at first": [
        "review priority",
        "recommended next action",
        "review urgency",
    ],
    "useful explanation": [
        "detailed description",
        "description word count",
        "description quality",
    ],
    "who opened": [
        "author",
        "created at",
        "pull request title",
    ],
}

METADATA_CONDITION_PATTERNS = {
    "manual_review_required": {
        "phrases": {
            "manual review",
            "reviewed manually",
            "human review",
            "human governance",
            "must be reviewed",
        },
        "value": True,
    },
    "critical_policy_risk": {
        "phrases": {
            "critical risk",
            "critical pull requests",
            "critical policy",
        },
        "field": "policy_risk_band",
        "value": "Critical",
    },
    "high_review_priority": {
        "phrases": {
            "high review priority",
            "high priority",
            "high urgency",
            "high or critical",
            "critical review urgency",
        },
        "field": "review_priority",
        "value": {"High", "Critical"},
    },
    "low_merge_probability": {
        "phrases": {
            "low chance of being merged",
            "unlikely to merge",
            "unlikely to get accepted",
            "low merge probability",
        },
        "field": "merge_probability",
        "operator": "lt",
        "value": 0.40,
    },
    "delay_prediction": {
        "phrases": {
            "delay risk",
            "delayed merge",
            "take longer",
            "remain open",
            "long time before",
            "merge delay",
        },
        "field": "delay_prediction",
        "value": 1,
    },
    "multiple_rules": {
        "phrases": {
            "several rules",
            "multiple rules",
            "several different repository rules",
            "many rules",
        },
        "field": "triggered_rule_count",
        "operator": "gte",
        "value": 3,
    },
    "security_or_governance": {
        "phrases": {
            "security concern",
            "governance concern",
            "security or governance",
        },
        "field": "triggered_categories",
        "operator": "contains_any",
        "value": {"Security", "Governance"},
    },
}


@dataclass(frozen=True)
class MetadataCondition:
    field: str
    operator: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        value = self.value

        if isinstance(value, set):
            value = sorted(value)

        return {
            "field": self.field,
            "operator": self.operator,
            "value": value,
        }


@dataclass(frozen=True)
class QueryUnderstandingResult:
    original_query: str
    normalised_query: str
    expanded_query: str
    pr_number: int | None
    is_repository_query: bool
    is_out_of_domain: bool
    domain_confidence: float
    detected_sections: list[str]
    metadata_conditions: list[MetadataCondition] = field(
        default_factory=list
    )
    matched_repository_terms: list[str] = field(
        default_factory=list
    )
    matched_out_of_domain_terms: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalised_query": self.normalised_query,
            "expanded_query": self.expanded_query,
            "pr_number": self.pr_number,
            "is_repository_query": self.is_repository_query,
            "is_out_of_domain": self.is_out_of_domain,
            "domain_confidence": self.domain_confidence,
            "detected_sections": self.detected_sections,
            "metadata_conditions": [
                condition.to_dict()
                for condition in self.metadata_conditions
            ],
            "matched_repository_terms": self.matched_repository_terms,
            "matched_out_of_domain_terms": (
                self.matched_out_of_domain_terms
            ),
        }


def normalise_query(query: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(query or "").strip().lower(),
    )


def extract_pr_number(query: str) -> int | None:
    for pattern in PR_NUMBER_PATTERNS:
        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    stripped = query.strip()

    if stripped.isdigit():
        return int(stripped)

    return None


def find_matching_terms(
    query: str,
    terms: set[str],
) -> list[str]:
    return sorted(
        term
        for term in terms
        if term in query
    )


def detect_sections(query: str) -> list[str]:
    section_scores: dict[str, int] = {}

    for section, synonyms in SECTION_SYNONYMS.items():
        score = sum(
            1
            for synonym in synonyms
            if synonym in query
        )

        if score > 0:
            section_scores[section] = score

    return [
        section
        for section, _ in sorted(
            section_scores.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
            reverse=True,
        )
    ]


def expand_query(
    query: str,
    detected_sections: list[str],
) -> str:
    expansion_terms: list[str] = []

    for phrase, additions in QUERY_EXPANSIONS.items():
        if phrase in query:
            expansion_terms.extend(additions)

    for section in detected_sections:
        if section == "PR identity":
            expansion_terms.extend(
                [
                    "author",
                    "title",
                    "created at",
                    "repository",
                ]
            )
        elif section == "PR description":
            expansion_terms.extend(
                [
                    "description",
                    "description quality",
                    "description word count",
                ]
            )
        elif section == "Change evidence":
            expansion_terms.extend(
                [
                    "changed files",
                    "total changed lines",
                    "additions",
                    "deletions",
                ]
            )
        elif section == "Predictive intelligence":
            expansion_terms.extend(
                [
                    "merge probability",
                    "merge prediction",
                    "delay prediction",
                ]
            )
        elif section == "Deterministic policy intelligence":
            expansion_terms.extend(
                [
                    "policy risk",
                    "triggered rules",
                    "manual review required",
                ]
            )
        elif section == "Unified review priority":
            expansion_terms.extend(
                [
                    "review priority",
                    "review priority score",
                    "recommended next action",
                ]
            )

    unique_terms: list[str] = []

    for term in expansion_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    if not unique_terms:
        return query

    return f"{query} {' '.join(unique_terms)}"


def detect_metadata_conditions(
    query: str,
) -> list[MetadataCondition]:
    conditions: list[MetadataCondition] = []

    for condition_name, specification in (
        METADATA_CONDITION_PATTERNS.items()
    ):
        phrases = specification["phrases"]

        if not any(
            phrase in query
            for phrase in phrases
        ):
            continue

        if condition_name == "manual_review_required":
            conditions.append(
                MetadataCondition(
                    field="manual_review_required",
                    operator="eq",
                    value=True,
                )
            )
            continue

        conditions.append(
            MetadataCondition(
                field=specification["field"],
                operator=specification.get(
                    "operator",
                    "eq",
                ),
                value=specification["value"],
            )
        )

    unique_conditions: list[MetadataCondition] = []
    seen: set[tuple[str, str, str]] = set()

    for condition in conditions:
        signature = (
            condition.field,
            condition.operator,
            repr(condition.value),
        )

        if signature in seen:
            continue

        seen.add(signature)
        unique_conditions.append(condition)

    return unique_conditions


def calculate_domain_confidence(
    pr_number: int | None,
    repository_terms: list[str],
    out_of_domain_terms: list[str],
    detected_sections: list[str],
    metadata_conditions: list[MetadataCondition],
) -> float:
    score = 0.0

    if pr_number is not None:
        score += 0.65

    score += min(
        0.32,
        len(repository_terms) * 0.08,
    )

    if detected_sections:
        score += min(
            0.32,
            len(detected_sections) * 0.16,
        )

    if metadata_conditions:
        score += min(
            0.24,
            len(metadata_conditions) * 0.12,
        )

    score -= min(
        0.80,
        len(out_of_domain_terms) * 0.35,
    )

    return round(
        max(0.0, min(1.0, score)),
        6,
    )


def understand_query(
    query: str,
) -> QueryUnderstandingResult:
    if not query or not query.strip():
        raise ValueError(
            "Query must not be empty."
        )

    normalised = normalise_query(query)
    pr_number = extract_pr_number(normalised)

    repository_terms = find_matching_terms(
        normalised,
        REPOSITORY_TERMS,
    )

    out_of_domain_terms = find_matching_terms(
        normalised,
        OUT_OF_DOMAIN_TERMS,
    )

    detected_sections = detect_sections(
        normalised
    )

    metadata_conditions = (
        detect_metadata_conditions(normalised)
    )

    repository_signal_present = bool(
        pr_number is not None
        or repository_terms
        or detected_sections
        or metadata_conditions
    )

    explicit_out_of_domain = bool(
        out_of_domain_terms
    )

    is_out_of_domain = (
        explicit_out_of_domain
        and not repository_signal_present
    )

    domain_confidence = calculate_domain_confidence(
        pr_number=pr_number,
        repository_terms=repository_terms,
        out_of_domain_terms=out_of_domain_terms,
        detected_sections=detected_sections,
        metadata_conditions=metadata_conditions,
    )

    is_repository_query = (
        repository_signal_present
        and not is_out_of_domain
        and domain_confidence >= 0.12
    )

    expanded_query = expand_query(
        query=normalised,
        detected_sections=detected_sections,
    )

    return QueryUnderstandingResult(
        original_query=query,
        normalised_query=normalised,
        expanded_query=expanded_query,
        pr_number=pr_number,
        is_repository_query=is_repository_query,
        is_out_of_domain=is_out_of_domain,
        domain_confidence=domain_confidence,
        detected_sections=detected_sections,
        metadata_conditions=metadata_conditions,
        matched_repository_terms=repository_terms,
        matched_out_of_domain_terms=out_of_domain_terms,
    )