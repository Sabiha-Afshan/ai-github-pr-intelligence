from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.rag.hybrid_retrieval import (
    HybridRetriever,
    RetrievalResult,
)
from src.rag.query_understanding import (
    MetadataCondition,
    QueryUnderstandingResult,
    understand_query,
)


OUT_OF_DOMAIN_MESSAGE = (
    "This question is outside the scope of the GitHub PR Intelligence "
    "system. Please ask about pull requests, merge outcomes, review "
    "priority, policy risk, repository activity or PR evidence."
)

NO_EVIDENCE_MESSAGE = (
    "No sufficiently relevant pull-request evidence was found for this "
    "question. Try mentioning a PR number, review concern, policy risk, "
    "merge prediction or repository detail."
)


@dataclass(frozen=True)
class GovernedRetrievalResult:
    rank: int
    chunk_id: str
    pr_number: int | None
    document_id: str
    section: str
    text: str
    base_hybrid_score: float
    section_boost: float
    metadata_boost: float
    governed_score: float
    metadata_condition_matches: int
    metadata_condition_total: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "pr_number": self.pr_number,
            "document_id": self.document_id,
            "section": self.section,
            "text": self.text,
            "base_hybrid_score": self.base_hybrid_score,
            "section_boost": self.section_boost,
            "metadata_boost": self.metadata_boost,
            "governed_score": self.governed_score,
            "metadata_condition_matches": (
                self.metadata_condition_matches
            ),
            "metadata_condition_total": (
                self.metadata_condition_total
            ),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GovernedRetrievalResponse:
    query: str
    action: str
    message: str
    retrieval_executed: bool
    query_understanding: QueryUnderstandingResult
    results: list[GovernedRetrievalResult]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "action": self.action,
            "message": self.message,
            "retrieval_executed": self.retrieval_executed,
            "query_understanding": (
                self.query_understanding.to_dict()
            ),
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "trace": self.trace,
        }


def _normalise_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metadata_value(
    result: RetrievalResult,
    field: str,
    default: Any = None,
) -> Any:
    if field == "pr_number":
        return result.pr_number

    if field == "section_name":
        return result.section

    return result.metadata.get(field, default)


def _condition_matches(
    result: RetrievalResult,
    condition: MetadataCondition,
) -> bool:
    actual_value = _metadata_value(
        result=result,
        field=condition.field,
    )

    expected_value = condition.value
    operator = condition.operator

    if operator == "eq":
        if isinstance(expected_value, set):
            expected_items = {
                _normalise_text(item)
                for item in expected_value
            }

            return (
                _normalise_text(actual_value)
                in expected_items
            )

        if isinstance(expected_value, str):
            return (
                _normalise_text(actual_value)
                == _normalise_text(expected_value)
            )

        return actual_value == expected_value

    if operator == "lt":
        return (
            _safe_float(actual_value, float("inf"))
            < _safe_float(expected_value)
        )

    if operator == "lte":
        return (
            _safe_float(actual_value, float("inf"))
            <= _safe_float(expected_value)
        )

    if operator == "gt":
        return (
            _safe_float(actual_value, float("-inf"))
            > _safe_float(expected_value)
        )

    if operator == "gte":
        return (
            _safe_float(actual_value, float("-inf"))
            >= _safe_float(expected_value)
        )

    if operator == "contains_any":
        if actual_value is None:
            return False

        if isinstance(actual_value, str):
            actual_items = {
                _normalise_text(actual_value)
            }
        else:
            try:
                actual_items = {
                    _normalise_text(item)
                    for item in actual_value
                }
            except TypeError:
                actual_items = {
                    _normalise_text(actual_value)
                }

        if isinstance(expected_value, str):
            expected_items = {
                _normalise_text(expected_value)
            }
        else:
            expected_items = {
                _normalise_text(item)
                for item in expected_value
            }

        return bool(
            actual_items.intersection(expected_items)
        )

    raise ValueError(
        f"Unsupported metadata operator: {operator}"
    )


def _count_condition_matches(
    result: RetrievalResult,
    conditions: list[MetadataCondition],
) -> int:
    return sum(
        1
        for condition in conditions
        if _condition_matches(
            result=result,
            condition=condition,
        )
    )


def _calculate_section_boost(
    result: RetrievalResult,
    detected_sections: list[str],
) -> float:
    if not detected_sections:
        return 0.0

    result_section = _normalise_text(
        result.section
    )

    for position, section in enumerate(
        detected_sections
    ):
        if result_section == _normalise_text(section):
            if position == 0:
                return 0.22

            return 0.12

    return 0.0


def _calculate_metadata_boost(
    matched_count: int,
    total_count: int,
) -> float:
    if total_count == 0:
        return 0.0

    match_ratio = matched_count / total_count

    if match_ratio == 1.0:
        return 0.30

    if match_ratio >= 0.50:
        return 0.10

    return -0.20


def _build_trace(
    query_understanding: QueryUnderstandingResult,
    action: str,
    retrieval_executed: bool,
    candidate_count: int,
    returned_count: int,
    retrieval_query: str | None = None,
    strict_metadata_filter_applied: bool = False,
) -> dict[str, Any]:
    return {
        "original_query": (
            query_understanding.original_query
        ),
        "expanded_query": (
            query_understanding.expanded_query
        ),
        "retrieval_query": retrieval_query,
        "pr_number": query_understanding.pr_number,
        "is_repository_query": (
            query_understanding.is_repository_query
        ),
        "is_out_of_domain": (
            query_understanding.is_out_of_domain
        ),
        "domain_confidence": (
            query_understanding.domain_confidence
        ),
        "detected_sections": (
            query_understanding.detected_sections
        ),
        "metadata_conditions": [
            condition.to_dict()
            for condition in (
                query_understanding.metadata_conditions
            )
        ],
        "strict_metadata_filter_applied": (
            strict_metadata_filter_applied
        ),
        "action": action,
        "retrieval_executed": retrieval_executed,
        "candidate_count": candidate_count,
        "returned_count": returned_count,
    }


class GovernedRetriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        candidate_pool_size: int = 200,
        minimum_governed_score: float = 0.20,
    ) -> None:
        self.hybrid_retriever = (
            hybrid_retriever
            if hybrid_retriever is not None
            else HybridRetriever()
        )

        self.candidate_pool_size = (
            candidate_pool_size
        )

        self.minimum_governed_score = (
            minimum_governed_score
        )

    def _build_retrieval_query(
        self,
        understanding: QueryUnderstandingResult,
    ) -> str:
        if understanding.pr_number is not None:
            return f"PR #{understanding.pr_number}"

        return understanding.expanded_query

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> GovernedRetrievalResponse:
        query_understanding = understand_query(query)

        if (
            query_understanding.is_out_of_domain
            or not query_understanding.is_repository_query
        ):
            action = "abstain_out_of_domain"

            return GovernedRetrievalResponse(
                query=query,
                action=action,
                message=OUT_OF_DOMAIN_MESSAGE,
                retrieval_executed=False,
                query_understanding=query_understanding,
                results=[],
                trace=_build_trace(
                    query_understanding=(
                        query_understanding
                    ),
                    action=action,
                    retrieval_executed=False,
                    candidate_count=0,
                    returned_count=0,
                ),
            )

        retrieval_query = self._build_retrieval_query(
            query_understanding
        )

        candidates = self.hybrid_retriever.retrieve(
            query=retrieval_query,
            top_k=max(
                top_k,
                self.candidate_pool_size,
            ),
            semantic_candidate_count=max(
                200,
                self.candidate_pool_size,
            ),
        )

        if query_understanding.pr_number is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.pr_number
                == query_understanding.pr_number
            ]

        conditions = (
            query_understanding.metadata_conditions
        )

        scored_candidates: list[
            tuple[
                RetrievalResult,
                float,
                float,
                float,
                int,
                int,
            ]
        ] = []

        for candidate in candidates:
            section_boost = (
                _calculate_section_boost(
                    result=candidate,
                    detected_sections=(
                        query_understanding
                        .detected_sections
                    ),
                )
            )

            metadata_match_count = (
                _count_condition_matches(
                    result=candidate,
                    conditions=conditions,
                )
            )

            metadata_boost = (
                _calculate_metadata_boost(
                    matched_count=(
                        metadata_match_count
                    ),
                    total_count=len(conditions),
                )
            )

            governed_score = (
                candidate.hybrid_score
                + section_boost
                + metadata_boost
            )

            if (
                query_understanding.pr_number
                is not None
                and candidate.pr_number
                == query_understanding.pr_number
            ):
                governed_score += 0.35

            scored_candidates.append(
                (
                    candidate,
                    section_boost,
                    metadata_boost,
                    governed_score,
                    metadata_match_count,
                    len(conditions),
                )
            )

        strict_metadata_filter_applied = False

        if conditions:
            fully_matching_candidates = [
                item
                for item in scored_candidates
                if item[4] == item[5]
            ]

            if fully_matching_candidates:
                scored_candidates = (
                    fully_matching_candidates
                )

                strict_metadata_filter_applied = True

        scored_candidates.sort(
            key=lambda item: (
                item[4],
                item[3],
                item[0].exact_match_score,
                item[0].hybrid_score,
            ),
            reverse=True,
        )

        selected_candidates = [
            item
            for item in scored_candidates
            if item[3]
            >= self.minimum_governed_score
        ][:top_k]

        if not selected_candidates:
            action = "abstain_no_evidence"

            return GovernedRetrievalResponse(
                query=query,
                action=action,
                message=NO_EVIDENCE_MESSAGE,
                retrieval_executed=True,
                query_understanding=query_understanding,
                results=[],
                trace=_build_trace(
                    query_understanding=(
                        query_understanding
                    ),
                    action=action,
                    retrieval_executed=True,
                    candidate_count=len(candidates),
                    returned_count=0,
                    retrieval_query=retrieval_query,
                    strict_metadata_filter_applied=(
                        strict_metadata_filter_applied
                    ),
                ),
            )

        governed_results: list[
            GovernedRetrievalResult
        ] = []

        for rank, item in enumerate(
            selected_candidates,
            start=1,
        ):
            (
                candidate,
                section_boost,
                metadata_boost,
                governed_score,
                metadata_match_count,
                metadata_condition_total,
            ) = item

            governed_results.append(
                GovernedRetrievalResult(
                    rank=rank,
                    chunk_id=candidate.chunk_id,
                    pr_number=candidate.pr_number,
                    document_id=(
                        candidate.document_id
                    ),
                    section=candidate.section,
                    text=candidate.text,
                    base_hybrid_score=round(
                        candidate.hybrid_score,
                        6,
                    ),
                    section_boost=round(
                        section_boost,
                        6,
                    ),
                    metadata_boost=round(
                        metadata_boost,
                        6,
                    ),
                    governed_score=round(
                        governed_score,
                        6,
                    ),
                    metadata_condition_matches=(
                        metadata_match_count
                    ),
                    metadata_condition_total=(
                        metadata_condition_total
                    ),
                    metadata=candidate.metadata,
                )
            )

        action = "retrieve"

        return GovernedRetrievalResponse(
            query=query,
            action=action,
            message=(
                "Relevant pull-request evidence "
                "was retrieved successfully."
            ),
            retrieval_executed=True,
            query_understanding=query_understanding,
            results=governed_results,
            trace=_build_trace(
                query_understanding=(
                    query_understanding
                ),
                action=action,
                retrieval_executed=True,
                candidate_count=len(candidates),
                returned_count=len(
                    governed_results
                ),
                retrieval_query=retrieval_query,
                strict_metadata_filter_applied=(
                    strict_metadata_filter_applied
                ),
            ),
        )