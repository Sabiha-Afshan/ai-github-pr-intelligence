from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.paths import PROJECT_ROOT


VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIRECTORY / "pr_chunks.faiss"
EMBEDDINGS_PATH = VECTOR_STORE_DIRECTORY / "pr_chunk_embeddings.npy"
METADATA_PATH = VECTOR_STORE_DIRECTORY / "pr_chunk_metadata.jsonl"
INDEX_CONFIGURATION_PATH = (
    VECTOR_STORE_DIRECTORY / "index_configuration.json"
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+#.-]+")


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    pr_number: int | None
    document_id: str
    section: str
    text: str
    semantic_score: float
    keyword_score: float
    exact_match_score: float
    hybrid_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "pr_number": self.pr_number,
            "document_id": self.document_id,
            "section": self.section,
            "text": self.text,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "exact_match_score": self.exact_match_score,
            "hybrid_score": self.hybrid_score,
            "metadata": self.metadata,
        }


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(str(text))
    ]


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_scores(
    scores: Sequence[float],
) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if math.isclose(minimum, maximum):
        return [
            1.0 if maximum > 0 else 0.0
            for _ in scores
        ]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def _extract_pr_number(
    query: str,
) -> int | None:
    patterns = [
        r"\bPR\s*#?\s*(\d+)\b",
        r"\bpull request\s*#?\s*(\d+)\b",
        r"#(\d+)\b",
    ]

    for pattern in patterns:
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


def _get_nested_metadata(
    record: dict[str, Any],
) -> dict[str, Any]:
    nested_metadata = record.get("metadata", {})

    if isinstance(nested_metadata, dict):
        return nested_metadata

    return {}


def _get_record_value(
    record: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    value = record.get(key)

    if value is not None:
        return value

    nested_metadata = _get_nested_metadata(record)

    return nested_metadata.get(key, default)


def _get_embedding_text(
    record: dict[str, Any],
) -> str:
    value = (
        record.get("embedding_content")
        or record.get("original_content")
        or record.get("text")
        or ""
    )

    return str(value)


def _get_display_text(
    record: dict[str, Any],
) -> str:
    value = (
        record.get("original_content")
        or record.get("embedding_content")
        or record.get("text")
        or ""
    )

    return str(value)


def _get_section_name(
    record: dict[str, Any],
) -> str:
    value = (
        record.get("section_name")
        or record.get("section")
        or _get_record_value(
            record,
            "section_name",
            "",
        )
    )

    return str(value)


class HybridRetriever:
    def __init__(
        self,
        model_name: str | None = None,
        semantic_weight: float = 0.60,
        keyword_weight: float = 0.25,
        exact_match_weight: float = 0.15,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.exact_match_weight = exact_match_weight

        total_weight = (
            semantic_weight
            + keyword_weight
            + exact_match_weight
        )

        if not math.isclose(
            total_weight,
            1.0,
            rel_tol=1e-6,
        ):
            raise ValueError(
                "semantic_weight, keyword_weight and "
                "exact_match_weight must sum to 1.0."
            )

        self.index_configuration = (
            self._load_index_configuration()
        )

        configured_model_name = (
            self.index_configuration.get(
                "embedding_model_name",
                "all-MiniLM-L6-v2",
            )
        )

        self.model_name = (
            model_name or configured_model_name
        )

        self.embedding_model = SentenceTransformer(
            self.model_name
        )

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        )

        self.metadata = self._load_metadata()

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "Metadata count does not match the "
                "number of FAISS vectors."
            )

        if self.embeddings.shape[0] != self.index.ntotal:
            raise ValueError(
                "Embedding count does not match the "
                "number of FAISS vectors."
            )

        self.document_frequency = (
            self._build_document_frequency()
        )

        self.average_document_length = (
            self._calculate_average_document_length()
        )

        self.pr_number_to_indices = (
            self._build_pr_number_lookup()
        )

    @staticmethod
    def _load_index_configuration() -> dict[str, Any]:
        if not INDEX_CONFIGURATION_PATH.exists():
            raise FileNotFoundError(
                "Index configuration not found: "
                f"{INDEX_CONFIGURATION_PATH}"
            )

        with INDEX_CONFIGURATION_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _load_metadata() -> list[dict[str, Any]]:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {METADATA_PATH}"
            )

        records: list[dict[str, Any]] = []

        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Invalid JSON on metadata line "
                        f"{line_number}."
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        "Metadata line "
                        f"{line_number} is not a JSON object."
                    )

                records.append(record)

        return records

    def _build_document_frequency(
        self,
    ) -> Counter[str]:
        document_frequency: Counter[str] = Counter()

        for record in self.metadata:
            text = _get_embedding_text(record)
            unique_tokens = set(_tokenize(text))
            document_frequency.update(unique_tokens)

        return document_frequency

    def _calculate_average_document_length(
        self,
    ) -> float:
        lengths = [
            len(
                _tokenize(
                    _get_embedding_text(record)
                )
            )
            for record in self.metadata
        ]

        if not lengths:
            return 0.0

        return float(
            sum(lengths) / len(lengths)
        )

    def _build_pr_number_lookup(
        self,
    ) -> dict[int, list[int]]:
        lookup: dict[int, list[int]] = defaultdict(list)

        for index, record in enumerate(self.metadata):
            pr_number = _get_record_value(
                record,
                "pr_number",
            )

            try:
                if pr_number is not None:
                    lookup[int(pr_number)].append(index)
            except (TypeError, ValueError):
                continue

        return dict(lookup)

    def _semantic_search(
        self,
        query: str,
        candidate_count: int,
    ) -> dict[int, float]:
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        result_count = min(
            candidate_count,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            result_count,
        )

        results: dict[int, float] = {}

        for index, score in zip(
            indices[0],
            scores[0],
        ):
            if index < 0:
                continue

            results[int(index)] = _safe_float(score)

        return results

    def _bm25_score(
        self,
        query_tokens: Iterable[str],
        document_tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if not document_tokens:
            return 0.0

        query_token_counts = Counter(query_tokens)
        document_token_counts = Counter(
            document_tokens
        )

        score = 0.0
        document_count = len(self.metadata)
        document_length = len(document_tokens)

        for token, query_frequency in (
            query_token_counts.items()
        ):
            term_frequency = (
                document_token_counts.get(token, 0)
            )

            if term_frequency == 0:
                continue

            document_frequency = (
                self.document_frequency.get(token, 0)
            )

            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            length_normalisation = (
                1
                - b
                + b
                * document_length
                / max(
                    self.average_document_length,
                    1.0,
                )
            )

            denominator = (
                term_frequency
                + k1 * length_normalisation
            )

            score += (
                inverse_document_frequency
                * (
                    term_frequency
                    * (k1 + 1)
                )
                / denominator
                * query_frequency
            )

        return float(score)

    def _keyword_search(
        self,
        query: str,
        candidate_indices: Iterable[int] | None = None,
    ) -> dict[int, float]:
        query_tokens = _tokenize(query)

        if not query_tokens:
            return {}

        if candidate_indices is None:
            indices: Iterable[int] = range(
                len(self.metadata)
            )
        else:
            indices = candidate_indices

        scores: dict[int, float] = {}

        for index in indices:
            record = self.metadata[index]
            document_text = _get_embedding_text(record)
            document_tokens = _tokenize(document_text)

            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=document_tokens,
            )

            if score > 0:
                scores[index] = score

        return scores

    def _exact_match_search(
        self,
        query: str,
    ) -> dict[int, float]:
        pr_number = _extract_pr_number(query)

        if pr_number is None:
            return {}

        matching_indices = (
            self.pr_number_to_indices.get(
                pr_number,
                [],
            )
        )

        return {
            index: 1.0
            for index in matching_indices
        }

    @staticmethod
    def _passes_filters(
        record: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected_value in filters.items():
            actual_value = _get_record_value(
                record,
                key,
            )

            if isinstance(
                expected_value,
                (list, tuple, set),
            ):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        semantic_candidate_count: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        semantic_scores = self._semantic_search(
            query=query,
            candidate_count=semantic_candidate_count,
        )

        exact_match_scores = (
            self._exact_match_search(query)
        )

        explicit_pr_number = _extract_pr_number(query)

        if (
            explicit_pr_number is not None
            and exact_match_scores
        ):
            candidate_indices = set(
                exact_match_scores.keys()
            )

            keyword_scores = self._keyword_search(
                query=query,
                candidate_indices=candidate_indices,
            )

            semantic_scores = {
                index: semantic_scores.get(
                    index,
                    0.0,
                )
                for index in candidate_indices
            }
        else:
            keyword_scores = self._keyword_search(
                query=query
            )

            candidate_indices = set(
                semantic_scores.keys()
            )

            candidate_indices.update(
                keyword_scores.keys()
            )

            candidate_indices.update(
                exact_match_scores.keys()
            )

        filtered_indices = [
            index
            for index in candidate_indices
            if self._passes_filters(
                self.metadata[index],
                filters,
            )
        ]

        if not filtered_indices:
            return []

        semantic_values = [
            semantic_scores.get(index, 0.0)
            for index in filtered_indices
        ]

        keyword_values = [
            keyword_scores.get(index, 0.0)
            for index in filtered_indices
        ]

        normalised_semantic_scores = dict(
            zip(
                filtered_indices,
                _normalise_scores(
                    semantic_values
                ),
            )
        )

        normalised_keyword_scores = dict(
            zip(
                filtered_indices,
                _normalise_scores(
                    keyword_values
                ),
            )
        )

        results: list[RetrievalResult] = []

        for index in filtered_indices:
            record = self.metadata[index]
            nested_metadata = (
                _get_nested_metadata(record)
            )

            semantic_score = (
                normalised_semantic_scores.get(
                    index,
                    0.0,
                )
            )

            keyword_score = (
                normalised_keyword_scores.get(
                    index,
                    0.0,
                )
            )

            exact_match_score = (
                exact_match_scores.get(
                    index,
                    0.0,
                )
            )

            hybrid_score = (
                self.semantic_weight
                * semantic_score
                + self.keyword_weight
                * keyword_score
                + self.exact_match_weight
                * exact_match_score
            )

            raw_pr_number = _get_record_value(
                record,
                "pr_number",
            )

            try:
                pr_number = (
                    int(raw_pr_number)
                    if raw_pr_number is not None
                    else None
                )
            except (TypeError, ValueError):
                pr_number = None

            document_id = str(
                _get_record_value(
                    record,
                    "document_id",
                    "",
                )
            )

            results.append(
                RetrievalResult(
                    chunk_id=str(
                        record.get(
                            "chunk_id",
                            index,
                        )
                    ),
                    pr_number=pr_number,
                    document_id=document_id,
                    section=_get_section_name(record),
                    text=_get_display_text(record),
                    semantic_score=round(
                        semantic_score,
                        6,
                    ),
                    keyword_score=round(
                        keyword_score,
                        6,
                    ),
                    exact_match_score=round(
                        exact_match_score,
                        6,
                    ),
                    hybrid_score=round(
                        hybrid_score,
                        6,
                    ),
                    metadata=nested_metadata,
                )
            )

        results.sort(
            key=lambda result: (
                result.hybrid_score,
                result.exact_match_score,
                result.semantic_score,
                result.keyword_score,
            ),
            reverse=True,
        )

        deduplicated_results: list[
            RetrievalResult
        ] = []

        seen_chunk_ids: set[str] = set()

        for result in results:
            if result.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(result.chunk_id)
            deduplicated_results.append(result)

            if len(deduplicated_results) >= top_k:
                break

        return deduplicated_results