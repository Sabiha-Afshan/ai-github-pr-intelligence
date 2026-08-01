from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from src.rag.governed_retrieval import (
    GovernedRetrievalResponse,
    GovernedRetriever,
)


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:3b"

OLLAMA_CHAT_ENDPOINT = "/api/chat"

GENERATION_SYSTEM_PROMPT = """
You are a governed GitHub pull-request intelligence assistant.

Your role is to answer questions using only the supplied pull-request
evidence.

Mandatory rules:

1. Use only facts explicitly present in the supplied evidence.
2. Do not invent repository facts, PR facts, predictions, risks, dates,
   authors, titles, review decisions, policies, or recommendations.
3. Every factual sentence must end with at least one inline evidence
   citation such as [E1].
4. Evidence citations must use the exact format [E1], [E2], [E3], etc.
5. Cite only evidence identifiers supplied in the prompt.
6. Do not cite an evidence item that does not support the statement.
7. The answer field itself must contain the inline citations.
8. Listing citations only inside evidence_used is not sufficient.
9. Clearly distinguish:
   - observed repository facts;
   - model predictions;
   - deterministic policy results;
   - recommended review actions.
10. Do not describe a prediction as a confirmed outcome.
11. Do not describe a governed ranking score as a probability.
12. When the evidence is insufficient, say so directly.
13. Do not answer unrelated questions.
14. Do not include markdown tables.
15. Keep the answer concise but sufficiently explanatory.

Example valid answer:

{
  "answer": "PR #123 was opened by the recorded author [E1]. The model predicts a lower merge probability, but this is not a confirmed outcome [E2].",
  "evidence_used": ["E1", "E2"],
  "insufficient_evidence": false,
  "limitations": [
    "The merge result is a model prediction rather than a confirmed outcome."
  ]
}

Return valid JSON using exactly this structure:

{
  "answer": "Grounded answer with inline citations.",
  "evidence_used": ["E1", "E2"],
  "insufficient_evidence": false,
  "limitations": [
    "Relevant limitation when applicable."
  ]
}
""".strip()


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    rank: int
    pr_number: int | None
    section: str
    text: str
    governed_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "pr_number": self.pr_number,
            "section": self.section,
            "text": self.text,
            "governed_score": self.governed_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CitationValidation:
    citations_found: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    uncited_factual_answer: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations_found": self.citations_found,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "uncited_factual_answer": self.uncited_factual_answer,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GroundedGenerationResponse:
    query: str
    action: str
    message: str
    answer: str
    evidence_used: list[str]
    insufficient_evidence: bool
    limitations: list[str]
    retrieval_response: GovernedRetrievalResponse
    evidence: list[EvidenceItem]
    citation_validation: CitationValidation
    model: str | None
    generation_executed: bool
    generation_latency_ms: float
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration_ms: float | None
    raw_model_response: dict[str, Any] | None
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "action": self.action,
            "message": self.message,
            "answer": self.answer,
            "evidence_used": self.evidence_used,
            "insufficient_evidence": self.insufficient_evidence,
            "limitations": self.limitations,
            "retrieval_response": (
                self.retrieval_response.to_dict()
            ),
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "citation_validation": (
                self.citation_validation.to_dict()
            ),
            "model": self.model,
            "generation_executed": (
                self.generation_executed
            ),
            "generation_latency_ms": (
                self.generation_latency_ms
            ),
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "total_duration_ms": self.total_duration_ms,
            "raw_model_response": self.raw_model_response,
            "trace": self.trace,
        }


class OllamaConnectionError(RuntimeError):
    pass


class OllamaGenerationError(RuntimeError):
    pass


class InvalidModelResponseError(RuntimeError):
    pass


def _truncate_text(
    text: str,
    maximum_characters: int,
) -> str:
    normalised_text = re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()

    if len(normalised_text) <= maximum_characters:
        return normalised_text

    return (
        normalised_text[:maximum_characters].rstrip()
        + "..."
    )


def _extract_json_object(
    content: str,
) -> dict[str, Any]:
    cleaned_content = str(content or "").strip()

    if cleaned_content.startswith("```"):
        cleaned_content = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_content,
            flags=re.IGNORECASE,
        )

        cleaned_content = re.sub(
            r"\s*```$",
            "",
            cleaned_content,
        )

    try:
        parsed = json.loads(cleaned_content)
    except json.JSONDecodeError:
        object_match = re.search(
            r"\{.*\}",
            cleaned_content,
            flags=re.DOTALL,
        )

        if object_match is None:
            raise InvalidModelResponseError(
                "The Ollama response did not contain a valid JSON object."
            )

        try:
            parsed = json.loads(
                object_match.group(0)
            )
        except json.JSONDecodeError as error:
            raise InvalidModelResponseError(
                "The Ollama response contained malformed JSON."
            ) from error

    if not isinstance(parsed, dict):
        raise InvalidModelResponseError(
            "The Ollama response JSON must be an object."
        )

    return parsed


def _normalise_evidence_used(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_items = re.findall(
            r"E\d+",
            value,
            flags=re.IGNORECASE,
        )
    elif isinstance(value, list):
        raw_items = [
            str(item)
            for item in value
        ]
    else:
        raw_items = [str(value)]

    normalised_items: list[str] = []

    for item in raw_items:
        match = re.search(
            r"E(\d+)",
            item,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        evidence_id = f"E{int(match.group(1))}"

        if evidence_id not in normalised_items:
            normalised_items.append(
                evidence_id
            )

    return normalised_items


def _normalise_limitations(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()

        return (
            [stripped]
            if stripped
            else []
        )

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    stripped = str(value).strip()

    return (
        [stripped]
        if stripped
        else []
    )


def _extract_answer_citations(
    answer: str,
) -> list[str]:
    citations = re.findall(
        r"\[(E\d+)\]",
        str(answer or ""),
        flags=re.IGNORECASE,
    )

    normalised_citations: list[str] = []

    for citation in citations:
        normalised = citation.upper()

        if normalised not in normalised_citations:
            normalised_citations.append(
                normalised
            )

    return normalised_citations


def _valid_selected_evidence(
    evidence_used: list[str],
    available_evidence_ids: set[str],
) -> list[str]:
    valid_ids: list[str] = []

    for evidence_id in evidence_used:
        normalised = evidence_id.upper()

        if (
            normalised in available_evidence_ids
            and normalised not in valid_ids
        ):
            valid_ids.append(normalised)

    return valid_ids


def _repair_missing_inline_citations(
    answer: str,
    evidence_used: list[str],
    available_evidence_ids: set[str],
) -> tuple[str, bool]:
    """
    Deterministically attaches the model-selected valid evidence IDs to
    uncited answer paragraphs.

    This repair is only permitted when:
    - the answer contains text;
    - there are no inline citations already;
    - the model selected at least one valid evidence ID;
    - no unsupported evidence ID is introduced.
    """
    stripped_answer = str(answer or "").strip()

    if not stripped_answer:
        return stripped_answer, False

    if _extract_answer_citations(stripped_answer):
        return stripped_answer, False

    selected_valid_ids = _valid_selected_evidence(
        evidence_used=evidence_used,
        available_evidence_ids=(
            available_evidence_ids
        ),
    )

    if not selected_valid_ids:
        return stripped_answer, False

    citation_suffix = " ".join(
        f"[{evidence_id}]"
        for evidence_id in selected_valid_ids
    )

    paragraphs = re.split(
        r"\n\s*\n",
        stripped_answer,
    )

    repaired_paragraphs: list[str] = []

    for paragraph in paragraphs:
        cleaned_paragraph = paragraph.strip()

        if not cleaned_paragraph:
            continue

        if _extract_answer_citations(
            cleaned_paragraph
        ):
            repaired_paragraphs.append(
                cleaned_paragraph
            )
            continue

        repaired_paragraphs.append(
            f"{cleaned_paragraph} {citation_suffix}"
        )

    repaired_answer = "\n\n".join(
        repaired_paragraphs
    )

    return repaired_answer, True


def _validate_citations(
    answer: str,
    evidence_used: list[str],
    available_evidence_ids: set[str],
    insufficient_evidence: bool,
) -> CitationValidation:
    answer_citations = (
        _extract_answer_citations(answer)
    )

    combined_citations: list[str] = []

    for citation in (
        answer_citations + evidence_used
    ):
        normalised = citation.upper()

        if normalised not in combined_citations:
            combined_citations.append(
                normalised
            )

    valid_citations = [
        citation
        for citation in combined_citations
        if citation in available_evidence_ids
    ]

    invalid_citations = [
        citation
        for citation in combined_citations
        if citation not in available_evidence_ids
    ]

    answer_has_meaningful_text = bool(
        str(answer or "").strip()
    )

    uncited_factual_answer = (
        answer_has_meaningful_text
        and not insufficient_evidence
        and len(answer_citations) == 0
    )

    passed = (
        len(invalid_citations) == 0
        and not uncited_factual_answer
    )

    return CitationValidation(
        citations_found=combined_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        uncited_factual_answer=(
            uncited_factual_answer
        ),
        passed=passed,
    )


def _nanoseconds_to_milliseconds(
    value: Any,
) -> float | None:
    try:
        return round(
            float(value) / 1_000_000,
            3,
        )
    except (TypeError, ValueError):
        return None


class GroundedResponseGenerator:
    def __init__(
        self,
        governed_retriever: (
            GovernedRetriever | None
        ) = None,
        model: str = DEFAULT_OLLAMA_MODEL,
        ollama_base_url: str = (
            DEFAULT_OLLAMA_BASE_URL
        ),
        request_timeout_seconds: int = 180,
        evidence_top_k: int = 5,
        maximum_evidence_characters: int = 2400,
        temperature: float = 0.0,
    ) -> None:
        self.governed_retriever = (
            governed_retriever
            if governed_retriever is not None
            else GovernedRetriever(
                candidate_pool_size=200,
                minimum_governed_score=0.20,
            )
        )

        self.model = model

        self.ollama_base_url = (
            ollama_base_url.rstrip("/")
        )

        self.request_timeout_seconds = (
            request_timeout_seconds
        )

        self.evidence_top_k = evidence_top_k

        self.maximum_evidence_characters = (
            maximum_evidence_characters
        )

        self.temperature = temperature

    def _build_evidence(
        self,
        retrieval_response: (
            GovernedRetrievalResponse
        ),
    ) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []

        for index, result in enumerate(
            retrieval_response.results,
            start=1,
        ):
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"E{index}",
                    rank=result.rank,
                    pr_number=result.pr_number,
                    section=result.section,
                    text=_truncate_text(
                        result.text,
                        self.maximum_evidence_characters,
                    ),
                    governed_score=(
                        result.governed_score
                    ),
                    metadata=result.metadata,
                )
            )

        return evidence_items

    def _build_user_prompt(
        self,
        query: str,
        retrieval_response: (
            GovernedRetrievalResponse
        ),
        evidence_items: list[EvidenceItem],
    ) -> str:
        evidence_blocks: list[str] = []

        for item in evidence_items:
            evidence_blocks.append(
                "\n".join(
                    [
                        (
                            f"[{item.evidence_id}] "
                            f"PR #{item.pr_number}"
                        ),
                        f"Section: {item.section}",
                        (
                            "Governed ranking score: "
                            f"{item.governed_score:.6f}"
                        ),
                        "Evidence:",
                        item.text,
                    ]
                )
            )

        detected_sections = (
            retrieval_response
            .query_understanding
            .detected_sections
        )

        metadata_conditions = [
            condition.to_dict()
            for condition in (
                retrieval_response
                .query_understanding
                .metadata_conditions
            )
        ]

        return "\n\n".join(
            [
                f"USER QUESTION:\n{query}",
                (
                    "QUERY INTERPRETATION:\n"
                    f"Detected PR number: "
                    f"{retrieval_response.query_understanding.pr_number}\n"
                    f"Detected sections: "
                    f"{detected_sections}\n"
                    f"Metadata conditions: "
                    f"{metadata_conditions}"
                ),
                (
                    "RETRIEVED EVIDENCE:\n"
                    + "\n\n".join(
                        evidence_blocks
                    )
                ),
                (
                    "Answer using only the retrieved evidence.\n"
                    "The answer field must include inline citations.\n"
                    "Every factual sentence must end with citations "
                    "such as [E1] or [E1] [E2].\n"
                    "Do not place citations only in evidence_used.\n"
                    "Return only the required JSON object."
                ),
            ]
        )

    def _call_ollama(
        self,
        user_prompt: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        float,
    ]:
        endpoint = (
            f"{self.ollama_base_url}"
            f"{OLLAMA_CHAT_ENDPOINT}"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        GENERATION_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "seed": 42,
            },
        }

        request_start = time.perf_counter()

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=(
                    self.request_timeout_seconds
                ),
            )
        except requests.ConnectionError as error:
            raise OllamaConnectionError(
                "Could not connect to Ollama at "
                f"{self.ollama_base_url}. "
                "Confirm that Ollama is running."
            ) from error
        except requests.Timeout as error:
            raise OllamaGenerationError(
                "The Ollama generation request "
                "timed out."
            ) from error
        except requests.RequestException as error:
            raise OllamaGenerationError(
                "The Ollama generation request "
                "failed."
            ) from error

        generation_latency_ms = round(
            (
                time.perf_counter()
                - request_start
            )
            * 1000,
            3,
        )

        if response.status_code != 200:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {
                    "response_text": response.text
                }

            raise OllamaGenerationError(
                "Ollama returned HTTP "
                f"{response.status_code}: "
                f"{error_payload}"
            )

        try:
            response_payload = response.json()
        except ValueError as error:
            raise OllamaGenerationError(
                "Ollama returned a non-JSON "
                "HTTP response."
            ) from error

        message = response_payload.get(
            "message",
            {},
        )

        model_content = message.get(
            "content",
            "",
        )

        parsed_content = _extract_json_object(
            model_content
        )

        return (
            parsed_content,
            response_payload,
            generation_latency_ms,
        )

    def _build_skipped_response(
        self,
        query: str,
        retrieval_response: (
            GovernedRetrievalResponse
        ),
    ) -> GroundedGenerationResponse:
        empty_validation = CitationValidation(
            citations_found=[],
            valid_citations=[],
            invalid_citations=[],
            uncited_factual_answer=False,
            passed=True,
        )

        return GroundedGenerationResponse(
            query=query,
            action=retrieval_response.action,
            message=retrieval_response.message,
            answer=retrieval_response.message,
            evidence_used=[],
            insufficient_evidence=True,
            limitations=[
                (
                    "LLM generation was not executed "
                    "because governed retrieval "
                    "abstained."
                )
            ],
            retrieval_response=retrieval_response,
            evidence=[],
            citation_validation=(
                empty_validation
            ),
            model=None,
            generation_executed=False,
            generation_latency_ms=0.0,
            prompt_eval_count=None,
            eval_count=None,
            total_duration_ms=None,
            raw_model_response=None,
            trace={
                "retrieval_action": (
                    retrieval_response.action
                ),
                "generation_action": "skipped",
                "reason": (
                    retrieval_response.message
                ),
                "evidence_count": 0,
                "citation_repair_applied": False,
                "citation_validation_passed": True,
            },
        )

    def generate(
        self,
        query: str,
    ) -> GroundedGenerationResponse:
        retrieval_response = (
            self.governed_retriever.retrieve(
                query=query,
                top_k=self.evidence_top_k,
            )
        )

        if retrieval_response.action != "retrieve":
            return self._build_skipped_response(
                query=query,
                retrieval_response=(
                    retrieval_response
                ),
            )

        evidence_items = self._build_evidence(
            retrieval_response
        )

        if not evidence_items:
            empty_validation = CitationValidation(
                citations_found=[],
                valid_citations=[],
                invalid_citations=[],
                uncited_factual_answer=False,
                passed=True,
            )

            return GroundedGenerationResponse(
                query=query,
                action="abstain_no_evidence",
                message=(
                    "No governed evidence was "
                    "available for grounded response "
                    "generation."
                ),
                answer=(
                    "No governed evidence was "
                    "available for grounded response "
                    "generation."
                ),
                evidence_used=[],
                insufficient_evidence=True,
                limitations=[
                    (
                        "No evidence was passed to "
                        "the language model."
                    )
                ],
                retrieval_response=(
                    retrieval_response
                ),
                evidence=[],
                citation_validation=(
                    empty_validation
                ),
                model=None,
                generation_executed=False,
                generation_latency_ms=0.0,
                prompt_eval_count=None,
                eval_count=None,
                total_duration_ms=None,
                raw_model_response=None,
                trace={
                    "retrieval_action": (
                        retrieval_response.action
                    ),
                    "generation_action": "skipped",
                    "reason": "no_evidence",
                    "evidence_count": 0,
                    "citation_repair_applied": False,
                    "citation_validation_passed": True,
                },
            )

        user_prompt = self._build_user_prompt(
            query=query,
            retrieval_response=(
                retrieval_response
            ),
            evidence_items=evidence_items,
        )

        (
            parsed_content,
            raw_response,
            generation_latency_ms,
        ) = self._call_ollama(
            user_prompt=user_prompt
        )

        answer = str(
            parsed_content.get(
                "answer",
                "",
            )
        ).strip()

        insufficient_evidence = bool(
            parsed_content.get(
                "insufficient_evidence",
                False,
            )
        )

        evidence_used = _normalise_evidence_used(
            parsed_content.get(
                "evidence_used",
                [],
            )
        )

        limitations = _normalise_limitations(
            parsed_content.get(
                "limitations",
                [],
            )
        )

        available_evidence_ids = {
            item.evidence_id
            for item in evidence_items
        }

        initial_validation = _validate_citations(
            answer=answer,
            evidence_used=evidence_used,
            available_evidence_ids=(
                available_evidence_ids
            ),
            insufficient_evidence=(
                insufficient_evidence
            ),
        )

        citation_repair_applied = False

        if (
            not initial_validation.passed
            and initial_validation
            .uncited_factual_answer
            and not initial_validation
            .invalid_citations
        ):
            (
                repaired_answer,
                citation_repair_applied,
            ) = _repair_missing_inline_citations(
                answer=answer,
                evidence_used=evidence_used,
                available_evidence_ids=(
                    available_evidence_ids
                ),
            )

            answer = repaired_answer

        citation_validation = (
            _validate_citations(
                answer=answer,
                evidence_used=evidence_used,
                available_evidence_ids=(
                    available_evidence_ids
                ),
                insufficient_evidence=(
                    insufficient_evidence
                ),
            )
        )

        if not citation_validation.passed:
            invalid_parts: list[str] = []

            if (
                citation_validation
                .invalid_citations
            ):
                invalid_parts.append(
                    "invalid citations: "
                    + ", ".join(
                        citation_validation
                        .invalid_citations
                    )
                )

            if (
                citation_validation
                .uncited_factual_answer
            ):
                invalid_parts.append(
                    "factual answer contained no "
                    "inline evidence citations and "
                    "could not be safely repaired"
                )

            raise InvalidModelResponseError(
                "Citation validation failed: "
                + "; ".join(invalid_parts)
            )

        if not answer:
            raise InvalidModelResponseError(
                "The model returned an empty answer."
            )

        prompt_eval_count = raw_response.get(
            "prompt_eval_count"
        )

        eval_count = raw_response.get(
            "eval_count"
        )

        total_duration_ms = (
            _nanoseconds_to_milliseconds(
                raw_response.get(
                    "total_duration"
                )
            )
        )

        return GroundedGenerationResponse(
            query=query,
            action="answer",
            message=(
                "A grounded response was generated "
                "from governed pull-request evidence."
            ),
            answer=answer,
            evidence_used=(
                citation_validation.valid_citations
            ),
            insufficient_evidence=(
                insufficient_evidence
            ),
            limitations=limitations,
            retrieval_response=(
                retrieval_response
            ),
            evidence=evidence_items,
            citation_validation=(
                citation_validation
            ),
            model=self.model,
            generation_executed=True,
            generation_latency_ms=(
                generation_latency_ms
            ),
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            total_duration_ms=(
                total_duration_ms
            ),
            raw_model_response=raw_response,
            trace={
                "retrieval_action": (
                    retrieval_response.action
                ),
                "generation_action": "completed",
                "model": self.model,
                "evidence_count": len(
                    evidence_items
                ),
                "available_evidence_ids": sorted(
                    available_evidence_ids
                ),
                "model_selected_evidence": (
                    evidence_used
                ),
                "evidence_used": (
                    citation_validation
                    .valid_citations
                ),
                "initial_citation_validation_passed": (
                    initial_validation.passed
                ),
                "citation_repair_applied": (
                    citation_repair_applied
                ),
                "citation_validation_passed": (
                    citation_validation.passed
                ),
                "insufficient_evidence": (
                    insufficient_evidence
                ),
                "generation_latency_ms": (
                    generation_latency_ms
                ),
                "prompt_eval_count": (
                    prompt_eval_count
                ),
                "eval_count": eval_count,
                "total_duration_ms": (
                    total_duration_ms
                ),
            },
        )