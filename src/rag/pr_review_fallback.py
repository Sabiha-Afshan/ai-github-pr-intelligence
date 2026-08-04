"""Question-aware deterministic answers for governed PR review workflows."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceView:
    evidence_id: str
    section: str
    text: str
    pr_number: str


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def classify_question(question: str) -> str:
    """Classify a reviewer request using deterministic, ordered rules.

    Analytical uses of words such as ``merge``, ``approve`` and ``close`` must
    not be treated as write actions. A write-action intent is returned only
    when the question contains an explicit command to perform a GitHub action.
    """

    normalised = _normalise(question).lower()

    # Safety controls must be evaluated before ordinary question types.
    if any(
        term in normalised
        for term in [
            "ignore the governance rules",
            "ignore governance rules",
            "ignore previous instructions",
            "ignore the safety checks",
            "bypass governance",
            "bypass the governance",
            "override the policy",
            "override governance",
            "disregard the evidence",
            "say this pr is safe",
            "declare this pr safe",
        ]
    ):
        return "prompt_injection"

    if any(
        term in normalised
        for term in [
            "safe to approve automatically",
            "approve automatically",
            "automatic approval",
            "auto approve",
            "auto-approved",
            "auto approved",
            "safe for automatic approval",
        ]
    ):
        return "automatic_approval"

    if any(
        term in normalised
        for term in [
            "security review",
            "security approval",
            "security assessment completed",
            "security review completed",
            "has the security review",
        ]
    ):
        return "security_review_status"

    if any(
        term in normalised
        for term in [
            "definitely be merged",
            "definitely merge",
            "will it merge",
            "guaranteed to merge",
            "guarantee a merge",
            "certain to merge",
            "confirmed merge",
            "actual merge outcome",
        ]
    ):
        return "prediction_certainty"

    if any(
        term in normalised
        for term in [
            "test evidence",
            "testing evidence",
            "tests recorded",
            "tests were run",
            "test execution",
            "pytest",
            "tox",
        ]
    ):
        return "testing"

    has_documentation = any(
        term in normalised
        for term in [
            "documentation",
            "document changes",
            "docs changes",
        ]
    )
    has_configuration = any(
        term in normalised
        for term in [
            "configuration",
            "config changes",
        ]
    )
    if has_documentation or has_configuration:
        return "documentation_configuration"

    if any(
        term in normalised
        for term in [
            "security-sensitive",
            "security sensitive",
            "contain security",
            "security changes",
        ]
    ):
        return "security"

    if any(
        term in normalised
        for term in [
            "missing",
            "uncertain",
            "unknown",
            "not provided",
            "not available",
            "cannot be determined",
            "cannot be confirmed",
            "what cannot be confirmed",
            "not confirmed",
            "not stated",
            "not recorded",
        ]
    ):
        return "missing_information"

    if any(
        term in normalised
        for term in [
            "what should",
            "review action",
            "reviewer action",
            "before approving",
            "before approval",
            "review checklist",
            "reviewer check",
            "maintainer check",
        ]
    ):
        return "review_action"

    # Question-specific governance intents must be checked before generic risk.
    if any(
        term in normalised
        for term in [
            "which governance rules",
            "what governance rules",
            "which rules were triggered",
            "what rules were triggered",
            "triggered governance rules",
            "triggered policy rules",
        ]
    ):
        return "governance_rules"

    if any(
        term in normalised
        for term in [
            "require manual review",
            "requires manual review",
            "manual review required",
            "need manual review",
            "needs manual review",
            "does it require manual review",
            "does this pr require manual review",
        ]
    ):
        return "manual_review"

    # Prediction and threshold analysis must run before write-action detection.
    # For example, "merge probability" contains the characters "merge pr" but
    # is an analytical phrase, not an instruction to merge a PR.
    if any(
        term in normalised
        for term in [
            "prediction",
            "predict",
            "predicts",
            "predicted",
            "what does the model predict",
            "model prediction",
            "model output",
            "probability",
            "merge probability",
            "predicted to merge",
            "predicted not to merge",
            "delay",
            "delay prediction",
            "decision threshold",
            "classification threshold",
            "merge threshold",
            "delay threshold",
            "probability threshold",
            "model 1",
            "model 2",
        ]
    ):
        return "prediction"

    # Only explicit commands to perform a GitHub write operation are blocked.
    # Mere discussion of approval, merging, closure, or comments is not enough.
    explicit_write_patterns = [
        r"^\s*approve\s+(?:pr|pull request)\s*#?\d+\b",
        r"^\s*approve\s+and\s+merge\b",
        r"^\s*merge\s+(?:pr|pull request)\s*#?\d+\b",
        r"^\s*close\s+(?:pr|pull request)\s*#?\d+\b",
        r"^\s*reject\s+(?:pr|pull request)\s*#?\d+\b",
        r"^\s*post\s+(?:a\s+)?(?:review\s+)?comment\b",
        r"^\s*submit\s+(?:a\s+)?review\b",
        r"^\s*request\s+changes\b",
        r"^\s*modify\s+(?:the\s+)?repository\b",
        r"^\s*update\s+(?:the\s+)?repository\b",
    ]
    if any(re.search(pattern, normalised) for pattern in explicit_write_patterns):
        return "write_action_request"

    if any(
        term in normalised
        for term in [
            "why critical",
            "why is pr",
            "classified as critical",
            "risk",
            "policy",
            "governance",
            "security concern",
        ]
    ):
        return "risk"

    return "summary"


def force_question_aware_answer(intent: str) -> bool:
    """Use deterministic answers for intents requiring exact coverage or safety."""

    return intent != "summary"


def _extract_value(text: str, label: str) -> str | None:
    pattern = re.compile(
        rf"{re.escape(label)}\s*:\s*(.+?)(?:\s+-\s+[A-Z][A-Za-z _-]*\s*:|$)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _normalise_pr_number(value: Any) -> str:
    text = _normalise(value)
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def _evidence_from_payload(
    response_payload: dict[str, Any],
) -> list[EvidenceView]:
    final_response = response_payload.get("final_response", {}) or {}
    base_response = final_response.get("base_response", {}) or {}
    raw_evidence = base_response.get("evidence", []) or []

    evidence: list[EvidenceView] = []

    for item in raw_evidence:
        evidence_id = _normalise(item.get("evidence_id"))
        section = _normalise(item.get("section"))
        text = _normalise(item.get("text"))
        pr_number = _normalise_pr_number(item.get("pr_number"))

        if evidence_id and text:
            evidence.append(
                EvidenceView(
                    evidence_id=evidence_id,
                    section=section,
                    text=text,
                    pr_number=pr_number,
                )
            )

    return evidence


def scope_response_payload_to_pr(
    response_payload: dict[str, Any],
    selected_pr_number: Any,
) -> tuple[dict[str, Any], int]:
    """
    Return a deep-copied payload containing evidence only for the selected PR.

    The second return value is the number of removed cross-PR evidence items.
    """

    scoped = copy.deepcopy(response_payload)
    selected = _normalise_pr_number(selected_pr_number)

    final_response = scoped.setdefault("final_response", {})
    base_response = final_response.setdefault("base_response", {})
    raw_evidence = base_response.get("evidence", []) or []

    matching: list[dict[str, Any]] = []

    for item in raw_evidence:
        item_pr = _normalise_pr_number(item.get("pr_number"))

        if item_pr == selected:
            matching.append(item)

    removed_count = len(raw_evidence) - len(matching)
    base_response["evidence"] = matching

    retrieval_response = base_response.get("retrieval_response", {}) or {}
    retrieved_items = retrieval_response.get("evidence", None)

    if isinstance(retrieved_items, list):
        retrieval_response["evidence"] = [
            item
            for item in retrieved_items
            if _normalise_pr_number(item.get("pr_number")) == selected
        ]

    scoped["ui_cross_pr_evidence_removed"] = removed_count
    scoped["ui_selected_pr_number"] = selected

    return scoped, removed_count


def deduplicate_inline_citations(answer: Any) -> str:
    """Remove repeated adjacent evidence citations without changing claims."""

    text = str(answer or "")

    # [E1]. [E1] -> [E1].
    text = re.sub(
        r"(\[E\d+\])\s*[.]?\s+\1\b",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # [E1] [E1] -> [E1]
    text = re.sub(
        r"(\[E\d+\])(?:\s+\1)+",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def _cite(statement: str, evidence_ids: list[str]) -> str:
    unique_ids: list[str] = []

    for evidence_id in evidence_ids:
        if evidence_id and evidence_id not in unique_ids:
            unique_ids.append(evidence_id)

    citations = " ".join(f"[{evidence_id}]" for evidence_id in unique_ids)
    return f"{_normalise(statement).rstrip('.')}. {citations}".strip()


def _section_map(
    evidence: list[EvidenceView],
) -> dict[str, list[EvidenceView]]:
    mapped: dict[str, list[EvidenceView]] = {}

    for item in evidence:
        mapped.setdefault(item.section.lower(), []).append(item)

    return mapped


def _first(
    mapped: dict[str, list[EvidenceView]],
    section: str,
) -> EvidenceView | None:
    matches = mapped.get(section.lower(), [])
    return matches[0] if matches else None


def _selected_pr_number(
    mapped: dict[str, list[EvidenceView]],
    fallback: Any,
) -> str:
    identity = _first(mapped, "PR identity")

    if identity:
        value = _extract_value(identity.text, "PR number")

        if value:
            return _normalise_pr_number(value)

    return _normalise_pr_number(fallback)


def _summary(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    identity = _first(mapped, "PR identity")
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")
    predictive = _first(mapped, "Predictive intelligence")
    descriptions = mapped.get("pr description", [])

    if identity:
        repo = _extract_value(identity.text, "Repository")
        number = _extract_value(identity.text, "PR number")
        title = _extract_value(identity.text, "Title")
        author = _extract_value(identity.text, "Author")
        parts: list[str] = []

        if number:
            parts.append(f"PR #{number}")

        if title:
            parts.append(f'is titled "{title}"')

        if repo:
            parts.append(f"in {repo}")

        if author:
            parts.append(f"and was opened by {author}")

        if parts:
            statements.append(_cite(" ".join(parts), [identity.evidence_id]))

    if change:
        total = _extract_value(change.text, "Total changed lines")
        files = _extract_value(change.text, "Changed files")
        commits = _extract_value(change.text, "Commit count")
        parts: list[str] = []

        if total:
            parts.append(f"{total} changed lines")

        if files:
            parts.append(f"{files} affected files")

        if commits:
            parts.append(f"{commits} commits")

        if parts:
            statements.append(
                _cite(
                    "The recorded change size is " + ", ".join(parts),
                    [change.evidence_id],
                )
            )

    for description in descriptions:
        description_text = _extract_value(
            description.text,
            "Description text",
        )

        if description_text and (
            "replace this comment with a description"
            in description_text.lower()
            or "before opening a pr" in description_text.lower()
        ):
            statements.append(
                _cite(
                    "The submitted description contains repository template instructions, so the actual problem and implementation approach are not clearly stated",
                    [description.evidence_id],
                )
            )
            break

    if policy:
        risk = _extract_value(policy.text, "Policy risk band")
        manual = _extract_value(policy.text, "Manual review required")
        rules = _extract_value(policy.text, "Triggered rules")
        parts: list[str] = []

        if risk:
            parts.append(f"{risk} policy risk")

        if manual and manual.lower() == "true":
            parts.append("mandatory manual review")

        if rules:
            parts.append(f"triggered rules {rules}")

        if parts:
            statements.append(
                _cite(
                    "The governance result records " + ", ".join(parts),
                    [policy.evidence_id],
                )
            )

    if predictive:
        outcome = _extract_value(predictive.text, "Merge outcome")
        probability = _extract_value(predictive.text, "Merge probability")
        delay = _extract_value(predictive.text, "Delay outcome")
        details: list[str] = []

        if outcome:
            details.append(outcome)

        if probability:
            details.append(f"merge probability {probability}")

        if delay:
            details.append(delay)

        if details:
            statements.append(
                _cite(
                    "The predictive evidence reports " + ", ".join(details),
                    [predictive.evidence_id],
                )
            )

    return statements[:5]


def _missing(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")

    for description in mapped.get("pr description", []):
        present = _extract_value(description.text, "Description present")
        detailed = _extract_value(
            description.text,
            "Detailed description detected",
        )
        description_text = _extract_value(
            description.text,
            "Description text",
        )

        if present and present.lower() == "false":
            statements.append(
                _cite(
                    "The evidence explicitly records that no PR description is present",
                    [description.evidence_id],
                )
            )
            break

        if detailed and detailed.lower() == "false":
            statements.append(
                _cite(
                    "A detailed PR description was not detected",
                    [description.evidence_id],
                )
            )

        if description_text and (
            "replace this comment with a description"
            in description_text.lower()
            or "before opening a pr" in description_text.lower()
        ):
            statements.append(
                _cite(
                    "The description contains repository template instructions, so the actual problem and implementation details are not clearly stated",
                    [description.evidence_id],
                )
            )

        if "- [ ]" in description.text and "pytest" in description.text.lower():
            statements.append(
                _cite(
                    "The description contains an unchecked test-related checklist item, so completed test execution cannot be confirmed",
                    [description.evidence_id],
                )
            )

    if change:
        reviewer_count = _extract_value(
            change.text,
            "Requested reviewer count",
        )
        test_changes = _extract_value(
            change.text,
            "Test changes detected",
        )
        docs_changes = _extract_value(
            change.text,
            "Documentation changes detected",
        )
        config_changes = _extract_value(
            change.text,
            "Configuration changes detected",
        )
        security_changes = _extract_value(
            change.text,
            "Security-sensitive changes detected",
        )

        if reviewer_count == "0":
            statements.append(
                _cite(
                    "No requested reviewer is recorded",
                    [change.evidence_id],
                )
            )

        absent: list[str] = []

        if test_changes and test_changes.lower() == "false":
            absent.append("test changes")

        if docs_changes and docs_changes.lower() == "false":
            absent.append("documentation changes")

        if config_changes and config_changes.lower() == "false":
            absent.append("configuration changes")

        if absent:
            if len(absent) == 1:
                absent_text = absent[0]
            elif len(absent) == 2:
                absent_text = f"{absent[0]} or {absent[1]}"
            else:
                absent_text = (
                    ", ".join(absent[:-1])
                    + f", or {absent[-1]}"
                )

            statements.append(
                _cite(
                    "The change evidence does not record "
                    + absent_text,
                    [change.evidence_id],
                )
            )

        if security_changes and security_changes.lower() == "true":
            statements.append(
                _cite(
                    "Security-sensitive changes are recorded, but completion of a security review cannot be confirmed from the retrieved evidence",
                    [change.evidence_id],
                )
            )

    if policy:
        risk = _extract_value(policy.text, "Policy risk band")
        manual = _extract_value(policy.text, "Manual review required")

        if manual and manual.lower() == "true":
            statement = (
                f"The PR is classified as {risk} policy risk and requires manual review, but the retrieved evidence does not confirm that the review has been completed"
                if risk
                else (
                    "Manual review is required, but the retrieved evidence "
                    "does not confirm that it has been completed"
                )
            )
            statements.append(_cite(statement, [policy.evidence_id]))

    return statements[:5]


def _governance_rules(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    policy = _first(mapped, "Deterministic policy intelligence")

    if not policy:
        return statements

    rules = _extract_value(policy.text, "Triggered rules")
    categories = _extract_value(policy.text, "Triggered categories")
    score = _extract_value(policy.text, "Policy risk score")
    band = _extract_value(policy.text, "Policy risk band")
    manual = _extract_value(policy.text, "Manual review required")

    if rules:
        statements.append(
            _cite(
                f"The triggered governance rules are {rules}",
                [policy.evidence_id],
            )
        )

    if categories:
        statements.append(
            _cite(
                f"These rules cover the categories {categories}",
                [policy.evidence_id],
            )
        )

    if score or band:
        parts: list[str] = []
        if score:
            parts.append(f"a policy risk score of {score}")
        if band:
            parts.append(f"a {band} risk band")
        statements.append(
            _cite(
                "The same policy assessment records " + " and ".join(parts),
                [policy.evidence_id],
            )
        )

    if manual and manual.lower() == "true":
        statements.append(
            _cite(
                "The governance result also requires manual review",
                [policy.evidence_id],
            )
        )

    return statements[:4]


def _manual_review(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")

    if not policy:
        return statements

    manual = _extract_value(policy.text, "Manual review required")
    score = _extract_value(policy.text, "Policy risk score")
    band = _extract_value(policy.text, "Policy risk band")
    rules = _extract_value(policy.text, "Triggered rules")
    categories = _extract_value(policy.text, "Triggered categories")

    if manual and manual.lower() == "true":
        statements.append(
            _cite(
                "Yes. The deterministic policy evidence explicitly requires manual review",
                [policy.evidence_id],
            )
        )
    elif manual and manual.lower() == "false":
        statements.append(
            _cite(
                "No. The deterministic policy evidence does not require manual review",
                [policy.evidence_id],
            )
        )
    else:
        statements.append(
            _cite(
                "The retrieved policy evidence does not explicitly confirm whether manual review is required",
                [policy.evidence_id],
            )
        )

    support_parts: list[str] = []
    if score:
        support_parts.append(f"a policy risk score of {score}")
    if band:
        support_parts.append(f"a {band} risk band")
    if rules:
        support_parts.append(f"triggered rules {rules}")
    if categories:
        support_parts.append(f"risk categories {categories}")

    if support_parts:
        statements.append(
            _cite(
                "The supporting governance evidence records " + ", ".join(support_parts),
                [policy.evidence_id],
            )
        )

    if change:
        security_changes = _extract_value(change.text, "Security-sensitive changes detected")
        if security_changes and security_changes.lower() == "true":
            statements.append(
                _cite(
                    "The change evidence also records security-sensitive changes",
                    [change.evidence_id],
                )
            )

    return statements[:3]


def _risk(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")

    if policy:
        score = _extract_value(policy.text, "Policy risk score")
        band = _extract_value(policy.text, "Policy risk band")
        rules = _extract_value(policy.text, "Triggered rules")
        categories = _extract_value(policy.text, "Triggered categories")
        manual = _extract_value(policy.text, "Manual review required")

        explanation_parts: list[str] = []

        if score:
            explanation_parts.append(f"a policy risk score of {score}")

        if rules:
            explanation_parts.append(f"triggered rules {rules}")

        if categories:
            explanation_parts.append(
                f"risk findings across {categories}"
            )

        if band and explanation_parts:
            statements.append(
                _cite(
                    f"The PR is classified as {band} risk because the "
                    "policy assessment records "
                    + ", ".join(explanation_parts),
                    [policy.evidence_id],
                )
            )
        elif band:
            statements.append(
                _cite(
                    f"The policy assessment classifies the PR as {band} risk",
                    [policy.evidence_id],
                )
            )
        elif explanation_parts:
            statements.append(
                _cite(
                    "The policy assessment records "
                    + ", ".join(explanation_parts),
                    [policy.evidence_id],
                )
            )

        if manual and manual.lower() == "true":
            statements.append(
                _cite(
                    "This governance result requires manual review before approval",
                    [policy.evidence_id],
                )
            )

    if change:
        security_changes = _extract_value(
            change.text,
            "Security-sensitive changes detected",
        )

        if security_changes and security_changes.lower() == "true":
            statements.append(
                _cite(
                    "The change evidence also records security-sensitive changes, which strengthens the need for careful human review",
                    [change.evidence_id],
                )
            )

    return statements[:5]


def _first_extracted_value(text: str, labels: list[str]) -> str | None:
    """Return the first available labelled value from an evidence string."""

    for label in labels:
        value = _extract_value(text, label)
        if value:
            return value
    return None


def _percentage_number(value: str | None) -> float | None:
    """Convert a percentage-like value such as ``43.71%`` to ``43.71``."""

    if not value:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None

    try:
        number = float(match.group(0))
    except ValueError:
        return None

    # Some stored thresholds may be represented as fractions, such as 0.425.
    if "%" not in value and 0.0 <= number <= 1.0:
        number *= 100.0

    return number


def _prediction(
    question: str,
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    predictive = _first(mapped, "Predictive intelligence")

    if not predictive:
        return []

    normalised_question = _normalise(question).lower()
    threshold_requested = any(
        term in normalised_question
        for term in [
            "decision threshold",
            "merge threshold",
            "classification threshold",
            "probability threshold",
            "threshold for",
            "above the threshold",
            "below the threshold",
        ]
    )

    merge_outcome = _extract_value(predictive.text, "Merge outcome")
    merge_probability = _extract_value(predictive.text, "Merge probability")
    merge_threshold = _first_extracted_value(
        predictive.text,
        [
            "Merge decision threshold",
            "Merge threshold",
            "Decision threshold",
            "Classification threshold",
        ],
    )
    delay_outcome = _extract_value(predictive.text, "Delay outcome")
    delay_probability = _extract_value(predictive.text, "Delay probability")
    delay_threshold = _first_extracted_value(
        predictive.text,
        ["Delay decision threshold", "Delay threshold"],
    )

    statements: list[str] = []

    if threshold_requested:
        if merge_probability:
            statements.append(
                _cite(
                    f"The model reports a merge probability of {merge_probability}",
                    [predictive.evidence_id],
                )
            )

        if merge_threshold:
            statements.append(
                _cite(
                    f"The merge decision threshold is {merge_threshold}",
                    [predictive.evidence_id],
                )
            )

        probability_number = _percentage_number(merge_probability)
        threshold_number = _percentage_number(merge_threshold)

        if probability_number is not None and threshold_number is not None:
            difference = probability_number - threshold_number
            relation = "above" if difference >= 0 else "below"
            outcome_text = (
                merge_outcome.lower()
                if merge_outcome
                else (
                    "predicted to merge"
                    if difference >= 0
                    else "predicted not to merge"
                )
            )
            statements.append(
                _cite(
                    f"The probability is {abs(difference):.2f} percentage points {relation} the threshold, so the model classifies the PR as {outcome_text}",
                    [predictive.evidence_id],
                )
            )
        elif merge_outcome:
            statements.append(
                _cite(
                    f"The recorded model classification is {merge_outcome.lower()}",
                    [predictive.evidence_id],
                )
            )
        elif not merge_threshold:
            statements.append(
                _cite(
                    "The retrieved predictive evidence does not explicitly record the merge decision threshold",
                    [predictive.evidence_id],
                )
            )

        statements.append(
            _cite(
                "This is a decision-support model output and not a confirmed future outcome",
                [predictive.evidence_id],
            )
        )
        return statements[:4]

    if merge_outcome or merge_probability:
        sentence = "The model"

        if merge_outcome:
            sentence += f" reports {merge_outcome.lower()}"

        if merge_probability:
            sentence += f" with a merge probability of {merge_probability}"

        statements.append(_cite(sentence, [predictive.evidence_id]))

    if delay_outcome or delay_probability:
        sentence = "For merge delay, the model"

        if delay_outcome:
            sentence += f" reports {delay_outcome.lower()}"

        if delay_probability:
            sentence += f" with a delay probability of {delay_probability}"

        if delay_threshold:
            sentence += f" against a decision threshold of {delay_threshold}"

        statements.append(_cite(sentence, [predictive.evidence_id]))

    statements.append(
        _cite(
            "These are decision-support model outputs and not confirmed future outcomes",
            [predictive.evidence_id],
        )
    )

    return statements[:3]



def _record_value(record: dict[str, Any] | None, aliases: list[str]) -> Any:
    """Return the first non-empty value from a PR record using aliases."""

    if not record:
        return None

    lower_lookup = {str(key).strip().lower(): key for key in record}

    for alias in aliases:
        key = lower_lookup.get(alias.strip().lower())
        if key is None:
            continue

        value = record.get(key)
        if value is None:
            continue

        text = _normalise(value)
        if text and text.lower() not in {"nan", "none", "null", "na", "n/a"}:
            return value

    return None


def _format_probability(value: Any) -> str:
    """Format a probability supplied as either 0-1 or 0-100."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""

    if 0.0 <= number <= 1.0:
        number *= 100.0

    return f"{number:.2f}%"


def _normalise_binary_prediction(value: Any, positive: str, negative: str) -> str:
    """Convert common binary prediction values into readable outcomes."""

    if isinstance(value, bool):
        return positive if value else negative

    text = _normalise(value).lower()

    if text in {"1", "1.0", "true", "yes", "positive", "merge", "merged"}:
        return positive
    if text in {"0", "0.0", "false", "no", "negative", "not merge", "not merged"}:
        return negative

    if "not" in text and "merge" in text:
        return negative
    if "merge" in text:
        return positive

    return _normalise(value)


def _append_prediction_evidence_from_record(
    response_payload: dict[str, Any],
    selected_pr_number: Any,
    selected_pr_record: dict[str, Any] | None,
) -> EvidenceView | None:
    """Add one governed predictive record when retrieval omitted it.

    Prediction-certainty questions require a predictive record that supports
    both the model output and the approved system limitation that predictions
    are decision-support signals rather than guaranteed future outcomes.
    """

    if not selected_pr_record:
        return None

    existing = _evidence_from_payload(response_payload)
    for item in existing:
        if item.section.lower() == "predictive intelligence":
            return item

    merge_prediction = _record_value(
        selected_pr_record,
        [
            "merge_prediction",
            "predicted_merge",
            "merge_outcome_prediction",
            "model_1_prediction",
            "model1_prediction",
            "predicted_merge_outcome",
            "predicted_merged",
        ],
    )
    merge_probability = _record_value(
        selected_pr_record,
        [
            "merge_probability",
            "predicted_merge_probability",
            "merge_prediction_probability",
            "model_1_probability",
            "model1_probability",
            "merge_outcome_probability",
            "probability_merged",
        ],
    )
    merge_threshold = _record_value(
        selected_pr_record,
        [
            "merge_decision_threshold",
            "merge_threshold",
            "model_1_threshold",
            "decision_threshold",
        ],
    )

    if merge_prediction is None and merge_probability is None:
        return None

    outcome = (
        _normalise_binary_prediction(
            merge_prediction,
            "Predicted to merge",
            "Predicted not to merge",
        )
        if merge_prediction is not None
        else ""
    )
    probability = _format_probability(merge_probability)
    threshold = _format_probability(merge_threshold)

    # If the explicit class is unavailable but probability and threshold are
    # present, derive the same deterministic class used by the production model.
    if not outcome and merge_probability is not None and merge_threshold is not None:
        try:
            probability_number = float(merge_probability)
            threshold_number = float(merge_threshold)
            if probability_number > 1.0:
                probability_number /= 100.0
            if threshold_number > 1.0:
                threshold_number /= 100.0
            outcome = (
                "Predicted to merge"
                if probability_number >= threshold_number
                else "Predicted not to merge"
            )
        except (TypeError, ValueError):
            outcome = ""

    parts = ["Predictive intelligence"]
    if outcome:
        parts.append(f"Merge outcome: {outcome}")
    if probability:
        parts.append(f"Merge probability: {probability}")
    if threshold:
        parts.append(f"Merge decision threshold: {threshold}")

    parts.append(
        "Prediction limitation: Model predictions are decision-support signals "
        "and cannot guarantee the eventual merge result"
    )
    parts.append(
        "Outcome dependency: The final merge result can depend on later human "
        "review, repository decisions, and events not represented by the model"
    )

    final_response = response_payload.setdefault("final_response", {})
    base_response = final_response.setdefault("base_response", {})
    raw_evidence = base_response.setdefault("evidence", [])

    existing_ids = {
        _normalise(item.get("evidence_id"))
        for item in raw_evidence
        if isinstance(item, dict)
    }
    next_number = 1
    while f"E{next_number}" in existing_ids:
        next_number += 1
    evidence_id = f"E{next_number}"

    evidence_item = {
        "evidence_id": evidence_id,
        "rank": len(raw_evidence) + 1,
        "pr_number": _normalise_pr_number(selected_pr_number),
        "section": "Predictive intelligence",
        "governed_score": 1.0,
        "source": "Selected PR predictive model record",
        "text": " - ".join(parts),
    }
    raw_evidence.append(evidence_item)

    retrieval_response = base_response.get("retrieval_response")
    if isinstance(retrieval_response, dict):
        retrieved_items = retrieval_response.get("evidence")
        if isinstance(retrieved_items, list):
            retrieved_items.append(copy.deepcopy(evidence_item))

    return EvidenceView(
        evidence_id=evidence_id,
        section="Predictive intelligence",
        text=evidence_item["text"],
        pr_number=_normalise_pr_number(selected_pr_number),
    )


def _prediction_certainty(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    """Build a fully grounded answer for merge-certainty questions."""

    predictive = _first(mapped, "Predictive intelligence")

    if predictive:
        outcome = _extract_value(predictive.text, "Merge outcome")
        probability = _extract_value(predictive.text, "Merge probability")

        statements = [
            _cite(
                "No. The model's merge result is only a decision-support "
                "prediction and cannot guarantee that the PR will eventually "
                "be merged",
                [predictive.evidence_id],
            )
        ]

        threshold = _extract_value(
            predictive.text,
            "Merge decision threshold",
        )

        if outcome or probability:
            if outcome and probability and threshold:
                detail = (
                    f"The model currently classifies the PR as {outcome.lower()} "
                    f"with a merge probability of {probability} against a "
                    f"decision threshold of {threshold}. The final outcome can "
                    "still depend on later human review, repository decisions, "
                    "and events not represented by the prediction"
                )
            elif outcome and probability:
                detail = (
                    f"The model currently classifies the PR as {outcome.lower()} "
                    f"with a recorded merge probability of {probability}. The "
                    "final outcome can still depend on later human review, "
                    "repository decisions, and events not represented by the "
                    "prediction"
                )
            elif probability:
                detail = (
                    f"The model currently records a merge probability of "
                    f"{probability}, but the retrieved predictive record does not "
                    "state the classification. The final outcome can still depend "
                    "on later human review, repository decisions, and events not "
                    "represented by the prediction"
                )
            else:
                detail = (
                    f"The model currently classifies the PR as {outcome.lower()}, "
                    "but the final outcome can still depend on later human review, "
                    "repository decisions, and events not represented by the "
                    "prediction"
                )

            statements.append(
                _cite(detail, [predictive.evidence_id])
            )

        return statements[:2]

    # No predictive record exists in the selected evidence. Do not attach
    # unrelated PR citations to a predictive claim. The UI presents the
    # approved prediction limitation separately in its limitations section.
    return [
        (
            "No Predictive intelligence record is available in the selected "
            "PR evidence, so the system cannot report a model-based merge "
            "assessment for this question"
        )
    ]


def _testing(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    change = _first(mapped, "Change evidence")

    for description in mapped.get("pr description", []):
        text_lower = description.text.lower()

        if "pytest" in text_lower or "tox" in text_lower:
            if "- [ ]" in description.text:
                statements.append(
                    _cite(
                        "The PR description contains an unchecked instruction to run pytest or tox, so successful test execution is not confirmed",
                        [description.evidence_id],
                    )
                )
            else:
                statements.append(
                    _cite(
                        "The PR description refers to pytest or tox, but the retrieved text does not provide a verified test result",
                        [description.evidence_id],
                    )
                )
            break

    if change:
        test_changes = _extract_value(
            change.text,
            "Test changes detected",
        )
        test_files = _extract_value(
            change.text,
            "Test files changed",
        )

        if test_changes:
            statements.append(
                _cite(
                    f"The change evidence records Test changes detected: {test_changes}",
                    [change.evidence_id],
                )
            )

        if test_files:
            statements.append(
                _cite(
                    f"The recorded number of test files changed is {test_files}",
                    [change.evidence_id],
                )
            )

    if not statements:
        return [
            "The retrieved evidence does not provide a confirmed test-execution result."
        ]

    return statements[:3]


def _documentation_configuration(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    change = _first(mapped, "Change evidence")

    if not change:
        return [
            "Documentation and configuration change status cannot be confirmed because no change-evidence record was retrieved."
        ]

    docs = _extract_value(
        change.text,
        "Documentation changes detected",
    )
    config = _extract_value(
        change.text,
        "Configuration changes detected",
    )
    statements: list[str] = []

    if docs is not None:
        statements.append(
            _cite(
                f"Documentation changes detected: {docs}",
                [change.evidence_id],
            )
        )

    if config is not None:
        statements.append(
            _cite(
                f"Configuration changes detected: {config}",
                [change.evidence_id],
            )
        )

    return statements or [
        _cite(
            "The retrieved change evidence does not state whether documentation or configuration changes were detected",
            [change.evidence_id],
        )
    ]


def _security(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    change = _first(mapped, "Change evidence")

    if not change:
        return [
            "Security-sensitive change status cannot be confirmed because no change-evidence record was retrieved."
        ]

    security = _extract_value(
        change.text,
        "Security-sensitive changes detected",
    )
    security_files = _extract_value(
        change.text,
        "Security-sensitive files changed",
    )

    statements: list[str] = []

    if security is not None:
        statements.append(
            _cite(
                f"Security-sensitive changes detected: {security}",
                [change.evidence_id],
            )
        )

    if security_files is not None:
        statements.append(
            _cite(
                f"Security-sensitive files changed: {security_files}",
                [change.evidence_id],
            )
        )

    return statements or [
        _cite(
            "The retrieved change evidence does not state whether security-sensitive changes were detected",
            [change.evidence_id],
        )
    ]


def _security_review_status(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")
    citations: list[str] = []
    context_parts: list[str] = []

    if change:
        security = _extract_value(
            change.text,
            "Security-sensitive changes detected",
        )

        if security:
            context_parts.append(
                f"security-sensitive changes detected: {security}"
            )
            citations.append(change.evidence_id)

    if policy:
        manual = _extract_value(policy.text, "Manual review required")

        if manual:
            context_parts.append(f"manual review required: {manual}")
            citations.append(policy.evidence_id)

    statement = (
        "The retrieved evidence does not state that a security review has been completed"
    )

    if context_parts:
        statement += "; it only records " + " and ".join(context_parts)

    if citations:
        return [_cite(statement, citations)]

    return [statement + "."]


def _split_recommendations(value: str) -> list[str]:
    cleaned = _normalise(value)
    parts = re.split(r"(?<=[.!?])\s+|;\s*", cleaned)

    return [
        part.strip().rstrip(".")
        for part in parts
        if part.strip()
    ]


def _review_action(mapped: dict[str, list[EvidenceView]]) -> list[str]:
    statements: list[str] = []
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")

    if policy:
        recommendations = _extract_value(
            policy.text,
            "Rule recommendations",
        )
        manual = _extract_value(policy.text, "Manual review required")
        rules = _extract_value(policy.text, "Triggered rules")

        if recommendations:
            for recommendation in _split_recommendations(recommendations):
                statements.append(
                    _cite(recommendation, [policy.evidence_id])
                )

        if manual and manual.lower() == "true":
            statements.insert(
                0,
                _cite(
                    "Complete the required manual review before approval",
                    [policy.evidence_id],
                ),
            )

        if rules:
            statements.append(
                _cite(
                    f"Confirm that the specifically triggered rules {rules} have been addressed",
                    [policy.evidence_id],
                )
            )

    if change:
        security = _extract_value(
            change.text,
            "Security-sensitive changes detected",
        )

        if security and security.lower() == "true":
            statements.append(
                _cite(
                    "Review the recorded security-sensitive changes before approval",
                    [change.evidence_id],
                )
            )

    return statements[:5]


def _automatic_approval(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    policy = _first(mapped, "Deterministic policy intelligence")
    change = _first(mapped, "Change evidence")
    citations: list[str] = []
    reasons: list[str] = []

    if policy:
        risk = _extract_value(policy.text, "Policy risk band")
        manual = _extract_value(policy.text, "Manual review required")
        rules = _extract_value(policy.text, "Triggered rules")

        if risk:
            reasons.append(f"{risk} policy risk")

        if manual and manual.lower() == "true":
            reasons.append("mandatory manual review")

        if rules:
            reasons.append(f"triggered rules {rules}")

        citations.append(policy.evidence_id)

    if change:
        security = _extract_value(
            change.text,
            "Security-sensitive changes detected",
        )

        if security and security.lower() == "true":
            reasons.append("security-sensitive changes")
            citations.append(change.evidence_id)

    statement = "No. This PR should not be approved automatically"

    if reasons:
        statement += " because the evidence records " + ", ".join(reasons)

    return [_cite(statement, citations)] if citations else [statement + "."]


def _write_action_request(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    statements = [
        (
            "This workflow cannot approve, merge, close, reject, request "
            "changes, or post comments on pull requests because it is read-only."
        )
    ]
    policy = _first(mapped, "Deterministic policy intelligence")

    if policy:
        risk = _extract_value(policy.text, "Policy risk band")
        manual = _extract_value(policy.text, "Manual review required")
        details: list[str] = []

        if risk:
            details.append(f"{risk} policy risk")

        if manual and manual.lower() == "true":
            details.append("mandatory manual review")

        if details:
            statements.append(
                _cite(
                    "The selected PR's governance evidence records "
                    + " and ".join(details),
                    [policy.evidence_id],
                )
            )

    return statements


def _prompt_injection(
    mapped: dict[str, list[EvidenceView]],
) -> list[str]:
    statements = [
        (
            "I cannot ignore, bypass, or override the governance controls, "
            "evidence requirements, or read-only restrictions."
        ),
        (
            "A PR can only be assessed using evidence for the selected PR; "
            "an instruction to declare it safe cannot replace that evidence."
        ),
    ]
    policy = _first(mapped, "Deterministic policy intelligence")

    if policy:
        risk = _extract_value(policy.text, "Policy risk band")
        manual = _extract_value(policy.text, "Manual review required")
        details: list[str] = []

        if risk:
            details.append(f"{risk} policy risk")

        if manual and manual.lower() == "true":
            details.append("mandatory manual review")

        if details:
            statements.append(
                _cite(
                    "The selected PR's evidence records "
                    + " and ".join(details),
                    [policy.evidence_id],
                )
            )

    return statements


def limitations_for_intent(intent: str) -> list[str]:
    mapping = {
        "prediction": [
            "Predictions are decision-support signals and are not confirmed future outcomes."
        ],
        "prediction_certainty": [
            "A model prediction cannot guarantee the eventual merge result."
        ],
        "testing": [
            "Checklist instructions or references to test commands do not prove that tests ran successfully."
        ],
        "documentation_configuration": [
            "A detected value of False means the feature extractor did not identify that change type; it does not prove the topic is irrelevant."
        ],
        "security": [
            "Security-sensitive change detection does not by itself establish that a vulnerability exists."
        ],
        "security_review_status": [
            "The absence of a recorded completion status does not prove that no review occurred outside the retrieved data."
        ],
        "automatic_approval": [
            "This workflow provides decision support only and cannot approve a PR automatically."
        ],
        "write_action_request": [
            "This workflow is read-only and performs no GitHub write actions."
        ],
        "prompt_injection": [
            "User or repository instructions cannot override governance rules or tool restrictions."
        ],
    }

    return mapping.get(intent, [])


def build_question_aware_fallback(
    question: str,
    response_payload: dict[str, Any],
    selected_pr_number: Any | None = None,
    selected_pr_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = classify_question(question)

    if intent in {"prediction", "prediction_certainty"}:
        _append_prediction_evidence_from_record(
            response_payload=response_payload,
            selected_pr_number=selected_pr_number,
            selected_pr_record=selected_pr_record,
        )

    evidence = _evidence_from_payload(response_payload)
    mapped = _section_map(evidence)
    pr_number = _selected_pr_number(mapped, selected_pr_number)

    if intent == "prompt_injection":
        statements = _prompt_injection(mapped)
        heading = "Governance instruction protected"
    elif intent == "write_action_request":
        statements = _write_action_request(mapped)
        heading = "Read-only action restriction"
    elif intent == "automatic_approval":
        statements = _automatic_approval(mapped)
        heading = "Automatic approval decision"
    elif intent == "security_review_status":
        statements = _security_review_status(mapped)
        heading = "Security review status"
    elif intent == "prediction_certainty":
        statements = _prediction_certainty(mapped)
        heading = "Prediction certainty"
    elif intent == "testing":
        statements = _testing(mapped)
        heading = "Test evidence"
    elif intent == "documentation_configuration":
        statements = _documentation_configuration(mapped)
        heading = "Documentation and configuration evidence"
    elif intent == "security":
        statements = _security(mapped)
        heading = "Security-sensitive change evidence"
    elif intent == "missing_information":
        statements = _missing(mapped)
        heading = "Missing or uncertain information"
    elif intent == "governance_rules":
        statements = _governance_rules(mapped)
        heading = "Triggered governance rules"
    elif intent == "manual_review":
        statements = _manual_review(mapped)
        heading = "Manual review requirement"
    elif intent == "risk":
        statements = _risk(mapped)
        heading = "Risk and governance evidence"
    elif intent == "prediction":
        statements = _prediction(question, mapped)
        heading = "Prediction evidence"
    elif intent == "review_action":
        statements = _review_action(mapped)
        heading = "Evidence-based review checklist"
    else:
        statements = _summary(mapped)
        heading = "Senior-maintainer summary"

    if not statements:
        if evidence:
            statements = [
                _cite(
                    "The requested detail cannot be determined safely from the current retrieved evidence",
                    [evidence[0].evidence_id],
                )
            ]
        else:
            statements = [
                (
                    f"No governed evidence for PR #{pr_number} was retrieved, "
                    "so the requested detail cannot be determined."
                    if pr_number
                    else (
                        "No governed evidence was retrieved, so the requested "
                        "detail cannot be determined."
                    )
                )
            ]

    answer = "\n".join(
        [
            f"**{heading}**",
            "",
            *[f"- {statement}" for statement in statements],
            "",
            "Details not explicitly stated in the selected PR evidence cannot be confirmed.",
        ]
    )

    return {
        "intent": intent,
        "answer": deduplicate_inline_citations(answer),
        "statement_count": len(statements),
        "evidence_ids": [item.evidence_id for item in evidence],
        "limitations": limitations_for_intent(intent),
    }