from __future__ import annotations

from typing import Any, Dict, List
import re


POLICY_KEYWORDS = {
    "policy",
    "policies",
    "pto",
    "leave",
    "holiday",
    "holidays",
    "remote",
    "work",
    "expense",
    "expenses",
    "reimbursement",
    "device",
    "devices",
    "security",
    "conduct",
    "handbook",
    "employee",
    "employees",
    "office",
    "vpn",
    "password",
    "incident",
    "incidents",
    "confidential",
    "data",
    "harassment",
    "disciplinary",
    "equipment",
}


REFUSAL_MESSAGE = (
    "I can only answer questions about the company policy documents in this system."
)


def normalize_text(text: str) -> str:
    """
    Lowercase and normalize whitespace for simple rule checks.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer for rule-based checks.
    """
    return re.findall(r"[a-zA-Z']+", normalize_text(text))


def is_policy_question(question: str) -> bool:
    """
    Detect whether a question is likely about the indexed company policy corpus.

    Strategy:
    - simple keyword overlap
    - intentionally conservative
    """
    tokens = set(tokenize(question))

    if not tokens:
        return False

    overlap = tokens.intersection(POLICY_KEYWORDS)

    # Require at least one meaningful overlap term
    return len(overlap) >= 1


def is_out_of_scope_question(question: str) -> bool:
    """
    Return True if the question appears unrelated to company policies.
    """
    return not is_policy_question(question)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Cap returned answer length.
    """
    text = text.strip()

    if len(text) <= max_chars:
        return text

    shortened = text[:max_chars].rstrip()

    # Try not to cut off mid-word
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]

    return shortened + "..."


def has_citations(result: Dict[str, Any]) -> bool:
    """
    Check whether the result contains at least one usable citation.
    """
    citations = result.get("citations", [])
    return isinstance(citations, list) and len(citations) > 0


def build_refusal_response() -> Dict[str, Any]:
    """
    Standard refusal response for out-of-scope questions.
    """
    return {
        "answer": REFUSAL_MESSAGE,
        "citations": [],
        "mode": "guardrail_refusal",
    }


def build_insufficient_support_response() -> Dict[str, Any]:
    """
    Response used when the system does not have enough support to answer safely.
    """
    return {
        "answer": (
            "I could not find enough support in the indexed company policy documents "
            "to answer that safely."
        ),
        "citations": [],
        "mode": "guardrail_insufficient_support",
    }


def apply_output_guardrails(
    result: Dict[str, Any],
    max_answer_chars: int = 500,
) -> Dict[str, Any]:
    """
    Apply output-side guardrails to a result.

    Rules:
    - require at least one citation for a full answer
    - cap answer length
    """
    guarded = dict(result)

    if not has_citations(guarded):
        return build_insufficient_support_response()

    answer = str(guarded.get("answer", "")).strip()
    guarded["answer"] = truncate_text(answer, max_answer_chars)

    return guarded