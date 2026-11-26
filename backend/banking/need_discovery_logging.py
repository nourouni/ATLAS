"""Privacy-safe prospective evidence capture for new-need discovery.

This module records only potentially useful gap signals. It intentionally does
not log normal responses, private customer context, or policy/security noise.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CHATBOT_GAP_EVENTS_PATH = REPO_ROOT / "archetype_data" / "artifacts" / "need_discovery" / "chatbot_gap_events.jsonl"

_BANKING_CONTENT_RE = re.compile(
    r"\b(account|bank|banking|card|cash|charge|credit|debit|fee|loan|mortgage|payment|product|"
    r"refund|saving|transfer|transaction|beneficiar|deposit|withdraw|atlas_demo|service|currency|spending|expense|category|cashflow|budget)\b",
    re.I,
)
_EXCLUDED_RE = re.compile(
    r"\b(unauthenticated|authentication required|another customer|other customer|prompt injection|"
    r"jailbreak|unsupported transaction|execute (a )?transfer|send money|internal exception|"
    r"server error|network error|malformed|empty message|message too long)\b",
    re.I,
)
_OPERATIONAL_HUMAN_RE = re.compile(
    r"\b(fraud|approve|approval|reject|rejected|freeze|frozen|appeal|staff-only|staff only|"
    r"bank decision|execute|modify account|change account)\b",
    re.I,
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _is_meaningful_question(question: str) -> bool:
    return bool(question and len(question) >= 8 and _BANKING_CONTENT_RE.search(question))


def _excluded_from_gap_logging(question: str, failure_code: str, route_reason: str) -> bool:
    combined = f"{question} {failure_code} {route_reason}"
    return bool(_EXCLUDED_RE.search(combined))


def _potential_gap_reason(result: dict[str, Any], question: str) -> str | None:
    failure_code = _text(result.get("failure_code")).lower()
    route = _text(result.get("route")).upper()
    intent = _text(result.get("intent")).upper()
    insufficient = bool(result.get("insufficient_evidence"))
    if "public_evidence_unavailable" in failure_code:
        return "public_evidence_unavailable"
    if insufficient and route == "PUBLIC_INFO":
        return "public information request completed without sufficient evidence"
    if route == "HUMAN_REQUIRED" or intent == "HUMAN_REQUIRED" or failure_code == "human_required":
        if _OPERATIONAL_HUMAN_RE.search(question):
            return None
        return _text(result.get("failure_reason")) or "human_required for a potentially service-related banking question"
    if route == "UNKNOWN" or intent == "UNKNOWN" or failure_code in {"unknown_intent", "low_confidence", "ambiguous_request"}:
        return _text(result.get("failure_reason")) or failure_code or "unknown banking intent"
    if failure_code in {"retrieval_failed", "retrieval_unavailable"} and _is_meaningful_question(question):
        return failure_code
    if failure_code == "customer_analytics_unavailable" and _is_meaningful_question(question):
        return failure_code
    return None


def build_chatbot_gap_event(
    *,
    question: str,
    result: dict[str, Any],
    request_id: str | None,
    event_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any] | None:
    """Return a prospective gap event, or None for ordinary/unsafe noise."""
    question = _text(question)[:2000]
    failure_code = _text(result.get("failure_code"))
    route_reason = _text(result.get("route_reason"))
    if not _is_meaningful_question(question) or _excluded_from_gap_logging(question, failure_code, route_reason):
        return None
    reason = _potential_gap_reason(result, question)
    if reason is None:
        return None
    evidence_status = "unavailable" if "unavailable" in reason else "insufficient" if "insufficient" in reason else "escalated"
    rag_context = result.get("rag_context") or []
    retrieval_status = None
    if rag_context:
        retrieval_status = "evidence_available" if any(item.get("evidence_count") for item in rag_context if isinstance(item, dict)) else "no_evidence"
    return {
        "event_id": event_id or f"chatgap_{uuid.uuid4().hex}",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "conversation_id_or_request_id": _text(request_id),
        "user_question": question,
        "actual_route": _text(result.get("route")),
        "intent": _text(result.get("intent")),
        "failure_code": failure_code,
        "retrieval_status": retrieval_status,
        "evidence_status": evidence_status,
        "potential_gap_reason": reason,
    }


def append_chatbot_gap_event(
    *,
    question: str,
    result: dict[str, Any],
    request_id: str | None,
    path: Path = CHATBOT_GAP_EVENTS_PATH,
) -> dict[str, Any] | None:
    event = build_chatbot_gap_event(question=question, result=result, request_id=request_id)
    if event is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return even
