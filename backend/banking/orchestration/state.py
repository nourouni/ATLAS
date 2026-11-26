from __future__ import annotations

from typing import TypedDic


class ChatState(TypedDict, total=False):
    message: str
    customer_id: int | None
    authenticated: bool
    request_id: str | None
    classifier_model: str | None
    generator_model: str | None
    history: list[dict] | None
    conversation_state: dict | None
    original_message: str | None
    resolved_query: str | None

    intent: str | None
    intent_confidence: float | None
    request_mode: str | None
    route: str | None
    route_reason: str | None

    rag_context: list[dict]
    customer_context: dict | None
    recommendation_context: dict | None

    blocked: bool
    requires_human: bool
    insufficient_evidence: bool
    evidence_sufficient: bool
    failure_code: str | None
    failure_reason: str | None
    structured_grounding_retry_used: bool
    structured_grounding_fallback_used: bool
    structured_grounding_unsupported: list[str]

    draft_answer: str | None
    final_answer: str | None
