from __future__ import annotations

from .intents import HUMAN_REQUIRED, UNKNOWN, SUPPORTED_RETRIEVAL_INTENTS
from .state import ChatState


def after_policy(state: ChatState) -> str:
    if state.get("final_answer"):
        return "finish"
    if state.get("blocked") or state.get("requires_human"):
        return "finish"
    return "classify"


def after_classification(state: ChatState) -> str:
    intent = state.get("intent")
    if state.get("blocked") or state.get("requires_human") or state.get("failure_code") or intent in {HUMAN_REQUIRED, UNKNOWN}:
        return "finish"
    if intent in SUPPORTED_RETRIEVAL_INTENTS:
        return "retrieve"
    return "finish"


def after_retrieval(state: ChatState) -> str:
    if state.get("blocked") or state.get("requires_human") or state.get("insufficient_evidence") or state.get("failure_code"):
        return "finish"
    return "generate"
