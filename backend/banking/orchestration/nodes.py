from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import ollama
from django.conf import settings
from pydantic import BaseModel, ValidationError

from banking import services
from banking.chatbot_models import get_chatbot_classifier_model, get_chatbot_generator_model
from banking.policy_patterns import (
    CUSTOMER_RE,
    HUMAN_RE,
    RECOMMENDATION_RE,
    is_identity_override_request,
    is_other_customer_reference,
    is_transaction_execution_request,
)
from banking.rag_tool import RAGToolError, rag_tool
from banking.serializers import chatbot_context, chatbot_recommendation

from .conversation_state import (
    PUBLIC_BANKING_RE,
    analyze_public_question,
    resolve_follow_up,
    sanitize_conversation_state,
    update_conversation_state,
)
from .grounding import deterministic_structured_summary, validate_structured_grounding
from .intents import (
    CUSTOMER_CONTEXT,
    HUMAN_REQUIRED,
    PUBLIC_INFO,
    RECOMMENDATION,
    RECOMMENDATION_EXPLANATION,
    UNKNOWN,
)
from .prompts import INTENT_PROMPT, generation_promp
from .state import ChatState


LOGGER = logging.getLogger(__name__)
MAX_MESSAGE_LENGTH = 2000

GREETING_RE = re.compile(r"^\s*(hello|hi|hey|good\s+(morning|afternoon|evening)|salam|salut|bonjour)\s*[.!?]*\s*$", re.I)
GENERIC_BANKING_RE = re.compile(
    r"^\s*what\s+(is|does|are)\s+(a\s+)?(debit card|atm withdrawal|savings account|current account|difference between a savings and current account)\??\s*$",
    re.I,
)
MODAL_RE = re.compile(r"\b(can|could|does|do|is there|will|would)\b", re.I)
ACTION_RE = re.compile(r"\b(set|create|enable|activate|reserve|move|send|execute|schedule|transfer|pay|withdraw|deposit)\b", re.I)
EXPLICIT_ACTION_START_RE = re.compile(
    r"^\s*(set|create|enable|activate|reserve|move|send|execute|schedule|transfer|pay|withdraw|deposit|notify|track|block|cancel)\b",
    re.I,
)
CONFIGURATION_RE = re.compile(r"\b(set(?:\s+up)?|create|enable|activate|configure|schedule|block|cancel)\b", re.I)
CAPABILITY_FEATURE_RE = re.compile(
    r"\b(alerts?|warn(?:ing)?s?|notifications?|monitor(?:ing)?|predict(?:ion|ed)?|forecast|track(?:ing|ed)?|"
    r"automatically|automatic|reserve|buffer|keep\s+aside|confirmation|confirm|short(?:age)?|insufficient|upcoming\s+bills?|"
    r"scheduled\s+(?:payments?|expenses?|bills?))\b",
    re.I,
)
ANALYTICS_RE = re.compile(
    r"\b(spending|spend|transactions?|merchant|category|categories|cashflow|cash flow|income|expenses?|budget|trend|changed|change|increased|decreased|average|compare|comparison|forecast|predict)\b",
    re.I,
)
BANKING_TOPIC_RE = re.compile(
    r"\b(alerts?|notifications?|warnings?|subscription|subscriptions|track(?:ing|ed)?|transfers?|payment|balance|funds?|money|account|bank|card|product|service|saving|loan|credit|reserve|scheduled)\b",
    re.I,
)

class IntentResult(BaseModel):
    intent: Literal[
        "PUBLIC_INFO",
        "CUSTOMER_CONTEXT",
        "RECOMMENDATION",
        "RECOMMENDATION_EXPLANATION",
        "HUMAN_REQUIRED",
        "UNKNOWN",
    ]
    request_mode: Literal[
        "information", "capability", "customer_context", "recommendation", "customer_analytics", "action"
    ] = "information"
    confidence: floa
    reason: str


def detect_request_mode(message: str) -> str:
    """Classify the request's operation without changing the intent taxonomy."""
    text = " ".join(str(message or "").split())
    lowered = text.lower()
    has_question_shape = "?" in text or bool(MODAL_RE.search(text)) or bool(re.match(r"^(what|which|how|why|where|when)\b", lowered))
    declarative_capability_shape = bool(re.match(
        r"^(after|before|do customers|i(?:'|’)?d like|i want(?: to know)?|it would be useful|"
        r"i need(?: a way)?|i wish)\b",
        lowered,
    ))
    if EXPLICIT_ACTION_START_RE.match(text):
        return "action"
    capability_shape = bool(
        has_question_shape
        or re.match(r"^(after|before|do customers|i(?:'|’)d like|it would be useful)\b", lowered)
    )
    capability_shape = capability_shape or declarative_capability_shape
    # A question about a feature remains a capability request even when the
    # feature would use private data or contains a transaction verb.
    asks_availability = bool(re.search(r"\b(whether|if|know|can\s+i|is\s+it\s+possible)\b", lowered))
    configures_now = bool(CONFIGURATION_RE.search(text)) and not asks_availability
    if capability_shape and CAPABILITY_FEATURE_RE.search(text) and not configures_now:
        return "capability"
    if is_transaction_execution_request(text) or (ACTION_RE.search(text) and not has_question_shape):
        return "action"
    if RECOMMENDATION_RE.search(text):
        return "recommendation"
    # Stored services contain forecasts and propensities, but not transaction
    # category/time-series analytics. Treat those requests explicitly.
    if ANALYTICS_RE.search(text) and (
        re.search(r"\b(my|mine|me|i|personal|recently|most)\b", lowered)
        or re.search(r"\b(spending\s+categor(?:y|ies)|how\s+.*spending|spending\s+.*(changed|increased|decreased))\b", lowered)
    ):
        return "customer_analytics"
    if MODAL_RE.search(text) and BANKING_TOPIC_RE.search(text):
        return "capability"
    if CUSTOMER_RE.search(text):
        return "customer_context"
    return "information"


def _ollama_model() -> str:
    return get_chatbot_classifier_model()


def _ollama_timeout() -> float:
    return float(getattr(settings, "CHATBOT_OLLAMA_TIMEOUT", 120))


def _ollama_host() -> str | None:
    return getattr(settings, "CHATBOT_OLLAMA_HOST", None)


def _llm_generate(prompt: str, model: str | None = None) -> str:
    client = ollama.Client(host=_ollama_host(), timeout=_ollama_timeout())
    response = client.chat(
        model=model or _ollama_model(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    return str(response["message"]["content"]).strip()


def _safe_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object in classifier response")
    return json.loads(text[start : end + 1])


def policy_check(state: ChatState) -> ChatState:
    message = " ".join(str(state.get("message") or "").split())
    LOGGER.info(
        "chatbot_policy_check request_id=%s authenticated=%s customer_id=%s",
        state.get("request_id"),
        bool(state.get("authenticated")),
        state.get("customer_id"),
    )
    original_message = message
    incoming_conversation_state = sanitize_conversation_state(state.get("conversation_state") or {})
    message = resolve_follow_up(message, state.get("history") or [], incoming_conversation_state)
    conversation_state = update_conversation_state(incoming_conversation_state, original_message, message)
    request_mode = detect_request_mode(message)
    if not message:
        return {"message": message, "conversation_state": conversation_state, "blocked": True, "route": "ESCALATE", "failure_code": "empty_message"}
    if len(message) > MAX_MESSAGE_LENGTH:
        return {"message": message, "blocked": True, "route": "ESCALATE", "failure_code": "message_too_long"}
    if GREETING_RE.match(message):
        return {
            "message": message,
            "conversation_state": conversation_state,
            "original_message": original_message,
            "resolved_query": message,
            "blocked": False,
            "requires_human": False,
            "insufficient_evidence": False,
            "intent": UNKNOWN,
            "route": "GREETING",
            "route_reason": "deterministic_greeting",
            "final_answer": "Hello. How can I help you with Atlas Demo Bank products or services?",
        }
    generic = _generic_banking_answer(message)
    if generic:
        return {
            "message": message,
            "conversation_state": conversation_state,
            "original_message": original_message,
            "resolved_query": message,
            "blocked": False,
            "requires_human": False,
            "insufficient_evidence": False,
            "intent": UNKNOWN,
            "route": "GENERAL_BANKING",
            "route_reason": "deterministic_general_banking",
            "final_answer": generic,
        }

    if is_other_customer_reference(message):
        return {
            "message": message,
            "conversation_state": conversation_state,
            "intent": UNKNOWN,
            "blocked": True,
            "route": "ESCALATE",
            "route_reason": "deterministic_other_customer_access_denied",
            "failure_code": "other_customer_access_denied",
        }
    if HUMAN_RE.search(message):
        return {
            "message": message,
            "conversation_state": conversation_state,
            "intent": HUMAN_REQUIRED,
            "requires_human": True,
            "route": "ESCALATE",
            "route_reason": "deterministic_human_required",
            "failure_code": "human_required",
        }
    if CUSTOMER_RE.search(message) and not state.get("authenticated"):
        if request_mode == "capability":
            pass
        else:
            return {
                "message": message,
                "conversation_state": conversation_state,
                "request_mode": request_mode,
                "intent": UNKNOWN,
                "blocked": True,
                "route": "ESCALATE",
                "route_reason": "unauthenticated_customer_specific_request",
                "failure_code": "authentication_required",
            }
    if request_mode == "customer_analytics" and not state.get("authenticated"):
        return {
            "message": message,
            "conversation_state": conversation_state,
            "request_mode": request_mode,
            "intent": CUSTOMER_CONTEXT,
            "blocked": True,
            "route": "ESCALATE",
            "route_reason": "unauthenticated_customer_analytics_request",
            "failure_code": "authentication_required",
        }
    if is_transaction_execution_request(message) and request_mode != "capability":
        return {
            "message": message,
            "request_mode": "action",
            "intent": UNKNOWN,
            "blocked": True,
            "route": "ESCALATE",
            "route_reason": "unsupported_transactional_action",
            "failure_code": "unsupported_transaction",
        }
    if request_mode == "action":
        return {
            "message": message,
            "request_mode": request_mode,
            "original_message": original_message,
            "resolved_query": message,
            "conversation_state": conversation_state,
            "blocked": True,
            "route": "ESCALATE",
            "route_reason": "unsupported_action",
            "failure_code": "unsupported_action",
        }
    return {"message": message, "request_mode": request_mode, "original_message": original_message, "resolved_query": message, "conversation_state": conversation_state, "blocked": False, "requires_human": False, "insufficient_evidence": False}


def _generic_banking_answer(message: str) -> str | None:
    if not GENERIC_BANKING_RE.match(message) or re.search(r"\b(atlas_demo|carte|pack)\b", message, re.I):
        return None
    query = message.lower()
    if "debit card" in query:
        return "A debit card lets you pay or withdraw cash using money available in a linked bank account. It is general banking information, not a statement about a specific Atlas Demo product."
    if "atm withdrawal" in query:
        return "An ATM withdrawal means taking cash from your bank account using a card at an automated teller machine. This is a general banking explanation, not a specific Atlas Demo rule."
    if "savings" in query and "current" in query:
        return "A savings account is usually for storing money and earning interest, while a current account is usually for day-to-day payments, deposits, and withdrawals. This is general banking information, not an Atlas Demo-specific product condition."
    if "savings account" in query:
        return "A savings account is a bank account intended for setting money aside and often earning interest. This is general banking information, not a specific Atlas Demo product detail."
    if "current account" in query:
        return "A current account is typically used for everyday banking such as receiving money, making payments, and withdrawing cash. This is general banking information, not a specific Atlas Demo product detail."
    return None


def classify_intent(state: ChatState) -> ChatState:
    threshold = float(getattr(settings, "CHATBOT_INTENT_CONFIDENCE_THRESHOLD", 0.70))
    request_mode = state.get("request_mode") or detect_request_mode(state["message"])
    deterministic_intent = _deterministic_supported_intent(state["message"], request_mode=request_mode)
    if request_mode == "customer_analytics":
        return {
            "intent": CUSTOMER_CONTEXT,
            "request_mode": request_mode,
            "intent_confidence": 1.0,
            "route": "ESCALATE",
            "failure_code": "customer_analytics_unavailable",
            "route_reason": "stored_context_has_no_transaction_analytics",
        }
    try:
        payload = _safe_json(_llm_generate(INTENT_PROMPT.format(message=state["message"]), model=get_chatbot_classifier_model(state)))
        if hasattr(IntentResult, "model_validate"):
            parsed = IntentResult.model_validate(payload)
        else:
            parsed = IntentResult.parse_obj(payload)
    except (Exception, ValidationError, ValueError):
        if deterministic_intent:
            if request_mode == "customer_analytics":
                return {"intent": CUSTOMER_CONTEXT, "request_mode": request_mode, "intent_confidence": 1.0, "route": "ESCALATE", "failure_code": "customer_analytics_unavailable", "route_reason": "stored_context_has_no_transaction_analytics"}
            return {"intent": deterministic_intent, "request_mode": request_mode, "intent_confidence": 1.0, "route": deterministic_intent, "route_reason": "deterministic_supported_intent"}
        return {
            "intent": UNKNOWN,
            "request_mode": request_mode,
            "intent_confidence": 0.0,
            "route": "ESCALATE",
            "failure_code": "unknown_intent",
        }

    low_confidence = parsed.confidence < threshold
    intent = parsed.intent if not low_confidence else (deterministic_intent or UNKNOWN)
    request_mode = request_mode if request_mode != "information" else parsed.request_mode
    if request_mode == "capability":
        intent = PUBLIC_INFO
    elif request_mode == "customer_analytics":
        intent = CUSTOMER_CONTEXT
    elif deterministic_intent in {CUSTOMER_CONTEXT, RECOMMENDATION, RECOMMENDATION_EXPLANATION} and CUSTOMER_RE.search(state["message"]):
        intent = deterministic_inten
    if deterministic_intent == PUBLIC_INFO and intent in {CUSTOMER_CONTEXT, RECOMMENDATION, RECOMMENDATION_EXPLANATION}:
        if not CUSTOMER_RE.search(state["message"]):
            intent = PUBLIC_INFO
    if intent in {CUSTOMER_CONTEXT, RECOMMENDATION, RECOMMENDATION_EXPLANATION} and not state.get("authenticated"):
        return {
            "intent": intent,
            "request_mode": request_mode,
            "intent_confidence": parsed.confidence,
            "blocked": True,
            "route": "ESCALATE",
            "failure_code": "authentication_required",
        }
    if intent == HUMAN_REQUIRED:
        return {
            "intent": intent,
            "request_mode": request_mode,
            "intent_confidence": parsed.confidence,
            "requires_human": True,
            "route": "ESCALATE",
            "failure_code": "human_required",
        }
    if intent == UNKNOWN:
        return {
            "intent": UNKNOWN,
            "request_mode": request_mode,
            "intent_confidence": parsed.confidence,
            "route": "ESCALATE",
            "failure_code": "low_confidence" if low_confidence else "ambiguous_request",
        }
    if request_mode == "customer_analytics":
        return {"intent": CUSTOMER_CONTEXT, "request_mode": request_mode, "intent_confidence": parsed.confidence, "route": "ESCALATE", "failure_code": "customer_analytics_unavailable", "route_reason": "stored_context_has_no_transaction_analytics"}
    return {"intent": intent, "request_mode": request_mode, "intent_confidence": parsed.confidence, "route": intent, "route_reason": parsed.reason[:160]}


def _deterministic_supported_intent(message: str, *, request_mode: str | None = None) -> str | None:
    if is_identity_override_request(message):
        return None
    mode = request_mode or detect_request_mode(message)
    if mode == "customer_analytics":
        return CUSTOMER_CONTEXT
    if mode == "capability":
        return PUBLIC_INFO
    if CUSTOMER_RE.search(message):
        if RECOMMENDATION_RE.search(message):
            if re.search(r"\b(why|explain|reason)\b", message, re.I):
                return RECOMMENDATION_EXPLANATION
            return RECOMMENDATION
        return CUSTOMER_CONTEXT
    if analyze_public_question(message).get("mentioned_products"):
        return PUBLIC_INFO
    if PUBLIC_BANKING_RE.search(message):
        return PUBLIC_INFO
    if BANKING_TOPIC_RE.search(message) and ("?" in message or MODAL_RE.search(message)):
        return PUBLIC_INFO
    return None


_RELEVANCE_STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "do", "does", "can", "could", "would", "will",
    "i", "me", "my", "we", "you", "bank", "banking", "provide", "offer", "please", "what", "how",
    "which", "when", "where", "for", "to", "of", "in", "on", "after", "with", "there", "this",
}
_ANSWERABILITY_CONCEPTS = {
    "alert": {"alert", "alerts", "warn", "warning", "warnings", "notification", "notifications"},
    "tracking": {"track", "tracking", "tracked", "tracking", "recipient", "received", "receive"},
    "prediction": {"predict", "prediction", "predicted", "forecast", "likely"},
    "balance_shortage": {"balance", "fund", "funds", "low", "short", "shortage", "insufficient", "enough"},
    "bill_buffer": {"reserve", "reserved", "buffer", "aside", "automatic", "automatically", "bill", "bills", "payment", "payments"},
    "subscription_monitoring": {"subscription", "subscriptions", "recurring", "monitor", "monitoring"},
}


def _relevance_tokens(value: Any) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z0-9]+", str(value or "").lower()):
        if len(token) <= 2 or token in _RELEVANCE_STOPWORDS:
            continue
        token = re.sub(r"(ing|ed|es|s)$", "", token)
        if len(token) > 2:
            tokens.add(token)
    return tokens


def _public_evidence_answerability(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Assess whether retrieved evidence can answer the requested objective."""
    citations = payload.get("citations") or []
    inspectable = any(
        isinstance(item, dict) and any(key in item for key in ("source_title", "document_id", "product", "question", "content", "score"))
        for item in citations
    ) or bool(payload.get("evidence") or payload.get("results") or payload.get("retrieval_score"))
    if not inspectable:
        return {"relevant": True, "answerable": True, "reason_code": "metadata_not_exposed"}

    evidence_text = [payload.get("answer")]
    evidence_text.extend(
        value
        for item in citations
        if isinstance(item, dict)
        for value in (item.get("source_title"), item.get("document_id"), item.get("product"), item.get("question"), item.get("content"))
    )
    query_terms = _relevance_tokens(query)
    evidence_terms = _relevance_tokens(" ".join(str(value or "") for value in evidence_text))
    overlap = query_terms & evidence_terms
    if len(overlap) < (1 if len(query_terms) <= 2 else 2):
        return {"relevant": False, "answerable": False, "reason_code": "topic_mismatch"}

    lowered_query = query.lower()
    if CAPABILITY_FEATURE_RE.search(query):
        requested_concepts = {
            concept for concept, terms in _ANSWERABILITY_CONCEPTS.items()
            if any(term in query_terms or term in lowered_query for term in terms)
        }
        supported_concepts = {
            concept for concept, terms in _ANSWERABILITY_CONCEPTS.items()
            if _relevance_tokens(" ".join(terms)) & evidence_terms
        }
        if requested_concepts and not requested_concepts.issubset(supported_concepts):
            return {"relevant": True, "answerable": False, "reason_code": "related_but_not_answering"}
    return {"relevant": True, "answerable": True, "reason_code": "supports_requested_capability" if CAPABILITY_FEATURE_RE.search(query) else "supports_requested_question"}


def _public_evidence_relevance(query: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Compatibility wrapper for callers that need a boolean gate."""
    assessment = _public_evidence_answerability(query, payload)
    return bool(assessment["relevant"] and assessment["answerable"]), str(assessment["reason_code"])


def _public_context(state: ChatState) -> dict[str, Any]:
    payload = rag_tool.query(
        query=state["message"],
        top_k=3,
        request_id=state.get("request_id"),
        generator_model=get_chatbot_generator_model(state),
    )
    citations = payload.get("citations") or []
    answer = _sanitize_customer_answer(payload.get("answer")) if payload.get("answered") else None
    assessment = _public_evidence_answerability(state["message"], payload)
    evidence_sufficient = bool(assessment["relevant"] and assessment["answerable"])
    relevance_reason = str(assessment["reason_code"])
    if not evidence_sufficient:
        return {
            "answer": None,
            "citations": [],
            "evidence_count": 0,
            "answered": False,
            "evidence_sufficient": False,
            "relevance_reason": relevance_reason,
            "failure_reason": relevance_reason,
        }
    return {
        "answer": answer,
        "citations": citations,
        "evidence_count": payload.get("evidence_count", len(citations)),
        "failure_reason": payload.get("reason"),
        "answered": bool(payload.get("answered")),
        "evidence_sufficient": evidence_sufficient,
        "relevance_reason": relevance_reason,
        "unanswered_answer": _sanitize_customer_answer(payload.get("answer")) if not payload.get("answered") else None,
    }


def _sanitize_customer_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    cleaned = re.sub(r"\bThe cited source confirms:\s*", "", str(answer), flags=re.I)
    cleaned = re.sub(r"\bDetails not confirmed by the cited evidence are not included\.?\s*(?:\[S\d+\])?", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bThe verified KB contains\s*", "I could verify ", cleaned, flags=re.I)
    cleaned = re.sub(r"\bretrieved evidence\b", "available information", cleaned, flags=re.I)
    return "\n".join(" ".join(line.split()) for line in cleaned.splitlines()).strip()


def build_context_for_intent(intent: str, customer_id: int | None, full_context: dict[str, Any] | None = None) -> dict[str, Any]:
    if intent == CUSTOMER_CONTEXT and full_context:
        safe = chatbot_context(full_context)
        return {
            "customer": safe.get("customer"),
            "segment": safe.get("segment"),
            "owned_products": safe.get("owned_products", []),
        }
    if intent == RECOMMENDATION and customer_id is not None:
        return {"recommendations": [chatbot_recommendation(item) for item in services.get_customer_recommendations(customer_id)[:3]]}
    if intent == RECOMMENDATION_EXPLANATION and customer_id is not None:
        recs = [chatbot_recommendation(item) for item in services.get_customer_recommendations(customer_id)[:3]]
        return {"recommendations": recs, "explanation_fields": ["product_name", "rank", "score", "reason"]}
    return {}


def retrieve_context(state: ChatState) -> ChatState:
    intent = state.get("intent")
    customer_id = state.get("customer_id")
    try:
        if intent == PUBLIC_INFO:
            public_context = _public_context(state)
            if not public_context.get("evidence_sufficient", True) or (not public_context.get("answer") and not public_context.get("evidence_count")):
                response: dict[str, Any] = {
                    # Do not carry rejected evidence into generation.
                    "rag_context": [public_context] if public_context.get("evidence_sufficient", True) else [],
                    "insufficient_evidence": True,
                    "evidence_sufficient": False,
                    "failure_code": "public_evidence_unavailable",
                    "failure_reason": public_context.get("failure_reason"),
                }
                if public_context.get("unanswered_answer"):
                    response["final_answer"] = public_context["unanswered_answer"]
                return response
            return {"rag_context": [public_context], "evidence_sufficient": True}
        if intent == CUSTOMER_CONTEXT:
            if customer_id is None:
                return {"blocked": True, "failure_code": "authentication_required"}
            LOGGER.info(
                "chatbot_customer_context_fetch request_id=%s customer_id=%s",
                state.get("request_id"),
                customer_id,
            )
            full_context = {
                "customer": services.get_customer_profile(customer_id),
                "segment": services.get_customer_segment(customer_id),
                "owned_products": services.get_customer_products(customer_id),
                "propensities": [],
                "forecasts": [],
                "recommendations": [],
            }
            return {"customer_context": build_context_for_intent(intent, customer_id, full_context)}
        if intent in {RECOMMENDATION, RECOMMENDATION_EXPLANATION}:
            if customer_id is None:
                return {"blocked": True, "failure_code": "authentication_required"}
            recommendation_context = build_context_for_intent(intent, customer_id)
            if intent == RECOMMENDATION_EXPLANATION:
                recommendation_context["segment"] = services.get_customer_segment(customer_id)
                provenance = services.get_customer_recommendation_provenance(customer_id)
                for recommendation in recommendation_context.get("recommendations", []):
                    recommendation["provenance"] = provenance.get(recommendation.get("product_name"), {})
            return {"recommendation_context": recommendation_context}
    except RAGToolError:
        return {"insufficient_evidence": True, "failure_code": "public_evidence_unavailable"}
    except services.CustomerNotFoundError:
        return {"insufficient_evidence": True, "failure_code": "customer_data_unavailable"}
    except Exception:
        LOGGER.exception("chatbot_retrieval_error request_id=%s intent=%s", state.get("request_id"), intent)
        return {"insufficient_evidence": True, "failure_code": "retrieval_failed"}
    return {}


def generate_answer(state: ChatState) -> ChatState:
    intent = state.get("intent")
    evidence: dict[str, Any] = {}
    if intent == PUBLIC_INFO:
        public_answers = [item.get("answer") for item in state.get("rag_context", []) if item.get("answer")]
        if public_answers:
            return {"draft_answer": public_answers[0]}
        evidence = {"public_information": state.get("rag_context", [])}
    elif intent == CUSTOMER_CONTEXT:
        if re.search(r"\b(personally eligible|eligible for a loan|qualify for a loan)\b", state["message"], re.I):
            return {
                "draft_answer": (
                    "I can use your authenticated customer context only for permitted account information, "
                    "but this chatbot does not have an authorized rule or model to decide loan eligibility or approval. "
                    "A bank advisor must review the required application details."
                )
            }
        evidence = {"customer_context": state.get("customer_context")}
    elif intent in {RECOMMENDATION, RECOMMENDATION_EXPLANATION}:
        evidence = {"recommendation_context": state.get("recommendation_context")}
    if intent == CUSTOMER_CONTEXT and re.search(r"\b(do i own|my banking segment|my segment|what products do i own|my current products)\b", state["message"], re.I):
        return {"draft_answer": deterministic_structured_summary(intent, state)}
    try:
        return {"draft_answer": _llm_generate(generation_prompt(state["message"], evidence), model=get_chatbot_generator_model(state))}
    except Exception:
        LOGGER.warning(
            "chatbot_generation_fallback request_id=%s intent=%s",
            state.get("request_id"),
            intent,
        )
        # Customer/recommendation data has already been fetched from the
        # trusted services. Keep the response grounded when Ollama is down.
        if intent in {CUSTOMER_CONTEXT, RECOMMENDATION, RECOMMENDATION_EXPLANATION}:
            return {
                "draft_answer": deterministic_structured_summary(intent, state),
                "structured_grounding_fallback_used": True,
            }
        return {"insufficient_evidence": True, "failure_code": "generation_unavailable"}


def validate_grounding(state: ChatState) -> ChatState:
    if state.get("insufficient_evidence") or state.get("blocked") or state.get("requires_human"):
        return {}
    intent = state.get("intent")
    if state.get("rag_context") and intent != PUBLIC_INFO:
        return {"insufficient_evidence": True, "failure_code": "protected_source_mismatch"}
    if state.get("customer_context") and intent != CUSTOMER_CONTEXT:
        return {"insufficient_evidence": True, "failure_code": "protected_source_mismatch"}
    if state.get("recommendation_context") and intent not in {RECOMMENDATION, RECOMMENDATION_EXPLANATION}:
        return {"insufficient_evidence": True, "failure_code": "protected_source_mismatch"}
    if intent == PUBLIC_INFO and not any(item.get("evidence_count") for item in state.get("rag_context", [])):
        return {"insufficient_evidence": True, "failure_code": "public_evidence_unavailable"}
    if intent == CUSTOMER_CONTEXT:
        ctx = state.get("customer_context") or {}
        if not ctx.get("customer") or "owned_products" not in ctx:
            return {"insufficient_evidence": True, "failure_code": "customer_data_unavailable"}
    if intent == RECOMMENDATION and not (state.get("recommendation_context") or {}).get("recommendations"):
        return {"insufficient_evidence": True, "failure_code": "recommendation_unavailable"}
    if intent == RECOMMENDATION_EXPLANATION:
        recs = (state.get("recommendation_context") or {}).get("recommendations") or []
        if not recs or not any(item.get("reason") for item in recs):
            return {"insufficient_evidence": True, "failure_code": "recommendation_explanation_unavailable"}
    if not str(state.get("draft_answer") or "").strip():
        return {"insufficient_evidence": True, "failure_code": "generation_unavailable"}
    structured_validation = validate_structured_grounding(intent, state.get("draft_answer"), state)
    if not structured_validation["passed"]:
        return {
            "draft_answer": deterministic_structured_summary(intent, state),
            "structured_grounding_retry_used": False,
            "structured_grounding_fallback_used": True,
            "structured_grounding_unsupported": structured_validation["unsupported"],
        }
    return {"insufficient_evidence": False}


SAFE_RESPONSES = {
    "empty_message": "Please send a banking question so I can help.",
    "message_too_long": "Please shorten your question and try again.",
    "authentication_required": "Please sign in so I can access customer-specific information.",
    "human_required": "This request needs help from an Atlas Demo Bank advisor.",
    "unsupported_transaction": "I can explain how this works, but I can't make the transaction for you.",
    "unsupported_action": "I can explain the option, but I can't make that change for you.",
    "other_customer_access_denied": "I can't access or share another customer's information.",
    "public_evidence_unavailable": "I don't have verified information about that yet.",
    "customer_data_unavailable": "I couldn't access your banking information for this request.",
    "customer_analytics_unavailable": "I can show your available banking information, but I can't analyze spending by category yet.",
    "recommendation_unavailable": "No stored recommendation is available for this customer.",
    "recommendation_explanation_unavailable": "No stored recommendation explanation is available for this customer.",
    "unknown_intent": "Could you rephrase your banking question?",
    "low_confidence": "I can help, but I need a bit more detail. Which Atlas Demo product, service, or account topic do you mean?",
    "ambiguous_request": "Could you clarify which Atlas Demo product, service, or account topic you mean?",
    "generation_unavailable": "I couldn't prepare an answer just now. Please try again.",
    "retrieval_failed": "I couldn't access the information needed for this answer. Please try again.",
    "protected_source_mismatch": "I couldn't safely verify that answer. Please try another question.",
}


def escalate_or_finish(state: ChatState) -> ChatState:
    if state.get("final_answer"):
        return {"final_answer": state.get("final_answer")}
    if state.get("blocked") or state.get("requires_human") or state.get("insufficient_evidence") or state.get("failure_code"):
        code = state.get("failure_code") or "unknown_intent"
        return {"final_answer": SAFE_RESPONSES.get(code, SAFE_RESPONSES["unknown_intent"])}
    return {"final_answer": state.get("draft_answer")}
