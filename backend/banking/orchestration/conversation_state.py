from __future__ import annotations

import re
from typing import Any


PUBLIC_BANKING_RE = re.compile(
    r"\b("
    r"atlas_demo|bank|banking|products?|carte|cards?|cib|gold|visa|platinum|pack|account|compte|"
    r"savings?|loans?|credits?|financ(?:e|ing|ement)?|invest(?:ment|ments)?|business|corporate|sme|"
    r"mobile banking|online banking|atlas_demo real time|documents?|eligibility|eligible|"
    r"apply|subscribe|open|request|features?|overview|compare|conditions?|fees?|replacement|lost card"
    r")\b",
    re.I,
)
CONVERSATION_STATE_KEYS = {"current_topic", "current_product", "current_audience", "last_comparison", "last_public_query"}


def analyze_public_question(message: str) -> dict[str, Any]:
    try:
        from RAG.retrieval import retriever

        return retriever.analyze_public_question(message)
    except Exception:
        return {"question_type": "factual", "mentioned_products": []}


def sanitize_conversation_state(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    state = {key: value.get(key) for key in CONVERSATION_STATE_KEYS if value.get(key)}
    if "last_comparison" in state and not isinstance(state["last_comparison"], list):
        state.pop("last_comparison", None)
    return state


def update_conversation_state(previous_state: dict[str, Any], original_message: str, resolved_query: str) -> dict[str, Any]:
    state = dict(previous_state)
    text = f"{original_message} {resolved_query}"
    analysis = analyze_public_question(resolved_query)
    topic = _topic_from_text(resolved_query) or _topic_from_text(original_message)
    audience = _audience_from_text(text)
    products = analysis.get("mentioned_products") or []
    if topic:
        state["current_topic"] = topic
        if not products and "compare" not in resolved_query.lower():
            state.pop("current_product", None)
            state.pop("last_comparison", None)
    if products:
        state["current_product"] = products[-1]
    if audience:
        state["current_audience"] = audience
    if analysis.get("is_comparison") and products:
        state["last_comparison"] = products
    if PUBLIC_BANKING_RE.search(resolved_query) or products:
        state["last_public_query"] = resolved_query
    return {key: state[key] for key in ("current_topic", "current_product", "current_audience", "last_comparison", "last_public_query") if state.get(key)}


def resolve_follow_up(message: str, history: list[dict], conversation_state: dict[str, Any] | None = None) -> str:
    text = " ".join((message or "").split())
    lowered = text.lower().strip(" ?!.")
    state_resolution = _resolve_from_conversation_state(text, conversation_state or {})
    if state_resolution:
        return state_resolution
    if not history:
        return tex
    previous = _recent_topic_message(history)
    if not previous:
        return tex
    prev_analysis = analyze_public_question(previous)
    prev_type = prev_analysis.get("question_type", "factual")
    if re.search(r"\b(which one|better|everyday use)\b", lowered) and prev_analysis.get("mentioned_products"):
        products = " and ".join(prev_analysis["mentioned_products"])
        return f"compare {products} for everyday use"
    if len(lowered.split()) > 6:
        return tex
    if re.match(r"^(and|what about)\s+", lowered):
        subject = re.sub(r"^(and|what about)\s+", "", lowered, flags=re.I).strip()
        if subject:
            if "retail customer" in subject or subject == "retail":
                return f"{previous} for retail customers"
            lead = {
                "factual": "what is",
                "eligibility": "who is eligible for",
                "procedure": "how do I use",
                "fees_limits": "how much does",
                "required_documents": "what documents are needed for",
                "comparison": "what is",
            }.get(prev_type, "what is")
            return f"{lead} {subject}?"
    if lowered in {"for jeune", "for young customers", "for young retail customers", "jeune"}:
        return f"{previous} for young retail customers"
    if lowered in {"and the international one", "what about the international one"}:
        return "what is the international version?"
    if lowered in {"how much does it cost", "how much does it cost?"}:
        products = prev_analysis.get("mentioned_products") or []
        if products:
            return f"how much does {products[-1]} cost?"
    if "eligibility" in lowered or "eligible" in lowered:
        products = prev_analysis.get("mentioned_products") or []
        if products:
            return f"who is eligible for {products[-1]}?"
    if "documents" in lowered:
        products = prev_analysis.get("mentioned_products") or []
        target = products[-1] if products else _previous_topic(previous)
        return f"what documents are needed for {target}?"
    if lowered in {"not just tre", "not only tre", "not just for tre", "not only for tre", "not tre"}:
        return f"{previous} Include non-TRE options too."
    if re.fullmatch(r"(i am |i'm |)?(jeune|young),?\s+not\s+tre", lowered):
        return f"{previous} for young retail customers, not TRE"
    if lowered in {"tell me more", "i want to know more", "more details", "know more"}:
        return f"tell me more about {_previous_topic(previous)}"
    return tex


def _last_user_message(history: list[dict]) -> str:
    for item in reversed(history or []):
        if item.get("role") == "user" and str(item.get("content") or "").strip():
            return str(item["content"])
    return ""


def _recent_topic_message(history: list[dict]) -> str:
    for item in reversed((history or [])[-6:]):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lowered = content.lower().strip(" ?!.")
        if lowered in {
            "tell me more",
            "i want to know more",
            "what about that one",
            "and for retail customers",
            "not tre",
            "not just tre",
            "i am jeune not tre",
            "i am jeune, not tre",
        }:
            continue
        if analyze_public_question(content).get("mentioned_products") or PUBLIC_BANKING_RE.search(content):
            return conten
    return _last_user_message(history)


def _audience_from_text(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(not tre|non-tre|jeune,?\s+not\s+tre|young,?\s+not\s+tre)\b", lowered):
        return "young_retail"
    if re.search(r"\b(young|jeune|student|22|started working)\b", lowered):
        return "young_retail"
    if re.search(r"\b(retail|individual|particulier)\b", lowered):
        return "retail"
    if re.search(r"\b(tre|democityian.*abroad|résidents|residents abroad)\b", lowered):
        return "TRE"
    if re.search(r"\b(business|businesses|company|corporate|sme|pme|enterprise|merchant)\b", lowered):
        return "business"
    return None


def _topic_from_text(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(cards?|carte|cib|gold|visa|platinum)\b", lowered):
        return "cards"
    if re.search(r"\b(savings?|saving|compte épargne|compte epargne)\b", lowered):
        return "savings"
    if re.search(r"\b(investment|investments|placement|placements|sicav|fcp)\b", lowered):
        return "investments"
    if re.search(r"\b(loans?|credits?|financing|financement)\b", lowered):
        return "loans"
    if re.search(r"\b(account|compte|open|opening|sign up|signup)\b", lowered):
        return "accounts"
    if re.search(r"\b(products?|offer|available|range)\b", lowered):
        return "products"
    return None


def _resolve_from_conversation_state(message: str, conversation_state: dict[str, Any]) -> str | None:
    lowered = message.lower().strip(" ?!.")
    topic = conversation_state.get("current_topic")
    product = conversation_state.get("current_product")
    audience = conversation_state.get("current_audience")
    comparison = conversation_state.get("last_comparison") or []
    last_query = conversation_state.get("last_public_query")
    if re.search(r"\b(which one|better|everyday use)\b", lowered) and comparison:
        return f"compare {' and '.join(comparison)} for everyday use"
    if lowered in {"tell me more", "i want to know more", "more details", "know more", "what about that one"}:
        if product:
            return f"tell me more about {product}"
        if topic in {"savings", "investments"}:
            qualifier = f" for {audience}" if audience else ""
            return f"tell me more about Atlas Demo savings and investment products{qualifier}"
        if topic:
            qualifier = f" for {audience}" if audience else ""
            return f"tell me more about Atlas Demo {topic}{qualifier}"
        return "Please tell me which product or service you want to know more about."
    if lowered in {"why", "why?", "why that one", "why this one", "why that product"}:
        if topic == "products" or last_query:
            return "Why do you recommend that product for me?"
    if lowered in {"and my segment", "and my segment?"}:
        return "What is my banking segment?"
    if re.match(r"^(and|what about)\s+", lowered):
        subject = re.sub(r"^(and|what about)\s+", "", lowered, flags=re.I).strip()
        if "retail customer" in subject or subject == "retail":
            base = last_query or f"what {topic or 'products'} does Atlas Demo offer?"
            return f"{base} for retail customers"
    if lowered in {"for jeune", "for young customers", "for young retail customers", "jeune"}:
        base = last_query or f"what {topic or 'products'} does Atlas Demo offer?"
        return f"{base} for young retail customers"
    if lowered in {"not tre", "not just tre", "not only tre", "not just for tre", "not only for tre"} and last_query:
        return f"{last_query} Include non-TRE options too."
    if re.fullmatch(r"(i am |i'm |)?(jeune|young),?\s+not\s+tre", lowered):
        base = last_query or f"what {topic or 'products'} does Atlas Demo offer?"
        return f"{base} for young retail customers, not TRE"
    if "documents" in lowered:
        if product:
            return f"what documents are needed for {product}?"
        if topic == "accounts":
            return "what documents are needed to open an Atlas Demo bank account?"
    if re.search(r"\b(do i own|does this belong to me)\b", lowered) and product:
        return f"Do I own {product}?"
    return None


def _previous_topic(previous: str) -> str:
    analysis = analyze_public_question(previous)
    products = analysis.get("mentioned_products") or []
    if products:
        return products[-1]
    previous_l = previous.lower()
    if re.search(r"\b(cards?|carte)\b", previous_l):
        return "Atlas Demo card products"
    if re.search(r"\b(loans?|credit|financing|financement)\b", previous_l):
        return "Atlas Demo loans and financing products"
    if re.search(r"\b(account|compte|opening|sign up|signup)\b", previous_l):
        return "opening an Atlas Demo bank account"
    if re.search(r"\b(savings?|saving|invest|placement)\b", previous_l):
        return "Atlas Demo savings and investment products"
    return previous
