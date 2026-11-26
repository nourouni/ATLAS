from __future__ import annotations

import re
from typing import Any


PRODUCT_NAME_RE = re.compile(r"\b(?:Carte|Pack|Compte|Cr[eé]dobliga|Webank|Bons\s+de\s+Tr[eé]sor|Attijari\s+Real\s+Time)(?:\s+[A-ZÀ-ÿ][\wÀ-ÿ+-]*){0,4}", re.I)
SCORE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%?\b")


def validate_structured_grounding(intent: str | None, answer: str | None, state: dict[str, Any]) -> dict[str, Any]:
    if intent not in {"CUSTOMER_CONTEXT", "RECOMMENDATION", "RECOMMENDATION_EXPLANATION"} or not answer:
        return {"passed": True, "unsupported": []}

    allowed = _allowed_terms(state)
    unsupported = []
    for name in PRODUCT_NAME_RE.findall(answer):
        normalized = _norm(name)
        if normalized and normalized not in allowed["products"]:
            unsupported.append(name)

    if "segment" in answer.lower():
        segment_names = allowed["segments"]
        if segment_names and not any(term in _norm(answer) for term in segment_names):
            unsupported.append("segment")

    if intent in {"RECOMMENDATION", "RECOMMENDATION_EXPLANATION"}:
        allowed_scores = allowed["scores"]
        for score in SCORE_RE.findall(answer):
            if score not in allowed_scores and any(char.isdigit() for char in score):
                unsupported.append(score)

    return {"passed": not unsupported, "unsupported": sorted(set(unsupported))}


def deterministic_structured_summary(intent: str | None, state: dict[str, Any]) -> str:
    if intent == "CUSTOMER_CONTEXT":
        ctx = state.get("customer_context") or {}
        products = ", ".join(item.get("product_name", "") for item in ctx.get("owned_products", []) if item.get("product_name"))
        segment = (ctx.get("segment") or {}).get("segment_name") or (ctx.get("segment") or {}).get("segment_label")
        message = str(state.get("message") or "").lower()
        if "segment" in message and "product" not in message:
            if not segment:
                return "I could not find your banking segment in the permitted customer information."
            explanation = (ctx.get("segment") or {}).get("customer_friendly_explanation") or {}
            summary = explanation.get("customer_friendly_summary")
            evolution = explanation.get("evolution_message")
            return f"Your behavioral segment is **{segment}**.\n\n{summary}\n\n{evolution}" if summary else f"Your behavioral segment is **{segment}**."
        ownership = re.search(r"\bdo i own\s+(.+?)[?.!]*$", message, re.I)
        if ownership:
            requested = _norm(ownership.group(1))
            owned = [item.get("product_name", "") for item in ctx.get("owned_products", [])]
            match = next((name for name in owned if _norm(name) == requested), None)
            if match:
                return f"Yes. **{match}** is one of your current products."
            return f"No. **{ownership.group(1).strip(' ?.!')}** is not listed among your current products."
        if "product" in message or "own" in message:
            names = [item.get("product_name", "") for item in ctx.get("owned_products", []) if item.get("product_name")]
            if not names:
                return "I could not find any owned products in your permitted customer information."
            lines = "\n".join(f"* **{name}**" for name in names)
            return f"You currently have **{len(names)} products**:\n{lines}"
        parts = []
        if products:
            parts.append(f"You currently have these products: {products}.")
        if segment:
            parts.append(f"Your banking segment is {segment}.")
        return " ".join(parts) or "I can access your authenticated customer context, but there is not enough permitted structured data to answer that detail."

    recs = ((state.get("recommendation_context") or {}).get("recommendations") or [])[:3]
    if not recs:
        return "No stored recommendation is available for this customer."
    if intent == "RECOMMENDATION_EXPLANATION":
        question = _norm(state.get("message"))
        targeted = [item for item in recs if _norm(item.get("product_name")) in question]
        if targeted:
            recs = targeted
    explanation = intent == "RECOMMENDATION_EXPLANATION"
    heading = "**Why it may fit**" if explanation else "**Recommended for you**"
    sections = [heading]
    for item in recs:
        name = item.get("product_name") or item.get("product_id") or "Recommended product"
        reason = _customer_recommendation_reason(item.get("reason"))
        provenance = item.get("provenance") or {}
        bullets = _recommendation_reason_bullets(reason, provenance)
        product_block = [f"**{name}**", "Why it may fit:"]
        product_block.extend(f"* {bullet}" for bullet in bullets[:3])
        sections.append("\n".join(product_block))
    return "\n\n".join(sections)


def _customer_recommendation_reason(reason: Any) -> str:
    """Remove internal score/provenance wording while retaining stored facts."""
    text = str(reason or "").strip()
    text = re.sub(r"\s+with a?\s*\d+(?:[.,]\d+)?%\s+product propensity score", "", text, flags=re.I)
    text = re.sub(r"\bpropensity portfolio affinity\s*\d+(?:[.,]\d+)?\.?", "", text, flags=re.I)
    text = re.sub(r"\b(?:from|based on) ontology evidence\b", "based on the available product information", text, flags=re.I)
    text = re.sub(r"\bneed_\d+\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.strip(" .")


def _recommendation_reason_bullets(reason: str, provenance: dict[str, Any]) -> list[str]:
    bullets = [item.strip() for item in re.split(r"\s*;\s*|(?<=[.!?])\s+(?=[A-Z])", reason) if item.strip()]
    if provenance.get("matched_needs"):
        needs = [re.sub(r"^need_\d+\s*:\s*", "", str(value), flags=re.I).strip() for value in provenance["matched_needs"]]
        needs = [value for value in needs if value]
        if needs and not any("need" in item.lower() for item in bullets):
            bullets.insert(0, "It matches this need: " + ", ".join(needs))
    if not bullets and provenance.get("eligible") == "eligible":
        bullets.append("It passed the available eligibility and compatibility checks.")
    if not bullets:
        bullets.append("It is included in your stored recommendations.")
    return bullets


def _allowed_terms(state: dict[str, Any]) -> dict[str, set[str]]:
    products: set[str] = set()
    segments: set[str] = set()
    scores: set[str] = set()
    customer_context = state.get("customer_context") or {}
    for item in customer_context.get("owned_products") or []:
        _add(products, item.get("product_name"))
        _add(products, item.get("product_id"))
    segment = customer_context.get("segment") or {}
    _add(segments, segment.get("segment_name"))
    _add(segments, segment.get("segment_label"))

    for item in (state.get("recommendation_context") or {}).get("recommendations") or []:
        _add(products, item.get("product_name"))
        _add(products, item.get("product_id"))
        if item.get("score") is not None:
            scores.add(str(item.get("score")))
    return {"products": products, "segments": segments, "scores": scores}


def _add(values: set[str], value: Any) -> None:
    normalized = _norm(value)
    if normalized:
        values.add(normalized)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())
