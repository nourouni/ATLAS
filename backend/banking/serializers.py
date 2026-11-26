from __future__ import annotations

from typing import Any


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def chatbot_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "customer_id": profile.get("customer_id"),
            "customer_class": profile.get("customer_class"),
            "employment": profile.get("employment"),
            "income_decile": profile.get("income_decile"),
            "tenure_days": profile.get("tenure_days"),
            "currency": profile.get("currency") or "TND",
        }
    )


def chatbot_segment(segment: dict[str, Any] | None) -> dict[str, Any] | None:
    if not segment:
        return None
    from .segment_explanations import customer_segment_explanation

    return _compact(
        {
            "segment_id": segment.get("segment_id"),
            "segment_label": segment.get("segment_label"),
            "segment_name": segment.get("display_name") or segment.get("segment_name"),
            "description": segment.get("description"),
            "likely_customer_objectives": segment.get("likely_customer_objectives"),
            "potential_risks": segment.get("potential_risks"),
            "customer_friendly_explanation": customer_segment_explanation(segment),
        }
    )


def chatbot_product(product: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "product_id": product.get("product_id"),
            "product_name": product.get("product_name"),
            "category": product.get("category"),
            "family": product.get("family"),
        }
    )


def chatbot_propensity(propensity: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "product_id": propensity.get("product_id"),
            "product_name": propensity.get("product_name"),
            "rank": propensity.get("rank"),
            "propensity_score": propensity.get("propensity_score"),
        }
    )


def chatbot_recommendation(recommendation: dict[str, Any]) -> dict[str, Any]:
    return _compact(
        {
            "product_id": recommendation.get("product_id"),
            "product_name": recommendation.get("product_name"),
            "category": recommendation.get("category"),
            "product_type": recommendation.get("product_type"),
            "classification": recommendation.get("classification"),
            "rank": recommendation.get("rank"),
            "score": recommendation.get("score"),
            "reason": recommendation.get("reason"),
        }
    )


def chatbot_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer": chatbot_profile(context["customer"]),
        "segment": chatbot_segment(context.get("segment")),
        "owned_products": [chatbot_product(item) for item in context.get("owned_products", [])],
        "propensities": [chatbot_propensity(item) for item in context.get("propensities", [])[:5]],
        "recommendations": [chatbot_recommendation(item) for item in context.get("recommendations", [])],
    }


def dashboard_context(context: dict[str, Any]) -> dict[str, Any]:
    return contex
