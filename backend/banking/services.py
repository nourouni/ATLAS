from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

from django.conf import settings
from django.db import connection
from django.db.models import QuerySe

from .models import (
    Customer,
    CustomerProduct,
    CustomerSegment,
    ModelRun,
    PropensityPrediction,
    Recommendation,
    SegmentForecast,
)
from .behavioral_segment_presentation import (
    resolve_behavioral_segment,
)


class CustomerNotFoundError(ValueError):
    pass


def _decimal(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def get_customer_profile(customer_id: int) -> dict[str, Any]:
    try:
        customer = Customer.objects.get(customer_id=customer_id)
    except Customer.DoesNotExist as exc:
        raise CustomerNotFoundError(f"Customer {customer_id} was not found.") from exc

    return {
        "customer_id": customer.customer_id,
        "segment_label": customer.segment_label,
        "behavioral_segment_id": customer.behavioral_segment_id,
        "forecast_pool": customer.forecast_pool,
        "latent_archetype": customer.latent_archetype,
        "customer_class": customer.customer_class,
        "employment": customer.employment,
        "income_decile": customer.income_decile,
        "tenure_days": customer.tenure_days,
        "source_system": customer.source_system,
        "created_at": customer.created_at.isoformat() if customer.created_at else None,
        "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
    }


def customer_exists(customer_id: int) -> bool:
    return Customer.objects.filter(customer_id=customer_id).exists()


def latest_recommendation_run() -> ModelRun | None:
    return (
        ModelRun.objects.filter(
            run_name="latest_recommendation_refresh",
            model_type="recommendation",
        )
        .exclude(completed_at__isnull=True)
        .order_by("-completed_at", "-run_id")
        .first()
    )


def latest_forecast_run() -> ModelRun | None:
    """Resolve the approved temporal customer-outlook forecast run."""
    return (
        ModelRun.objects.filter(
            run_name="temporal_feature_filled_cluster_pooled",
            model_type="segment_forecast",
        )
        .exclude(completed_at__isnull=True)
        .order_by("-completed_at", "-run_id")
        .first()
    )


def current_ownership_run() -> ModelRun | None:
    """Resolve the ownership snapshot explicitly associated with the current recommendations.

    Older databases do not have the association column yet. They must return
    unavailable rather than silently treating every historical ownership row as current.
    """
    recommendation_run = latest_recommendation_run()
    if recommendation_run is None:
        return None
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'model_runs'
              AND column_name = 'ownership_run_id'
            """
        )
        if cursor.fetchone() is None:
            return None
        cursor.execute("SELECT ownership_run_id FROM model_runs WHERE run_id = %s", [recommendation_run.run_id])
        row = cursor.fetchone()
    ownership_run_id = row[0] if row else None
    if ownership_run_id is None:
        return None
    return (
        ModelRun.objects.filter(run_id=ownership_run_id, model_type="product_ownership")
        .exclude(completed_at__isnull=True)
        .order_by("-completed_at", "-run_id")
        .first()
    )


def get_customer_products(customer_id: int) -> list[dict[str, Any]]:
    ownership_run = current_ownership_run()
    if ownership_run is None:
        return []
    rows = (
        CustomerProduct.objects.filter(customer_id=customer_id, run=ownership_run)
        .select_related("product", "run")
        .order_by("product__product_name", "-loaded_at")
    )
    return [
        {
            "product_id": row.product_id,
            "product_name": row.product.product_name,
            "category": row.product.category,
            "classification": row.product.classification,
            "product_type": row.product.product_type,
            "family": row.family,
            "product_model_segment": row.product_model_segment,
            "run_id": row.run_id,
            "run_name": row.run.run_name,
            "run_completed_at": row.run.completed_at.isoformat() if row.run.completed_at else None,
            "loaded_at": row.loaded_at.isoformat() if row.loaded_at else None,
        }
        for row in rows
    ]


def get_customer_segment(customer_id: int) -> dict[str, Any] | None:
    profile = get_customer_profile(customer_id)
    behavioral_segment_id = profile.get("behavioral_segment_id")
    label = profile.get("segment_label")
    if not behavioral_segment_id and not label:
        return None

    segment = CustomerSegment.objects.filter(segment_id=behavioral_segment_id).first() if behavioral_segment_id else None
    if segment is None and label:
        segment = CustomerSegment.objects.filter(segment_id=label).first() or CustomerSegment.objects.filter(segment_name=label).first()
    if segment is None:
        presentation = resolve_behavioral_segment(behavioral_segment_id)
        return {
            "segment_label": label,
            "behavioral_segment_id": behavioral_segment_id,
            "display_name": presentation.get("display_name"),
            "description": presentation.get("description"),
        }

    result = {
        "segment_id": segment.segment_id,
        "segment_name": segment.segment_name,
        "source_group": segment.source_group,
        "cluster_id": segment.cluster_id,
        "description": segment.description,
        "transaction_behavior": segment.transaction_behavior,
        "balance_and_cashflow_behavior": segment.balance_and_cashflow_behavior,
        "digital_channel_behavior": segment.digital_channel_behavior,
        "likely_customer_objectives": segment.likely_customer_objectives,
        "potential_risks": segment.potential_risks,
        "interpretation_confidence": segment.interpretation_confidence,
        "requires_manual_review": segment.requires_manual_review,
    }
    # Resolve presentation metadata without changing the stored assignment or
    # the original technical segment_name field.
    presentation = resolve_behavioral_segment(segment.segment_id, result)
    result.update(
        {
            "display_name": presentation.get("display_name"),
            "description": presentation.get("description") or result.get("description"),
            "strengths": presentation.get("strengths", []),
            "watch_outs": presentation.get("watch_outs", []),
        }
    )
    return resul


def get_customer_propensities(customer_id: int) -> list[dict[str, Any]]:
    run = latest_recommendation_run()
    if run is None:
        return []
    rows = (
        PropensityPrediction.objects.filter(customer_id=customer_id, run=run)
        .select_related("product", "run")
        .order_by("rank", "-scored_at")
    )
    return [
        {
            "product_id": row.product_id,
            "product_name": row.product.product_name,
            "category": row.product.category,
            "product_type": row.product.product_type,
            "classification": row.product.classification,
            "rank": row.rank,
            "propensity_score": _decimal(row.propensity_score),
            "segment_product_prior": _decimal(row.segment_product_prior),
            "max_owned_to_candidate_affinity": _decimal(row.max_owned_to_candidate_affinity),
            "mean_owned_to_candidate_affinity": _decimal(row.mean_owned_to_candidate_affinity),
            "family": row.family,
            "product_model_segment": row.product_model_segment,
            "run_id": row.run_id,
            "run_name": row.run.run_name,
            "run_completed_at": row.run.completed_at.isoformat() if row.run.completed_at else None,
            "scored_at": row.scored_at.isoformat() if row.scored_at else None,
        }
        for row in rows
    ]


def get_customer_forecast(customer_id: int) -> list[dict[str, Any]]:
    profile = get_customer_profile(customer_id)
    pools = [profile["forecast_pool"]] if profile.get("forecast_pool") else []
    rows: QuerySet[SegmentForecast] = SegmentForecast.objects.none()
    forecast_run = latest_forecast_run()
    if pools and forecast_run is not None:
        rows = (
            SegmentForecast.objects.filter(run=forecast_run, pool__in=pools, metric="avg_balance")
            .select_related("run")
            .order_by("forecast_date", "scenario_case", "model_name")
        )
    return [
        {
            "pool": row.pool,
            "metric": row.metric,
            "forecast_date": row.forecast_date.isoformat(),
            "scenario_case": row.scenario_case,
            "model_name": row.model_name,
            "actual_value": _decimal(row.actual_value),
            "predicted_value": _decimal(row.predicted_value),
            "abs_error": _decimal(row.abs_error),
            "run_id": row.run_id,
            "run_name": row.run.run_name,
            "run_completed_at": row.run.completed_at.isoformat() if row.run.completed_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def get_customer_recommendations(customer_id: int) -> list[dict[str, Any]]:
    run = latest_recommendation_run()
    if run is None:
        return []
    rows = (
        Recommendation.objects.filter(customer_id=customer_id, run=run)
        .select_related("product", "run")
        .order_by("rank", "-created_at")
    )
    return [
        {
            "product_id": row.product_id,
            "product_name": row.product.product_name,
            "category": row.product.category,
            "product_type": row.product.product_type,
            "classification": row.product.classification,
            "rank": row.rank,
            "score": _decimal(row.score),
            "reason": row.reason,
            "run_id": row.run_id,
            "run_name": row.run.run_name,
            "run_completed_at": row.run.completed_at.isoformat() if row.run.completed_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


def get_customer_recommendation_provenance(customer_id: int) -> dict[str, dict[str, Any]]:
    """Rebuild explanation evidence from the existing archetype recommender.

    The operational recommendations table intentionally stores the customer
    output, but not the recommender's intermediate rule-check package.
    """
    try:
        project_root = Path(settings.BASE_DIR).paren
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        recommender = _hybrid_recommender()
        customer_state = recommender.get_customer_state(customer_id)
        checked = recommender.apply_ontology_rules(customer_state, recommender.generate_candidates(customer_state))
        return {
            row["product"]: {
                "matched_needs": row.get("matched_needs", []),
                "eligible": row.get("eligible"),
                "eligibility_basis": row.get("eligibility_basis"),
                "eligibility_rules": row.get("eligibility_rules", []),
                "already_owned": row.get("already_owned", False),
                "missing_requirements": row.get("missing_requirements", []),
                "conflicts": row.get("conflicts", []),
                "alternatives": row.get("alternatives", []),
                "supporting_relations": row.get("supporting_relations", []),
                "relation_evidence": row.get("relation_evidence", []),
                "sources": row.get("sources", []),
                "propensity_score": row.get("propensity_score"),
                "segment_product_prior": row.get("segment_product_prior"),
                "max_owned_to_candidate_affinity": row.get("max_owned_to_candidate_affinity"),
                "need_match": row.get("need_match", 0),
            }
            for row in checked
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _hybrid_recommender():
    """Load the canonical hybrid recommender once per process for explanations."""
    from archetype_data.recommendation.hybrid_recommender import HybridRecommender

    return HybridRecommender()


def get_customer_context(customer_id: int) -> dict[str, Any]:
    profile = get_customer_profile(customer_id)
    return {
        "customer": profile,
        "segment": get_customer_segment(customer_id),
        "owned_products": get_customer_products(customer_id),
        "propensities": get_customer_propensities(customer_id),
        "forecasts": get_customer_forecast(customer_id),
        "recommendations": get_customer_recommendations(customer_id),
    }


def get_customer_dashboard_analytics(customer_id: int) -> dict[str, Any]:
    context = get_customer_context(customer_id)
    return {
        "customer_id": context["customer"]["customer_id"],
        "segment": context["segment"],
        "owned_products": context["owned_products"],
        "propensities": context["propensities"],
        "forecasts": context["forecasts"],
        "recommendations": context["recommendations"],
        "model_runs": _model_runs_from_context(context),
        "missing_outputs": {
            "owned_products": len(context["owned_products"]) == 0,
            "propensities": len(context["propensities"]) == 0,
            "forecasts": len(context["forecasts"]) == 0,
            "recommendations": len(context["recommendations"]) == 0,
        },
    }


def _model_runs_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    runs: list[dict[str, Any]] = []
    for section in ("owned_products", "propensities", "forecasts", "recommendations"):
        for item in context.get(section, []):
            run_id = item.get("run_id")
            if run_id is None or run_id in seen:
                continue
            seen.add(run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "run_name": item.get("run_name"),
                    "run_completed_at": item.get("run_completed_at"),
                    "source_section": section,
                }
            )
    return runs
