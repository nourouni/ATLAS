from __future__ import annotations

from decimal import Decimal
import csv
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from django.conf import settings
from django.db import connection
from django.db.models import Avg, Count, Q

from .models import (
    Customer,
    CustomerProduct,
    CustomerSegment,
    ModelRun,
    PropensityPrediction,
    Recommendation,
    SegmentForecast,
    Product,
)
from .services import CustomerNotFoundError, current_ownership_run, get_customer_context, latest_forecast_run, latest_recommendation_run
from .behavioral_segment_presentation import (
    behavioral_segment_definitions as _presentation_segment_definitions,
    resolve_behavioral_segment,
)


def _number(value: Decimal | int | float | None) -> float | int | None:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _pct(value: Decimal | int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _score(value: Decimal | int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


class _FeaturedLastList(list):
    """Keep normal iteration order while exposing the last item as index 0."""

    def __getitem__(self, index):
        if index == 0 and self:
            return super().__getitem__(-1)
        return super().__getitem__(index)


def _money(value: Decimal | int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.2f}"


def _current_recommendation_run_id() -> int | None:
    run = latest_recommendation_run()
    return run.run_id if run else None


def _current_propensity_rows():
    run_id = _current_recommendation_run_id()
    return PropensityPrediction.objects.filter(run_id=run_id) if run_id is not None else PropensityPrediction.objects.none()


def _current_recommendation_rows():
    run_id = _current_recommendation_run_id()
    return Recommendation.objects.filter(run_id=run_id) if run_id is not None else Recommendation.objects.none()


def _current_ownership_run_id() -> int | None:
    run = current_ownership_run()
    return run.run_id if run else None


def _current_ownership_rows():
    run_id = _current_ownership_run_id()
    return CustomerProduct.objects.filter(run_id=run_id) if run_id is not None else CustomerProduct.objects.none()


RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS_NS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL_NAMED_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"
ONTOLOGY_PATH = settings.BASE_DIR.parent / "atlas_demo_banking_ontology_fixed.owl"
KB_CSV_DIR = settings.BASE_DIR.parent / "knowledge_base" / "csv"
PRODUCT_BROCHURES_PATH = settings.BASE_DIR.parent / "product_brochures_rag.csv"
ARCHETYPE_WEEKLY_PANEL = settings.BASE_DIR.parent / "archetype_data" / "outputs" / "archetype_weekly_panel.parquet"
STATIC_SEGMENT_ASSIGNMENTS = (
    settings.BASE_DIR.parent / "artifacts" / "lookup_data" / "clustering" / "static_segment_assignments.csv"
)
TS_BEST_PROCESSES = settings.BASE_DIR.parent / "archetype_data" / "ts" / "model_registry" / "best_processes_by_metric.csv"
STATIC_FORECAST_POINTS = Path(settings.ATLAS_STATIC_FORECAST_POINTS)
PROMOTED_FORECAST_POINTS = settings.BASE_DIR.parent / "artifacts" / "forecasting" / "forecast_points.csv"
STATIC_SEGMENT_FORECAST_POOLS = {
    "seg_ret_0": "retail_static_2000",
    "seg_ret_1": "retail_static_2001",
    "seg_ret_2": "retail_static_2002",
    "seg_ret_3": "retail_static_2003",
    "seg_ret_4": "retail_static_2004",
    "seg_ent_0": "enterprise_static_1000",
    "seg_ent_1": "enterprise_static_1001",
    "seg_ent_2": "enterprise_static_1002",
}
STATIC_CLUSTER_PROFILE_PATHS = {
    "retail": settings.BASE_DIR.parent / "archetype_data" / "static_clustering" / "retail_cluster_profiles.csv",
    "enterprise": settings.BASE_DIR.parent / "archetype_data" / "static_clustering" / "entreprise_cluster_profiles.csv",
}

GROUP_PRIORITY = [
    "Package",
    "Account",
    "Card",
    "FinancingSolution",
    "DigitalService",
    "Investment",
    "Insurance",
    "Service",
    "ProductFamily",
    "CustomerNeed",
    "BehavioralSegment",
    "EligibilityRule",
    "OwnershipConstraint",
    "EvidenceSource",
    "Product",
]

def _short_id(uri: str | None) -> str:
    if not uri:
        return ""
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _rdf_attr(element: ET.Element, name: str) -> str:
    return element.attrib.get(f"{RDF_NS}{name}", "")


def _choose_group(types: list[str]) -> str:
    names = [_short_id(item) for item in types]
    for group in GROUP_PRIORITY:
        if group in names:
            return group
    return next((name for name in names if name and name != "NamedIndividual"), "Unknown")


def _read_csv(name: str) -> list[dict[str, str]]:
    path = KB_CSV_DIR / name
    return _read_csv_path(path)


def _read_csv_path(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: str | Decimal | int | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def behavioral_segment_definitions() -> dict[str, dict[str, Any]]:
    return _presentation_segment_definitions()


@lru_cache(maxsize=1)
def product_brochure_summaries() -> dict[str, str]:
    """Return concise customer-facing product summaries keyed by ID and name."""
    summaries: dict[str, str] = {}
    for row in _read_csv_path(PRODUCT_BROCHURES_PATH):
        summary = (row.get("short_description") or "").strip()
        if not summary:
            continue
        for key in (row.get("product_id"), row.get("product_name")):
            if key:
                summaries[key.strip().lower()] = summary
    return summaries


def _resolved_segment(segment: dict[str, Any] | None) -> dict[str, Any] | None:
    if not segment:
        return segmen
    # A segment assignment may exist without a matching descriptive CSV row.
    # Keep the assignment and alias usable instead of dereferencing None.
    definition = behavioral_segment_definitions().get(
        segment.get("segment_id") or segment.get("behavioral_segment_id") or ""
    ) or {}
    resolved = {**segment}
    field_map = {
        "cluster_id": "cluster_id",
        "segment_name": "segment_name",
        "description": "description",
        "transaction_behavior": "transaction_behavior",
        "balance_and_cashflow_behavior": "balance_and_cashflow_behavior",
        "digital_channel_behavior": "digital_channel_behavior",
        "likely_customer_objectives": "likely_customer_objectives",
        "potential_risks": "potential_risks",
        "interpretation_confidence": "interpretation_confidence",
        "source_group": "source_group",
    }
    for target, source in field_map.items():
        if definition.get(source):
            resolved[target] = definition[source]
    resolved = resolve_behavioral_segment(resolved.get("segment_id"), resolved)
    if resolved.get("cluster_id") not in (None, ""):
        try:
            resolved["cluster_id"] = int(resolved["cluster_id"])
        except (TypeError, ValueError):
            pass
    return resolved


def _segment_alias(segment_id: str | None, fallback: str = "Unknown") -> str:
    return behavioral_segment_definitions().get(segment_id or "", {}).get("display_name") or (
        "Unclassified behavior" if segment_id else fallback
    )


def _split_indicators(value: str | None, limit: int = 4) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()][:limit]


@lru_cache(maxsize=1)
def product_ontology_graph() -> dict[str, Any]:
    if not ONTOLOGY_PATH.exists():
        return {"nodes": [], "edges": [], "summary": {"source": str(ONTOLOGY_PATH), "node_count": 0, "edge_count": 0}}

    root = ET.parse(ONTOLOGY_PATH).getroot()
    individuals: dict[str, dict[str, Any]] = {}
    for desc in root.findall(f".//{RDF_NS}Description"):
        node_id = _rdf_attr(desc, "about")
        if not node_id:
            continue
        children = list(desc)
        types = [_rdf_attr(child, "resource") for child in children if child.tag == f"{RDF_NS}type" and _rdf_attr(child, "resource")]
        if OWL_NAMED_INDIVIDUAL not in types:
            continue

        label = next(
            (child.text.strip() for child in children if child.tag == f"{RDFS_NS}label" and child.text and child.text.strip()),
            _short_id(node_id),
        )
        props = {
            _local_name(child.tag): child.text.strip()
            for child in children
            if child.text and child.text.strip() and not _rdf_attr(child, "resource")
        }
        individuals[node_id] = {
            "id": node_id,
            "label": label,
            "group": _choose_group(types),
            "description": props.get("description") or props.get("justification") or "",
            "properties": props,
        }

    products = {
        row.product_name.lower(): row
        for row in Product.objects.only("product_id", "product_name", "category", "classification", "product_type")
    }
    products.update(
        {
            row.product_id.lower(): row
            for row in Product.objects.only("product_id", "product_name", "category", "classification", "product_type")
        }
    )
    for node in individuals.values():
        if node.get("group") not in {"Product", "Package", "Account", "Card", "FinancingSolution", "DigitalService", "Investment", "Insurance", "Service"}:
            continue
        product = products.get(str(node.get("label", "")).lower()) or products.get(_short_id(node.get("id")).lower())
        if not product:
            continue
        node["properties"] = {
            **node.get("properties", {}),
            "product_id": product.product_id,
            "category": product.category or "",
            "classification": product.classification or "",
            "product_type": product.product_type or "",
        }

    edges = []
    seen: set[tuple[str, str, str]] = set()
    for desc in root.findall(f".//{RDF_NS}Description"):
        source = _rdf_attr(desc, "about")
        if source not in individuals:
            continue
        for child in list(desc):
            if child.tag == f"{RDF_NS}type":
                continue
            target = _rdf_attr(child, "resource")
            if target not in individuals:
                continue
            key = (source, _local_name(child.tag), target)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": source, "to": target, "label": key[1]})

    return {
        "nodes": list(individuals.values()),
        "edges": edges,
        "summary": {"source": ONTOLOGY_PATH.name, "node_count": len(individuals), "edge_count": len(edges)},
    }


def _signed_pct(from_value: Decimal | int | float | None, to_value: Decimal | int | float | None) -> str:
    if from_value in (None, 0) or to_value is None:
        return "n/a"
    change = ((float(to_value) - float(from_value)) / abs(float(from_value))) * 100
    return f"{change:+.1f}%"


def employee_authorized(user) -> bool:
    if not getattr(user, "is_authenticated", False):
        return False
    try:
        role = user.demo_profile.role
    except Exception:
        return bool(user.is_staff or user.is_superuser)
    return role in {"staff", "admin"} or bool(user.is_staff or user.is_superuser)


def employee_role(user) -> str:
    try:
        return user.demo_profile.role
    except Exception:
        if user.is_superuser:
            return "admin"
        if user.is_staff:
            return "staff"
        return "user"


def portfolio_overview() -> dict[str, Any]:
    total_customers = Customer.objects.count()
    class_counts = {
        row["customer_class"]: row["count"]
        for row in Customer.objects.values("customer_class").annotate(count=Count("*"))
    }
    retail_customers = class_counts.get("retail")
    enterprise_customers = class_counts.get("enterprise")

    latest_run = latest_recommendation_run()
    propensity_rows = list(
        _current_propensity_rows().values("product_id", "product__product_name")
        .annotate(customer_count=Count("customer_id"), average_propensity=Avg("propensity_score"))
        .order_by("-average_propensity", "product__product_name")
    )

    segment_summary = _behavioral_segment_summary()
    assigned_customer_count = _valid_assigned_customers().count()
    current_recommendations = _current_recommendation_rows()
    customers_with_recommendations = current_recommendations.values("customer_id").distinct().count()
    recommendation_coverage = customers_with_recommendations / total_customers if total_customers else None
    result = {
        "summary": {
            "total_customers": total_customers,
            "retail_customers": retail_customers,
            "enterprise_customers": enterprise_customers,
            "customer_class_available": any(key for key in class_counts),
            "behavioral_segments": segment_summary["total_segments"],
            "retail_behavioral_segments": segment_summary["retail_segments"],
            "enterprise_behavioral_segments": segment_summary["enterprise_segments"],
            "assigned_behavioral_segments": segment_summary["assigned_segments"],
            "assigned_customer_count": assigned_customer_count,
            "assigned_retail_segments": segment_summary["assigned_retail_segments"],
            "assigned_enterprise_segments": segment_summary["assigned_enterprise_segments"],
            "distinct_products": _current_ownership_rows().values("product_id").distinct().count(),
            "distinct_propensity_products": len(propensity_rows),
            "total_recommendations": current_recommendations.count(),
            "customers_with_recommendations": customers_with_recommendations,
            "customers_without_recommendations": max(0, total_customers - customers_with_recommendations),
            "recommendation_coverage": recommendation_coverage,
            "recommendation_coverage_display": _pct(recommendation_coverage),
            "latest_run": latest_run,
        },
        "segment_distribution": _segment_distribution(),
        "segment_distribution_by_population": {
            "retail": _segment_distribution("Retail"),
            "enterprise": _segment_distribution("Enterprise"),
        },
        "product_ownership": _product_ownership(),
        "propensity_overview": _propensity_by_population(),
        "recommendation_overview": _recommendation_overview(),
        "recommendation_action_summary": _recommendation_action_summary(),
        "model_runs": list(ModelRun.objects.order_by("-completed_at", "-created_at")[:12]),
    }
    result["headline"] = _overview_headline(result)
    result["opportunity_teaser"] = {
        "propensity": (result["propensity_overview"].get("retail") or result["propensity_overview"].get("enterprise") or [])[:1],
        "recommendation": (result["recommendation_action_summary"].get("retail") or result["recommendation_action_summary"].get("enterprise") or [])[:1],
    }
    result["insights"] = _overview_insights(result)
    return resul


def _overview_headline(overview: dict[str, Any]) -> str:
    summary = overview["summary"]
    retail = overview["segment_distribution_by_population"].get("retail") or []
    enterprise = overview["segment_distribution_by_population"].get("enterprise") or []
    parts = [f"{summary['total_customers']:,} customers are analyzed across the portfolio."]
    if retail:
        largest = max(retail, key=lambda row: row["percentage"])
        parts.append(f"Among classified retail customers, {largest['segment_name']} is the largest behavioral group ({largest['percentage_display']}).")
    if enterprise:
        largest = max(enterprise, key=lambda row: row["percentage"])
        parts.append(f"Enterprise customers are most concentrated in {largest['segment_name']} ({largest['percentage_display']}).")
    return " ".join(parts)


def _overview_insights(overview: dict[str, Any]) -> list[str]:
    insights = []
    summary = overview["summary"]
    for population in ("retail", "enterprise"):
        rows = overview["segment_distribution_by_population"].get(population) or []
        if rows:
            largest = max(rows, key=lambda row: row["customer_count"])
            if population == "retail" and largest["percentage"] < 0.5:
                insights.append(f"No single behavioral group dominates the retail portfolio; {largest['segment_name']} represents {largest['percentage_display']} of valid retail assignments.")
            else:
                insights.append(f"{largest['segment_name']} represents {largest['percentage_display']} of valid {population} assignments, making this the most concentrated group to investigate.")
    if summary["total_customers"]:
        insights.append(
            f"Recommendations are stored for {summary['customers_with_recommendations']:,} customers "
            f"({summary['recommendation_coverage'] * 100:.1f}% of all customers)."
        )
        if summary["customers_without_recommendations"]:
            insights.append(
                f"{summary['customers_without_recommendations']:,} customers have no stored recommendation output."
            )
    if summary["assigned_enterprise_segments"] == 0:
        insights.append("No valid enterprise segment assignments are currently available in the dashboard outputs.")
    return insights[:5]


def segments_analytics() -> dict[str, Any]:
    distribution = _segment_distribution()
    distribution_by_population = {
        "retail": [row for row in _segment_distribution("Retail") if row.get("cluster_id") in {0, 1, 2, 3, 4}],
        # Enterprise static clustering uses all valid stored clusters (currently
        # eight); do not truncate the population to the first three IDs.
        "enterprise": [row for row in _segment_distribution("Enterprise") if row.get("cluster_id") != -1],
    }
    exclusion_counts = _artifact_exclusion_counts()
    active_segments = sum(len(rows) for rows in distribution_by_population.values())
    assigned_customers = sum(row["customer_count"] for rows in distribution_by_population.values() for row in rows)
    ownership_rows = _products_by_segment()
    result = {
        "summary": {
            "active_segments": active_segments,
            "total_customers": Customer.objects.count(),
            "assigned_customers": assigned_customers,
            "outlier_customers": exclusion_counts["outlier_customers"],
            "unmapped_customers": exclusion_counts["unmapped_customers"],
        },
        "distribution": distribution,
        "distribution_by_population": distribution_by_population,
        "profile_cards_by_population": _segment_profile_cards_by_population(distribution_by_population),
        "largest_segments": sorted(distribution, key=lambda row: row["customer_count"], reverse=True)[:10],
        "segments": [_resolved_segment(_segment_model_to_dict(segment)) for segment in CustomerSegment.objects.order_by("segment_id")],
        "products_by_segment": ownership_rows,
        "ownership_explorer": _ownership_explorer(ownership_rows),
    }
    result["largest_insight_by_population"] = {
        population: (f"{rows[0]['segment_name']} is the largest {population} behavior group at {rows[0]['percentage_display']} of valid {population} assignments." if rows else "")
        for population, rows in distribution_by_population.items()
    }
    result["ownership_insight_by_population"] = {
        population: _highest_penetration_sentence(result["ownership_explorer"].get(population, {}), population)
        for population in ("retail", "enterprise")
    }
    result["workspace_by_population"] = _segments_workspace(
        distribution_by_population,
        result["profile_cards_by_population"],
        ownership_rows,
    )
    return resul


def _segments_workspace(
    distribution_by_population: dict[str, list[dict[str, Any]]],
    profiles_by_population: dict[str, list[dict[str, Any]]],
    ownership_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the business-facing Segments workspace from existing stored outputs."""
    # Segments are final static clusters. Do not use SegmentForecast here because
    # the database table currently contains temporal/rolling pools as well.
    forecasts_by_pool = _static_forecasts_by_pool()

    workspaces: dict[str, dict[str, Any]] = {}
    for population in ("retail", "enterprise"):
        distribution = distribution_by_population.get(population, [])
        clustering_features = _static_cluster_feature_medians(population)
        profiles = {item["segment_id"]: item for item in profiles_by_population.get(population, [])}
        segment_rows = []
        comparison = []
        key_differences = []
        for row in distribution:
            segment_id = row["segment_id"]
            profile = profiles.get(segment_id, {})
            forecast = _segment_forecast_for(segment_id, population, forecasts_by_pool)
            products = sorted(
                [item for item in ownership_rows if item.get("segment_id") == segment_id],
                key=lambda item: item.get("penetration_value", 0),
                reverse=True,
            )[:5]
            detail = {
                **row,
                **profile,
                "display_name": profile.get("display_name") or row.get("segment_name"),
                "forecast": forecast,
                "products": products,
                "clustering_features": clustering_features.get(segment_id, {}),
                "technical_details": {
                    "segment_id": segment_id,
                    "source_name": profile.get("technical_name") or row.get("segment_name"),
                    "cluster_id": row.get("cluster_id"),
                    "confidence": profile.get("interpretation_confidence") or "Not recorded",
                },
            }
            segment_rows.append(detail)
            comparison.append(
                {
                    "segment_id": segment_id,
                    "segment_name": detail["display_name"],
                    "transaction_behavior": profile.get("transaction_behavior") or "Not available",
                    "balance_and_cashflow_behavior": profile.get("balance_and_cashflow_behavior") or "Not available",
                    "digital_channel_behavior": profile.get("digital_channel_behavior") or "Not available",
                }
            )
            for indicator in profile.get("indicators") or []:
                if indicator not in key_differences and len(key_differences) < 4:
                    key_differences.append(indicator)

        workspaces[population] = {
            "population_label": population.title(),
            "assigned_customer_count": sum(item["customer_count"] for item in distribution),
            "segment_count": len(distribution),
            "distribution": distribution,
            "comparison": comparison,
            "key_differences": key_differences,
            "forecast_comparison": [
                item["forecast"] | {"segment_id": item["segment_id"], "segment_name": item["display_name"]}
                for item in segment_rows
                if item.get("forecast", {}).get("available")
            ],
            "segments": segment_rows,
        }
    return workspaces


@lru_cache(maxsize=2)
def _static_cluster_feature_medians(population: str) -> dict[str, dict[str, float]]:
    """Read the stored median values for the shared clustering inputs."""
    path = STATIC_CLUSTER_PROFILE_PATHS.get(population.lower())
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 4:
        return {}
    features = rows[0][1:]
    statistics = rows[1][1:]
    selected = {"Monetary", "Frequency", "avg_balance", "net_cash_flow", "channel_entropy"}
    prefix = "seg_ent_" if population.lower() == "enterprise" else "seg_ret_"
    result: dict[str, dict[str, float]] = {}
    for row in rows[3:]:
        cluster = row[0].strip() if row else ""
        if not cluster or cluster == "-1":
            continue
        values: dict[str, float] = {}
        for feature, statistic, raw in zip(features, statistics, row[1:]):
            if feature not in selected or statistic != "median" or raw == "":
                continue
            try:
                values[feature] = float(raw)
            except ValueError:
                continue
        if values:
            result[f"{prefix}{cluster}"] = values
    return resul


@lru_cache(maxsize=1)
def _static_customer_clustering_features() -> dict[int, dict[str, float]]:
    """Read the actual clustering inputs stored for each customer."""
    if not STATIC_SEGMENT_ASSIGNMENTS.exists():
        return {}
    selected = {"Monetary", "Frequency", "avg_balance", "net_cash_flow", "channel_entropy"}
    result: dict[int, dict[str, float]] = {}
    for row in _read_csv_path(STATIC_SEGMENT_ASSIGNMENTS):
        try:
            customer_id = int(row.get("client_id", ""))
        except (TypeError, ValueError):
            continue
        values = {}
        for feature in selected:
            raw = row.get(feature, "")
            try:
                values[feature] = float(raw)
            except (TypeError, ValueError):
                continue
        if values:
            result[customer_id] = values
    return resul


@lru_cache(maxsize=1)
def _static_forecasts_by_pool() -> dict[str, list[dict[str, Any]]]:
    """Read the final static-segment average-balance forecast artifact."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_csv_path(STATIC_FORECAST_POINTS):
        pool = row.get("pool")
        if not pool or row.get("metric") != "avg_balance":
            continue
        grouped.setdefault(pool, []).append(
            {
                "date": row.get("date"),
                "actual": _float_or_none(row.get("actual")),
                "forecast": _float_or_none(row.get("prediction")),
                "method": row.get("method"),
                "model": row.get("model"),
                "scenario": row.get("case"),
                "metric": row.get("metric"),
                "manual_review": (row.get("model") or "").strip().lower() == "manual_review",
            }
        )
    return grouped


def _segment_forecast_for(segment_id: str, population: str, forecasts_by_pool: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Resolve only a known final static segment; never substitute temporal pools."""
    pool = STATIC_SEGMENT_FORECAST_POOLS.get(segment_id)
    if not pool or pool not in forecasts_by_pool:
        return {"available": False, "manual_review": False, "points": [], "model": None, "scenario": None, "status": "Forecast unavailable for this segment; no promoted forecast artifact was found."}
    rows = forecasts_by_pool[pool]
    selected = (rows[0].get("model"), rows[0].get("scenario")) if rows else (None, None)
    points = [{"date": row.get("date"), "actual": row.get("actual"), "forecast": row.get("forecast")} for row in rows]
    manual_review = any(row.get("manual_review") for row in rows)
    if manual_review:
        return {"available": False, "manual_review": True, "points": points, "model": selected[0], "scenario": selected[1], "status": "Forecast unavailable for this segment pending model review."}
    observed = next((point for point in reversed(points) if point.get("actual") is not None), None)
    projected = next((point for point in reversed(points) if point.get("forecast") is not None), None)
    change = _change_pct(observed.get("actual") if observed else None, projected.get("forecast") if projected else None)
    return {
        "available": bool(points),
        "manual_review": False,
        "pool": pool,
        "model": selected[0] or None,
        "scenario": selected[1] or None,
        "points": points,
        "current_average": _money(observed.get("actual") if observed else None),
        "forecast_end": _money(projected.get("forecast") if projected else None),
        "change": _signed_pct(observed.get("actual") if observed else None, projected.get("forecast") if projected else None),
        "change_value": change,
        "direction": "Increasing" if change is not None and change > 0 else "Decreasing" if change is not None and change < 0 else "Stable" if change is not None else "Unavailable",
        "status": "",
    }


def _static_forecast_overview() -> dict[str, list[dict[str, Any]]]:
    overview = {"retail": [], "enterprise": []}
    for pool, rows in _static_forecasts_by_pool().items():
        population = "enterprise" if pool.startswith("enterprise_") else "retail" if pool.startswith("retail_") else None
        if not population:
            continue
        cluster = pool.rsplit("_", 1)[-1]
        segment_id = f"seg_{'ent' if population == 'enterprise' else 'ret'}_{int(cluster)}"
        forecast = _segment_forecast_for(segment_id, population, _static_forecasts_by_pool())
        overview[population].append({"pool": pool, "segment_id": segment_id, "segment_name": _segment_alias(segment_id, segment_id), **forecast})
    return overview


def _highest_penetration_sentence(rows: dict[str, Any], population: str) -> str:
    items = [item for group in rows.get("by_product", {}).values() for item in group.get("items", [])]
    if not items:
        return "No product ownership data is available for this population."
    row = max(items, key=lambda item: float(item.get("penetration_value") or item.get("penetration", 0) or 0))
    return f"{row.get('product_name', 'This product')} has the highest ownership penetration in the valid {population} assignments ({row.get('penetration', 'n/a')})."


def products_analytics() -> dict[str, Any]:
    ownership = {row["product_id"]: row for row in _product_ownership(limit=None)}
    total_customers = Customer.objects.count()
    current_recommendations = _current_recommendation_rows()
    total_recommendations = current_recommendations.count()
    recommendations = {
        row["product_id"]: row
        for row in current_recommendations.values("product_id", "product__product_name")
        .annotate(recommendation_count=Count("*"), average_score=Avg("score"))
    }
    propensities = (
        _current_propensity_rows().values("product_id", "product__product_name")
        .annotate(customer_count=Count("customer_id"), average_propensity=Avg("propensity_score"))
        .order_by("product__product_name")
    )
    product_ids = sorted(set(ownership) | set(recommendations) | {row["product_id"] for row in propensities})
    propensity_by_id = {row["product_id"]: row for row in propensities}
    rows = []
    for product_id in product_ids:
        product_name = (
            ownership.get(product_id, {}).get("product_name")
            or recommendations.get(product_id, {}).get("product__product_name")
            or propensity_by_id.get(product_id, {}).get("product__product_name")
            or product_id
        )
        owner_count = ownership.get(product_id, {}).get("owner_count", 0)
        recommendation_count = recommendations.get(product_id, {}).get("recommendation_count", 0)
        ownership_rate = owner_count / total_customers if total_customers else None
        recommendation_share = recommendation_count / total_recommendations if total_recommendations else None
        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "owner_count": owner_count,
                "ownership_rate": ownership_rate or 0,
                "ownership_rate_display": _pct(ownership_rate),
                "ownership_bar_width": f"{max(2, min(100, float(ownership_rate or 0) * 100)):.1f}%",
                "scored_customer_count": propensity_by_id.get(product_id, {}).get("customer_count", 0),
                "average_propensity": _number(propensity_by_id.get(product_id, {}).get("average_propensity")),
                "average_propensity_display": _score(propensity_by_id.get(product_id, {}).get("average_propensity")),
                "average_propensity_bar_width": f"{max(0, min(100, float(propensity_by_id.get(product_id, {}).get('average_propensity') or 0) * 100)):.1f}%",
                "recommendation_count": recommendation_count,
                "recommendation_share": recommendation_share or 0,
                "recommendation_share_display": _pct(recommendation_share),
                "recommendation_bar_width": f"{max(2, min(100, float(recommendation_share or 0) * 100)):.1f}%",
                "average_score": _score(recommendations.get(product_id, {}).get("average_score")),
            }
        )
    propensity_ranked = sorted(rows, key=lambda row: (row.get("average_propensity") is not None, row.get("average_propensity") or -1), reverse=True)
    ownership_ranked = sorted(rows, key=lambda row: row.get("ownership_rate", 0))
    top_n = max(1, (len(rows) + 2) // 3) if rows else 0
    top_propensity = {row["product_id"] for row in propensity_ranked[:top_n]}
    low_ownership = {row["product_id"] for row in ownership_ranked[:top_n]}
    opportunity_candidates = [row for row in rows if row["product_id"] in top_propensity & low_ownership]
    opportunity_candidates.sort(key=lambda row: ((row.get("average_propensity") or -1), -(row.get("ownership_rate") or 0)), reverse=True)
    family_rows = (
        _current_ownership_rows()
        .exclude(family__isnull=True)
        .exclude(family__exact="")
        .values("family")
        .annotate(customer_count=Count("customer_id", distinct=True))
        .order_by("-customer_count", "family")
    )
    max_family_count = max((row["customer_count"] for row in family_rows), default=1)
    product_families = [
        {
            "family": row["family"],
            "customer_count": row["customer_count"],
            "share_display": _pct(row["customer_count"] / total_customers if total_customers else None),
            "bar_width": f"{max(2, min(100, row['customer_count'] / max_family_count * 100)):.1f}%",
        }
        for row in family_rows
    ]
    return {
        "summary": {"total_customers": total_customers, "total_recommendations": total_recommendations},
        "rows": rows,
        "top_owned": sorted(rows, key=lambda row: row["ownership_rate"], reverse=True)[:18],
        "top_recommended": sorted(rows, key=lambda row: row["recommendation_count"], reverse=True)[:18],
        "opportunity_candidates": opportunity_candidates,
        "product_families": product_families,
        "ontology_graph": product_ontology_graph(),
    }


def forecasting_analytics(pool: str | None = None) -> dict[str, Any]:
    forecast_run = latest_forecast_run()
    forecast_rows = SegmentForecast.objects.filter(run=forecast_run) if forecast_run is not None else SegmentForecast.objects.none()
    pools = list(forecast_rows.filter(metric="avg_balance").values_list("pool", flat=True).distinct().order_by("pool"))
    selected_pool = pool if pool in pools else (pools[0] if pools else None)
    forecasts = []
    selected_scenario = None
    selected_model = None
    if selected_pool:
        rows = forecast_rows.filter(pool=selected_pool, metric="avg_balance").select_related("run").order_by("forecast_date")
        row_list = list(rows)
        combinations = {}
        for row in row_list:
            key = (row.model_name or "", row.scenario_case or "")
            combinations[key] = combinations.get(key, 0) + 1
        if combinations:
            selected_model, selected_scenario = max(combinations, key=combinations.get)
            row_list = [
                row for row in row_lis
                if (row.model_name or "", row.scenario_case or "") == (selected_model, selected_scenario)
            ]
        forecasts = [
            {
                "forecast_date": row.forecast_date,
                "scenario_case": row.scenario_case,
                "model_name": row.model_name,
                "actual_value_display": _money(row.actual_value),
                "predicted_value_display": _money(row.predicted_value),
                "actual_value": _number(row.actual_value),
                "predicted_value": _number(row.predicted_value),
                "run_name": row.run.run_name,
            }
            for row in row_lis
        ]
    observed = next((row for row in reversed(forecasts) if row.get("actual_value") is not None), None)
    projected = next((row for row in reversed(forecasts) if row.get("predicted_value") is not None), None)
    change = _change_pct(observed.get("actual_value") if observed else None, projected.get("predicted_value") if projected else None)
    comparison = []
    for candidate in pools:
        latest = forecast_rows.filter(pool=candidate, metric="avg_balance").order_by("-forecast_date").first()
        first = forecast_rows.filter(pool=candidate, metric="avg_balance").order_by("forecast_date").first()
        if latest:
            predicted = _number(latest.predicted_value) or 0
            trend_value = _change_pct(first.predicted_value if first else None, latest.predicted_value)
            comparison.append(
                {
                    "pool": candidate,
                    "pool_label": _forecast_pool_label(candidate),
                    "population": _forecast_pool_population(candidate),
                    "latest_date": latest.forecast_date,
                    "forecast_average_balance_value": predicted,
                    "forecast_average_balance": _money(latest.predicted_value),
                    "trend_value": trend_value or 0,
                    "trend": _signed_pct(first.predicted_value if first else None, latest.predicted_value),
                }
            )
    comparison_by_population = _forecast_comparison_by_population(comparison)
    return {
        "pools": pools,
        "pool_options": [{"value": candidate, "label": _forecast_pool_label(candidate)} for candidate in pools],
        "selected_pool": selected_pool,
        "selected_pool_label": _forecast_pool_label(selected_pool) if selected_pool else None,
        "selected_model": selected_model,
        "selected_scenario": selected_scenario,
        "forecasts": forecasts,
        "forecast_summary": {
            "last_observed": _money(observed.get("actual_value")) if observed else None,
            "end_forecast": _money(projected.get("predicted_value")) if projected else None,
            "change": _signed_pct(observed.get("actual_value") if observed else None, projected.get("predicted_value") if projected else None),
            "direction": "increase" if change is not None and change > 0 else "decrease" if change is not None and change < 0 else "remain broadly stable" if change is not None else None,
        },
        "forecast_chart": _forecast_chart(forecasts),
        "forecast_metrics": _forecast_model_metrics(),
        "comparison": comparison,
        "comparison_by_population": comparison_by_population,
        "comparison_chart_by_population": _forecast_comparison_chart_by_population(pools, forecast_run),
        "model_runs": [forecast_run] if forecast_run is not None else [],
        "static_overview": _static_forecast_overview(),
    }


def _change_pct(from_value: Decimal | int | float | None, to_value: Decimal | int | float | None) -> float | None:
    if from_value in (None, 0) or to_value is None:
        return None
    return ((float(to_value) - float(from_value)) / abs(float(from_value))) * 100


def _select_forecast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep chart/table output on one model/scenario when stored outputs overlap."""
    if not rows:
        return rows
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.get("model_name") or "", row.get("scenario_case") or "")
        counts[key] = counts.get(key, 0) + 1
    selected = max(counts, key=counts.get)
    return [row for row in rows if (row.get("model_name") or "", row.get("scenario_case") or "") == selected]


def _forecast_pool_population(pool: str) -> str:
    if pool.startswith("enterprise_"):
        return "enterprise"
    if pool.startswith("retail_"):
        return "retail"
    return "other"


def _forecast_comparison_by_population(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"retail": [], "enterprise": []}
    max_abs_balance = max((abs(float(row.get("forecast_average_balance_value") or 0)) for row in rows), default=1) or 1
    max_abs_trend = max((abs(float(row.get("trend_value") or 0)) for row in rows), default=1) or 1
    for row in rows:
        population = row.get("population")
        if population not in grouped:
            continue
        grouped[population].append(
            {
                **row,
                "balance_bar_width": f"{max(2, min(100, abs(float(row.get('forecast_average_balance_value') or 0)) / max_abs_balance * 100)):.1f}%",
                "trend_bar_width": f"{max(2, min(100, abs(float(row.get('trend_value') or 0)) / max_abs_trend * 100)):.1f}%",
                "trend_direction": "positive" if float(row.get("trend_value") or 0) >= 0 else "negative",
            }
        )
    for population in grouped:
        grouped[population] = sorted(grouped[population], key=lambda item: abs(float(item["forecast_average_balance_value"])), reverse=True)
    return grouped


def _forecast_comparison_chart_by_population(pools: list[str], forecast_run: ModelRun | None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"retail": [], "enterprise": []}
    if forecast_run is None:
        return grouped
    rows = SegmentForecast.objects.filter(run=forecast_run, metric="avg_balance", pool__in=pools).order_by("pool", "forecast_date")
    by_pool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = _number(row.predicted_value)
        if value is None:
            continue
        by_pool.setdefault(row.pool, []).append({"date": row.forecast_date.isoformat(), "value": value})
    for pool, points in by_pool.items():
        population = _forecast_pool_population(pool)
        if population in grouped:
            grouped[population].append({"id": pool, "label": _forecast_pool_label(pool), "points": points})
    return grouped


def _forecast_pool_label(pool: str) -> str:
    if not pool:
        return "Unavailable"
    parts = pool.split("_")
    population = "Enterprise" if parts and parts[0] == "enterprise" else "Retail" if parts and parts[0] == "retail" else None
    cluster = parts[-1] if parts and parts[-1].lstrip("-").isdigit() else None
    method = "rolling temporal" if "rolling" in parts else "temporal" if "temporal" in parts else "static"
    filled = " filled" if "filled" in parts else ""
    if population and cluster is not None:
        return f"{population} {method}{filled} segment {cluster}"
    return pool.replace("_", " ").title()


def _forecast_model_metrics() -> list[dict[str, Any]]:
    rows = []
    if not TS_BEST_PROCESSES.exists():
        return rows
    for row in _read_csv_path(TS_BEST_PROCESSES):
        if (row.get("target_metric") or row.get("metric")) != "avg_balance":
            continue
        mae = _float_or_none(row.get("mae_mean"))
        rmse = _float_or_none(row.get("rmse_mean"))
        mase = _float_or_none(row.get("mase_mean") or row.get("mase"))
        rows.append(
            {
                "label": (row.get("Forecasting Model") or row.get("model") or "Selected model").replace("_", " ").title(),
                "process": (row.get("Clustering Strategy") or row.get("method") or "").replace("_", " ").title(),
                "model_family": (row.get("model_family") or row.get("model") or "").replace("_", " ").upper(),
                "mae": mae,
                "rmse": rmse,
                "mase": mase,
                "mae_display": _score(mae),
                "rmse_display": _score(rmse),
                "mase_display": _score(mase),
                "mae_width": _metric_width(mae, 0.02),
                "rmse_width": _metric_width(rmse, 0.03),
                "mase_width": _metric_width(mase, 0.2),
                "selected": True,
            }
        )
    return rows


def _metric_width(value: float | None, scale: float) -> str:
    if value is None or scale <= 0:
        return "0%"
    return f"{max(2, min(100, (value / scale) * 100)):.1f}%"


def recommendations_analytics(customer_id: int | None = None) -> dict[str, Any]:
    eligible_customers = Customer.objects.count()
    selected_customer = customer_dashboard(customer_id) if customer_id else None
    by_product = _recommendation_overview()["by_product"]
    by_segment = _recommendations_by_segment()
    return {
        "summary": {
            "total_recommendations": _current_recommendation_rows().count(),
            "customers_with_recommendations": _current_recommendation_rows().values("customer_id").distinct().count(),
            "coverage": _pct(
                _current_recommendation_rows().values("customer_id").distinct().count() / Customer.objects.count()
                if Customer.objects.count()
                else None
            ),
            "coverage_denominator": eligible_customers,
            "coverage_population_label": f"{eligible_customers:,} eligible portfolio customers",
        },
        "by_product": by_product,
        "by_segment": by_segment,
        "by_segment_explorer": _recommendation_segment_explorer(by_segment),
        "selected_customer": selected_customer,
    }


def _recommendation_reason_text(item: dict[str, Any], propensity: dict[str, Any] | None) -> str:
    if item.get("reason"):
        return item["reason"]
    if propensity:
        return (
            "Ranked from this customer's product propensity signal, then checked against "
            "the recommendation rules and ontology evidence."
        )
    return "Ranked from stored recommendation evidence; no matching propensity row is available for this product."


def _valid_segment_filter() -> Q:
    return (
        Q(behavioral_segment__isnull=False)
        & ~Q(behavioral_segment__cluster_id=-1)
        & ~Q(behavioral_segment__segment_name__istartswith="Unmapped ")
        & ~Q(behavioral_segment__segment_name__icontains="Outlier")
        & ~Q(behavioral_segment__segment_name__icontains="Noise")
    )


def _valid_assigned_customers(population: str | None = None):
    qs = Customer.objects.filter(_valid_segment_filter())
    if population:
        qs = qs.filter(behavioral_segment__source_group__iexact=population)
    return qs


def _behavioral_segment_summary() -> dict[str, Any]:
    valid_segments = CustomerSegment.objects.exclude(cluster_id=-1).exclude(segment_name__istartswith="Unmapped ")
    valid_segments = valid_segments.exclude(segment_name__icontains="Outlier").exclude(segment_name__icontains="Noise")
    retail_segments = valid_segments.filter(source_group__iexact="Retail").count()
    enterprise_segments = valid_segments.filter(source_group__iexact="Enterprise").count()
    assigned = _valid_assigned_customers()
    assigned_retail = _valid_assigned_customers("Retail").values("behavioral_segment_id").distinct().count()
    assigned_enterprise = _valid_assigned_customers("Enterprise").values("behavioral_segment_id").distinct().count()
    if assigned_enterprise == 0:
        assigned_enterprise = len(_artifact_segment_distribution("Enterprise"))
    return {
        "total_segments": retail_segments + enterprise_segments,
        "retail_segments": retail_segments,
        "enterprise_segments": enterprise_segments,
        "assigned_segments": assigned.values("behavioral_segment_id").distinct().count(),
        "assigned_retail_segments": assigned_retail,
        "assigned_enterprise_segments": assigned_enterprise,
    }


def _segment_distribution(population: str | None = None) -> list[dict[str, Any]]:
    qs = _valid_assigned_customers(population)
    rows = (
        qs.values(
            "behavioral_segment_id",
            "behavioral_segment__segment_name",
            "behavioral_segment__source_group",
            "behavioral_segment__cluster_id",
        )
        .annotate(customer_count=Count("*"))
        .order_by("behavioral_segment__source_group", "behavioral_segment__cluster_id")
    )
    total_assigned = sum(row["customer_count"] for row in rows)
    result = []
    for row in rows:
        segment = _resolved_segment(
            {
                "segment_id": row["behavioral_segment_id"],
                "segment_name": row["behavioral_segment__segment_name"] or "",
                "source_group": row["behavioral_segment__source_group"] or "Unspecified",
                "cluster_id": row["behavioral_segment__cluster_id"],
            }
        ) or {}
        result.append(
            {
                "segment_id": row["behavioral_segment_id"],
                "segment_name": segment.get("display_name") or segment.get("segment_name") or row["behavioral_segment_id"],
                "description": segment.get("description") or "",
                "customer_class": segment.get("source_group") or "Unspecified",
                "cluster_id": segment.get("cluster_id"),
                "customer_count": row["customer_count"],
                "percentage": row["customer_count"] / total_assigned if total_assigned else 0,
                "percentage_display": _pct(row["customer_count"] / total_assigned if total_assigned else None),
            }
        )
    if result or not population:
        return resul
    return _artifact_segment_distribution(population)


@lru_cache(maxsize=4)
def _artifact_segment_distribution(population: str) -> list[dict[str, Any]]:
    target = population.lower()
    counts: dict[str, int] = {}
    for row in _read_csv_path(STATIC_SEGMENT_ASSIGNMENTS):
        source = (row.get("customer_class") or row.get("source_group") or "").strip().lower()
        is_enterprise = source in {"enterprise", "entreprise"}
        row_population = "enterprise" if is_enterprise else "retail"
        if row_population != target:
            continue
        cluster = row.get("cluster")
        if cluster in (None, "", "-1"):
            continue
        prefix = "seg_ent_" if is_enterprise else "seg_ret_"
        segment_id = f"{prefix}{cluster}"
        counts[segment_id] = counts.get(segment_id, 0) + 1

    total = sum(counts.values())
    result = []
    for segment_id, count in sorted(counts.items()):
        definition = _resolved_segment({"segment_id": segment_id, "segment_name": segment_id, "source_group": population}) or {}
        result.append(
            {
                "segment_id": segment_id,
                "segment_name": definition.get("display_name") or definition.get("segment_name") or segment_id,
                "description": definition.get("description") or "",
                "customer_class": definition.get("source_group") or population,
                "cluster_id": definition.get("cluster_id"),
                "customer_count": count,
                "percentage": count / total if total else 0,
                "percentage_display": _pct(count / total if total else None),
                "source_note": STATIC_SEGMENT_ASSIGNMENTS.name,
            }
        )
    return resul


def _artifact_segment_assignments(population: str) -> list[tuple[int, str]]:
    target = population.lower()
    assignments = []
    for row in _read_csv_path(STATIC_SEGMENT_ASSIGNMENTS):
        source = (row.get("customer_class") or row.get("source_group") or "").strip().lower()
        is_enterprise = source in {"enterprise", "entreprise"}
        row_population = "enterprise" if is_enterprise else "retail"
        if row_population != target:
            continue
        cluster = row.get("cluster")
        customer_id = row.get("client_id")
        if cluster in (None, "", "-1") or not customer_id:
            continue
        prefix = "seg_ent_" if is_enterprise else "seg_ret_"
        try:
            assignments.append((int(customer_id), f"{prefix}{cluster}"))
        except ValueError:
            continue
    return assignments


@lru_cache(maxsize=1)
def _artifact_exclusion_counts() -> dict[str, int]:
    definitions = behavioral_segment_definitions()
    outlier_customers = 0
    unmapped_customers = 0
    for row in _read_csv_path(STATIC_SEGMENT_ASSIGNMENTS):
        source = (row.get("customer_class") or row.get("source_group") or "").strip().lower()
        is_enterprise = source in {"enterprise", "entreprise"}
        cluster = row.get("cluster")
        if cluster in (None, ""):
            continue
        if cluster == "-1":
            outlier_customers += 1
            continue
        segment_id = f"{'seg_ent_' if is_enterprise else 'seg_ret_'}{cluster}"
        if segment_id not in definitions:
            unmapped_customers += 1
    return {"outlier_customers": outlier_customers, "unmapped_customers": unmapped_customers}


def _enterprise_assignment_values_sql() -> tuple[str, list[Any]]:
    assignments = _artifact_segment_assignments("Enterprise")
    if not assignments:
        return "", []
    placeholders = ",".join(["(%s,%s)"] * len(assignments))
    params = [value for pair in assignments for value in pair]
    return placeholders, params


def _segment_model_to_dict(segment: CustomerSegment) -> dict[str, Any]:
    return {
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


def _product_ownership(limit: int | None = 25) -> list[dict[str, Any]]:
    rows = (
        _current_ownership_rows().values("product_id", "product__product_name")
        .annotate(owner_count=Count("customer_id", distinct=True))
        .order_by("-owner_count", "product__product_name")
    )
    if limit is not None:
        rows = rows[:limit]
    return [
        {
            "product_id": row["product_id"],
            "product_name": row["product__product_name"],
            "owner_count": row["owner_count"],
        }
        for row in rows
    ]


def _recommendation_overview() -> dict[str, list[dict[str, Any]]]:
    by_product = (
        _current_recommendation_rows().values("product_id", "product__product_name")
        .annotate(recommendation_count=Count("*"), average_score=Avg("score"))
        .order_by("-recommendation_count", "product__product_name")[:25]
    )
    max_count = max((row["recommendation_count"] for row in by_product), default=1)
    return {
        "by_product": [
            {
                "product_id": row["product_id"],
                "product_name": row["product__product_name"],
                "recommendation_count": row["recommendation_count"],
                "average_score_value": _number(row["average_score"]) or 0,
                "average_score": _score(row["average_score"]),
                "count_bar_width": f"{(row['recommendation_count'] / max_count) * 100:.1f}%",
                "score_bar_width": f"{max(2, min(100, float(row['average_score'] or 0) * 100)):.1f}%",
            }
            for row in by_produc
        ],
        "by_segment": [],
    }


def _population_product_rows(table: str, score_column: str, limit: int = 6) -> dict[str, list[dict[str, Any]]]:
    if table not in {"propensity_predictions", "recommendations"}:
        raise ValueError("Unsupported overview aggregation table.")
    run_id = _current_recommendation_run_id()
    if run_id is None:
        return {"retail": [], "enterprise": []}
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT lower(cs.source_group) AS population,
                   src.product_id,
                   p.product_name,
                   COUNT(*) AS row_count,
                   COUNT(DISTINCT src.customer_id) AS customer_count,
                   AVG(src.{score_column}) AS average_score
            FROM {table} src
            JOIN customers c ON c.customer_id = src.customer_id
            JOIN customer_segments cs ON cs.segment_id = c.behavioral_segment_id
            JOIN products p ON p.product_id = src.product_id
            WHERE src.run_id = %s
              AND cs.cluster_id <> -1
              AND cs.segment_name NOT ILIKE 'Unmapped %%'
              AND cs.segment_name NOT ILIKE '%%Outlier%%'
              AND cs.segment_name NOT ILIKE '%%Noise%%'
            GROUP BY lower(cs.source_group), src.product_id, p.product_name
            ORDER BY lower(cs.source_group), AVG(src.{score_column}) DESC, COUNT(*) DESC, p.product_name
            """,
            [run_id],
        )
        rows = cursor.fetchall()

    grouped = {"retail": [], "enterprise": []}
    for population, product_id, product_name, row_count, customer_count, average_score in rows:
        key = population or "unknown"
        if key not in grouped or len(grouped[key]) >= limit:
            continue
        grouped[key].append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "row_count": row_count,
                "customer_count": customer_count,
                "average_score": _number(average_score),
                "average_score_display": _score(average_score),
            }
        )
    if not grouped["enterprise"]:
        grouped["enterprise"] = _artifact_population_product_rows(table, score_column, limit)
    return grouped


def _artifact_population_product_rows(table: str, score_column: str, limit: int = 6) -> list[dict[str, Any]]:
    values_sql, params = _enterprise_assignment_values_sql()
    run_id = _current_recommendation_run_id()
    if not values_sql or run_id is None:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH artifact_assignments(customer_id, segment_id) AS (VALUES {values_sql})
            SELECT src.product_id,
                   p.product_name,
                   COUNT(*) AS row_count,
                   COUNT(DISTINCT src.customer_id) AS customer_count,
                   AVG(src.{score_column}) AS average_score
            FROM {table} src
            JOIN artifact_assignments aa ON aa.customer_id = src.customer_id
            JOIN products p ON p.product_id = src.product_id
            WHERE src.run_id = %s
            GROUP BY src.product_id, p.product_name
            ORDER BY AVG(src.{score_column}) DESC, COUNT(*) DESC, p.product_name
            LIMIT %s
            """,
            [*params, run_id, limit],
        )
        rows = cursor.fetchall()
    return [
        {
            "product_id": product_id,
            "product_name": product_name,
            "row_count": row_count,
            "customer_count": customer_count,
            "average_score": _number(average_score),
            "average_score_display": _score(average_score),
        }
        for product_id, product_name, row_count, customer_count, average_score in rows
    ]


def _propensity_by_population() -> dict[str, list[dict[str, Any]]]:
    return _population_product_rows("propensity_predictions", "propensity_score")


def _recommendation_action_summary(limit: int = 6) -> dict[str, list[dict[str, Any]]]:
    run_id = _current_recommendation_run_id()
    if run_id is None:
        return {"retail": [], "enterprise": []}
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT lower(cs.source_group) AS population,
                   r.product_id,
                   p.product_name,
                   COUNT(*) AS recommendation_slots,
                   COUNT(DISTINCT r.customer_id) AS recommended_customers,
                   AVG(r.score) AS average_score
            FROM recommendations r
            JOIN customers c ON c.customer_id = r.customer_id
            JOIN customer_segments cs ON cs.segment_id = c.behavioral_segment_id
            JOIN products p ON p.product_id = r.product_id
            WHERE r.run_id = %s
              AND cs.cluster_id <> -1
              AND cs.segment_name NOT ILIKE 'Unmapped %%'
              AND cs.segment_name NOT ILIKE '%%Outlier%%'
              AND cs.segment_name NOT ILIKE '%%Noise%%'
            GROUP BY lower(cs.source_group), r.product_id, p.product_name
            ORDER BY lower(cs.source_group), COUNT(*) DESC, AVG(r.score) DESC, p.product_name
            """,
            [run_id],
        )
        rows = cursor.fetchall()

    grouped = {"retail": [], "enterprise": []}
    max_slots_by_population = {"retail": 0, "enterprise": 0}
    for population, _product_id, _product_name, recommendation_slots, _recommended_customers, _average_score in rows:
        key = population or "unknown"
        if key in max_slots_by_population:
            max_slots_by_population[key] = max(max_slots_by_population[key], recommendation_slots)

    for population, product_id, product_name, recommendation_slots, recommended_customers, average_score in rows:
        key = population or "unknown"
        if key not in grouped or len(grouped[key]) >= limit:
            continue
        max_slots = max_slots_by_population.get(key) or recommendation_slots or 1
        grouped[key].append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "recommendation_slots": recommendation_slots,
                "recommended_customers": recommended_customers,
                "average_score": _number(average_score),
                "average_score_display": _score(average_score),
                "bar_width": f"{(recommendation_slots / max_slots) * 100:.1f}%",
            }
        )
    if not grouped["enterprise"]:
        grouped["enterprise"] = _artifact_recommendation_action_summary(limit)
    return grouped


def _artifact_recommendation_action_summary(limit: int = 6) -> list[dict[str, Any]]:
    values_sql, params = _enterprise_assignment_values_sql()
    run_id = _current_recommendation_run_id()
    if not values_sql or run_id is None:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH artifact_assignments(customer_id, segment_id) AS (VALUES {values_sql})
            SELECT r.product_id,
                   p.product_name,
                   COUNT(*) AS recommendation_slots,
                   COUNT(DISTINCT r.customer_id) AS recommended_customers,
                   AVG(r.score) AS average_score
            FROM recommendations r
            JOIN artifact_assignments aa ON aa.customer_id = r.customer_id
            JOIN products p ON p.product_id = r.product_id
            WHERE r.run_id = %s
            GROUP BY r.product_id, p.product_name
            ORDER BY COUNT(*) DESC, AVG(r.score) DESC, p.product_name
            LIMIT %s
            """,
            [*params, run_id, limit],
        )
        rows = cursor.fetchall()
    max_slots = max((row[2] for row in rows), default=1)
    return [
        {
            "product_id": product_id,
            "product_name": product_name,
            "recommendation_slots": recommendation_slots,
            "recommended_customers": recommended_customers,
            "average_score": _number(average_score),
            "average_score_display": _score(average_score),
            "bar_width": f"{(recommendation_slots / max_slots) * 100:.1f}%",
        }
        for product_id, product_name, recommendation_slots, recommended_customers, average_score in rows
    ]


def _segment_profile_cards_by_population(distribution_by_population: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    cards = {"retail": [], "enterprise": []}
    for population, rows in distribution_by_population.items():
        for row in rows:
            definition = behavioral_segment_definitions().get(row["segment_id"], {})
            cards[population].append(
                {
                    "segment_id": row["segment_id"],
                    "segment_name": _segment_alias(row["segment_id"], row["segment_name"]),
                    "description": row.get("description") or definition.get("description") or "",
                    "customer_count": row["customer_count"],
                    "percentage_display": row["percentage_display"],
                    "transaction_behavior": definition.get("transaction_behavior") or "",
                    "balance_and_cashflow_behavior": definition.get("balance_and_cashflow_behavior") or "",
                    "digital_channel_behavior": definition.get("digital_channel_behavior") or "",
                    "strengths": definition.get("strengths") or [],
                    "watch_outs": definition.get("watch_outs") or [],
                    "supporting_statistics": definition.get("supporting_statistics") or "",
                    "interpretation_confidence": definition.get("interpretation_confidence") or "",
                    "indicators": _split_indicators(definition.get("supporting_statistics")),
                    "technical_name": definition.get("segment_name") or row["segment_name"],
                }
            )
    return cards


def _ownership_explorer(rows: list[dict[str, Any]] | None = None, limit_per_group: int = 10) -> dict[str, Any]:
    rows = rows if rows is not None else _products_by_segment()
    by_population = {"retail": {"by_segment": {}, "by_product": {}}, "enterprise": {"by_segment": {}, "by_product": {}}}
    for row in rows:
        population = (row.get("population") or "").lower()
        if population not in by_population:
            continue
        segment_id = row["segment_id"]
        product_id = row["product_id"]
        by_population[population]["by_segment"].setdefault(segment_id, {"segment_name": row["segment_name"], "items": []})
        by_population[population]["by_segment"][segment_id]["items"].append(row)
        by_population[population]["by_product"].setdefault(product_id, {"product_name": row["product_name"], "items": []})
        by_population[population]["by_product"][product_id]["items"].append(row)

    for population in by_population.values():
        for group_name in ("by_segment", "by_product"):
            for group in population[group_name].values():
                group["items"] = sorted(group["items"], key=lambda item: item["penetration_value"], reverse=True)[:limit_per_group]
    return by_population


def _products_by_segment() -> list[dict[str, Any]]:
    ownership_run_id = _current_ownership_run_id()
    if ownership_run_id is None:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH valid_customers AS (
                SELECT c.customer_id, c.behavioral_segment_id, cs.segment_name, cs.source_group
                FROM customers c
                JOIN customer_segments cs ON cs.segment_id = c.behavioral_segment_id
                WHERE cs.cluster_id <> -1
                  AND cs.segment_name NOT ILIKE 'Unmapped %'
                  AND cs.segment_name NOT ILIKE '%Outlier%'
                  AND cs.segment_name NOT ILIKE '%Noise%'
            ),
            segment_sizes AS (
                SELECT behavioral_segment_id, COUNT(DISTINCT customer_id) AS segment_customers
                FROM valid_customers
                GROUP BY behavioral_segment_id
            )
            SELECT vc.behavioral_segment_id, vc.segment_name, vc.source_group, cp.product_id, p.product_name,
                   COUNT(DISTINCT cp.customer_id) AS owner_count,
                   ss.segment_customers
            FROM customer_products cp
            JOIN valid_customers vc ON vc.customer_id = cp.customer_id
            JOIN segment_sizes ss ON ss.behavioral_segment_id = vc.behavioral_segment_id
            JOIN products p ON p.product_id = cp.product_id
            WHERE cp.run_id = %s
            GROUP BY vc.behavioral_segment_id, vc.segment_name, vc.source_group, cp.product_id, p.product_name, ss.segment_customers
            ORDER BY vc.behavioral_segment_id, owner_count DESC, p.product_name
            """,
            [ownership_run_id],
        )
        rows = cursor.fetchall()
    result = []
    for segment_id, segment_name, population, product_id, product_name, owner_count, segment_customers in rows:
        resolved = _resolved_segment({"segment_id": segment_id, "segment_name": segment_name, "source_group": population}) or {}
        penetration = owner_count / segment_customers if segment_customers else None
        result.append(
            {
                "segment_id": segment_id,
                "segment_name": resolved.get("display_name") or segment_name,
                "population": resolved.get("source_group") or population,
                "product_id": product_id,
                "product_name": product_name,
                "owner_count": owner_count,
                "segment_customers": segment_customers,
                "penetration_value": penetration or 0,
                "penetration": _pct(penetration),
            }
        )
    if not any((row.get("population") or "").lower() == "enterprise" for row in result):
        result.extend(_artifact_products_by_segment())
    return resul


def _artifact_products_by_segment() -> list[dict[str, Any]]:
    ownership_run_id = _current_ownership_run_id()
    if ownership_run_id is None:
        return []
    values_sql, params = _enterprise_assignment_values_sql()
    if not values_sql:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH artifact_assignments(customer_id, segment_id) AS (VALUES {values_sql}),
            segment_sizes AS (
                SELECT segment_id, COUNT(DISTINCT customer_id) AS segment_customers
                FROM artifact_assignments
                GROUP BY segment_id
            )
            SELECT aa.segment_id,
                   cp.product_id,
                   p.product_name,
                   COUNT(DISTINCT cp.customer_id) AS owner_count,
                   ss.segment_customers
            FROM customer_products cp
            JOIN artifact_assignments aa ON aa.customer_id = cp.customer_id
            JOIN segment_sizes ss ON ss.segment_id = aa.segment_id
            JOIN products p ON p.product_id = cp.product_id
            WHERE cp.run_id = %s
            GROUP BY aa.segment_id, cp.product_id, p.product_name, ss.segment_customers
            ORDER BY aa.segment_id, owner_count DESC, p.product_name
            """,
            [*params, ownership_run_id],
        )
        rows = cursor.fetchall()
    result = []
    for segment_id, product_id, product_name, owner_count, segment_customers in rows:
        resolved = _resolved_segment({"segment_id": segment_id, "segment_name": segment_id, "source_group": "Enterprise"}) or {}
        penetration = owner_count / segment_customers if segment_customers else None
        result.append(
            {
                "segment_id": segment_id,
                "segment_name": resolved.get("display_name") or "Unclassified behavior",
                "population": "Enterprise",
                "product_id": product_id,
                "product_name": product_name,
                "owner_count": owner_count,
                "segment_customers": segment_customers,
                "penetration_value": penetration or 0,
                "penetration": _pct(penetration),
            }
        )
    return resul


def _recommendations_by_segment() -> list[dict[str, Any]]:
    run_id = _current_recommendation_run_id()
    if run_id is None:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.behavioral_segment_id, cs.segment_name, cs.source_group, p.product_name,
                   COUNT(*) AS recommendation_count, AVG(r.score) AS average_score
            FROM recommendations r
            JOIN customers c ON c.customer_id = r.customer_id
            JOIN customer_segments cs ON cs.segment_id = c.behavioral_segment_id
            JOIN products p ON p.product_id = r.product_id
            WHERE r.run_id = %s
              AND cs.cluster_id <> -1
              AND cs.segment_name NOT ILIKE 'Unmapped %%'
              AND cs.segment_name NOT ILIKE '%%Outlier%%'
              AND cs.segment_name NOT ILIKE '%%Noise%%'
            GROUP BY c.behavioral_segment_id, cs.segment_name, cs.source_group, p.product_name
            ORDER BY c.behavioral_segment_id, recommendation_count DESC, p.product_name
            """,
            [run_id],
        )
        rows = cursor.fetchall()
    max_count_by_segment: dict[str, int] = {}
    for segment_id, _segment_name, _population, _product_name, count, _score_value in rows:
        max_count_by_segment[segment_id] = max(max_count_by_segment.get(segment_id, 0), count)

    result = []
    for segment_id, segment_name, population, product_name, count, score in rows:
        resolved = _resolved_segment({"segment_id": segment_id, "segment_name": segment_name, "source_group": population}) or {}
        max_count = max_count_by_segment.get(segment_id) or count or 1
        result.append(
            {
                "segment_id": segment_id,
                "segment_name": resolved.get("display_name") or segment_name,
                "population": resolved.get("source_group") or population,
                "product_name": product_name,
                "recommendation_count": count,
                "average_score_value": _number(score) or 0,
                "average_score": _score(score),
                "count_bar_width": f"{(count / max_count) * 100:.1f}%",
                "score_bar_width": f"{max(2, min(100, float(score or 0) * 100)):.1f}%",
            }
        )
    if not any((row.get("population") or "").lower() == "enterprise" for row in result):
        result.extend(_artifact_recommendations_by_segment())
    return resul


def _artifact_recommendations_by_segment() -> list[dict[str, Any]]:
    values_sql, params = _enterprise_assignment_values_sql()
    run_id = _current_recommendation_run_id()
    if not values_sql or run_id is None:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            WITH artifact_assignments(customer_id, segment_id) AS (VALUES {values_sql})
            SELECT aa.segment_id,
                   p.product_name,
                   COUNT(*) AS recommendation_count,
                   AVG(r.score) AS average_score
            FROM recommendations r
            JOIN artifact_assignments aa ON aa.customer_id = r.customer_id
            JOIN products p ON p.product_id = r.product_id
            WHERE r.run_id = %s
            GROUP BY aa.segment_id, p.product_name
            ORDER BY aa.segment_id, recommendation_count DESC, p.product_name
            """,
            [*params, run_id],
        )
        rows = cursor.fetchall()
    max_count_by_segment: dict[str, int] = {}
    for segment_id, _product_name, count, _score_value in rows:
        max_count_by_segment[segment_id] = max(max_count_by_segment.get(segment_id, 0), count)
    result = []
    for segment_id, product_name, count, score in rows:
        resolved = _resolved_segment({"segment_id": segment_id, "segment_name": segment_id, "source_group": "Enterprise"}) or {}
        max_count = max_count_by_segment.get(segment_id) or count or 1
        result.append(
            {
                "segment_id": segment_id,
                "segment_name": resolved.get("display_name") or "Unclassified behavior",
                "population": "Enterprise",
                "product_name": product_name,
                "recommendation_count": count,
                "average_score_value": _number(score) or 0,
                "average_score": _score(score),
                "count_bar_width": f"{(count / max_count) * 100:.1f}%",
                "score_bar_width": f"{max(2, min(100, float(score or 0) * 100)):.1f}%",
            }
        )
    return resul


def _recommendation_segment_explorer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {"retail": {}, "enterprise": {}}
    for row in rows:
        population = str(row.get("population") or "").lower()
        if population not in grouped:
            continue
        segment = grouped[population].setdefault(
            row["segment_id"],
            {"segment_id": row["segment_id"], "segment_name": row["segment_name"], "items": []},
        )
        segment["items"].append(row)
    for segments in grouped.values():
        for segment in segments.values():
            segment["items"] = sorted(segment["items"], key=lambda item: item["recommendation_count"], reverse=True)
    return grouped


def customer_dashboard(customer_id: int) -> dict[str, Any]:
    context = get_customer_context(customer_id)
    customer = context["customer"]
    segment = _resolved_segment(context["segment"])
    if segment:
        segment["clustering_features"] = _static_customer_clustering_features().get(customer_id, {})
        population = str(segment.get("source_group") or customer.get("customer_class") or "").lower()
        prefix = "seg_ent_" if population.startswith("ent") else "seg_ret_"
        segment["segment_median_features"] = _static_cluster_feature_medians(population).get(
            f"{prefix}{segment.get('cluster_id')}", {}
        )
    propensities = sorted(
        context["propensities"],
        key=lambda item: item["propensity_score"] if item["propensity_score"] is not None else -1,
        reverse=True,
    )
    display_propensities = propensities
    display_recommendations = context["recommendations"]
    selected_recommendations = _selected_recommendations(display_recommendations, propensities)
    forecasts = _select_forecast_rows([item for item in context["forecasts"] if item.get("metric") == "avg_balance"])
    forecast_pool = customer.get("forecast_pool")
    forecast_run = latest_forecast_run()
    latest_forecast_row = None
    if forecast_pool and forecast_run is not None:
        latest_forecast_row = (
            SegmentForecast.objects.filter(run=forecast_run, pool=forecast_pool, metric="avg_balance")
            .select_related("run")
            .order_by("-forecast_date", "-created_at")
            .first()
        )

    return {
        "customer": customer,
        "segment": segment,
        "owned_products": context["owned_products"],
        "recommendation_owned_products": context["owned_products"],
        "propensities": [
            {
                **item,
                "propensity_display": _score(item.get("propensity_score")),
                "bar_width": f"{max(0, min(100, float(item.get('propensity_score') or 0) * 100)):.1f}%",
            }
            for item in display_propensities
        ],
        "forecasts": [
            {
                **item,
                "predicted_value_display": _money(item.get("predicted_value")),
                "actual_value_display": _money(item.get("actual_value")),
            }
            for item in forecasts
        ],
        "recommendations": _FeaturedLastList(selected_recommendations),
        "featured_recommendation": selected_recommendations[-1] if selected_recommendations else None,
        "decision_support_products": _decision_support_products(propensities, context["recommendations"]),
        "customer_feature_series": _customer_feature_series(customer_id),
        "forecast_chart": _forecast_chart(forecasts),
        "customer_product_graph": customer_product_graph(customer_id, context),
        "forecast_run": latest_forecast_row.run if latest_forecast_row else None,
        "missing_outputs": {
            "owned_products": not context["owned_products"],
            "propensities": not propensities,
            "forecasts": not forecasts,
            "recommendations": not context["recommendations"],
        },
        "owned_product_family_count": len({item.get("family") for item in context["owned_products"] if item.get("family")}),
        "segment_is_assigned": bool(segment and segment.get("segment_id")),
    }


def _selected_recommendations(
    recommendations: list[dict[str, Any]],
    propensities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    propensity_by_product = {item["product_id"]: item for item in propensities}
    rows = []
    for display_rank, item in enumerate(recommendations, start=1):
        propensity = propensity_by_product.get(item["product_id"])
        rows.append(
            {
                **item,
                "rank": display_rank,
                "stored_rank": item.get("rank"),
                "score_display": _score(item.get("score")),
                "score_bar_width": f"{max(0, min(100, float(item.get('score') or 0) * 100)):.1f}%",
                "propensity_display": _pct(propensity.get("propensity_score")) if propensity else "n/a",
                "segment_prior_display": _score(propensity.get("segment_product_prior")) if propensity else "n/a",
                "max_affinity_display": _score(propensity.get("max_owned_to_candidate_affinity")) if propensity else "n/a",
                "reason_display": _recommendation_reason_text(item, propensity),
            }
        )
    return rows


def _decision_support_products(propensities: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rec_by_product = {item["product_id"]: item for item in recommendations}
    product_ids = []
    for item in recommendations + propensities:
        product_id = item.get("product_id")
        if product_id and product_id not in product_ids:
            product_ids.append(product_id)

    prop_by_product = {item["product_id"]: item for item in propensities}
    rows = []
    for product_id in product_ids[:12]:
        propensity = prop_by_product.get(product_id, {})
        recommendation = rec_by_product.get(product_id, {})
        score = recommendation.get("score")
        propensity_score = propensity.get("propensity_score")
        product_name = recommendation.get("product_name") or propensity.get("product_name") or product_id
        product_summary = product_brochure_summaries().get(str(product_id).lower()) or product_brochure_summaries().get(str(product_name).lower())
        rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "product_summary": product_summary or "A recommended product selected from the bank's available product portfolio.",
                "propensity_score": propensity_score,
                "propensity_display": _score(propensity_score),
                "propensity_bar_width": f"{max(0, min(100, float(propensity_score or 0) * 100)):.1f}%",
                "recommendation_rank": recommendation.get("rank"),
                "recommendation_score": score,
                "recommendation_score_display": _score(score),
                "recommendation_bar_width": f"{max(0, min(100, float(score or 0) * 100)):.1f}%",
                "segment_prior_display": _score(propensity.get("segment_product_prior")) if propensity else "n/a",
                "max_affinity_display": _score(propensity.get("max_owned_to_candidate_affinity")) if propensity else "n/a",
                "reason_display": _recommendation_reason_text(recommendation, propensity),
                "is_recommended": bool(recommendation),
            }
        )
    return rows


def _customer_feature_series(customer_id: int, limit: int = 80) -> dict[str, Any]:
    if not ARCHETYPE_WEEKLY_PANEL.exists():
        return {"points": [], "channel_mix": [], "source": ""}
    try:
        import polars as pl

        df = (
            pl.scan_parquet(ARCHETYPE_WEEKLY_PANEL)
            .filter(pl.col("client_id") == customer_id)
            .select(
                "date",
                "weekly_deposit",
                "weekly_withdrawal",
                "weekly_net_cash_flow",
                "ending_balance",
                "tx_count",
                "digital_tx_pct",
                "atm_tx_pct",
                "branch_tx_pct",
            )
            .sort("date")
            .tail(limit)
            .collect()
        )
    except Exception:
        return {"points": [], "channel_mix": [], "source": ARCHETYPE_WEEKLY_PANEL.name}

    points = []
    for row in df.iter_rows(named=True):
        points.append(
            {
                "date": str(row["date"]),
                "deposit": _number(row["weekly_deposit"]),
                "withdrawal": _number(row["weekly_withdrawal"]),
                "net_cash_flow": _number(row["weekly_net_cash_flow"]),
                "ending_balance": _number(row["ending_balance"]),
                "tx_count": _number(row["tx_count"]),
                "digital_tx_pct": _number(row["digital_tx_pct"]),
                "atm_tx_pct": _number(row["atm_tx_pct"]),
                "branch_tx_pct": _number(row["branch_tx_pct"]),
            }
        )

    if points:
        avg_digital = sum(point["digital_tx_pct"] or 0 for point in points) / len(points)
        avg_atm = sum(point["atm_tx_pct"] or 0 for point in points) / len(points)
        avg_branch = sum(point["branch_tx_pct"] or 0 for point in points) / len(points)
    else:
        avg_digital = avg_atm = avg_branch = 0
    latest_balance = points[-1].get("ending_balance") if points else None
    first_balance = points[0].get("ending_balance") if points else None
    balance_change = _change_pct(first_balance, latest_balance)
    positive_cashflow_periods = sum(1 for point in points if (point.get("net_cash_flow") or 0) > 0)
    dominant_channel = max(
        (("Digital", avg_digital), ("ATM", avg_atm), ("Branch", avg_branch)),
        key=lambda item: item[1],
    )[0] if points else None
    return {
        "points": points,
        "channel_mix": [
            {"label": "Digital", "value": avg_digital, "display": _pct(avg_digital)},
            {"label": "ATM", "value": avg_atm, "display": _pct(avg_atm)},
            {"label": "Branch", "value": avg_branch, "display": _pct(avg_branch)},
        ],
        "summary": {
            "observed_periods": len(points),
            "balance_change": _signed_pct(first_balance, latest_balance),
            "positive_cashflow_periods": positive_cashflow_periods,
            "dominant_channel": dominant_channel,
        },
        "source": ARCHETYPE_WEEKLY_PANEL.name,
    }


def _forecast_chart(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    by_date: dict[str, dict[str, Any]] = {}
    for item in forecasts:
        point = by_date.setdefault(item["forecast_date"], {"date": item["forecast_date"]})
        if item.get("actual_value") is not None:
            point["actual"] = item["actual_value"]
        if item.get("predicted_value") is not None:
            point["forecast"] = item["predicted_value"]
        point["model"] = item.get("model_name")
        point["scenario"] = item.get("scenario_case")
    points = [by_date[key] for key in sorted(by_date)]
    return {"points": points, "has_confidence_interval": False}


def customer_product_graph(customer_id: int, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or get_customer_context(customer_id)
    customer = context["customer"]
    segment = context.get("segment") or {}
    segment_id = customer.get("behavioral_segment_id") or segment.get("segment_id") or customer.get("segment_label")

    products = {row.product_id: row.product_name for row in Product.objects.all()}
    needs = {row["need_id"]: row for row in _read_csv("customer_needs.csv")}
    product_need_edges = _read_csv("product_need_edges.csv")
    segment_need_edges = [row for row in _read_csv("segment_need_edges.csv") if row.get("segment_id") == segment_id]
    assigned_need_ids = {row.get("need_id") for row in segment_need_edges}
    owned_ids = {item["product_id"] for item in context.get("owned_products", [])}
    rec_ids = {item["product_id"] for item in context.get("recommendations", [])}
    product_ids = owned_ids | rec_ids
    graph_need_ids = assigned_need_ids | {
        row.get("need_id") for row in product_need_edges if row.get("product_id") in product_ids and row.get("need_id")
    }

    nodes = [{"id": f"customer:{customer_id}", "label": f"Customer {customer_id}", "group": "Customer"}]
    if segment_id:
        nodes.append({"id": f"segment:{segment_id}", "label": segment.get("display_name") or "Unclassified behavior", "group": "BehavioralSegment"})
    for need_id in sorted(graph_need_ids):
        row = needs.get(need_id, {})
        nodes.append({"id": f"need:{need_id}", "label": row.get("need_label") or row.get("need_name") or need_id, "group": "CustomerNeed"})
    for product_id in sorted(product_ids):
        group = "OwnedProduct" if product_id in owned_ids else "RecommendedProduct"
        if product_id in owned_ids and product_id in rec_ids:
            group = "OwnedAndRecommendedProduct"
        nodes.append({"id": f"product:{product_id}", "label": products.get(product_id, product_id), "group": group})

    edges = []
    if segment_id:
        edges.append({"from": f"customer:{customer_id}", "to": f"segment:{segment_id}", "label": "assignedSegment"})
        for row in segment_need_edges:
            need_id = row.get("need_id")
            if need_id:
                edges.append({"from": f"segment:{segment_id}", "to": f"need:{need_id}", "label": row.get("relationship_strength") or "LIKELY_NEEDS"})
    for item in context.get("owned_products", []):
        edges.append({"from": f"customer:{customer_id}", "to": f"product:{item['product_id']}", "label": "owns"})
    for item in context.get("recommendations", []):
        edges.append({"from": f"customer:{customer_id}", "to": f"product:{item['product_id']}", "label": f"recommended #{item.get('rank', '')}".strip()})
    for row in product_need_edges:
        product_id = row.get("product_id")
        need_id = row.get("need_id")
        if product_id in product_ids and need_id in graph_need_ids:
            edges.append({"from": f"product:{product_id}", "to": f"need:{need_id}", "label": row.get("relationship_type") or "servesNeed"})

    return {"nodes": nodes, "edges": edges}


def dashboard_context_for_customer_id(raw_customer_id: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not raw_customer_id:
        return None, None
    try:
        customer_id = int(raw_customer_id)
    except (TypeError, ValueError):
        return None, "Enter a valid numeric customer ID."
    try:
        return customer_dashboard(customer_id), None
    except CustomerNotFoundError:
        return None, f"Customer {customer_id} was not found."
