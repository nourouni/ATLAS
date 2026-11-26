"""Artifact-only hybrid recommendation generation and validation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

from archetype_data.propensity.product_propensity_model import (
    BEHAVIORAL_SEGMENT_BRIDGE_PATH,
    CANONICAL_SEGMENTS,
)
from archetype_data.recommendation.hybrid_recommender import HybridRecommender
from ml_pipeline.paths import CUSTOMERS, FINAL_RECOMMENDATIONS, PRODUCT_PROPENSITY_TOP, PRODUCTS
from ml_pipeline.runtime import ensure_exists

REMOVED_PRODUCTS = PRODUCTS.parent / "products_removed.csv"


def _json(value):
    return json.dumps(value if value is not None else [], ensure_ascii=False, allow_nan=False)


def _load_metadata() -> tuple[dict[int, dict], dict[str, dict]]:
    customers = pl.read_csv(CUSTOMERS).select(["client_id", "segment", "latent_archetype"])
    bridge = pl.read_csv(BEHAVIORAL_SEGMENT_BRIDGE_PATH).select(["client_id", "behavioral_segment"])
    if customers.get_column("client_id").n_unique() != customers.height:
        raise ValueError(f"Duplicate client_id values in customer source: {CUSTOMERS}")
    if bridge.get_column("client_id").n_unique() != bridge.height:
        raise ValueError(f"Duplicate client_id values in behavioral bridge: {BEHAVIORAL_SEGMENT_BRIDGE_PATH}")
    invalid_segments = sorted(set(bridge.get_column("behavioral_segment").drop_nulls().to_list()) - CANONICAL_SEGMENTS)
    if invalid_segments:
        raise ValueError(f"Unknown behavioral segment IDs in canonical bridge: {invalid_segments}")
    metadata = customers.join(bridge, on="client_id", how="left", validate="1:1")
    return (
        {int(row["client_id"]): row for row in metadata.iter_rows(named=True)},
        {
            str(row["product_id"]): row
            for row in pl.read_csv(PRODUCTS).select(
                ["product_id", "product_name", "category", "classification", "type"]
            ).iter_rows(named=True)
        },
    )


def generate_and_save(output_path: Path = FINAL_RECOMMENDATIONS) -> dict:
    ensure_exists(PRODUCT_PROPENSITY_TOP)
    ensure_exists(PRODUCTS)
    customer_metadata, product_metadata = _load_metadata()
    recommender = HybridRecommender(propensity_path=PRODUCT_PROPENSITY_TOP)
    customer_ids = pl.read_parquet(PRODUCT_PROPENSITY_TOP).get_column("client_id").unique().sort().to_list()
    started = time.perf_counter()
    rows = []
    total_customers = len(customer_ids)
    for index, customer_id in enumerate(customer_ids, start=1):
        customer_meta = customer_metadata.get(int(customer_id))
        if customer_meta is None:
            raise ValueError(f"Customer {customer_id} is missing from {CUSTOMERS}")
        if customer_meta.get("behavioral_segment") is None:
            raise ValueError(f"Customer {customer_id} has no canonical behavioral segment")
        result = recommender.recommend_customer(int(customer_id), top_k=5, persist_provenance=False)
        for recommendation in result["recommendations"]:
            product_id = str(recommendation.get("product_id") or "")
            product_meta = product_metadata.get(product_id)
            if product_meta is None:
                raise ValueError(
                    "Final recommendation product_id does not resolve to the verified catalog: "
                    f"{product_id or recommendation.get('product')}"
                )
            rows.append({
                "client_id": int(customer_id),
                "product_id": product_id,
                "segment": customer_meta.get("segment"),
                "behavioral_segment": customer_meta.get("behavioral_segment"),
                "latent_archetype": customer_meta.get("latent_archetype"),
                "product_name": product_meta["product_name"],
                "category": product_meta["category"],
                "classification": product_meta["classification"],
                "type": product_meta["type"],
                "rank": int(recommendation["rank"]),
                "propensity_score": recommendation.get("propensity_score"),
                "final_score": recommendation.get("final_score"),
                "need_match": int(recommendation.get("need_match") or 0),
                "matched_needs": _json(recommendation.get("matched_needs")),
                "eligible": recommendation.get("eligible"),
                "already_owned": bool(recommendation.get("already_owned")),
                "missing_requirements": _json(recommendation.get("missing_requirements")),
                "conflicts": _json(recommendation.get("conflicts")),
                "alternatives": _json(recommendation.get("alternatives")),
                "supporting_relations": _json(recommendation.get("supporting_relations")),
                "sources": _json(recommendation.get("sources")),
                "explanation": recommendation.get("explanation") or "",
            })
        if index % 10000 == 0 or index == total_customers:
            elapsed = time.perf_counter() - started
            print(
                f"[recommendation] progress: {index}/{total_customers} customers "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    frame = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return {"path": str(output_path), "rows": frame.height, "customers": frame.get_column("client_id").n_unique() if frame.height else 0}


def validate_recommendations(path: Path = FINAL_RECOMMENDATIONS) -> dict:
    ensure_exists(path)
    frame = pl.read_parquet(path)
    required = [
        "client_id", "segment", "behavioral_segment", "latent_archetype",
        "product_id", "product_name", "category", "classification", "type",
        "propensity_score", "final_score", "rank", "explanation",
    ]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Recommendation artifact is missing required columns: {missing_columns}")
    catalog = pl.read_csv(PRODUCTS).select(["product_id", "product_name", "category", "classification", "type"])
    unknown = frame.join(catalog, on="product_id", how="anti").heigh
    removed = 0
    if REMOVED_PRODUCTS.exists():
        removed_ids = pl.read_csv(REMOVED_PRODUCTS).select("product_id")
        removed = frame.join(removed_ids, on="product_id", how="inner").heigh
    missing_segment = frame.filter(pl.col("segment").is_null() | (pl.col("segment").str.strip_chars() == "")).heigh
    missing_behavioral = frame.filter(
        pl.col("behavioral_segment").is_null() | (pl.col("behavioral_segment").str.strip_chars() == "")
    ).heigh
    unknown_behavioral = frame.filter(~pl.col("behavioral_segment").is_in(sorted(CANONICAL_SEGMENTS))).heigh
    missing_product_metadata = frame.filter(
        pl.any_horizontal(
            *(pl.col(field).is_null() for field in ("product_name", "category", "classification", "type"))
        )
    ).heigh
    score_cols = [column for column in ("propensity_score", "final_score") if column in frame.columns]
    null_scores = sum(frame.filter(pl.col(column).is_null()).height for column in score_cols)
    nonfinite = sum(frame.filter(~pl.col(column).is_null() & ~pl.col(column).is_finite()).height for column in score_cols)
    result = {
        "path": str(path),
        "rows": frame.height,
        "customers": frame.get_column("client_id").n_unique() if frame.height else 0,
        "products": frame.get_column("product_id").n_unique() if frame.height else 0,
        "recommendations_per_customer": frame.group_by("client_id").len().select(pl.col("len").value_counts()).to_dicts() if frame.height else [],
        "duplicate_customer_product_recommendations": frame.select(["client_id", "product_id"]).is_duplicated().sum(),
        "owned_product_violations": frame.filter(pl.col("already_owned")).height,
        "eligibility_violations": frame.filter(pl.col("eligible") == "ineligible").height,
        "rule_violations": frame.filter((pl.col("conflicts") != "[]") | (pl.col("missing_requirements") != "[]")).height,
        "null_or_nonfinite_scores": nonfinite,
        "null_scores": null_scores,
        "missing_explanations": frame.filter(pl.col("explanation").str.len_chars() == 0).height,
        "unknown_products": unknown,
        "removed_products": removed,
        "missing_segment": missing_segment,
        "missing_behavioral_segment": missing_behavioral,
        "unknown_behavioral_segments": unknown_behavioral,
        "seg_ret_5_count": frame.filter(pl.col("behavioral_segment") == "seg_ret_5").height,
        "seg_ret_6_count": frame.filter(pl.col("behavioral_segment") == "seg_ret_6").height,
        "missing_product_metadata": missing_product_metadata,
        "missing_latent_archetype": frame.filter(pl.col("latent_archetype").is_null()).height,
    }
    if any(result[key] for key in ("duplicate_customer_product_recommendations", "owned_product_violations", "eligibility_violations", "rule_violations", "null_or_nonfinite_scores", "missing_explanations", "unknown_products", "removed_products", "missing_segment", "missing_behavioral_segment", "unknown_behavioral_segments", "seg_ret_5_count", "seg_ret_6_count", "missing_product_metadata")):
        raise ValueError(json.dumps(result, indent=2, default=str))
    return resul


if __name__ == "__main__":
    print(json.dumps(generate_and_save(), indent=2))
