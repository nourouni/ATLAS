from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from ml_pipeline.paths import (
    ARCHETYPE_ROOT,
    ARTIFACTS_ROOT,
    CUSTOMERS,
    FINAL_RECOMMENDATIONS,
    OWNERSHIP_FEATURES,
    PRODUCT_PROPENSITY_TOP,
    PRODUCTS,
)
from ml_pipeline.runtime import copy_if_exists, ensure_exists, import_from_dir, promote_product_ownership_artifacts, with_db_transaction


def _read_columns(path: Path, kind: str) -> tuple[set[str], int]:
    if kind == "parquet":
        frame = pl.scan_parquet(path).head(1).collect()
    elif kind == "csv":
        frame = pl.scan_csv(path, infer_schema_length=1000).head(1).collect()
    else:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    return set(frame.columns), frame.heigh


def check_runtime_inputs() -> tuple[list[str], list[str]]:
    checks = [
        (
            ARTIFACTS_ROOT / "ontology" / "atlas_demo_banking_ontology_fixed.owl",
            (),
            "file",
        ),
        (
            FINAL_RECOMMENDATIONS,
            (
                "client_id",
                "segment",
                "behavioral_segment",
                "latent_archetype",
                "product_id",
                "product_name",
                "category",
                "classification",
                "type",
                "propensity_score",
                "rank",
            ),
            "parquet",
        ),
        (
            ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_long.parquet",
            ("client_id", "product_id", "product_name", "owned", "eligible"),
            "parquet",
        ),
        (
            OWNERSHIP_FEATURES,
            ("client_id", "n_products_owned", "n_families_owned"),
            "parquet",
        ),
        (CUSTOMERS, ("client_id", "segment"), "csv"),
        (PRODUCTS, ("product_id", "product_name", "category", "classification", "type"), "csv"),
    ]
    errors: list[str] = []
    warnings: list[str] = []
    for path, required_columns, kind in checks:
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")
            continue
        if kind == "file":
            continue
        try:
            columns, rows = _read_columns(path, kind)
        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")
            continue
        missing = [column for column in required_columns if column not in columns]
        if missing:
            errors.append(f"{path} is missing column(s): {missing}")
        if rows < 1:
            errors.append(f"{path} has no rows.")
    return errors, warnings


def promote_recommendation_artifacts() -> None:
    copy_if_exists(FINAL_RECOMMENDATIONS, ARTIFACTS_ROOT / "recommendation" / FINAL_RECOMMENDATIONS.name)
    for name in ["README.md", "chapter_product_recommendation.md", "config.py", "hybrid_recommender.py"]:
        copy_if_exists(ARCHETYPE_ROOT / "recommendation" / name, ARTIFACTS_ROOT / "recommendation" / "runtime_sources" / name)


def preload_validation(path: Path = FINAL_RECOMMENDATIONS) -> dict[str, int | str]:
    """Read-only reconciliation of final recommendation rows before DB loading."""
    ensure_exists(path)
    frame = pl.read_parquet(path)
    required = {
        "client_id", "product_id", "propensity_score", "final_score", "rank",
        "segment", "behavioral_segment", "latent_archetype", "product_name",
        "category", "classification", "type", "explanation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Recommendation artifact is missing preload columns: {missing}")

    customer_ids = pl.read_csv(CUSTOMERS).select("client_id").unique()
    product_ids = pl.read_csv(PRODUCTS).select("product_id").unique()
    missing_customers = frame.join(customer_ids, on="client_id", how="anti").heigh
    unknown_products = frame.join(product_ids, on="product_id", how="anti").heigh
    duplicate_pairs = frame.select(["client_id", "product_id"]).is_duplicated().sum()
    missing_final_score = frame.filter(pl.col("final_score").is_null()).heigh
    missing_rank = frame.filter(pl.col("rank").is_null()).heigh
    invalid_scores = (
        frame.filter(
            (pl.col("propensity_score").is_not_null() & ~pl.col("propensity_score").is_finite())
            | (pl.col("final_score").is_not_null() & ~pl.col("final_score").is_finite())
        ).heigh
    )
    owned_violations = frame.filter(pl.col("already_owned") == True).height if "already_owned" in frame.columns else 0
    eligibility_violations = (
        frame.filter(pl.col("eligible") == "ineligible").height if "eligible" in frame.columns else 0
    )
    invalid_rows = frame.filter(
        pl.col("final_score").is_null()
        | pl.col("rank").is_null()
        | pl.col("segment").is_null()
        | pl.col("behavioral_segment").is_null()
    ).heigh
    invalid_rows += missing_customers + unknown_products + duplicate_pairs + invalid_scores
    invalid_rows += owned_violations + eligibility_violations
    return {
        "path": str(path),
        "total_final_recommendation_rows": frame.height,
        "propensity_backed_rows": frame.filter(pl.col("propensity_score").is_not_null()).height,
        "ontology_only_rows": frame.filter(pl.col("propensity_score").is_null()).height,
        "rows_to_insert": frame.height - invalid_rows,
        "rows_dropped": invalid_rows,
        "duplicate_customer_product_rows": duplicate_pairs,
        "unknown_product_ids": unknown_products,
        "missing_customer_ids": missing_customers,
        "missing_final_score": missing_final_score,
        "missing_final_rank": missing_rank,
        "null_propensity_score": frame.filter(pl.col("propensity_score").is_null()).height,
        "null_final_score": missing_final_score,
        "invalid_or_nonfinite_scores": invalid_scores,
        "owned_product_violations": owned_violations,
        "eligibility_violations": eligibility_violations,
    }


def run(skip_schema: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    promote_product_ownership_artifacts()
    recommendation_dir = ARCHETYPE_ROOT / "recommendation"
    errors, warnings = check_runtime_inputs()
    if errors:
        raise RuntimeError("Recommendation readiness failed: " + "; ".join(errors))
    consistency = import_from_dir(
        "recommendation_output_consistency",
        recommendation_dir / "test_output_consistency.py",
        recommendation_dir,
    )
    sample_ids = consistency._sample_customer_ids(limit=100, path=PRODUCT_PROPENSITY_TOP)
    recommender = consistency.HybridRecommender(propensity_path=PRODUCT_PROPENSITY_TOP)
    for customer_id in sample_ids:
        result = recommender.recommend_customer(int(customer_id), top_k=5)
        json.dumps(result, allow_nan=False)
        consistency._assert_json_finite(result)
        consistency._assert_fields(result["rule_checks"], consistency.REQUIRED_RULE_FIELDS, "rule check")
        consistency._assert_fields(result["recommendations"], consistency.REQUIRED_RECOMMENDATION_FIELDS, "recommendation")
        consistency._assert_rule_consistency(result, int(customer_id))
        consistency._assert_rank_consistency(result["recommendations"], int(customer_id))

    for path in [PRODUCTS, CUSTOMERS, OWNERSHIP_FEATURES, FINAL_RECOMMENDATIONS]:
        ensure_exists(path)
    preload = preload_validation()
    if preload["rows_dropped"]:
        raise RuntimeError(f"Recommendation preload validation failed: {json.dumps(preload, indent=2)}")
    promote_recommendation_artifacts()

    def work(db, conn):
        db.load_products(conn, PRODUCTS)
        db.load_customers(conn, CUSTOMERS)
        db.load_output_products(conn, [FINAL_RECOMMENDATIONS])
        run_id = db.upsert_model_run(conn, "latest_recommendation_refresh", "recommendation", FINAL_RECOMMENDATIONS)
        predictions, recommendations = db.load_recommendations(conn, FINAL_RECOMMENDATIONS, run_id)
        return {
            "recommendation_run_id": run_id,
            "propensity_predictions": predictions,
        "recommendations": recommendations,
            "preload_validation": preload,
        }

    counts = with_db_transaction(work, skip_schema)
    counts["hybrid_recommender_validated_customers"] = len(sample_ids)
    counts["readiness_warnings"] = warnings
    counts["duration_seconds"] = round(time.perf_counter() - start, 3)
    return counts
