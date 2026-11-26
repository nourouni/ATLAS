"""Artifact-only propensity scoring and validation; never opens PostgreSQL."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import polars as pl

from ml_pipeline.paths import (
    ARCHETYPE_ROOT,
    OWNERSHIP_PAIRS,
    PRODUCT_PROPENSITY_MODEL,
    PRODUCT_PROPENSITY_OUTPUTS,
    PRODUCT_PROPENSITY_SCORES,
    PRODUCT_PROPENSITY_TOP,
    PRODUCTS,
)
from ml_pipeline.runtime import ensure_exists, import_from_dir
from ml_pipeline.workflows.propensity import _require_fresh_propensity_inputs
from archetype_data.propensity.product_propensity_model import ProductPropensityConfig


def readiness_report() -> dict:
    _require_fresh_propensity_inputs()
    validation_path = PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_input_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if isinstance(validation, list):
        validation = validation[0] if len(validation) == 1 else validation
    return {
        "status": "READY",
        "model": str(PRODUCT_PROPENSITY_MODEL),
        "model_modified_at_utc": datetime.fromtimestamp(PRODUCT_PROPENSITY_MODEL.stat().st_mtime, tz=timezone.utc).isoformat(),
        "ownership_source": str(OWNERSHIP_PAIRS),
        "validation_artifact": str(validation_path),
        "validation": validation,
    }


def score_and_save() -> dict:
    _require_fresh_propensity_inputs()
    ensure_exists(PRODUCTS)
    module = import_from_dir(
        "product_propensity_model_artifact_runner",
        ARCHETYPE_ROOT / "propensity" / "product_propensity_model.py",
        ARCHETYPE_ROOT / "propensity",
    )
    model = joblib.load(PRODUCT_PROPENSITY_MODEL)
    top = module.score_non_owned_products(
        model,
        data_dir=ARCHETYPE_ROOT / "outputs",
        output_dir=PRODUCT_PROPENSITY_OUTPUTS,
    )
    scores = pl.read_parquet(PRODUCT_PROPENSITY_SCORES)
    return {
        "score_artifact": str(PRODUCT_PROPENSITY_SCORES),
        "ranked_candidate_artifact": str(PRODUCT_PROPENSITY_TOP),
        "rows": scores.height,
        "customers": scores.get_column("client_id").n_unique(),
        "products": scores.get_column("product_id").n_unique(),
        "ranked_rows": top.height,
    }


def validate_scores(path: Path = PRODUCT_PROPENSITY_SCORES) -> dict:
    ensure_exists(path)
    frame = pl.read_parquet(path)
    catalog = pl.read_csv(PRODUCTS).select("product_id")
    canonical_ids = set(catalog.get_column("product_id").to_list())
    unknown = frame.join(catalog, on="product_id", how="anti").heigh
    score = frame.get_column("propensity_score")
    null_or_nonfinite = frame.filter(~pl.col("propensity_score").is_finite()).heigh
    result = {
        "path": str(path),
        "rows": frame.height,
        "customers": frame.get_column("client_id").n_unique(),
        "products": frame.get_column("product_id").n_unique(),
        "min_score": score.min(),
        "max_score": score.max(),
        "mean_score": score.mean(),
        "duplicate_pairs": frame.select(["client_id", "product_id"]).is_duplicated().sum(),
        "owned_candidates": frame.filter(pl.col("owned")).height,
        "ineligible_candidates": frame.filter(~pl.col("eligible")).height,
        "unknown_products": unknown,
        "removed_products": 0,
        "null_or_nonfinite_scores": null_or_nonfinite,
        "canonical_product_count": len(canonical_ids),
    }
    if any(result[key] for key in ("duplicate_pairs", "owned_candidates", "ineligible_candidates", "unknown_products", "removed_products", "null_or_nonfinite_scores")) or result["canonical_product_count"] != 79:
        raise ValueError(json.dumps(result, indent=2, default=str))
    return resul


if __name__ == "__main__":
    print(json.dumps(readiness_report(), indent=2))
