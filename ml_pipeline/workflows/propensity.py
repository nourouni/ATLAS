from __future__ import annotations

import time
import json
from datetime import datetime, timezone
from typing import Any

import joblib

from ml_pipeline.paths import (
    ARCHETYPE_ROOT,
    ARTIFACTS_ROOT,
    CUSTOMERS,
    OWNERSHIP_PAIRS,
    PRODUCT_PROPENSITY_MODEL,
PRODUCT_PROPENSITY_OUTPUTS,
    PRODUCTS,
    RECOMMENDATIONS,
)
from ml_pipeline.runtime import copy_if_exists, ensure_exists, import_from_dir, promote_product_ownership_artifacts, with_db_transaction


def promote_propensity_artifacts() -> None:
    copy_if_exists(
        PRODUCT_PROPENSITY_MODEL,
        ARTIFACTS_ROOT / "models" / "product_propensity" / "best_product_propensity_model.joblib",
    )
    copy_if_exists(
        PRODUCT_PROPENSITY_OUTPUTS / "product_feature_list.csv",
        ARTIFACTS_ROOT / "models" / "product_propensity" / "product_feature_list.csv",
    )
    copy_if_exists(PRODUCT_PROPENSITY_OUTPUTS / "top_non_owned_product_recommendations.parquet", RECOMMENDATIONS)
    for name in [
        "best_product_propensity_model.joblib",
        "product_feature_list.csv",
        "product_model_comparison.csv",
        "product_holdout_predictions.parquet",
        "product_holdout_support.csv",
        "product_holdout_metrics.csv",
        "product_calibration.csv",
        "product_propensity_metadata.json",
        "product_propensity_input_validation.json",
        "product_label_counts.csv",
        "product_propensity_scores.parquet",
        "top_non_owned_product_recommendations.parquet",
    ]:
        copy_if_exists(PRODUCT_PROPENSITY_OUTPUTS / name, ARTIFACTS_ROOT / "product_propensity" / name)


def _require_fresh_propensity_inputs() -> None:
    required = [
        PRODUCT_PROPENSITY_MODEL,
        PRODUCT_PROPENSITY_OUTPUTS / "product_feature_list.csv",
        PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_input_validation.json",
        PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_metadata.json",
        PRODUCT_PROPENSITY_OUTPUTS / "product_model_comparison.csv",
    ]
    for path in required:
        ensure_exists(path)
    ownership_mtime = OWNERSHIP_PAIRS.stat().st_mtime
    stale = [str(path) for path in required if path.stat().st_mtime < ownership_mtime]
    if stale:
        ownership_time = datetime.fromtimestamp(ownership_mtime, tz=timezone.utc).isoformat()
        raise RuntimeError(
            "Refusing to score with stale propensity artifacts. "
            f"Canonical ownership modified at {ownership_time}; stale artifacts: {stale}"
        )
    def read_record(path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            if len(value) != 1 or not isinstance(value[0], dict):
                raise RuntimeError(f"Propensity metadata must contain one JSON object: {path}")
            return value[0]
        if not isinstance(value, dict):
            raise RuntimeError(f"Propensity metadata must be a JSON object: {path}")
        return value

    metadata = read_record(PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_metadata.json")
    if metadata.get("label_type") != "synthetic product-ownership propensity":
        raise RuntimeError("Propensity metadata does not identify the canonical synthetic ownership run.")
    validation = read_record(PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_input_validation.json")
    if validation.get("unique_product_ids") != 79 or validation.get("unknown_ids") != 0 or validation.get("removed_ids") != 0:
        raise RuntimeError("Propensity validation artifact does not confirm the 79-product canonical universe.")


def run(skip_schema: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    for path in [
        PRODUCTS,
        ARCHETYPE_ROOT / "outputs" / "archetype_customers.csv",
        ARCHETYPE_ROOT / "outputs" / "archetype_features.csv",
        OWNERSHIP_PAIRS,
    ]:
        ensure_exists(path)

    _require_fresh_propensity_inputs()

    module = import_from_dir(
        "product_propensity_model",
        ARCHETYPE_ROOT / "propensity" / "product_propensity_model.py",
        ARCHETYPE_ROOT / "propensity",
    )
    model = joblib.load(PRODUCT_PROPENSITY_MODEL)
    top = module.score_non_owned_products(
        model,
        data_dir=ARCHETYPE_ROOT / "outputs",
        output_dir=PRODUCT_PROPENSITY_OUTPUTS,
    )
    promote_product_ownership_artifacts()
    promote_propensity_artifacts()

    def work(db, conn):
        db.load_products(conn, PRODUCTS)
        db.load_customers(conn, CUSTOMERS)
        db.load_output_products(conn, [RECOMMENDATIONS])
        propensity_run_id = db.upsert_model_run(
            conn,
            "latest_product_propensity",
            "product_propensity",
            ARTIFACTS_ROOT / "models" / "product_propensity" / "best_product_propensity_model.joblib",
        )
        recommendation_run_id = db.upsert_model_run(
            conn,
            "latest_product_propensity_recommendations",
            "recommendation",
            RECOMMENDATIONS,
            {"source_model_run_id": propensity_run_id},
        )
        predictions, recommendations = db.load_recommendations(conn, RECOMMENDATIONS, recommendation_run_id)
        return {
            "propensity_run_id": propensity_run_id,
            "recommendation_run_id": recommendation_run_id,
            "propensity_predictions": predictions,
            "recommendations": recommendations,
        }

    counts = with_db_transaction(work, skip_schema)
    counts["fresh_scored_rows"] = int(top.height)
    counts["duration_seconds"] = round(time.perf_counter() - start, 3)
    return counts
