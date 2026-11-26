from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_ROOT = PROJECT_ROOT / "archetype_data"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

PRODUCTS = ARTIFACTS_ROOT / "ontology" / "csv" / "products_verified.csv"
CUSTOMERS = ARCHETYPE_ROOT / "outputs" / "archetype_customers.csv"
SEGMENTS = ARTIFACTS_ROOT / "ontology" / "csv" / "behavioral_segments_clean.csv"
OWNERSHIP_FEATURES = ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_features.parquet"
OWNERSHIP_LONG = ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_long.parquet"
OWNERSHIP_PAIRS = ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_pairs.parquet"
RECOMMENDATIONS = ARTIFACTS_ROOT / "lookup_data" / "recommendation" / "top_non_owned_product_recommendations.parquet"

STATIC_CLUSTERING_OUTPUTS = ARCHETYPE_ROOT / "static_clustering" / "clustering_outputs"
PRODUCT_PROPENSITY_OUTPUTS = ARCHETYPE_ROOT / "propensity" / "outputs" / "product_propensity"
PRODUCT_PROPENSITY_MODEL = PRODUCT_PROPENSITY_OUTPUTS / "best_product_propensity_model.joblib"
PRODUCT_PROPENSITY_SCORES = PRODUCT_PROPENSITY_OUTPUTS / "product_propensity_scores.parquet"
PRODUCT_PROPENSITY_TOP = PRODUCT_PROPENSITY_OUTPUTS / "top_non_owned_product_recommendations.parquet"
FINAL_RECOMMENDATIONS = (
    ARCHETYPE_ROOT
    / "recommendation"
    / "outputs"
    / "final_hybrid_recommendations_current.parquet"
)
PRODUCT_SIM_OUTPUTS = ARTIFACTS_ROOT / "lookup_data" / "ownership"

FORECAST_METHOD = "rolling_temporal_feature_filled_cluster_pooled"
FORECAST_OUTPUTS = ARCHETYPE_ROOT / "ts" / "method_outputs" / FORECAST_METHOD / "data"
FORECAST_POINTS = FORECAST_OUTPUTS / "forecast_points.csv"
