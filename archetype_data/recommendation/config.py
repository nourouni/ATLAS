from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ARCHETYPE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARCHETYPE_ROOT.paren
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
CURRENT_PROPENSITY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "lookup_data"
    / "recommendation"
    / "top_non_owned_product_recommendations.parquet"
)
CURRENT_RECOMMENDATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "lookup_data"
    / "recommendation"
    / "final_hybrid_recommendations.parquet"
)


@dataclass(frozen=True)
class RecommendationPaths:
    project_root: Path = PROJECT_ROOT
    archetype_root: Path = ARCHETYPE_ROOT
    artifacts_root: Path = ARTIFACTS_ROOT
    ontology_path: Path = ARTIFACTS_ROOT / "ontology" / "atlas_demo_banking_ontology_fixed.owl"
    propensity_recommendations_path: Path = CURRENT_PROPENSITY_PATH
    final_recommendations_path: Path = CURRENT_RECOMMENDATION_OUTPUT_PATH
    product_propensity_model_path: Path = (
        ARTIFACTS_ROOT
        / "models"
        / "product_propensity"
        / "best_product_propensity_model.joblib"
    )
    product_feature_list_path: Path = (
        ARTIFACTS_ROOT
        / "models"
        / "product_propensity"
        / "product_feature_list.csv"
    )
    ownership_long_path: Path = (
        ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_long.parquet"
    )
    ownership_pairs_path: Path = (
        ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_pairs.parquet"
    )
    ownership_features_path: Path = (
        ARTIFACTS_ROOT / "lookup_data" / "ownership" / "product_ownership_features.parquet"
    )
    product_catalog_path: Path = ARTIFACTS_ROOT / "ontology" / "csv" / "products_verified.csv"
    behavioral_segments_path: Path = ARTIFACTS_ROOT / "mappings" / "behavioral_segments.csv"
    static_retail_clusters_path: Path = (
        ARCHETYPE_ROOT / "static_clustering" / "retail_clustered.csv"
    )
    temporal_retail_clusters_path: Path = (
        ARTIFACTS_ROOT / "lookup_data" / "clustering" / "temporal_feature_clusters_retail.parquet"
    )


DEFAULT_PATHS = RecommendationPaths()
