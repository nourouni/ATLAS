from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import sys
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import polars as pl
from sklearn.exceptions import InconsistentVersionWarning

try:
    from .config import DEFAULT_PATHS, RecommendationPaths
except ImportError:
    from config import DEFAULT_PATHS, RecommendationPaths


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    required_columns: tuple[str, ...]
    kind: str
    min_rows: int = 1


ARTIFACT_SPECS = (
    ArtifactSpec(
        DEFAULT_PATHS.final_recommendations_path,
        (
            "client_id", "product_id", "product_name", "rank",
            "propensity_score", "final_score", "explanation",
            "sources", "already_owned",
        ),
        "parquet",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.propensity_recommendations_path,
        (
            "client_id",
            "segment",
            "behavioral_segment",
            "latent_archetype",
            "product_id",
            "product_name",
            "category",
            "propensity_score",
            "rank",
        ),
        "parquet",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.ownership_long_path,
        ("client_id", "product_id", "product_name", "owned", "eligible"),
        "parquet",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.ownership_pairs_path,
        ("client_id", "product_id", "owned", "eligible"),
        "parquet",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.ownership_features_path,
        ("client_id", "n_products_owned", "n_families_owned"),
        "parquet",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.product_feature_list_path,
        ("feature", "type"),
        "csv",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.product_catalog_path,
        ("product_id", "product_name", "category", "classification", "type"),
        "csv",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.behavioral_segments_path,
        ("segment_id", "segment_name", "source_group", "cluster_id"),
        "csv",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.static_retail_clusters_path,
        ("client_id", "segment", "latent_archetype", "customer_class", "cluster"),
        "csv",
    ),
    ArtifactSpec(
        DEFAULT_PATHS.temporal_retail_clusters_path,
        ("client_id", "segment", "cluster", "cluster_filled"),
        "parquet",
    ),
)

RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def _ontology_product_ids(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    product_ids: set[str] = set()
    for description in root.findall(f".//{RDF_NS}Description"):
        types = {
            child.attrib.get(f"{RDF_NS}resource", "").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            for child in description
            if child.tag.rsplit("}", 1)[-1] == "type"
        }
        if "Product" in types:
            product_ids.add(
                description.attrib.get(f"{RDF_NS}about", "").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            )
    return product_ids


def _read_schema(spec: ArtifactSpec) -> tuple[set[str], int]:
    if spec.kind == "parquet":
        frame = pl.scan_parquet(spec.path).head(max(spec.min_rows, 1)).collect()
    elif spec.kind == "csv":
        frame = pl.scan_csv(spec.path, infer_schema_length=1000).head(max(spec.min_rows, 1)).collect()
    else:
        raise ValueError(f"Unsupported artifact kind: {spec.kind}")
    return set(frame.columns), frame.heigh


def check_readiness(paths: RecommendationPaths = DEFAULT_PATHS) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not paths.ontology_path.exists():
        errors.append(f"Missing OWL ontology: {paths.ontology_path}")

    for spec in ARTIFACT_SPECS:
        if not spec.path.exists():
            errors.append(f"Missing required artifact: {spec.path}")
            continue
        try:
            columns, rows = _read_schema(spec)
        except Exception as exc:
            errors.append(f"Could not read {spec.path}: {exc}")
            continue
        missing = [column for column in spec.required_columns if column not in columns]
        if missing:
            errors.append(f"{spec.path} is missing column(s): {missing}")
        if rows < spec.min_rows:
            errors.append(f"{spec.path} has no rows.")

    try:
        verified_ids = set(pl.read_csv(paths.product_catalog_path).get_column("product_id").cast(pl.Utf8))
        if len(verified_ids) != 79:
            errors.append(f"Verified catalog must contain exactly 79 product IDs; found {len(verified_ids)}.")

        propensity = pl.read_parquet(paths.propensity_recommendations_path)
        unknown_propensity = (
            propensity.select("product_id").unique().join(
                pl.DataFrame({"product_id": list(verified_ids)}), on="product_id", how="anti"
            ).heigh
        )
        if unknown_propensity:
            errors.append(f"Propensity artifact contains {unknown_propensity} unknown product ID(s).")

        ontology_ids = _ontology_product_ids(paths.ontology_path)
        if ontology_ids != verified_ids:
            errors.append(
                "Runtime ontology product IDs do not exactly match the verified catalog "
                f"(ontology-only={len(ontology_ids - verified_ids)}, catalog-only={len(verified_ids - ontology_ids)})."
            )

        if paths.final_recommendations_path.exists():
            recommendations = pl.read_parquet(paths.final_recommendations_path)
            unknown_recommendations = (
                recommendations.select("product_id").unique().join(
                    pl.DataFrame({"product_id": list(verified_ids)}), on="product_id", how="anti"
                ).heigh
            )
            if unknown_recommendations:
                errors.append(f"Recommendation artifact contains {unknown_recommendations} unknown product ID(s).")
            if recommendations.filter(pl.col("already_owned") == True).height:
                errors.append("Recommendation artifact contains already-owned products.")

            backed = recommendations.filter(pl.col("propensity_score").is_not_null()).select(
                ["client_id", "product_id", "propensity_score"]
            )
            current_pairs = propensity.select(
                ["client_id", "product_id", "propensity_score"]
            ).rename({"propensity_score": "source_propensity_score"})
            provenance = backed.join(current_pairs, on=["client_id", "product_id"], how="left")
            unmatched = provenance.filter(pl.col("source_propensity_score").is_null()).heigh
            mismatched = provenance.filter(
                (pl.col("source_propensity_score").is_not_null())
                & ((pl.col("propensity_score") - pl.col("source_propensity_score")).abs() > 1e-12)
            ).heigh
            if unmatched:
                errors.append(f"{unmatched} propensity-backed recommendation pair(s) are absent from current propensity input.")
            if mismatched:
                errors.append(f"{mismatched} propensity-backed recommendation score(s) disagree with current propensity input.")
    except Exception as exc:
        errors.append(f"Canonical recommendation artifact checks failed: {exc}")

    if not paths.product_propensity_model_path.exists():
        errors.append(f"Missing trained product propensity model: {paths.product_propensity_model_path}")
    else:
        if paths.ownership_pairs_path.exists() and paths.product_propensity_model_path.stat().st_mtime < paths.ownership_pairs_path.stat().st_mtime:
            errors.append("Product propensity model is older than canonical ownership labels; refusing stale model usage.")
        try:
            with py_warnings.catch_warnings(record=True) as caught:
                py_warnings.simplefilter("always", InconsistentVersionWarning)
                model = joblib.load(paths.product_propensity_model_path)
        except Exception as exc:
            errors.append(f"Could not reload product propensity model: {exc}")
        else:
            for warning in caught:
                if issubclass(warning.category, InconsistentVersionWarning):
                    warnings.append(str(warning.message).splitlines()[0])
            if not hasattr(model, "predict_proba"):
                errors.append("Product propensity model does not expose predict_proba.")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check recommendation artifacts before service startup.")
    parser.parse_args()
    errors, warnings = check_readiness()

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)
    print("Recommendation readiness check passed.")


if __name__ == "__main__":
    main()
