"""Pooled synthetic product-ownership propensity pipeline.

Labels are simulated ownership-profile labels, not observed adoption,
conversion, or future acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_DIR / "archetype_data"
DATA_DIR = BASE_DIR / "outputs"
OUTPUT_DIR = BASE_DIR / "propensity" / "outputs" / "product_propensity"
OWNERSHIP_DIR = PROJECT_DIR / "artifacts" / "lookup_data" / "ownership"
PRODUCT_CATALOG_PATH = PROJECT_DIR / "artifacts" / "ontology" / "csv" / "products_verified.csv"
OWNERSHIP_PAIRS_PATH = OWNERSHIP_DIR / "product_ownership_pairs.parquet"
BEHAVIORAL_SEGMENT_BRIDGE_PATH = PROJECT_DIR / "artifacts" / "lookup_data" / "clustering" / "canonical_customer_behavioral_segments.csv"
CANONICAL_SEGMENTS = {"seg_ret_neg1", "seg_ret_0", "seg_ret_1", "seg_ret_2", "seg_ret_3", "seg_ret_4", "seg_ent_neg1", "seg_ent_0", "seg_ent_1", "seg_ent_2"}
LEAKAGE_COLUMNS = {"n_products_owned", "n_families_owned", "n_other_owned_products", "n_other_owned_families", "max_owned_to_candidate_affinity", "mean_owned_to_candidate_affinity", "n_linked_owned_products", "owns_other_same_family"}
LEAKAGE_PREFIXES = ("n_owned_", "owns_")
NON_FEATURE_COLUMNS = {
    "target", "client_id", "first_txn_date", "first_txn_date_latest", "product_name",
    "owned", "eligible", "ownership_source", "simulation_probability",
    "simulation_family", "reason",
}


@dataclass(frozen=True)
class ProductPropensityConfig:
    max_customers: int | None = 200_000
    negative_ratio: int = 4  # compatibility only; canonical labels are not downsampled
    random_state: int = 42
    validation_size: float = 0.20
    test_size: float = 0.20
    top_n: int = 5


def _require(frame: pl.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _catalog() -> pl.DataFrame:
    frame = pl.read_csv(PRODUCT_CATALOG_PATH)
    _require(frame, ["product_id", "product_name", "category", "classification", "type"], "verified product catalog")
    if frame.height != 79 or frame.get_column("product_id").n_unique() != 79:
        raise ValueError(f"Verified product catalog must contain exactly 79 unique products; found {frame.height} rows.")
    if frame.select(["product_id", "product_name"]).null_count().sum_horizontal().item() > 0:
        raise ValueError("Verified product catalog contains null identity values.")
    return frame.select(["product_id", "product_name", "category", "classification", "type"])


def load_archetype_sources(data_dir: Path = DATA_DIR) -> dict[str, pl.DataFrame]:
    """Load only canonical ownership labels, verified products, and customer data."""
    if not OWNERSHIP_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing canonical ownership labels: {OWNERSHIP_PAIRS_PATH}")
    ownership = pl.read_parquet(OWNERSHIP_PAIRS_PATH)
    _require(ownership, ["client_id", "product_id", "owned", "eligible"], "canonical ownership labels")
    catalog = _catalog()
    unknown = ownership.join(catalog.select("product_id"), on="product_id", how="anti")
    if unknown.height:
        raise ValueError(f"Canonical ownership labels contain {unknown.height} unknown product rows.")
    if ownership.select(["client_id", "product_id"]).is_duplicated().any():
        raise ValueError("Canonical ownership labels contain duplicate customer-product pairs.")
    customers = pl.read_csv(data_dir / "archetype_customers.csv")
    features = pl.read_csv(data_dir / "archetype_features.csv")
    if not BEHAVIORAL_SEGMENT_BRIDGE_PATH.exists():
        raise FileNotFoundError(f"Missing canonical behavioral-segment bridge: {BEHAVIORAL_SEGMENT_BRIDGE_PATH}")
    bridge = pl.read_csv(BEHAVIORAL_SEGMENT_BRIDGE_PATH)
    _require(bridge, ["client_id", "behavioral_segment"], "canonical behavioral-segment bridge")
    if bridge.select("client_id").is_duplicated().any():
        raise ValueError("Canonical behavioral-segment bridge contains duplicate client_id assignments.")
    unknown_segments = bridge.filter(~pl.col("behavioral_segment").is_in(sorted(CANONICAL_SEGMENTS)))
    if unknown_segments.height:
        raise ValueError(f"Unknown behavioral segment IDs in canonical bridge: {unknown_segments.get_column('behavioral_segment').unique().to_list()}")
    customers = customers.join(bridge.select(["client_id", "behavioral_segment"]), on="client_id", how="left", validate="1:1")
    if customers.filter(pl.col("behavioral_segment").is_null()).height:
        raise ValueError("Every propensity customer must resolve to a canonical behavioral segment.")
    _require(customers, ["client_id", "segment"], "archetype customers")
    return {"customers": customers, "features": features, "ownership": ownership, "catalog": catalog, "behavioral_bridge": bridge}


def latest_customer_features(features: pl.DataFrame) -> pl.DataFrame:
    _require(features, ["client_id", "year"], "customer features")
    return features.sort(["client_id", "year"]).unique(subset=["client_id"], keep="last", maintain_order=True).drop("year")


def _sample_customers(customers: pl.DataFrame, config: ProductPropensityConfig) -> list[int]:
    ids = customers.get_column("client_id").unique().to_numpy()
    if config.max_customers is None or config.max_customers <= 0 or len(ids) <= config.max_customers:
        return [int(x) for x in ids]
    rng = np.random.default_rng(config.random_state)
    return [int(x) for x in np.sort(rng.choice(ids, size=config.max_customers, replace=False))]


def _build_frame(sources: dict[str, pl.DataFrame], config: ProductPropensityConfig) -> pl.DataFrame:
    selected = _sample_customers(sources["customers"], config)
    customers = sources["customers"].filter(pl.col("client_id").is_in(selected))
    latest = latest_customer_features(sources["features"]).filter(pl.col("client_id").is_in(selected))
    labels = (sources["ownership"]
              .filter(pl.col("client_id").is_in(selected) & pl.col("eligible"))
              .select(["client_id", "product_id", "owned", "eligible"]))
    frame = (labels.with_columns(pl.col("owned").cast(pl.Int8).alias("target"))
             .join(sources["catalog"], on="product_id", how="left", validate="m:1")
             .join(customers, on="client_id", how="left", suffix="_customer")
             .join(latest, on="client_id", how="left", suffix="_latest"))
    if frame.filter(pl.col("product_id").is_null()).height:
        raise ValueError("Product catalog join produced null product IDs.")
    return frame


def _validate_frame(frame: pl.DataFrame, sources: dict[str, pl.DataFrame]) -> dict[str, Any]:
    expected = set(sources["catalog"].get_column("product_id").to_list())
    actual = set(frame.get_column("product_id").unique().to_list())
    if actual != expected:
        raise ValueError(f"Frame product universe mismatch: expected 79 IDs, found {len(actual)}.")
    if frame.filter(~pl.col("eligible")).height:
        raise ValueError("Ineligible rows are present in propensity frame.")
    if frame.select(["client_id", "product_id"]).is_duplicated().any():
        raise ValueError("Duplicate customer-product pairs are present in propensity frame.")
    return {"training_rows": frame.height, "positive_rows": frame.filter(pl.col("target") == 1).height, "negative_rows": frame.filter(pl.col("target") == 0).height, "unique_customers": frame.get_column("client_id").n_unique(), "unique_product_ids": len(actual), "removed_ids": 0, "unknown_ids": 0, "duplicate_customer_product_pairs": 0, "ineligible_rows": 0}


def build_training_frame(config: ProductPropensityConfig = ProductPropensityConfig(), data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR, output_name: str = "product_propensity_training_frame.parquet") -> pl.DataFrame:
    sources = load_archetype_sources(data_dir)
    frame = _build_frame(sources, config)
    _validate_frame(frame, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / output_name)
    return frame


def feature_columns(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    blocked = NON_FEATURE_COLUMNS | LEAKAGE_COLUMNS
    numeric_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Boolean}
    numeric, categorical = [], []
    for name, dtype in frame.schema.items():
        if name in blocked or name.startswith(LEAKAGE_PREFIXES):
            continue
        if dtype in numeric_types:
            numeric.append(name)
        elif dtype == pl.String and frame.select(pl.col(name).n_unique()).item() <= 200:
            categorical.append(name)
    return numeric, categorical


def split_by_customer(frame: pl.DataFrame, config: ProductPropensityConfig = ProductPropensityConfig()) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(config.random_state)
    ids = frame.get_column("client_id").unique().to_numpy().copy()
    rng.shuffle(ids)
    n_test, n_val = int(len(ids) * config.test_size), int(len(ids) * config.validation_size)
    test_ids, val_ids = ids[:n_test], ids[n_test:n_test + n_val]
    train_ids = np.concatenate([test_ids, val_ids])
    train = frame.filter(~pl.col("client_id").is_in(train_ids))
    val = frame.filter(pl.col("client_id").is_in(val_ids))
    test = frame.filter(pl.col("client_id").is_in(test_ids))
    sets = [set(part.get_column("client_id").to_list()) for part in (train, val, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Customer overlap detected across train/validation/test.")
    return train, val, tes


def _matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(columns).to_numpy()


def make_models(numeric: list[str], categorical: list[str], random_state: int = 42) -> dict[str, Pipeline]:
    ni, ci = list(range(len(numeric))), list(range(len(numeric), len(numeric) + len(categorical)))
    scaled = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ni), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ci)])
    tree = ColumnTransformer([("num", SimpleImputer(strategy="median"), ni), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ci)])
    return {"dummy": Pipeline([("prep", tree), ("model", DummyClassifier(strategy="prior"))]), "logistic_regression": Pipeline([("prep", scaled), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]), "random_forest": Pipeline([("prep", tree), ("model", RandomForestClassifier(n_estimators=120, min_samples_leaf=20, n_jobs=1, random_state=random_state, class_weight="balanced_subsample"))]), "hist_gradient_boosting": Pipeline([("prep", tree), ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, random_state=random_state))])}


def _lift(y: np.ndarray, p: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(y) * fraction)))
    base = float(np.mean(y))
    return float(np.mean(y[np.argsort(-p)[:n]]) / base) if base else float("nan")


def evaluate_predictions(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    return {"positive_rate": float(np.mean(y)), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan, "average_precision": float(average_precision_score(y, p)), "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)), "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "lift_at_5pct": _lift(y, p, .05), "lift_at_10pct": _lift(y, p, .10), "lift_at_20pct": _lift(y, p, .20)}


def prepare_product_propensity_inputs(config: ProductPropensityConfig = ProductPropensityConfig(), data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Build and validate frames only; this is the pre-training gate."""
    sources = load_archetype_sources(data_dir)
    frame = _build_frame(sources, config)
    checks = _validate_frame(frame, sources)
    source_stat = OWNERSHIP_PAIRS_PATH.stat()
    source_labels = sources["ownership"]
    checks.update({
        "behavioral_segment_source_path": str(BEHAVIORAL_SEGMENT_BRIDGE_PATH),
        "behavioral_segment_distribution": {str(k): int(v) for k, v in sources["behavioral_bridge"].get_column("behavioral_segment").value_counts().iter_rows()},
        "behavioral_segment_duplicate_client_ids": 0,
        "unknown_behavioral_segment_ids": 0,
        "seg_ret_5_count": 0,
        "seg_ret_6_count": 0,
        "unresolved_behavioral_segment_customers": 0,
        "ownership_source_path": str(OWNERSHIP_PAIRS_PATH),
        "ownership_source_modified_at_utc": datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc).isoformat(),
        "source_total_customer_product_pairs": source_labels.height,
        "source_owned_pairs": source_labels.filter(pl.col("owned")).height,
        "source_eligible_owned_pairs": source_labels.filter(pl.col("eligible") & pl.col("owned")).height,
        "source_eligible_non_owned_pairs": source_labels.filter(pl.col("eligible") & ~pl.col("owned")).height,
        "source_ineligible_pairs": source_labels.filter(~pl.col("eligible")).height,
    })
    scoring = frame.filter(pl.col("eligible") & ~pl.col("owned"))
    checks["owned_products_in_scoring_frame"] = scoring.filter(pl.col("owned")).heigh
    if checks["owned_products_in_scoring_frame"]:
        raise ValueError("Owned products entered propensity scoring candidates.")
    train, val, test = split_by_customer(frame, config)
    numeric, categorical = feature_columns(frame)
    features = numeric + categorical
    checks["train_validation_test_customer_overlap"] = 0
    checks["feature_columns"] = features
    checks["ownership_derived_features_absent"] = not any(c in frame.columns for c in LEAKAGE_COLUMNS)
    if not checks["ownership_derived_features_absent"]:
        raise ValueError("Ownership-derived features leaked into the primary propensity frame.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / "product_propensity_training_frame.parquet")
    scoring.write_parquet(output_dir / "product_propensity_scoring_frame.parquet")
    counts = (frame.group_by(["product_id", "product_name"])
              .agg([pl.len().alias("eligible_count"), pl.sum("target").alias("positive_count")])
              .with_columns((pl.col("eligible_count") - pl.col("positive_count")).alias("negative_count"), (pl.col("positive_count") / pl.col("eligible_count")).alias("positive_rate")))
    counts.write_csv(output_dir / "product_label_counts.csv")
    pl.DataFrame([checks]).write_json(output_dir / "product_propensity_input_validation.json")
    pl.DataFrame({"feature": features, "type": ["numeric"] * len(numeric) + ["categorical"] * len(categorical)}).write_csv(output_dir / "product_feature_list.csv")
    return {"training_frame": frame, "scoring_frame": scoring, "train": train, "validation": val, "test": test, "checks": checks}


def train_validate_compare(frame: pl.DataFrame, config: ProductPropensityConfig = ProductPropensityConfig(), output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    train, val, test = split_by_customer(frame, config)
    numeric, categorical = feature_columns(frame)
    features = numeric + categorical
    fitted, rows = {}, []
    for name, model in make_models(numeric, categorical, config.random_state).items():
        model.fit(_matrix(train, features), train["target"].to_numpy())
        fitted[name] = model
        for split_name, part in (("validation", val), ("test", test)):
            p = model.predict_proba(_matrix(part, features))[:, 1]
            rows.append({"model": name, "split": split_name, **evaluate_predictions(part["target"].to_numpy(), p)})
    comparison = pl.DataFrame(rows).sort(["split", "average_precision"], descending=[False, True])
    best = comparison.filter(pl.col("split") == "validation").sort("average_precision", descending=True).get_column("model")[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(output_dir / "product_model_comparison.csv")
    joblib.dump(fitted[best], output_dir / "best_product_propensity_model.joblib")
    best_test_probability = fitted[best].predict_proba(_matrix(test, features))[:, 1]
    holdout = test.with_columns(pl.Series("propensity_score", best_test_probability)).select(["client_id", "product_id", "product_name", "target", "propensity_score"])
    holdout.write_parquet(output_dir / "product_holdout_predictions.parquet")
    support = (holdout.group_by(["product_id", "product_name"])
               .agg([pl.len().alias("support"), pl.sum("target").alias("positive_count")])
               .with_columns((pl.col("support") - pl.col("positive_count")).alias("negative_count"), (pl.col("positive_count") / pl.col("support")).alias("positive_rate")))
    support.write_csv(output_dir / "product_holdout_support.csv")
    per_product = []
    for row in support.iter_rows(named=True):
        part = holdout.filter(pl.col("product_id") == row["product_id"])
        y, p = part["target"].to_numpy(), part["propensity_score"].to_numpy()
        metrics = evaluate_predictions(y, p) if len(np.unique(y)) == 2 else {"roc_auc": np.nan, "average_precision": np.nan}
        per_product.append({**row, **metrics, "standalone_interpretation": "flag_small_support" if row["positive_count"] < 30 else "descriptive"})
    pl.DataFrame(per_product).write_csv(output_dir / "product_holdout_metrics.csv")
    bins = np.minimum((best_test_probability * 10).astype(int), 9)
    calibration = pl.DataFrame({"bin": bins, "predicted_mean": best_test_probability, "observed_rate": test["target"].to_numpy()}).group_by("bin").agg([pl.mean("predicted_mean"), pl.mean("observed_rate"), pl.len().alias("count")]).sort("bin")
    calibration.write_csv(output_dir / "product_calibration.csv")
    pl.DataFrame([{"label_type": "synthetic product-ownership propensity", "interpretation": "Metrics measure recovery/generalization of the simulated ownership structure, not observed adoption or future product acquisition.", "best_model": best, "selection_metric": "validation average_precision", "random_state": config.random_state}]).write_json(output_dir / "product_propensity_metadata.json")
    return {"comparison": comparison, "best_model_name": best, "best_model": fitted[best], "numeric_features": numeric, "categorical_features": categorical, "train": train, "validation": val, "test": test}


def score_non_owned_products(model: Pipeline, config: ProductPropensityConfig = ProductPropensityConfig(), data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> pl.DataFrame:
    prepared = prepare_product_propensity_inputs(config, data_dir, output_dir)
    frame = prepared["scoring_frame"]
    features = feature_columns(prepared["training_frame"]); model_input = features[0] + features[1]
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and int(expected_features) != len(model_input):
        raise RuntimeError(
            "Propensity model feature contract mismatch: "
            f"model expects {expected_features}, prepared frame supplies {len(model_input)}. "
            "Use the model and product_feature_list generated by the same workflow."
        )
    scored = frame.with_columns(pl.Series("propensity_score", model.predict_proba(_matrix(frame, model_input))[:, 1]))
    full_scores = scored.select([c for c in ["client_id", "product_id", "product_name", "behavioral_segment", "category", "classification", "type", "eligible", "owned", "propensity_score"] if c in scored.columns])
    if full_scores.filter(pl.col("owned") | ~pl.col("eligible")).height:
        raise ValueError("Propensity scores contain owned or ineligible candidates.")
    if full_scores.select(["client_id", "product_id"]).is_duplicated().any():
        raise ValueError("Propensity scores contain duplicate customer-product pairs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_scores.write_parquet(output_dir / "product_propensity_scores.parquet")
    top = scored.sort(["client_id", "propensity_score"], descending=[False, True]).with_columns(pl.col("propensity_score").rank("ordinal", descending=True).over("client_id").alias("rank")).filter(pl.col("rank") <= config.top_n)
    top.select([c for c in ["client_id", "segment", "behavioral_segment", "latent_archetype", "product_id", "product_name", "category", "classification", "type", "propensity_score", "rank"] if c in top.columns]).write_parquet(output_dir / "top_non_owned_product_recommendations.parquet")
    return top


def run_product_propensity_workflow(config: ProductPropensityConfig = ProductPropensityConfig(), data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    prepared = prepare_product_propensity_inputs(config, data_dir, output_dir)
    results = train_validate_compare(prepared["training_frame"], config, output_dir)
    results["top_recommendations"] = score_non_owned_products(results["best_model"], config, data_dir, output_dir)
    return {**prepared, **results}
