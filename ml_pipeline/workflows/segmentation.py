from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from ml_pipeline.paths import (
    ARTIFACTS_ROOT,
    OWNERSHIP_FEATURES,
    OWNERSHIP_LONG,
    PRODUCTS,
    SEGMENTS,
    STATIC_CLUSTERING_OUTPUTS,
)
from ml_pipeline.runtime import copy_if_exists, ensure_exists, promote_product_ownership_artifacts, with_db_transaction


PRODUCT_CATEGORICAL_FEATURES = [
    "owns_investment",
    "owns_pack",
    "owns_business_support",
    "owns_digital_niche",
    "owns_credit",
    "owns_insurance_savings",
    "owns_premium_card",
    "owns_transfer",
    "owns_study_admin",
    "owns_digital_core",
    "owns_assistance",
]

BASE_STATIC_FEATURES = [
    "Monetary",
    "Frequency",
    "avg_balance",
    "net_cash_flow",
    "channel_entropy",
    "balance_volatility",
]

CLUSTER_ENRICHMENT_NUMERIC_FEATURES = [
    "positive_cashflow_month_share",
    "deposit_active_month_share",
    "digital_tx_pct",
    "atm_tx_pct",
    "deposit_cv",
    "low_balance_month_share",
    "n_families_owned",
]

FEATURES = BASE_STATIC_FEATURES + CLUSTER_ENRICHMENT_NUMERIC_FEATURES
LOG_FEATURES = [
    "Monetary",
    "avg_balance",
    "balance_volatility",
    "channel_entropy",
    "digital_tx_pct",
    "atm_tx_pct",
    "low_balance_month_share",
    "n_families_owned",
]

CORR_THRESHOLD = 0.85
MIN_TARGET_CLUSTERS = 6
DATASET_NAME = "archetype_static"

ETA_SQUARED = {
    "avg_balance": 0.9469,
    "max_balance": 0.9471,
    "min_balance": 0.9466,
    "Monetary": 0.3452,
    "channel_entropy": 0.2893,
    "total_deposit": 0.2646,
    "total_withdrawal": 0.2161,
    "net_cash_flow": 0.1459,
    "balance_volatility": 0.1450,
    "balance_growth": 0.1358,
    "Frequency": 0.1133,
    "withdrawal_deposit_ratio": 0.0359,
    "tenure_days": 0.0133,
    "Recency": 0.0010,
    "txn_count": 0.0000,
    "positive_cashflow_month_share": 0.0500,
    "deposit_active_month_share": 0.0500,
    "digital_tx_pct": 0.0500,
    "atm_tx_pct": 0.0500,
    "deposit_cv": 0.0500,
    "low_balance_month_share": 0.0500,
    "n_families_owned": 0.0500,
}


def load_enriched_rfm(path: Path | None = None) -> pd.DataFrame:
    path = path or STATIC_CLUSTERING_OUTPUTS.parents[1] / "rfm.csv"
    ensure_exists(path)
    df = pd.read_csv(path)
    if "customer_class" not in df.columns:
        df["customer_class"] = df["segment"].replace(
            {"PME": "Entreprise", "TPE": "Entreprise", "Grande_Entreprise": "Entreprise"}
        )
        df["customer_class"] = df["customer_class"].where(df["customer_class"] == "Entreprise", "Retail")
    missing = [col for col in ["client_id", "segment", "customer_class", *FEATURES] if col not in df.columns]
    if missing:
        raise ValueError(f"Static clustering RFM is missing required column(s): {missing}")
    return df


def deduplicate_features(
    df: pd.DataFrame,
    features: list[str],
    eta_squared_map: dict[str, float] | None = None,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list[str], dict[str, str]]:
    X = df[features].copy()
    dropped: dict[str, str] = {}
    to_drop: set[str] = set()
    for i, left in enumerate(features):
        if left in to_drop:
            continue
        for right in features[i + 1 :]:
            if right in to_drop:
                continue
            corr, _ = spearmanr(X[left], X[right], nan_policy="omit")
            if not np.isfinite(corr) or abs(corr) < threshold:
                continue
            if eta_squared_map and eta_squared_map.get(right, 0) > eta_squared_map.get(left, 0):
                to_drop.add(left)
                dropped[left] = righ
                break
            to_drop.add(right)
            dropped[right] = lef
    return [feature for feature in features if feature not in to_drop], dropped


def split_outliers(df: pd.DataFrame, features: list[str], quantile: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["outlier_score"] = work[features].rank(pct=True).mean(axis=1)
    threshold = work["outlier_score"].quantile(quantile)
    outliers = work[work["outlier_score"] >= threshold].copy()
    normal = work[work["outlier_score"] < threshold].copy()
    outliers["cluster"] = -1
    return normal, outliers


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in LOG_FEATURES:
        if col not in work.columns:
            continue
        shift = max(0.0, -float(work[col].min()) + 1e-6)
        work[col] = np.log1p(work[col] + shift)
    return work


def safe_scale(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, RobustScaler, list[str]]:
    X = df[features].copy()
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    scaler = RobustScaler()
    return scaler.fit_transform(X), scaler, list(X.columns)


def apply_pca(
    X_scaled: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 2,
) -> tuple[np.ndarray, PCA, pd.DataFrame]:
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = max(min_components, int(np.searchsorted(cumulative, variance_threshold) + 1))
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return X_pca, pca, variance


def create_model_sample(X: np.ndarray, max_samples: int = 50_000) -> np.ndarray:
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        return X[idx]
    return X


def run_gmm(X: np.ndarray, k: int) -> tuple[np.ndarray, GaussianMixture]:
    model = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        reg_covar=1e-3,
        n_init=5,
        max_iter=300,
        random_state=42,
    )
    labels = model.fit_predict(X.astype(np.float64))
    return labels, model


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict[str, Any] | None:
    labels = np.asarray(labels)
    mask = labels != -1
    X_clean = X[mask]
    y_clean = labels[mask]
    if len(set(y_clean)) < 2 or len(X_clean) < 10:
        return None
    counts = pd.Series(y_clean).value_counts()
    min_cluster_size = int(counts.min())
    return {
        "silhouette": silhouette_score(X_clean, y_clean, sample_size=min(10_000, len(X_clean))),
        "davies_bouldin": davies_bouldin_score(X_clean, y_clean),
        "calinski_harabasz": calinski_harabasz_score(X_clean, y_clean),
        "n_clusters": len(set(y_clean)),
        "noise_ratio": float(np.mean(labels == -1)),
        "min_cluster_size": min_cluster_size,
        "min_cluster_frac": float(min_cluster_size / len(y_clean)),
    }


def compute_normalized_score(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()

    def norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else pd.Series(np.ones(len(series)), index=series.index)

    df["sil_n"] = norm(df["silhouette"])
    df["db_n"] = norm(-df["davies_bouldin"])
    df["ch_n"] = norm(np.log1p(df["calinski_harabasz"]))
    df["noise_n"] = 1 - df["noise_ratio"]
    df["cluster_count_n"] = np.minimum(df["n_clusters"], MIN_TARGET_CLUSTERS) / MIN_TARGET_CLUSTERS
    df["too_few_cluster_penalty"] = np.where(df["n_clusters"] < MIN_TARGET_CLUSTERS, 0.25, 0.0)
    df["tiny_cluster_penalty"] = np.where(df["min_cluster_frac"] < 0.01, 0.35, 0.0)
    df["composite_score"] = (
        0.30 * df["sil_n"]
        + 0.20 * df["db_n"]
        + 0.20 * df["ch_n"]
        + 0.20 * df["cluster_count_n"]
        + 0.10 * df["noise_n"]
        - df["too_few_cluster_penalty"]
        - df["tiny_cluster_penalty"]
    )
    return df


def run_model_search(X: np.ndarray, k_values: tuple[int, ...] = (3, 4, 5, 6, 7, 8)) -> pd.DataFrame:
    X_sample = create_model_sample(X)
    rows: list[dict[str, Any]] = []
    for k in k_values:
        labels, model = run_gmm(X_sample, k)
        metrics = evaluate_clustering(X_sample, labels)
        if metrics is None:
            continue
        metrics.update({"model": "GMM", "k": k, "bic": model.bic(X_sample)})
        rows.append(metrics)
    if not rows:
        raise RuntimeError("No valid static clustering configurations found.")
    return compute_normalized_score(pd.DataFrame(rows))


def cluster_profile(df: pd.DataFrame, features: list[str], cluster_col: str = "cluster") -> pd.DataFrame:
    return df.groupby(cluster_col)[features].agg(["mean", "median", "std", "min", "max"])


def categorical_cluster_profile(
    df: pd.DataFrame,
    categorical_features: list[str],
    cluster_col: str = "cluster",
    normalize: bool = True,
) -> pd.DataFrame:
    rows = []
    work = df[df[cluster_col].notna()].copy()
    for feature in categorical_features:
        if feature not in work.columns:
            continue
        table = pd.crosstab(work[cluster_col], work[feature])
        if normalize:
            table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
        table = table.add_prefix(f"{feature}=").reset_index()
        table.insert(0, "feature", feature)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cramers_v_from_table(table: pd.DataFrame) -> tuple[float, float, int, float]:
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return chi2, p, dof, np.nan
    rows, cols = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    rows_corr = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    cols_corr = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denom = min(cols_corr - 1, rows_corr - 1)
    return chi2, p, dof, np.sqrt(phi2_corr / denom) if denom > 0 else np.nan


def full_categorical_stat_report(
    df: pd.DataFrame,
    categorical_features: list[str],
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    rows = []
    for feature in categorical_features:
        if feature not in df.columns:
            continue
        work = df[[cluster_col, feature]].dropna()
        work = work[work[cluster_col] != -1]
        if work[cluster_col].nunique() < 2 or work[feature].nunique() < 2:
            rows.append({"feature": feature, "test": "skip", "p_value": np.nan, "cramers_v": np.nan})
            continue
        table = pd.crosstab(work[cluster_col], work[feature])
        chi2, p, dof, expected = chi2_contingency(table)
        _, _, _, v = cramers_v_from_table(table)
        rows.append(
            {
                "feature": feature,
                "test": "chi_square",
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": v,
                "min_expected_count": float(expected.min()),
            }
        )
    return pd.DataFrame(rows)


def full_stat_report(df: pd.DataFrame, features: list[str], cluster_col: str = "cluster") -> pd.DataFrame:
    rows = []
    for feature in features:
        groups = [g[feature].dropna() for _, g in df.groupby(cluster_col) if g[feature].notna().sum() > 1]
        if len(groups) < 2:
            rows.append({"feature": feature, "test": "skip", "stat": np.nan, "p_value": np.nan})
            continue
        if abs(df[feature].skew()) > 1:
            stat, p = kruskal(*groups)
            test_name = "kruskal"
        else:
            stat, p = f_oneway(*groups)
            test_name = "anova"
        rows.append({"feature": feature, "test": test_name, "stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def clustering_pipeline(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    variance_threshold: float = 0.95,
    k_values: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    normal, outliers = split_outliers(df, features)
    normal = transform_features(normal)
    kept_features, dropped = deduplicate_features(normal, features, eta_squared_map=ETA_SQUARED)
    X_scaled, scaler, used_features = safe_scale(normal, kept_features)
    X_pca, pca, variance_df = apply_pca(X_scaled.astype(np.float64), variance_threshold=variance_threshold)
    results = run_model_search(X_pca.astype(np.float64), k_values=k_values)
    best_row = results.sort_values("composite_score", ascending=False).iloc[0]
    labels, final_model = run_gmm(X_pca, int(best_row["k"]))
    normal = normal.copy()
    normal["cluster"] = labels
    outliers = outliers.copy()
    outliers["cluster"] = -1
    final_df = pd.concat([normal, outliers], ignore_index=True)
    meta = {
        "model": str(best_row["model"]),
        "k": int(best_row["k"]),
        "pca_components": int(X_pca.shape[1]),
        "features_used": used_features,
        "features_dropped": dropped,
        "pca": pca,
        "scaler": scaler,
        "model_object": final_model,
        "pca_variance": variance_df,
    }
    return final_df, results, meta, outliers


def run_static_segmentation(output_dir: Path = STATIC_CLUSTERING_OUTPUTS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rfm = load_enriched_rfm()
    outputs: dict[str, Any] = {}
    for label, subset in [
        ("retail", rfm[rfm["customer_class"] == "Retail"].copy()),
        ("entreprise", rfm[rfm["customer_class"] == "Entreprise"].copy()),
    ]:
        if subset.empty:
            raise ValueError(f"No rows found for static clustering subset {label!r}.")
        clustered, results, meta, _ = clustering_pipeline(subset, FEATURES)
        profile = cluster_profile(clustered, meta["features_used"])
        stats = full_stat_report(clustered, meta["features_used"])
        categorical_stats = full_categorical_stat_report(clustered, PRODUCT_CATEGORICAL_FEATURES)
        categorical_profile = categorical_cluster_profile(clustered, PRODUCT_CATEGORICAL_FEATURES)
        clustered.to_csv(output_dir / f"{label}_clusters.csv", index=False)
        results.to_csv(output_dir / f"{label}_model_results.csv", index=False)
        profile.to_csv(output_dir / f"{label}_cluster_profiles.csv")
        stats.to_csv(output_dir / f"{label}_stat_tests.csv", index=False)
        categorical_stats.to_csv(output_dir / f"{label}_categorical_stat_tests.csv", index=False)
        categorical_profile.to_csv(output_dir / f"{label}_categorical_cluster_profiles.csv", index=False)
        joblib.dump(meta, output_dir / f"{label}_meta.pkl")
        outputs[label] = {
            "rows": int(len(clustered)),
            "model": meta["model"],
            "k": meta["k"],
            "clusters": {str(k): int(v) for k, v in clustered["cluster"].value_counts().sort_index().items()},
        }
    combined = pd.concat(
        [
            pd.read_csv(output_dir / "retail_clusters.csv"),
            pd.read_csv(output_dir / "entreprise_clusters.csv"),
        ],
        ignore_index=True,
    )
    combined.to_csv(output_dir / "static_segment_assignments.csv", index=False)
    outputs["static_segment_assignments"] = int(len(combined))
    return outputs


def promote_static_artifacts() -> None:
    clustering_lookup_dir = ARTIFACTS_ROOT / "lookup_data" / "clustering"
    segmentation_dir = ARTIFACTS_ROOT / "segmentation"
    model_dir = ARTIFACTS_ROOT / "models" / "segmentation"
    for name in [
        "retail_clusters.csv",
        "entreprise_clusters.csv",
        "static_segment_assignments.csv",
    ]:
        copy_if_exists(STATIC_CLUSTERING_OUTPUTS / name, clustering_lookup_dir / name)
        copy_if_exists(STATIC_CLUSTERING_OUTPUTS / name, segmentation_dir / name)
    for name in [
        "retail_model_results.csv",
        "entreprise_model_results.csv",
        "retail_cluster_profiles.csv",
        "entreprise_cluster_profiles.csv",
        "retail_stat_tests.csv",
        "entreprise_stat_tests.csv",
        "retail_categorical_stat_tests.csv",
        "entreprise_categorical_stat_tests.csv",
        "retail_categorical_cluster_profiles.csv",
        "entreprise_categorical_cluster_profiles.csv",
    ]:
        copy_if_exists(STATIC_CLUSTERING_OUTPUTS / name, segmentation_dir / name)
    for name in ["retail_meta.pkl", "entreprise_meta.pkl"]:
        copy_if_exists(STATIC_CLUSTERING_OUTPUTS / name, model_dir / name)
        copy_if_exists(STATIC_CLUSTERING_OUTPUTS / name, segmentation_dir / name)


def run(skip_schema: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    segmentation_outputs = run_static_segmentation()
    promote_static_artifacts()
    promote_product_ownership_artifacts()
    customer_source = ARTIFACTS_ROOT / "lookup_data" / "clustering" / "static_segment_assignments.csv"
    for path in [PRODUCTS, SEGMENTS, customer_source, OWNERSHIP_LONG]:
        ensure_exists(path)

    def work(db, conn):
        counts = {
            "products": db.load_products(conn, PRODUCTS),
            "customer_segments": db.load_segments(conn, SEGMENTS),
            "customers": db.load_customers(conn, customer_source),
            "output_only_products": db.load_output_products(conn, [OWNERSHIP_LONG]),
        }
        run_id = db.upsert_model_run(
            conn,
            "latest_static_behavioral_segmentation",
            "segmentation",
            customer_source,
            {"workflow": "static_clustering", "outputs": segmentation_outputs},
        )
        ownership_run_id = db.upsert_model_run(conn, "latest_product_ownership", "product_ownership", OWNERSHIP_LONG)
        counts["segmentation_run_id"] = run_id
        counts["customer_products"] = db.load_customer_products(conn, OWNERSHIP_LONG, ownership_run_id)
        return counts

    counts = with_db_transaction(work, skip_schema)
    counts["segmentation_workflow"] = "static_clustering"
    counts["segmentation_outputs"] = segmentation_outputs
    counts["duration_seconds"] = round(time.perf_counter() - start, 3)
    return counts
