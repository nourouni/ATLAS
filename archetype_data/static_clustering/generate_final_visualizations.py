"""Generate presentation-ready plots from the frozen final assignments.

This script reuses the plotting calculations from
``archetype_clustering_initial_method_reproduction.ipynb``:

* median cluster profiles;
* per-feature 0-1 radar normalization;
* z-score heatmaps with raw median annotations; and
* PCA projection through the saved final scaler/PCA metadata.

It never fits a model, selects K, remaps labels, or writes outside
``final_visualizations``.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "archetype_data" / "static_clustering"
OUT = BASE / "final_visualizations"
REPORT_RADAR_OUT = ROOT / "report" / "figures" / "segmentation" / "clusters"
RETAIL_INPUT = BASE / "retail_clustered.csv"
ENTERPRISE_INPUT = BASE / "enterprise_clustered.csv"

EXPECTED = {"Retail": {-1, 0, 1, 2, 3, 4}, "Enterprise": {-1, 0, 1, 2}}
REGULAR = {"Retail": [0, 1, 2, 3, 4], "Enterprise": [0, 1, 2]}
COLORS = {-1: "#7f8791", 0: "#4C72B0", 1: "#2ca02c", 2: "#d6531a", 3: "#9467bd", 4: "#8c564b"}
NAMES = {
    "Retail": {
        -1: "Exceptional Multi-Channel Retail Activity",
        0: "Balanced Everyday Users",
        1: "Established Multi-Channel Customers",
        2: "High-Value Digital Customers",
        3: "Digital Frequent Users",
        4: "ATM-Focused Light Users",
    },
    "Enterprise": {
        -1: "Exceptional Enterprise Activity",
        0: "High-Value Digital Enterprises",
        1: "Mixed-Channel Smaller Enterprises",
        2: "Branch-Oriented Core Enterprises",
    },
}

FEATURES = [
    "Monetary",
    "Frequency",
    "avg_balance",
    "total_deposit",
    "total_withdrawal",
    "net_cash_flow",
    "channel_entropy",
    "balance_volatility",
]
DISPLAY = {
    "Monetary": "Monetary",
    "Frequency": "Frequency",
    "avg_balance": "Avg Balance",
    "total_deposit": "Total Deposit",
    "total_withdrawal": "Total Withdrawal",
    "net_cash_flow": "Net Cash Flow",
    "channel_entropy": "Channel Entropy",
    "balance_volatility": "Balance Volatility",
}
RADAR_DISPLAY = {
    "Monetary": "Monetary",
    "Frequency": "Frequency",
    "avg_balance": "Avg Balance",
    "total_deposit": "Total\nDeposit",
    "total_withdrawal": "Total\nWithdrawal",
    "net_cash_flow": "Net Cash\nFlow",
    "channel_entropy": "Channel\nEntropy",
    "balance_volatility": "Balance\nVolatility",
}
LOG_FEATURES = set(FEATURES)
PALETTE = "RdBu_r"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f6",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def validate_assignments(df: pd.DataFrame, population: str) -> None:
    if "cluster" not in df.columns:
        raise ValueError(f"{population} assignment file has no 'cluster' column")
    found = set(pd.to_numeric(df["cluster"], errors="raise").astype(int).unique())
    if found != EXPECTED[population]:
        raise ValueError(
            f"{population} clusters are {sorted(found)}, expected {sorted(EXPECTED[population])}. "
            "Refusing to generate figures."
        )
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{population} is missing profile features: {missing}")


def label(population: str, cluster: int) -> str:
    return f"Cluster {cluster}: {NAMES[population][cluster]}"


def save_figure(fig: plt.Figure, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    transparent = path.with_name(path.stem + "_transparent.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(transparent, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return [str(path.relative_to(OUT)), str(transparent.relative_to(OUT))]


def profile(df: pd.DataFrame, include_outlier: bool = False) -> pd.DataFrame:
    work = df if include_outlier else df[df.cluster != -1]
    return work.groupby("cluster")[FEATURES].median().sort_index()


def plot_distribution(df: pd.DataFrame, population: str, files: list[str]) -> None:
    counts = df.cluster.value_counts().reindex([-1] + REGULAR[population]).dropna().astype(int)
    shares = counts / len(df) * 100
    labels = [f"{c}\n{NAMES[population][c]}" for c in counts.index]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, counts.values, color=[COLORS[int(c)] for c in counts.index], width=0.72)
    for bar, count, share in zip(bars, counts, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}\n({share:.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{population} segment population", fontweight="bold")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    files += save_figure(fig, OUT / population.lower() / f"{population.lower()}_cluster_distribution.png")


def plot_profile_heatmap(df: pd.DataFrame, population: str, files: list[str]) -> None:
    raw = profile(df)
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    raw.index = [f"{c}: {NAMES[population][int(c)]}" for c in raw.index]
    z.index = raw.index
    z.columns = [DISPLAY[f] for f in FEATURES]
    fig, ax = plt.subplots(figsize=(15, max(4.5, len(raw) * 0.9)))
    sns.heatmap(z, ax=ax, cmap=PALETTE, center=0, vmin=-2, vmax=2,
                annot=raw.round(1), fmt="g", linewidths=.5, linecolor="white",
                cbar_kws={"label": "Z-score of cluster median"}, annot_kws={"size": 8})
    ax.set_title(f"{population} cluster profile heatmap\nColor = z-score; annotation = raw median", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Regular clusters only; outlier -1 excluded")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    files += save_figure(fig, OUT / population.lower() / f"{population.lower()}_cluster_profile_heatmap.png")


def plot_radar(df: pd.DataFrame, population: str, files: list[str]) -> None:
    raw = profile(df)
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    for cluster_id in REGULAR[population]:
        values = norm.loc[cluster_id].tolist()
        values += values[:1]
        fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
        color = COLORS[cluster_id]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.plot(angles, values, color=color, linewidth=3.0)
        ax.fill(angles, values, color=color, alpha=.20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([RADAR_DISPLAY[f] for f in FEATURES], fontsize=22)
        ax.tick_params(axis="x", pad=8)
        ax.set_yticklabels([])
        ax.grid(color="#c7c7c7", linewidth=0.8, alpha=0.8)
        for angle, value in zip(angles[:-1], values[:-1]):
            radius = min(max(value + 0.055, 0.065), 0.965)
            ax.text(angle, radius, f"{value:.2f}", ha="center", va="center",
                    fontsize=15, color=color)
        # The composite figure supplies the segment and cluster captions;
        # removing the redundant PNG title gives the radar and its labels more room.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        artifact = OUT / population.lower() / f"{population.lower()}_cluster_{cluster_id}_radar.png"
        report_asset = REPORT_RADAR_OUT / artifact.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        report_asset.parent.mkdir(parents=True, exist_ok=True)
        for target in (artifact, report_asset):
            fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
        transparent = artifact.with_name(artifact.stem + "_transparent.png")
        fig.savefig(transparent, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        files += [str(artifact.relative_to(OUT)), str(transparent.relative_to(OUT))]


def plot_outlier(df: pd.DataFrame, population: str, files: list[str]) -> None:
    outlier = df[df.cluster == -1]
    regular = df[df.cluster != -1]
    raw = pd.concat([regular, outlier]).groupby("cluster")[FEATURES].median().sort_index()
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist(); angles += angles[:1]
    values = norm.loc[-1].tolist(); values += values[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(0, 1)
    ax.plot(angles, values, color=COLORS[-1], linewidth=2.4); ax.fill(angles, values, color=COLORS[-1], alpha=.2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([DISPLAY[f] for f in FEATURES], fontsize=8); ax.set_yticklabels([])
    ax.set_title(f"{population} — OUTLIER / DIAGNOSTIC (Cluster -1)\nNormalized median profile", fontsize=12, fontweight="bold", pad=25)
    fig.tight_layout()
    files += save_figure(fig, OUT / population.lower() / f"{population.lower()}_outlier_neg1_profile.png")


def transformed_for_pca(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = df[features].copy().replace([np.inf, -np.inf], np.nan)
    work = work.fillna(work.median(numeric_only=True))
    for feature in LOG_FEATURES.intersection(features):
        shift = max(0, -work[feature].min() + 1e-6)
        work[feature] = np.log1p(work[feature] + shift)
    return work


def plot_pca(df: pd.DataFrame, population: str, files: list[str]) -> None:
    meta_path = BASE / ("retail_meta.joblib" if population == "Retail" else "enterprise_meta.joblib")
    meta = joblib.load(meta_path)
    normal = df[df.cluster != -1].copy()
    if len(normal) > 20000:
        normal = normal.sample(20000, random_state=42)
    used = [f for f in meta["features_used"] if f in FEATURES]
    X = transformed_for_pca(normal, used)
    coords = meta["pca"].transform(meta["scaler"].transform(X))
    pairs = [(0, 1), (0, 2 if coords.shape[1] > 2 else 1), (1, 2 if coords.shape[1] > 2 else 0)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x, y) in zip(axes, pairs):
        for cluster_id in REGULAR[population]:
            mask = normal.cluster.to_numpy() == cluster_id
            ax.scatter(coords[mask, x], coords[mask, y], s=5, alpha=.32, rasterized=True,
                       color=COLORS[cluster_id], label=f"{cluster_id}: {NAMES[population][cluster_id]}")
        ax.set_xlabel(f"PC{x + 1} ({meta['pca'].explained_variance_ratio_[x]:.1%} var)")
        ax.set_ylabel(f"PC{y + 1} ({meta['pca'].explained_variance_ratio_[y]:.1%} var)")
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"{population} final clusters in saved PCA space\nDeterministic sample; projection only", fontweight="bold")
    fig.tight_layout()
    files += save_figure(fig, OUT / population.lower() / f"{population.lower()}_pca_clusters.png")


def plot_comparison(retail: pd.DataFrame, enterprise: pd.DataFrame, files: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, df, population in zip(axes, [retail, enterprise], ["Retail", "Enterprise"]):
        counts = df.cluster.value_counts().reindex([-1] + REGULAR[population]).dropna()
        ax.bar([str(int(c)) for c in counts.index], counts.values, color=[COLORS[int(c)] for c in counts.index])
        ax.set_title(population, fontweight="bold")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Customers")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x):,}")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Final segmentation population overview", fontweight="bold")
    fig.tight_layout()
    files += save_figure(fig, OUT / "comparison" / "final_population_overview.png")


def write_readme(files: list[str]) -> None:
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    text = f"""# Final segmentation visualizations

Generated: `{generated}`

## Inputs

- Retail: `../retail_clustered.csv`
- Enterprise: `../enterprise_clustered.csv`
- Saved projection metadata: `../retail_meta.joblib`, `../enterprise_meta.joblib`

The script validates Retail IDs `[-1, 0, 1, 2, 3, 4]` and Enterprise IDs `[-1, 0, 1, 2]` before writing. It stops on any mismatch and never remaps labels.

## Method and features

The plotting calculations are reused from `archetype_clustering_initial_method_reproduction.ipynb`: median profiles, 0–1 normalization for radar charts, and feature-wise z-scores for heatmaps. Heatmap colors are z-scores of cluster medians; annotations are raw medians. Radar values are normalized medians within each population. Features: {', '.join(FEATURES)}.

Main heatmaps and regular-cluster radars exclude `-1`, because it is an outlier/residual group rather than a stable archetype. Separate `*_outlier_neg1_profile.png` diagnostics are generated. PCA figures transform data using the saved final scaler/PCA objects; PCA is not refit.

## Final mappings

Retail: `-1` Exceptional Multi-Channel Retail Activity; `0` Balanced Everyday Users; `1` Established Multi-Channel Customers; `2` High-Value Digital Customers; `3` Digital Frequent Users; `4` ATM-Focused Light Users.

Enterprise: `-1` Exceptional Enterprise Activity; `0` High-Value Digital Enterprises; `1` Mixed-Channel Smaller Enterprises; `2` Branch-Oriented Core Enterprises.

## Regenerate

```powershell
cd "{ROOT}"
python archetype_data/static_clustering/generate_final_visualizations.py
```

Generated files: {len(files)} PNGs plus this README.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    configure_style()
    retail = pd.read_csv(RETAIL_INPUT)
    enterprise = pd.read_csv(ENTERPRISE_INPUT)
    validate_assignments(retail, "Retail")
    validate_assignments(enterprise, "Enterprise")
    OUT.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for df, population in [(retail, "Retail"), (enterprise, "Enterprise")]:
        plot_distribution(df, population, files)
        plot_profile_heatmap(df, population, files)
        plot_radar(df, population, files)
        plot_outlier(df, population, files)
        plot_pca(df, population, files)
    plot_comparison(retail, enterprise, files)
    write_readme(files)
    forbidden = [p for p in OUT.rglob("*") if p.is_file() and any(
        token in p.name.lower() for token in ("enterprise_cluster_3", "enterprise_cluster_4")
    )]
    if forbidden:
        raise RuntimeError(f"Forbidden Enterprise radar filenames found: {forbidden}")
    retail_clusters = sorted(int(value) for value in retail.cluster.astype(int).unique())
    enterprise_clusters = sorted(int(value) for value in enterprise.cluster.astype(int).unique())
    print(f"Retail clusters found: {retail_clusters}")
    print(f"Enterprise clusters found: {enterprise_clusters}")
    print("Generated visualizations:")
    for path in sorted(files):
        print(f" - {path}")
    print(f"\nRegenerate with: C:\\Users\\author\\anaconda3\\python.exe archetype_data\\static_clustering\\generate_final_visualizations.py")


if __name__ == "__main__":
    main()
