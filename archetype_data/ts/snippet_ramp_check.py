import matplotlib.pyplot as pl

def infer_cluster_col(dataset_name):
    """Infer the correct cluster column name based on dataset name."""
    if "temporal" in dataset_name:
        return "temporal_feature_cluster_ns"
    if "static" in dataset_name:
        return "static_cluster_ns"
    if "final" in dataset_name:
        return "final_cluster_ns"
    return "segment_cluster_ns"

def plot_series_raw(datasets, dataset_name, feature, cluster, highlight_first_n=4):
    """
    Quick visual check for one (dataset, feature, cluster) series.
    Highlights the first `highlight_first_n` points in red so you can see
    at a glance whether they're a smooth ramp (real trend) or a discontinuity
    (genuine data issue) relative to the rest of the series.
    """
    df = datasets[dataset_name]
    col = infer_cluster_col(dataset_name)
    ts = df[df[col] == cluster].sort_values("date").set_index("date")[feature]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(ts.index, ts.values, color="#3b6ea5", marker="o", markersize=4, linewidth=1.2)
    ax.scatter(ts.index[:highlight_first_n], ts.values[:highlight_first_n],
               color="#d6604d", zorder=5, s=50, label=f"first {highlight_first_n} points")
    ax.set_title(f"{dataset_name} / {feature} / {cluster}  (n={len(ts)})")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return ts


def find_and_plot_early_outliers(datasets, max_plots=12):
    """
    Scans every (dataset, feature, cluster) series, flags ones with an
    IQR outlier in the first 4 points (same check data_prep.py uses),
    and plots each one so you can eyeball the pattern directly withou
    needing to manually identify which cluster matches which warning.
    """
    import numpy as np
    flagged = []

    for name, df in datasets.items():
        col = infer_cluster_col(name)
        for cluster, grp in df.groupby(col):
            grp = grp.sort_values("date")
            for feature in ["frequency", "monetary", "avg_balance"]:
                s = grp[feature].values.astype(float)
                if len(s) < 8:
                    continue
                q1, q3 = np.percentile(s, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower, upper = q1 - 3*iqr, q3 + 3*iqr
                early_flagged = [i for i in range(min(4, len(s))) if s[i] < lower or s[i] > upper]
                if early_flagged:
                    flagged.append((name, feature, cluster))

    print(f"Found {len(flagged)} series with early-point outliers.")
    for name, feature, cluster in flagged[:max_plots]:
        plot_series_raw(datasets, name, feature, cluster)
    return flagged
