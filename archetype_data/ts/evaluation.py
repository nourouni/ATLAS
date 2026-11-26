"""
Evaluation utilities for the case-driven forecasting pipeline.

Deliberately scoped to what's actually informative at n=36:
  - Actual-vs-predicted TIME SERIES OVERLAY per series (not a scatter AvP —
    too few points at this sample size for a scatter to mean anything;
    an overlay shows whether the model tracks shape, lags, or flatlines)
  - Residuals plotted over time (not vs. fitted, not as a histogram —
    3 backtest residuals per series is not enough for a distributional plot)
  - A cross-case leaderboard: backtest MAE vs persistence MAE, so you can
    see which case/model combos are actually earning their complexity

No Q-Q plots, no residual histograms, no scatter AvP — all of those need
more backtest points than this sample size can honestly provide, and will
just show visual noise dressed up as a "standard" diagnostic.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.dates as mdates

from forecast_dispatch import (
    walk_forward_mae,
    fit_naive_persistence,
    CASE_FITTERS,
    GATED_CASE_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Per-series backtest trace: actual vs predicted at each walk-forward origin
# (separate from fit_model's single aggregate MAE — this keeps the per-origin
# detail needed for plotting, rather than just the mean)
# ---------------------------------------------------------------------------

def backtest_trace(series, fit_fn, n_origins=3, h=1, min_train=52):
    """
    Same walk-forward loop as walk_forward_mae, but returns the actual and
    predicted value at each origin individually, instead of just the mean error.
    Needed for plotting; walk_forward_mae alone only gives you the aggregate.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    trace = []  # list of (origin_index, actual, predicted)

    for i in range(n_origins, 0, -1):
        split = n - i
        if split < min_train:
            continue
        train, actual = series[:split], series[split:split + h]
        if len(actual) < h:
            continue
        try:
            forecast = fit_fn(train, h)
            trace.append((split, float(actual[0]), float(np.asarray(forecast)[0])))
        except Exception:
            continue

    return trace


def get_fit_fn(case, model_used):
    """Resolve the actual fitter function used, for re-running the backtest trace.
    Needed because fit_model() only returns the aggregate MAE, not the fitter itself."""
    if case in GATED_CASE_CANDIDATES:
        return GATED_CASE_CANDIDATES[case].get(model_used, fit_naive_persistence)
    return CASE_FITTERS.get(case, fit_naive_persistence)


# ---------------------------------------------------------------------------
# Per-series AvP overlay + residual-over-time plo
# ---------------------------------------------------------------------------

def plot_avp_single(ax, series, case, model_used, dates=None, n_origins=3, min_train=52, label=""):
    """
    Draws one actual-vs-predicted overlay onto a given matplotlib axis:
      - full actual series as a line
      - backtest origins marked with the model's predicted value
    Returns the trace (for residual plotting) so it isn't recomputed twice.
    """
    series = np.asarray(series, dtype=float)
    fit_fn = get_fit_fn(case, model_used)
    trace = backtest_trace(series, fit_fn, n_origins=n_origins, min_train=min_train)

    x = dates if dates is not None else np.arange(len(series))
    ax.plot(x, series, color="#3b6ea5", linewidth=1.3, label="actual")

    if trace:
        origin_idx = [t[0] for t in trace]
        preds = [t[2] for t in trace]
        origin_x = [x[i] for i in origin_idx]
        ax.scatter(origin_x, preds, color="#d6604d", zorder=5, s=28, label="predicted (backtest)")
        ax.plot(origin_x, preds, color="#d6604d", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    return trace


def plot_residuals_single(ax, trace, label=""):
    """Residuals over time (origin index), NOT vs fitted and NOT a histogram —
    both are uninformative at n_origins=3."""
    if not trace:
        ax.set_title(f"{label} (no backtest)", fontsize=9)
        return
    residuals = [t[1] - t[2] for t in trace]  # actual - predicted
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.bar(range(len(residuals)), residuals, color=["#d6604d" if r < 0 else "#3b6ea5" for r in residuals])
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(len(residuals)))
    ax.set_xticklabels([f"t-{len(residuals)-i}" for i in range(len(residuals))], fontsize=6)


# ---------------------------------------------------------------------------
# Grid view: one AvP overlay per (case, dataset, feature, cluster) row
# ---------------------------------------------------------------------------

def plot_avp_grid(results_df, series_lookup, n_origins=3, min_train=52,
                   max_cols=4, save_path=None):
    """
    results_df    : the dataframe built from your case-store loop (must have
                     case, dataset, feature, cluster, model columns)
    series_lookup : dict mapping (dataset, feature, cluster) -> array-like series,
                     so this function can re-derive the backtest trace for plotting
    """
    n = len(results_df)
    if n == 0:
        print("No rows to plot.")
        return

    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, results_df.iterrows()):
        key = (row["dataset"], row["feature"], row["cluster"])
        series = series_lookup.get(key)
        if series is None:
            ax.set_title(f"{row['case']}\n(no series found)", fontsize=8)
            continue
        label = f"{row['case']}\n{row['dataset']}/{row['feature']}/{row['cluster']}"
        plot_avp_single(ax, series, row["case"], row["model"],
                         n_origins=n_origins, min_train=min_train, label=label)
        ax.legend(fontsize=6, loc="upper left")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved AvP grid to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Leaderboard: backtest MAE vs persistence MAE, grouped by case
# ---------------------------------------------------------------------------

def plot_leaderboard(results_df, save_path=None):
    """
    Horizontal bar chart, one bar per row, comparing model MAE to persistence MAE.
    Rows are sorted by case, then by how much the model beats (or loses to) persistence —
    this surfaces underperforming model choices at a glance, which is the point.
    """
    df = results_df.copy()
    df = df[df["mae"].notna()]  # drop manual_review / failed-prep rows from this char
    if df.empty:
        print("No rows with valid backtest MAE to plot.")
        return

    df["improvement"] = df["persistence_mae"] - df["mae"]  # positive = model beats persistence
    df = df.sort_values(["case", "improvement"])
    df["row_label"] = df["case"] + " | " + df["dataset"] + "/" + df["feature"] + "/" + df["cluster"].astype(str)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    y = np.arange(len(df))
    colors = ["#3b6ea5" if v >= 0 else "#d6604d" for v in df["improvement"]]
    ax.barh(y, df["improvement"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["row_label"], fontsize=7)
    ax.set_xlabel("Persistence MAE - Model MAE  (positive = model wins)")
    ax.set_title("Model vs. naive-persistence backtest performance")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved leaderboard to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Demo with synthetic data, mirroring the case-store loop structure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Functions that work directly off a results_df like the one you actually have
# (case, dataset, feature, cluster, model, mae, persistence_mae, forecast list).
# These do NOT require re-running backtests or having the raw series on hand —
# they visualize what's already in the table.
# ---------------------------------------------------------------------------

def plot_win_rate_by_case(results_df, save_path=None):
    """
    Bar chart: % of series where the model beat naive persistence, grouped by case.
    This is the fastest way to see which case->model mappings are actually
    earning their complexity across your whole case store, in one glance.
    Rows with beats_persistence == None (manual_review) are excluded from the
    denominator, since they were never given a real model to evaluate.
    """
    df = results_df.copy()
    df = df[df["beats_persistence"].notna()]
    if df.empty:
        print("No rows with a valid beats_persistence flag to plot.")
        return

    summary = (
        df.groupby("case")
        .agg(win_rate=("beats_persistence", "mean"), n=("beats_persistence", "size"))
        .sort_values("win_rate")
    )

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(summary))))
    colors = ["#d6604d" if v < 0.5 else "#3b6ea5" for v in summary["win_rate"]]
    bars = ax.barh(summary.index, summary["win_rate"], color=colors)
    ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of series beating naive persistence")
    ax.set_title("Win rate vs. persistence, by case")

    for bar, (case, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"n={int(row['n'])}", va="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved win-rate chart to {save_path}")
    return fig


def plot_mae_ratio_leaderboard(results_df, save_path=None, log_scale=True):
    """
    Horizontal bar chart of mae / persistence_mae per row (< 1 means model wins).
    Uses a RATIO instead of the raw difference used in plot_leaderboard(), because
    your series span wildly different scales (single digits up to millions) —
    a raw MAE difference chart would be dominated by 2-3 large-scale rows and
    everything else would look like zero. The ratio is scale-free, so a
    frequency series and a monetary series in the millions plot side by side
    meaningfully. log_scale=True additionally helps since ratios can range from
    ~0.1 to ~10+ in your data.
    """
    df = results_df.copy()
    df = df[df["mae"].notna() & df["persistence_mae"].notna() & (df["persistence_mae"] != 0)]
    if df.empty:
        print("No rows with valid mae/persistence_mae to plot.")
        return

    df["ratio"] = df["mae"] / df["persistence_mae"]
    df = df.sort_values("ratio", ascending=False)
    df["row_label"] = df["case"] + " | " + df["dataset"] + "/" + df["feature"] + "/" + df["cluster"].astype(str)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    y = np.arange(len(df))
    colors = ["#d6604d" if r > 1 else "#3b6ea5" for r in df["ratio"]]
    ax.barh(y, df["ratio"], color=colors)
    ax.axvline(1, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["row_label"], fontsize=7)
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("Model MAE / Persistence MAE  (< 1 = model wins, log scale)" if log_scale
                   else "Model MAE / Persistence MAE  (< 1 = model wins)")
    ax.set_title("Relative backtest performance (scale-free across series)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved MAE-ratio leaderboard to {save_path}")
    return fig


def plot_forecast_paths_by_case(results_df, case, max_series=12, save_path=None):
    """
    Small-multiples grid showing each series' forward FORECAST PATH (the
    'forecast' column — your actual h-step-ahead prediction, not a backtes
    trace) for one case at a time. Flags any series whose forecast has gone
    sign-flipped or exploded in magnitude relative to its own first step,
    since divergence like that (seen in several of your ETS_damped rows,
    e.g. forecasts in the millions or strongly negative) is a model-fi
    problem worth catching visually, not just a number to skim past in a table.
    """
    df = results_df[(results_df["case"] == case) & results_df["forecast"].notna()].copy()
    if df.empty:
        print(f"No plottable rows for case={case!r} (forecast is null for all, e.g. manual_review).")
        return

    df = df.head(max_series)
    n = len(df)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, df.iterrows()):
        fc = np.asarray(row["forecast"], dtype=float)
        h = np.arange(1, len(fc) + 1)
        ax.plot(h, fc, marker="o", color="#3b6ea5")
        ax.axhline(0, color="gray", linewidth=0.6)

        diverging = len(fc) > 1 and (np.sign(fc[0]) != np.sign(fc[-1]) or
                                      abs(fc[-1]) > 5 * (abs(fc[0]) + 1e-9))
        title_color = "#d6604d" if diverging else "black"
        flag = " \u26a0 diverging" if diverging else ""
        ax.set_title(f"{row['dataset']}/{row['feature']}/{row['cluster']}{flag}",
                     fontsize=8, color=title_color)
        ax.tick_params(labelsize=7)
        ax.set_xticks(h)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Forecast paths — case: {case}", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved forecast-path grid to {save_path}")
    return fig


def flag_diverging_forecasts(results_df):
    """
    Returns a filtered dataframe of rows whose forward forecast looks
    pathological: sign flip across the horizon, or final step more than
    5x the magnitude of the first step. This is a quick non-visual check —
    run it before plotting to know which rows need the most attention.
    Several of your trend_seasonal/ETS_damped rows will likely show up here.
    """
    flagged = []
    for _, row in results_df.iterrows():
        fc = row.get("forecast")
        if fc is None or (isinstance(fc, float) and np.isnan(fc)):
            continue
        fc = np.asarray(fc, dtype=float)
        if len(fc) < 2:
            continue
        sign_flip = np.sign(fc[0]) != np.sign(fc[-1])
        magnitude_explosion = abs(fc[-1]) > 5 * (abs(fc[0]) + 1e-9)
        if sign_flip or magnitude_explosion:
            flagged.append({
                "case": row["case"], "dataset": row["dataset"], "feature": row["feature"],
                "cluster": row["cluster"], "model": row["model"],
                "forecast_first": fc[0], "forecast_last": fc[-1],
                "sign_flip": sign_flip, "magnitude_explosion": magnitude_explosion,
            })
    return pd.DataFrame(flagged)


if __name__ == "__main__":
    from forecast_dispatch import fit_model

    rng = np.random.default_rng(1)
    n = 36
    t = np.arange(n)

    synthetic_rows = [
        ("trend_seasonal", "ts_final_retail", "frequency", "retail_2",
         50 + 0.8 * t + 10 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, n)),
        ("trend_only", "ts_final_retail", "monetary", "retail_1",
         20 + 1.2 * t + rng.normal(0, 3, n)),
        ("seasonal_only", "ts_final_retail", "frequency", "retail_4",
         100 + 15 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, n)),
        ("stationary_noise", "ts_final_retail", "avg_balance", "retail_3",
         rng.normal(50, 3, n)),
        ("complex_mixed", "ts_seg_retail", "avg_balance", "retail_-1",
         40 + 0.2 * t + rng.normal(0, 6, n)),
    ]

    series_lookup = {}
    final_results = []

    for case, dataset, feature, cluster, series in synthetic_rows:
        series_lookup[(dataset, feature, cluster)] = series
        result = fit_model(case, series, h=3, verbose_prep=False)
        final_results.append({
            "case": case, "dataset": dataset, "feature": feature, "cluster": cluster,
            "model": result["model_used"],
            "mae": result.get("backtest_mae_h1"),
            "persistence_mae": result.get("persistence_mae"),
            "beats_persistence": result.get("beats_persistence"),
            "forecast": result.get("forecast"),
        })

    results_df = pd.DataFrame(final_results)
    print(results_df)

    plot_avp_grid(results_df, series_lookup, save_path="/tmp/avp_grid_demo.png")
    plot_leaderboard(results_df, save_path="/tmp/leaderboard_demo.png")
    plot_win_rate_by_case(results_df, save_path="/tmp/win_rate_demo.png")
    plot_mae_ratio_leaderboard(results_df, save_path="/tmp/mae_ratio_demo.png")
    plot_forecast_paths_by_case(results_df, "trend_seasonal", save_path="/tmp/forecast_paths_demo.png")
    print(flag_diverging_forecasts(results_df))
