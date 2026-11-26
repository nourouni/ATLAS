"""
Method-specific pooled time-series forecasting workflow.

Each notebook sets one pooling method, then calls these functions in order:
load data -> statistical tests and visualizations -> case/model assignment ->
model-specific prep/forecasting -> evaluation and actual-vs-predicted plots.
"""

from __future__ import annotations

from collections import defaultdic
import as
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import polars as pl

from forecast_dispatch import GATED_CASE_CANDIDATES, CASE_FITTERS, fit_naive_persistence
from forecast_dispatch import fit_model
from statistical_tests import (
    adf_test,
    acf_significance,
    jarque_bera_test,
    kpss_test,
    linear_trend_test,
    ljung_box_test,
    movable_holiday_effect_placeholders,
    pacf_significance,
    seasonal_strength_multi,
    seasonal_strength_stl,
    SEASONAL_STRENGTH_THRESHOLD,
    shapiro_test,
    trend_strength_stl,
)
from evaluation import backtest_trace
import plotting as tsplo


BASE_DIR = Path(__file__).resolve().parents[1]
TS_DIR = BASE_DIR / "ts"
TEMPORAL_DIR = BASE_DIR / "temporal clustering"
COMPARISON_LABELS_PATH = BASE_DIR / "clustering comparison" / "data" / "unified_client_cluster_comparison.parquet"
# Static assignments are the promoted outputs of the final clustering notebook:
# Retail = deduplicated GMM k=5; Enterprise = all-8 HDBSCAN k=3.
STATIC_CLUSTER_OUTPUTS = {
    "retail": BASE_DIR / "static_clustering" / "retail_clustered.csv",
    "enterprise": BASE_DIR / "static_clustering" / "enterprise_clustered.csv",
}
ROLLING_LABEL_CANDIDATES = [
    TEMPORAL_DIR / "rolling_temporal_clustering" / "data" / "rolling_temporal_cluster_labels.parquet",
    BASE_DIR / "forecasting comparison" / "rolling_temporal_clustering" / "data" / "rolling_temporal_cluster_labels.parquet",
]

METRICS = ["frequency", "monetary", "avg_balance"]
STATIC_METRICS = ["avg_balance"]
EXCLUDE_STATIC_UNASSIGNED = True
SEASONAL_PERIOD = 52
FORECAST_HORIZON = 3
MIN_TRAIN = 52
STRENGTH_THRESHOLD = SEASONAL_STRENGTH_THRESHOLD

POOLING_METHODS = {
    "global_pooled": {
        "label_col": None,
        "display_name": "Global pooled",
        "cluster_prefix": "global",
    },
    "static_cluster_pooled": {
        "label_col": "static_cluster_forecast_label",
        "display_name": "Static cluster pooled",
        "cluster_prefix": "static",
    },
    "dtw_cluster_pooled": {
        "label_col": "dtw_cluster_forecast_label",
        "display_name": "DTW cluster pooled",
        "cluster_prefix": "dtw",
    },
    "temporal_feature_cluster_pooled": {
        "label_col": "temporal_feature_cluster_forecast_label",
        "display_name": "Temporal-feature cluster pooled",
        "cluster_prefix": "temporal_feature",
    },
    "temporal_feature_filled_cluster_pooled": {
        "label_col": "temporal_feature_cluster_filled_forecast_label",
        "display_name": "Temporal-feature filled cluster pooled",
        "cluster_prefix": "temporal_feature_filled",
    },
    "rolling_temporal_feature_cluster_pooled": {
        "rolling_label_col": "cluster",
        "display_name": "Rolling temporal-feature cluster pooled",
        "cluster_prefix": "rolling_temporal_feature",
        "origin_train_periods": 24,
    },
    "rolling_temporal_feature_filled_cluster_pooled": {
        "rolling_label_col": "cluster_filled",
        "display_name": "Rolling temporal-feature filled cluster pooled",
        "cluster_prefix": "rolling_temporal_feature_filled",
        "origin_train_periods": 24,
    },
}

MODEL_COLORS = {
    "ETS_trend": "#1f77b4",
    "AR_auto": "#ff7f0e",
    "ARIMA_0_1_0": "#2ca02c",
    "mean_naive": "#9467bd",
    "median_naive": "#8c564b",
    "seasonal_naive": "#e377c2",
    "ets_trend": "#1f77b4",
    "ets_damped": "#17becf",
    "sklearn_hgb_lag": "#d62728",
    "manual_review": "#7f7f7f",
    "skipped_failed_prep": "#7f7f7f",
}


def ensure_output_dirs(method: str) -> dict[str, Path]:
    root = TS_DIR / "method_outputs" / method
    paths = {
        "root": root,
        "data": root / "data",
        "plots": root / "plots",
        "diagnostics": root / "plots" / "cluster_diagnostics",
        "avp": root / "plots" / "actual_vs_predicted",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _read_segment_weekly(segment: str) -> pl.DataFrame:
    rfm = (
        pl.read_parquet(TEMPORAL_DIR / "data" / "panels" / f"weekly_rfm_panel_{segment}.parquet")
        .select(
            pl.col("client_id").cast(pl.Utf8),
            "week_idx",
            "week_start",
            "week_end",
            pl.col("Frequency").alias("frequency"),
            pl.col("Monetary").alias("monetary"),
        )
    )
    balance = (
        pl.read_parquet(TEMPORAL_DIR / "data" / "panels" / f"weekly_panel_{segment}.parquet")
        .select(
            pl.col("client_id").cast(pl.Utf8),
            "week_idx",
            "week_start",
            "week_end",
            "avg_balance",
        )
    )
    return (
        rfm.join(balance, on=["client_id", "week_idx", "week_start", "week_end"], how="left")
        .with_columns(pl.lit(segment).alias("segment"))
    )


def load_weekly_client_panel_pl() -> pl.DataFrame:
    return pl.concat([_read_segment_weekly("retail"), _read_segment_weekly("enterprise")], how="diagonal_relaxed")


def load_weekly_client_panel() -> pd.DataFrame:
    """Small-use convenience loader; main workflow aggregates in Polars first."""
    panel = load_weekly_client_panel_pl()
    pdf = pd.DataFrame(panel.to_dicts())
    pdf["date"] = pd.to_datetime(pdf["week_start"])
    return pdf


def _labels_with_fallbacks() -> pl.DataFrame:
    path = comparison_labels_path()
    labels = pl.read_parquet(path).with_columns(
        pl.col("client_id").cast(pl.Utf8),
        pl.col("comparison_segment").str.to_lowercase().alias("comparison_segment"),
    )
    offset = (
        pl.when(pl.col("comparison_segment") == "enterprise")
        .then(pl.lit(1000))
        .when(pl.col("comparison_segment") == "retail")
        .then(pl.lit(2000))
        .otherwise(pl.lit(9000))
    )
    exprs = []
    for raw_col, forecast_col in [
        ("static_cluster", "static_cluster_forecast_label"),
        ("dtw_cluster", "dtw_cluster_forecast_label"),
        ("temporal_feature_cluster", "temporal_feature_cluster_forecast_label"),
        ("temporal_feature_cluster_filled", "temporal_feature_cluster_filled_forecast_label"),
    ]:
        if forecast_col not in labels.columns and raw_col in labels.columns:
            raw = pl.col(raw_col).fill_null(-1).cast(pl.Int64)
            exprs.append(pl.when(raw == -1).then(pl.lit(-1)).otherwise(raw + offset).alias(forecast_col))
    return labels.with_columns(exprs) if exprs else labels


def comparison_labels_path() -> Path:
    if COMPARISON_LABELS_PATH.exists():
        return COMPARISON_LABELS_PATH
    raise FileNotFoundError(f"Missing unified cluster comparison labels: {COMPARISON_LABELS_PATH}")


def _static_labels_from_outputs() -> pl.DataFrame:
    frames = []
    expected_clusters = {"retail": {0, 1, 2, 3, 4}, "enterprise": {0, 1, 2}}
    for segment, path in STATIC_CLUSTER_OUTPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing final static assignment artifact for {segment}: {path}")
        offset = 2000 if segment == "retail" else 1000
        raw = pl.read_csv(path, columns=["client_id", "cluster"])
        detected = set(raw.get_column("cluster").drop_nulls().cast(pl.Int64).unique().to_list())
        behavioral = detected - {-1}
        if behavioral != expected_clusters[segment]:
            raise ValueError(
                f"Unexpected final {segment} cluster IDs in {path}: "
                f"detected={sorted(detected)}, expected behavioral={sorted(expected_clusters[segment])} plus -1"
            )
        frame = (
            raw
            .with_columns(
                pl.col("client_id").cast(pl.Utf8),
                pl.lit(segment).alias("comparison_segment"),
                pl.col("cluster").fill_null(-1).cast(pl.Int64).alias("static_cluster"),
            )
            .filter(~pl.col("static_cluster").eq(-1) if EXCLUDE_STATIC_UNASSIGNED else pl.lit(True))
            .with_columns(
                pl.when(pl.col("static_cluster") == -1)
                .then(pl.lit(-1))
                .otherwise(pl.col("static_cluster") + offset)
                .alias("static_cluster_forecast_label")
            )
        )
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed")


def metrics_for_method(method: str) -> list[str]:
    """Return the target metrics without changing requirements of other workflows."""
    return STATIC_METRICS if method == "static_cluster_pooled" else METRICS


def load_pool_lookup(method: str) -> pd.DataFrame:
    config = POOLING_METHODS[method]
    if "rolling_label_col" in config:
        return load_rolling_pool_lookup(method)
    if config["label_col"] is None:
        return pd.DataFrame(columns=["client_id", "segment", "pool"])

    labels = _static_labels_from_outputs() if method == "static_cluster_pooled" else _labels_with_fallbacks()
    label_col = config["label_col"]
    lookup = (
        labels.select(
            "client_id",
            pl.col("comparison_segment").alias("segment"),
            pl.col(label_col).fill_null(-1).cast(pl.Int64).alias("pool_id"),
        )
        .with_columns(
            (
                pl.col("segment")
                + pl.lit("_")
                + pl.lit(config["cluster_prefix"])
                + pl.lit("_")
                + pl.col("pool_id").cast(pl.Utf8)
            ).alias("pool")
        )
        .select("client_id", "segment", "pool")
    )
    return pd.DataFrame(lookup.to_dicts())


def rolling_labels_path() -> Path:
    for path in ROLLING_LABEL_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in ROLLING_LABEL_CANDIDATES)
    raise FileNotFoundError(
        "Missing rolling temporal cluster labels. Run the rolling workflow first. Searched:\n"
        + searched
    )


def load_rolling_pool_lookup(method: str, origin_train_months: int | None = None) -> pd.DataFrame:
    config = POOLING_METHODS[method]
    # Rolling labels still store the historical column name
    # `origin_train_months`; in this weekly workflow it means period count.
    origin = int(origin_train_months or config.get("origin_train_periods", config.get("origin_train_months", 33)))
    label_col = config["rolling_label_col"]
    path = rolling_labels_path()
    labels = (
        pl.read_parquet(path)
        .filter(pl.col("origin_train_months") == origin)
        .select(
            pl.col("client_id").cast(pl.Utf8),
            pl.col("segment").str.to_lowercase().alias("segment"),
            pl.col(label_col).fill_null(-1).cast(pl.Int64).alias("pool_id"),
        )
        .with_columns(
            (
                pl.col("segment")
                + pl.lit("_")
                + pl.lit(config["cluster_prefix"])
                + pl.lit("_origin")
                + pl.lit(origin).cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("pool_id").cast(pl.Utf8)
            ).alias("pool")
        )
        .select("client_id", "segment", "pool")
    )
    if labels.height == 0:
        raise ValueError(f"No rolling labels found for origin_train_months={origin} in {path}")
    return pd.DataFrame(labels.to_dicts())


def build_pool_timeseries(method: str) -> pd.DataFrame:
    segment_frames = []
    lookup_by_segment: dict[str, pl.DataFrame] = {}
    if method != "global_pooled":
        lookup = load_pool_lookup(method)
        for segment in lookup["segment"].dropna().unique():
            sub = lookup[lookup["segment"] == segment]
            lookup_by_segment[str(segment)] = pl.DataFrame(
                {
                    "client_id": sub["client_id"].astype(str).to_list(),
                    "pool": sub["pool"].astype(str).to_list(),
                }
            )

    for segment in ["retail", "enterprise"]:
        panel = _read_segment_weekly(segment)
        if method == "global_pooled":
            panel = panel.with_columns(pl.lit(f"{segment}_global_pool").alias("pool"))
        else:
            lookup_pl = lookup_by_segment.get(segment)
            if lookup_pl is None or lookup_pl.height == 0:
                continue
            panel = panel.join(lookup_pl, on="client_id", how="inner")
        segment_frames.append(
            panel.group_by(["pool", "week_start"])
            .agg(
                pl.col("frequency").sum().alias("frequency"),
                pl.col("monetary").sum().alias("monetary"),
                pl.col("avg_balance").mean().alias("avg_balance"),
                pl.col("client_id").n_unique().alias("n_clients"),
            )
        )

    if not segment_frames:
        raise ValueError(f"No pooled time-series rows built for method={method!r}")

    ts_pl = (
        pl.concat(segment_frames, how="diagonal_relaxed")
        .group_by(["pool", "week_start"])
        .agg(
            pl.col("frequency").sum(),
            pl.col("monetary").sum(),
            pl.col("avg_balance").mean(),
            pl.col("n_clients").sum(),
        )
        .rename({"week_start": "date"})
        .sort(["pool", "date"])
    )
    ts = pd.DataFrame(ts_pl.to_dicts())
    ts["date"] = pd.to_datetime(ts["date"])
    ts["method"] = method
    return ts


def classify_ts_case(row: pd.Series) -> str:
    stationary = bool(row.get("ADF stationary", False)) and bool(row.get("KPSS stationary", False))
    trend_strength = float(row.get("Trend strength", 0) or 0)
    trend = bool(row.get("Trend exists", False)) or trend_strength > STRENGTH_THRESHOLD
    seasonal = bool(row.get("Strong seasonality", False)) or bool(row.get("Any strong seasonality", False))
    pacf_lag_count = int(row.get("PACF significant lag count", 0) or 0)
    acf_lag_count = int(row.get("ACF significant lag count", 0) or 0)
    joint_autocorr = not bool(row.get("No autocorr", True))
    repeated_lag_evidence = max(pacf_lag_count, acf_lag_count) >= 2
    autocorr = joint_autocorr or repeated_lag_evidence
    normal = bool(row.get("Normal (JB)", False)) and bool(row.get("Normal (Shapiro)", False))

    if trend and seasonal:
        return "trend_seasonal"
    if seasonal and not trend:
        return "seasonal_only"
    if trend and not seasonal:
        return "trend_only"
    if autocorr and not trend and not seasonal:
        return "stationary_ar_structure" if stationary else "nonstationary_ar_structure"
    if stationary and not autocorr:
        return "stationary_noise" if normal else "heavy_tailed_noise"
    return "complex_mixed"


def safe_stat(fn, series: pd.Series, default: dict, **kwargs) -> dict:
    try:
        return fn(series, **kwargs)
    except Exception:
        return defaul


def run_statistical_tests(ts_full: pd.DataFrame, method: str, output_dirs: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    rows = []
    case_store: dict[str, list[dict]] = defaultdict(list)
    for pool, group in ts_full.groupby("pool"):
        group = group.sort_values("date")
        for metric in metrics_for_method(method):
            series = group.set_index("date")[metric].dropna()
            if len(series) < 8:
                continue
            adf = safe_stat(adf_test, series, {"is_stationary": False, "pvalue": np.nan})
            kpss = safe_stat(kpss_test, series, {"is_stationary": False, "pvalue": np.nan})
            trend = safe_stat(linear_trend_test, series, {"trend_exists": False, "slope": np.nan, "pvalue": np.nan})
            season = safe_stat(
                seasonal_strength_stl,
                series,
                {"seasonal_strength": np.nan, "has_strong_seasonality": False},
                period=SEASONAL_PERIOD,
            )
            multi_season = safe_stat(
                seasonal_strength_multi,
                series,
                {
                    "strengths": {},
                    "flags": {},
                    "strongest_period_name": None,
                    "strongest_period": np.nan,
                    "strongest_strength": np.nan,
                    "has_any_strong_seasonality": False,
                },
            )
            trend_stl = safe_stat(
                trend_strength_stl,
                series,
                {"trend_strength": np.nan, "has_strong_trend": False},
                period=SEASONAL_PERIOD,
            )
            lb = safe_stat(
                ljung_box_test,
                series,
                {"pvalue": np.nan, "no_autocorrelation": True},
                lags=min(SEASONAL_PERIOD, max(1, len(series) // 3)),
            )
            jb = safe_stat(jarque_bera_test, series, {"is_normal": False, "pvalue": np.nan})
            shapiro = safe_stat(shapiro_test, series, {"is_normal": False, "pvalue": np.nan})
            pacf = safe_stat(
                pacf_significance,
                series,
                {
                    "has_partial_autocorrelation": False,
                    "significant_lags": [],
                    "n_significant_lags": 0,
                    "lag_window_hits": {},
                    "multiple_lag_false_positive_prob": np.nan,
                },
                nlags=max(1, min(SEASONAL_PERIOD, len(series) // 2 - 1)),
            )
            acf_sig = safe_stat(
                acf_significance,
                series,
                {
                    "has_autocorrelation": False,
                    "significant_lags": [],
                    "n_significant_lags": 0,
                    "lag_window_hits": {},
                    "multiple_lag_false_positive_prob": np.nan,
                },
                nlags=max(1, min(SEASONAL_PERIOD, len(series) // 2 - 1)),
            )
            holiday = movable_holiday_effect_placeholders(series)
            strengths = multi_season.get("strengths", {})
            flags = multi_season.get("flags", {})
            pacf_windows = pacf.get("lag_window_hits", {})
            acf_windows = acf_sig.get("lag_window_hits", {})
            row = {
                "method": method,
                "pool": pool,
                "metric": metric,
                "n_obs": len(series),
                "ADF stationary": adf["is_stationary"],
                "ADF p": adf.get("pvalue"),
                "KPSS stationary": kpss["is_stationary"],
                "KPSS p": kpss.get("pvalue"),
                "Trend exists": trend["trend_exists"],
                "Slope": trend.get("slope"),
                "Trend p": trend.get("pvalue"),
                "Seasonality strength": season["seasonal_strength"],
                "Strong seasonality": season["has_strong_seasonality"],
                "Seasonality strength threshold": STRENGTH_THRESHOLD,
                "Monthly-like 4w seasonality strength": strengths.get("monthly_like", np.nan),
                "Quarterly 13w seasonality strength": strengths.get("quarterly", np.nan),
                "Semiannual 26w seasonality strength": strengths.get("semiannual", np.nan),
                "Annual 52w seasonality strength": strengths.get("annual", season["seasonal_strength"]),
                "Strong monthly-like 4w seasonality": flags.get("monthly_like", False),
                "Strong quarterly 13w seasonality": flags.get("quarterly", False),
                "Strong semiannual 26w seasonality": flags.get("semiannual", False),
                "Strong annual 52w seasonality": flags.get("annual", season["has_strong_seasonality"]),
                "Any strong seasonality": multi_season.get("has_any_strong_seasonality", False),
                "Strongest seasonal period": multi_season.get("strongest_period"),
                "Strongest seasonal period name": multi_season.get("strongest_period_name"),
                "Strongest seasonal strength": multi_season.get("strongest_strength"),
                "Trend strength": trend_stl["trend_strength"],
                "Strong trend": trend_stl["has_strong_trend"],
                "Trend strength threshold": STRENGTH_THRESHOLD,
                "LB p": lb["pvalue"],
                "No autocorr": lb["no_autocorrelation"],
                "ACF signal": acf_sig["has_autocorrelation"],
                "ACF significant lag count": acf_sig.get("n_significant_lags", 0),
                "ACF significant lags": ",".join(map(str, acf_sig.get("significant_lags", []))),
                "ACF monthly-like lag hits": ",".join(map(str, acf_windows.get("monthly_like", []))),
                "ACF quarterly lag hits": ",".join(map(str, acf_windows.get("quarterly", []))),
                "ACF semiannual lag hits": ",".join(map(str, acf_windows.get("semiannual", []))),
                "ACF annual lag hits": ",".join(map(str, acf_windows.get("annual", []))),
                "ACF multi-lag false-positive probability": acf_sig.get("multiple_lag_false_positive_prob"),
                "PACF signal": pacf["has_partial_autocorrelation"],
                "PACF significant lag count": pacf.get("n_significant_lags", 0),
                "PACF significant lags": ",".join(map(str, pacf.get("significant_lags", []))),
                "PACF monthly-like lag hits": ",".join(map(str, pacf_windows.get("monthly_like", []))),
                "PACF quarterly lag hits": ",".join(map(str, pacf_windows.get("quarterly", []))),
                "PACF semiannual lag hits": ",".join(map(str, pacf_windows.get("semiannual", []))),
                "PACF annual lag hits": ",".join(map(str, pacf_windows.get("annual", []))),
                "PACF multi-lag false-positive probability": pacf.get("multiple_lag_false_positive_prob"),
                "Movable holiday regressors needed": holiday["requires_event_calendar"],
                "Movable holiday note": holiday["note"],
                "Normal (JB)": jb["is_normal"],
                "JB p": jb.get("pvalue"),
                "Normal (Shapiro)": shapiro["is_normal"],
                "Shapiro p": shapiro.get("pvalue"),
            }
            row["case"] = classify_ts_case(pd.Series(row))
            rows.append(row)
            case_store[row["case"]].append({"method": method, "pool": pool, "metric": metric})
            plot_cluster_metric_diagnostics(series, method, pool, metric, output_dirs["diagnostics"])

    report = pd.DataFrame(rows).sort_values(["pool", "metric"])
    report.to_csv(output_dirs["data"] / "statistical_report.csv", index=False)
    return report, case_store


def plot_cluster_metric_diagnostics(series: pd.Series, method: str, pool: str, metric: str, save_root: Path) -> None:
    safe_pool = str(pool).replace("/", "_").replace("\\", "_")
    save_dir = save_root / safe_pool / metric
    title = f"{method} | {pool} | {metric}"
    for plot_fn in [
        tsplot.plot_time_series,
        tsplot.plot_rolling_stats,
        tsplot.plot_acf,
        tsplot.plot_pacf,
        tsplot.plot_distribution,
        tsplot.plot_stl,
        tsplot.plot_diagnostic_summary,
    ]:
        try:
            plot_fn(series, save_dir, title=title)
            plt.close("all")
        except Exception:
            plt.close("all")


def resolve_fit_fn(case: str, model_used: str):
    if case in GATED_CASE_CANDIDATES:
        return GATED_CASE_CANDIDATES[case].get(model_used, fit_naive_persistence)
    return CASE_FITTERS.get(case, fit_naive_persistence)


def train_test_split_series(series: pd.Series, horizon: int = FORECAST_HORIZON) -> tuple[pd.Series, pd.Series]:
    clean = series.dropna()
    if len(clean) <= horizon:
        return clean, clean.iloc[0:0]
    return clean.iloc[:-horizon], clean.iloc[-horizon:]


def run_case_forecasts(
    ts_full: pd.DataFrame,
    statistical_report: pd.DataFrame,
    case_store: dict,
    method: str,
    output_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    forecast_rows = []
    forecasts_dict: dict[str, dict[str, dict[str, pd.Series]]] = defaultdict(lambda: defaultdict(dict))
    for case, items in case_store.items():
        for item in items:
            pool = item["pool"]
            metric = item["metric"]
            group = ts_full[ts_full["pool"] == pool].sort_values("date")
            series = group.set_index("date")[metric].dropna()
            train, test = train_test_split_series(series)
            if test.empty:
                continue
            result = fit_model(
                case,
                train.values,
                h=len(test),
                index=train.index,
                expected_length=max(len(train), 1),
                min_train=MIN_TRAIN,
                verbose_prep=False,
            )
            model_used = result["model_used"]
            forecast = result.get("forecast")
            if forecast is None:
                forecast_values = np.repeat(train.iloc[-1], len(test)) if len(train) else np.zeros(len(test))
            else:
                forecast_values = np.asarray(forecast, dtype=float)[: len(test)]
            forecast_series = pd.Series(forecast_values, index=test.index, name=metric)
            forecasts_dict[model_used][pool][metric] = forecast_series
            abs_error = np.abs(test.values - forecast_values)
            rows.append(
                {
                    "method": method,
                    "pool": pool,
                    "metric": metric,
                    "case": case,
                    "model": model_used,
                    "n_train": len(train),
                    "n_test": len(test),
                    "mae": float(np.mean(abs_error)),
                    "rmse": float(np.sqrt(np.mean(abs_error**2))),
                    "mape_like": float(np.mean(abs_error / np.maximum(np.abs(test.values), 1e-9))),
                    "backtest_mae_h1": result.get("backtest_mae_h1"),
                    "persistence_mae": result.get("persistence_mae"),
                    "beats_persistence": result.get("beats_persistence"),
                    "prep_warnings": result.get("prep_warnings", []),
                    "needs_manual_review": model_used in ("manual_review", "skipped_failed_prep"),
                }
            )
            for date, actual, pred in zip(test.index, test.values, forecast_values):
                forecast_rows.append(
                    {
                        "method": method,
                        "pool": pool,
                        "metric": metric,
                        "date": date,
                        "case": case,
                        "model": model_used,
                        "actual": float(actual),
                        "prediction": float(pred),
                        "abs_error": float(abs(actual - pred)),
                    }
                )
    results = pd.DataFrame(rows).sort_values(["pool", "metric"])
    forecast_df = pd.DataFrame(forecast_rows).sort_values(["pool", "metric", "date"])
    results.to_csv(output_dirs["data"] / "forecast_results.csv", index=False)
    forecast_df.to_csv(output_dirs["data"] / "forecast_points.csv", index=False)
    return results, forecast_df, forecasts_dic


def _parse_warning_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
        return [text]
    return [str(value)]


def _warning_issue_type(warning: str) -> str:
    text = warning.lower()
    if "zero values" in text:
        return "zero_values"
    if "possible outlier" in text:
        return "possible_outliers"
    if "missing from the date index" in text:
        return "missing_weeks"
    if "nan value" in text:
        return "nan_values"
    if "not a multiple of the seasonal" in text:
        return "seasonal_phase_check"
    if "variance is ~0" in text:
        return "near_constant"
    if "adf p-value" in text:
        return "case_recheck"
    return "other"


def summarize_prep_warnings(forecast_results: pd.DataFrame, output_dirs: dict[str, Path] | None = None) -> pd.DataFrame:
    """Flatten prep warnings into a checklist grouped by pool/metric/case."""
    rows = []
    for row in forecast_results.to_dict("records"):
        for warning in _parse_warning_list(row.get("prep_warnings")):
            issue_type = _warning_issue_type(warning)
            rows.append(
                {
                    "pool": row.get("pool"),
                    "metric": row.get("metric"),
                    "case": row.get("case"),
                    "model": row.get("model"),
                    "issue_type": issue_type,
                    "suggested_fix": WARNING_FIX_GUIDE.get(issue_type, WARNING_FIX_GUIDE["other"]),
                    "warning": warning,
                    "baseline_mae": row.get("mae"),
                    "baseline_beats_persistence": row.get("beats_persistence"),
                    "needs_manual_review": row.get("needs_manual_review"),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["issue_type", "pool", "metric"]).reset_index(drop=True)
    if output_dirs is not None:
        summary.to_csv(output_dirs["data"] / "prep_warning_action_items.csv", index=False)
    return summary


WARNING_FIX_GUIDE = {
    "zero_values": "If zeros are leading sentinel values, replace that leading block with interpolated values; otherwise keep them.",
    "possible_outliers": "Winsorize only extreme IQR outliers, then compare forecast MAE against the baseline.",
    "missing_weeks": "Reindex to weekly dates and interpolate short gaps before forecasting.",
    "nan_values": "Interpolate internal NaNs, then fill remaining edge NaNs from nearest observed values.",
    "seasonal_phase_check": "Check the start date and compare against a phase-neutral model if seasonal lookup is suspect.",
    "near_constant": "Prefer mean/median naive over parametric models for effectively constant series.",
    "case_recheck": "Re-check the assigned case; avoid differencing if the original series is already stationary.",
    "other": "Inspect the series plot and compare the repair experiment before changing the canonical data.",
}


def repair_series_for_prep_warnings(series: pd.Series, warnings: list[str]) -> tuple[pd.Series, list[str]]:
    """Apply conservative, explainable repairs driven by prep warning types."""
    repaired = series.astype(float).sort_index().copy()
    applied: list[str] = []
    issue_types = {_warning_issue_type(w) for w in warnings}

    if "missing_weeks" in issue_types and isinstance(repaired.index, pd.DatetimeIndex):
        full_index = pd.date_range(repaired.index.min(), repaired.index.max(), freq="7D")
        if len(full_index) > len(repaired):
            repaired = repaired.reindex(full_index).interpolate(limit_direction="both")
            applied.append("weekly_reindex_interpolate")

    if "nan_values" in issue_types and repaired.isna().any():
        repaired = repaired.interpolate(limit_direction="both")
        applied.append("nan_interpolate")

    if "zero_values" in issue_types:
        values = repaired.to_numpy(dtype=float)
        zero_mask = values == 0
        if zero_mask.any() and zero_mask[0]:
            first_nonzero = int(np.argmax(~zero_mask)) if np.any(~zero_mask) else len(values)
            if 0 < first_nonzero < len(values):
                repaired.iloc[:first_nonzero] = np.nan
                repaired = repaired.interpolate(limit_direction="both")
                applied.append(f"leading_zero_block_interpolate_n={first_nonzero}")

    if "possible_outliers" in issue_types and repaired.notna().sum() >= 8:
        q1, q3 = repaired.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            mask = (repaired < lower) | (repaired > upper)
            if mask.any():
                repaired = repaired.clip(lower=lower, upper=upper)
                applied.append(f"iqr_winsorize_n={int(mask.sum())}")

    return repaired, applied


def run_prep_warning_fix_experiment(
    ts_full: pd.DataFrame,
    forecast_results: pd.DataFrame,
    case_store: dict,
    method: str,
    output_dirs: dict[str, Path],
) -> pd.DataFrame:
    """
    Try warning-driven repairs on affected series and compare holdout scores.
    This does not mutate ts_full; it writes a before/after audit table.
    """
    warning_summary = summarize_prep_warnings(forecast_results, output_dirs)
    if warning_summary.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_dirs["data"] / "prep_warning_fix_experiment.csv", index=False)
        return empty

    baseline_lookup = {
        (row.pool, row.metric): row
        for row in forecast_results.itertuples(index=False)
    }
    case_lookup = {
        (item["pool"], item["metric"]): case
        for case, items in case_store.items()
        for item in items
    }
    rows = []
    for (pool, metric), group in warning_summary.groupby(["pool", "metric"]):
        base = baseline_lookup.get((pool, metric))
        case = case_lookup.get((pool, metric))
        if base is None or case is None:
            continue
        warnings = group["warning"].tolist()
        series = (
            ts_full[ts_full["pool"] == pool]
            .sort_values("date")
            .set_index("date")[metric]
            .dropna()
        )
        repaired, applied = repair_series_for_prep_warnings(series, warnings)
        if not applied:
            rows.append(
                {
                    "method": method,
                    "pool": pool,
                    "metric": metric,
                    "case": case,
                    "baseline_model": base.model,
                    "repair_model": None,
                    "fix_applied": "none_auto_safe",
                    "baseline_mae": base.mae,
                    "repair_mae": np.nan,
                    "mae_delta": np.nan,
                    "improved": False,
                    "warning_count": len(warnings),
                }
            )
            continue
        train, test = train_test_split_series(repaired)
        if test.empty:
            continue
        result = fit_model(
            case,
            train.values,
            h=len(test),
            index=train.index,
            expected_length=max(len(train), 1),
            min_train=MIN_TRAIN,
            verbose_prep=False,
        )
        forecast = result.get("forecast")
        if forecast is None:
            forecast_values = np.repeat(train.iloc[-1], len(test)) if len(train) else np.zeros(len(test))
        else:
            forecast_values = np.asarray(forecast, dtype=float)[: len(test)]
        abs_error = np.abs(test.values - forecast_values)
        repair_mae = float(np.mean(abs_error))
        rows.append(
            {
                "method": method,
                "pool": pool,
                "metric": metric,
                "case": case,
                "baseline_model": base.model,
                "repair_model": result.get("model_used"),
                "fix_applied": "; ".join(applied),
                "baseline_mae": base.mae,
                "repair_mae": repair_mae,
                "mae_delta": repair_mae - float(base.mae),
                "improved": repair_mae < float(base.mae),
                "warning_count": len(warnings),
            }
        )
    experiment = pd.DataFrame(rows)
    if not experiment.empty:
        experiment = experiment.sort_values(["improved", "mae_delta"], ascending=[False, True]).reset_index(drop=True)
    experiment.to_csv(output_dirs["data"] / "prep_warning_fix_experiment.csv", index=False)
    return experimen


def build_test_frame(ts_full: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    frames = []
    for pool, group in ts_full.groupby("pool"):
        group = group.sort_values("date")
        frames.append(group.tail(horizon))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_actual_vs_predicted(
    forecasts_dict: dict,
    ts_full: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_col: str = "pool",
    metrics: list[str] | None = None,
    segment_label: str = "",
    output_dir: str | Path | None = None,
) -> None:
    """Plot full history, actual test period, and each model forecast per pool."""
    if metrics is None:
        metrics = METRICS
    output_dir = Path(output_dir or TS_DIR / "method_outputs" / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = sorted(ts_full[cluster_col].dropna().unique())

    for cluster in clusters:
        full_grp = ts_full[ts_full[cluster_col] == cluster].sort_values("date")
        test_grp = test_df[test_df[cluster_col] == cluster].sort_values("date")
        if full_grp.empty or test_grp.empty:
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4), sharey=False)
        if len(metrics) == 1:
            axes = [axes]

        for j, metric in enumerate(metrics):
            ax = axes[j]
            ax.plot(full_grp["date"], full_grp[metric], color="lightgrey", linewidth=1, label="Full history", zorder=1)
            ax.plot(
                test_grp["date"],
                test_grp[metric],
                color="black",
                linewidth=2,
                label="Actual (test)",
                zorder=2,
            )
            for model_name, cluster_fcs in forecasts_dict.items():
                if cluster not in cluster_fcs or metric not in cluster_fcs[cluster]:
                    continue
                fc = cluster_fcs[cluster][metric]
                color = MODEL_COLORS.get(model_name, "purple")
                ax.plot(fc.index, fc.values, linestyle="--", linewidth=1.5, color=color, label=model_name, zorder=3)
            ax.set_title(f"{cluster} | {metric}", fontsize=9)
            ax.set_ylabel(metric, fontsize=8)
            ax.legend(fontsize=7, loc="best")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)

        fig.suptitle(f"{segment_label} - Actual vs Predicted - {cluster}", fontsize=11, y=1.01)
        fig.tight_layout()
        safe_cluster = str(cluster).replace("/", "_").replace("\\", "_")
        path = output_dir / f"avp_{safe_cluster}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def summarize_scores(results: pd.DataFrame, statistical_report: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_summary = (
        results.groupby(["method", "metric", "case", "model"], dropna=False)
        .agg(
            n_series=("pool", "count"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            backtest_mae_mean=("backtest_mae_h1", "mean"),
            persistence_mae_mean=("persistence_mae", "mean"),
            win_rate=("beats_persistence", "mean"),
            manual_review_rate=("needs_manual_review", "mean"),
        )
        .reset_index()
        .sort_values(["metric", "mae_mean"])
    )
    case_summary = (
        statistical_report.groupby(["method", "metric", "case"], dropna=False)
        .size()
        .reset_index(name="n_pools")
        .sort_values(["metric", "case"])
    )
    return score_summary, case_summary


def run_full_workflow(method: str) -> dict:
    if method not in POOLING_METHODS:
        raise ValueError(f"Unknown method {method!r}. Use one of: {sorted(POOLING_METHODS)}")
    output_dirs = ensure_output_dirs(method)
    ts_full = build_pool_timeseries(method)
    ts_full.to_csv(output_dirs["data"] / "pool_timeseries.csv", index=False)
    statistical_report, case_store = run_statistical_tests(ts_full, method, output_dirs)
    forecast_results, forecast_points, forecasts_dict = run_case_forecasts(
        ts_full, statistical_report, case_store, method, output_dirs
    )
    test_df = build_test_frame(ts_full)
    plot_actual_vs_predicted(
        forecasts_dict,
        ts_full,
        test_df,
        cluster_col="pool",
        metrics=metrics_for_method(method),
        segment_label=POOLING_METHODS[method]["display_name"],
        output_dir=output_dirs["avp"],
    )
    score_summary, case_summary = summarize_scores(forecast_results, statistical_report)
    score_summary.to_csv(output_dirs["data"] / "score_summary.csv", index=False)
    case_summary.to_csv(output_dirs["data"] / "case_summary.csv", index=False)
    return {
        "method": method,
        "output_dirs": output_dirs,
        "ts_full": ts_full,
        "statistical_report": statistical_report,
        "case_store": case_store,
        "forecast_results": forecast_results,
        "forecast_points": forecast_points,
        "score_summary": score_summary,
        "case_summary": case_summary,
    }
