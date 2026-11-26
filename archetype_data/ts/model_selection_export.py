"""
Model selection and export helpers for the TS pooling comparison notebook.

The notebook compares pooling methods using saved method_outputs, then this
module refits only the promoted winners on the full available pooled series and
writes deployable artifacts.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import forecast_dispatch as fd
import pooling_method_workflow as pmw


REGISTRY_DIR = pmw.TS_DIR / "model_registry"
ARTIFACT_DIR = REGISTRY_DIR / "artifacts"
METADATA_DIR = REGISTRY_DIR / "metadata"


def _safe_name(*parts: Any) -> str:
    text = "__".join(str(part) for part in parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _fit_hist_gradient_boosting_lag_model(series: np.ndarray, period: int = fd.SEASONAL_PERIOD):
    series = np.asarray(series, dtype=float)
    if len(series) < 64:
        return None
    max_lag = min(52, max(8, len(series) // 3))
    X, y = [], []
    for t in range(max_lag, len(series)):
        X.append(fd._lag_feature_row(series[:t], t, period=period))
        y.append(series[t])
    if len(y) < 24:
        return None
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(np.asarray(X), np.asarray(y))
    return model


def fit_deployable_model(model_name: str, series: pd.Series) -> tuple[Any | None, dict[str, Any]]:
    """Fit the selected model on the full available series.

    Returns a serializable model object when that is useful, plus a recipe/state
    dictionary that is always written for reproducibility and simple deployment.
    """
    clean = series.dropna().astype(float)
    values = clean.to_numpy(dtype=float)
    state: dict[str, Any] = {
        "model": model_name,
        "n_observations": int(len(clean)),
        "train_start": clean.index.min(),
        "train_end": clean.index.max(),
        "last_value": float(values[-1]) if len(values) else None,
        "mean_value": float(np.mean(values)) if len(values) else None,
        "median_value": float(np.median(values)) if len(values) else None,
        "seasonal_period": fd.SEASONAL_PERIOD,
        "tail_values": values[-fd.SEASONAL_PERIOD :].tolist(),
    }

    if len(values) == 0:
        return None, state

    if model_name in {"manual_review", "skipped_failed_prep"}:
        state["export_note"] = "No fitted model exported for manual-review rows."
        return None, state
    if model_name in {"mean_naive", "median_naive", "seasonal_naive", "persistence"}:
        state["export_note"] = "Naive model represented by recipe/state JSON."
        return None, state
    if model_name == "ets_trend":
        fitted = ExponentialSmoothing(values, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
        return fitted, state
    if model_name == "ets_damped":
        if len(values) < 2 * fd.SEASONAL_PERIOD:
            fitted = ExponentialSmoothing(values, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
            state["export_note"] = "Fell back to ETS trend because fewer than two seasonal cycles were available."
            return fitted, state
        fitted = ExponentialSmoothing(
            values,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=fd.SEASONAL_PERIOD,
        ).fit(optimized=True)
        return fitted, state
    if model_name == "AR_auto":
        best_aic, best_model = np.inf, None
        for lag in range(1, 3):
            if len(values) <= lag + 2:
                continue
            try:
                fitted = AutoReg(values, lags=lag, old_names=False).fit()
                if fitted.aic < best_aic:
                    best_aic, best_model = fitted.aic, fitted
            except Exception:
                continue
        return best_model, state
    if model_name == "ARIMA_0_1_0":
        return ARIMA(values, order=(0, 1, 0), trend="t").fit(), state
    if model_name == "sklearn_hgb_lag":
        fitted = _fit_hist_gradient_boosting_lag_model(values)
        if fitted is None:
            state["export_note"] = "HGB lag model could not be fit with available history."
        return fitted, state

    state["export_note"] = "Unknown model name; recipe/state JSON written without binary artifact."
    return None, state


def select_best_models(forecast_results: pd.DataFrame) -> pd.DataFrame:
    """Select deployment winners per method/pool/metric candidate surface."""
    candidates = forecast_results.copy()
    candidates = candidates[candidates["metric"].isin(pmw.METRICS)]
    candidates = candidates[~candidates["model"].isin(["manual_review", "skipped_failed_prep"])]
    candidates = candidates[candidates["mae"].notna()]
    candidates = candidates[candidates["backtest_mae_h1"].notna()]
    candidates = candidates[
        candidates["beats_persistence"].fillna(False).astype(bool)
        | (candidates["mae"] <= candidates["persistence_mae"])
    ].copy()
    candidates["selection_score"] = candidates["mae"]
    candidates["tie_breaker_backtest_mae"] = candidates["backtest_mae_h1"]
    return candidates.sort_values(
        ["metric", "selection_score", "tie_breaker_backtest_mae", "method", "pool"],
        na_position="last",
    ).reset_index(drop=True)


def select_best_processes(forecast_results: pd.DataFrame) -> pd.DataFrame:
    """Select the best pooling-method/model process per metric.

    This compares processes, not individual tiny pools, so deployment does no
    accidentally pick a method only because one small customer class has low
    absolute error.
    """
    candidates = select_best_models(forecast_results)
    process_scores = (
        candidates.groupby(["metric", "method", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            backtest_mae_h1_mean=("backtest_mae_h1", "mean"),
            persistence_mae_mean=("persistence_mae", "mean"),
            n_pools=("pool", "nunique"),
            win_rate=("beats_persistence", "mean"),
        )
        .replace([np.inf, -np.inf], np.nan)
    )
    process_scores["mase"] = process_scores["backtest_mae_h1_mean"] / process_scores["persistence_mae_mean"]
    process_scores["selection_score"] = process_scores["backtest_mae_h1_mean"]
    return (
        process_scores.sort_values(
            ["metric", "selection_score", "mae_mean", "win_rate", "method", "model"],
            ascending=[True, True, True, False, True, True],
            na_position="last",
        )
        .groupby("metric", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def registry_from_best_processes(forecast_results: pd.DataFrame, best_processes: pd.DataFrame) -> pd.DataFrame:
    """Return every pool/customer-class row belonging to the winning process."""
    candidates = select_best_models(forecast_results)
    registry_parts = []
    for process in best_processes.to_dict("records"):
        rows = candidates[
            (candidates["metric"] == process["metric"])
            & (candidates["method"] == process["method"])
            & (candidates["model"] == process["model"])
        ].copy()
        rows["process_mae_mean"] = process["mae_mean"]
        rows["process_backtest_mae_h1_mean"] = process["backtest_mae_h1_mean"]
        rows["process_mase"] = process["mase"]
        registry_parts.append(rows)
    if not registry_parts:
        return pd.DataFrame()
    registry = pd.concat(registry_parts, ignore_index=True)
    registry["promotion_rank"] = registry.groupby("metric").cumcount() + 1
    return registry.reset_index(drop=True)


def promote_top_models(
    selection_candidates: pd.DataFrame,
    group_cols: list[str] | tuple[str, ...] = ("metric",),
    top_n_per_group: int = 1,
) -> pd.DataFrame:
    promoted = selection_candidates.groupby(list(group_cols), as_index=False).head(top_n_per_group).copy()
    promoted["promotion_rank"] = promoted.groupby(list(group_cols)).cumcount() + 1
    return promoted.reset_index(drop=True)


def export_selected_models(registry: pd.DataFrame) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    exported_rows = []
    ts_cache: dict[str, pd.DataFrame] = {}

    for row in registry.to_dict("records"):
        method = row["method"]
        pool = row["pool"]
        metric = row["metric"]
        model_name = row["model"]
        if method not in ts_cache:
            ts_cache[method] = pmw.build_pool_timeseries(method)
        ts_full = ts_cache[method]
        series = ts_full[ts_full["pool"] == pool].sort_values("date").set_index("date")[metric]
        artifact_id = _safe_name(method, pool, metric, model_name)
        model_obj, state = fit_deployable_model(model_name, series)
        state.update(
            {
                "artifact_id": artifact_id,
                "method": method,
                "pool": pool,
                "metric": metric,
                "case": row.get("case"),
                "validation_mae": row.get("mae"),
                "validation_rmse": row.get("rmse"),
                "backtest_mae_h1": row.get("backtest_mae_h1"),
                "persistence_mae": row.get("persistence_mae"),
            }
        )
        metadata_path = METADATA_DIR / f"{artifact_id}.json"
        _write_json(metadata_path, state)

        artifact_path = None
        if model_obj is not None:
            artifact_path = ARTIFACT_DIR / f"{artifact_id}.pkl"
            with artifact_path.open("wb") as f:
                pickle.dump(model_obj, f)

        exported = dict(row)
        exported.update(
            {
                "artifact_id": artifact_id,
                "artifact_path": str(artifact_path) if artifact_path else "",
                "metadata_path": str(metadata_path),
                "train_start": state["train_start"],
                "train_end": state["train_end"],
                "n_observations": state["n_observations"],
                "export_note": state.get("export_note", ""),
            }
        )
        exported_rows.append(exported)

    exported_registry = pd.DataFrame(exported_rows)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    exported_registry.to_csv(REGISTRY_DIR / "best_model_registry.csv", index=False)
    return exported_registry
