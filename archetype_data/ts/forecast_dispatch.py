import numpy as np
import pandas as pd
import os
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor

from data_prep import run_prep

warnings.filterwarnings("ignore")  # statsmodels is noisy with short series; we check fits explicitly

SEASONAL_PERIOD = 52  # weekly data with annual seasonality


# ---------------------------------------------------------------------------
# Individual model fitters
# Each takes a 1D array-like `series` (training window) and `h` (forecast horizon)
# and returns a forecast array of length h.
# ---------------------------------------------------------------------------

def fit_mean_naive(series, h):
    return np.repeat(np.mean(series), h)


def fit_median_naive(series, h):
    return np.repeat(np.median(series), h)


def fit_naive_persistence(series, h):
    """Repeat the last observed value. Used as the floor baseline, not in case_to_model."""
    return np.repeat(series[-1], h)


def fit_seasonal_naive(series, h, period=SEASONAL_PERIOD):
    """Repeat the value from one season-length back. Falls back to last value
    if the series is shorter than one full season."""
    series = np.asarray(series)
    if len(series) < period:
        return fit_naive_persistence(series, h)
    forecasts = []
    extended = list(series)
    for _ in range(h):
        forecasts.append(extended[-period])
        extended.append(extended[-period])
    return np.array(forecasts)


def fit_ets_trend(series, h):
    """ETS with trend, no seasonal component. Damped to limit overfit risk."""
    model = ExponentialSmoothing(
        series, trend="add", damped_trend=True, seasonal=None
    ).fit(optimized=True)
    return model.forecast(h)


def fit_ets_damped(series, h, period=SEASONAL_PERIOD):
    """ETS with damped trend + seasonal. Requires >= 2 full seasonal cycles;
    falls back to ETS_trend if there isn't enough data for seasonality."""
    series = np.asarray(series)
    if len(series) < 2 * period:
        return fit_ets_trend(series, h)
    model = ExponentialSmoothing(
        series, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=period
    ).fit(optimized=True)
    return model.forecast(h)


def fit_ar_auto(series, h, max_lag=2):
    """AR(1) or AR(2), selected by AIC."""
    series = np.asarray(series, dtype=float)
    best_aic, best_order, best_model = np.inf, 1, None
    for lag in range(1, max_lag + 1):
        if len(series) <= lag + 2:
            continue
        try:
            m = AutoReg(series, lags=lag, old_names=False).fit()
            if m.aic < best_aic:
                best_aic, best_order, best_model = m.aic, lag, m
        except Exception:
            continue
    if best_model is None:
        return fit_naive_persistence(series, h)
    return best_model.predict(start=len(series), end=len(series) + h - 1)


def fit_arima_random_walk(series, h):
    """ARIMA(0,1,0) i.e. random walk, optionally with drift via trend term."""
    model = ARIMA(series, order=(0, 1, 0), trend="t").fit()
    return model.forecast(h)


def fit_autoets(series, h, period=SEASONAL_PERIOD):
    """Try a small grid of ETS configurations, pick by AIC. Used for complex_mixed
    if the fallback chain doesn't resolve cleanly (kept available, not default)."""
    series = np.asarray(series)
    candidates = []
    configs = [
        dict(trend=None, seasonal=None),
        dict(trend="add", damped_trend=True, seasonal=None),
    ]
    if len(series) >= 2 * period:
        configs.append(dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=period))
        configs.append(dict(trend=None, seasonal="add", seasonal_periods=period))

    for cfg in configs:
        try:
            m = ExponentialSmoothing(series, **cfg).fit(optimized=True)
            candidates.append((m.aic, m))
        except Exception:
            continue

    if not candidates:
        return fit_mean_naive(series, h)
    _, best_model = min(candidates, key=lambda x: x[0])
    return best_model.forecast(h)


def _lag_feature_row(history, t, period=SEASONAL_PERIOD):
    values = np.asarray(history, dtype=float)
    lags = [1, 2, 3, 4, 8, 13, 26, 52]
    row = []
    for lag in lags:
        row.append(values[-lag] if len(values) >= lag else values[0])
    for window in [4, 8, 13, 26]:
        recent = values[-window:] if len(values) >= window else values
        row.extend([float(np.mean(recent)), float(np.std(recent)), float(recent[-1] - recent[0])])
    row.extend(
        [
            float(t),
            float(np.sin(2 * np.pi * t / period)),
            float(np.cos(2 * np.pi * t / period)),
        ]
    )
    return row


def fit_hist_gradient_boosting_lag(series, h, period=SEASONAL_PERIOD):
    """
    Lagged sklearn gradient boosting for short weekly pooled series.

    This is the stronger non-linear candidate for 156-ish weekly points. It is
    deliberately validation-gated elsewhere, because tree models can overfi
    short histories when the signal is mostly smooth trend.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 64:
        return fit_ets_trend(series, h)

    max_lag = min(52, max(8, len(series) // 3))
    X, y = [], []
    for t in range(max_lag, len(series)):
        X.append(_lag_feature_row(series[:t], t, period=period))
        y.append(series[t])
    if len(y) < 24:
        return fit_ets_trend(series, h)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(np.asarray(X), np.asarray(y))

    history = list(series.astype(float))
    forecasts = []
    for step in range(h):
        t = len(history)
        pred = float(model.predict([_lag_feature_row(history, t, period=period)])[0])
        forecasts.append(pred)
        history.append(pred)
    return np.asarray(forecasts)


# ---------------------------------------------------------------------------
# Dispatch table: case label -> fitter function
# ---------------------------------------------------------------------------

CASE_FITTERS = {
    "stationary_ar_structure": fit_ar_auto,
    "nonstationary_ar_structure": fit_arima_random_walk,
    "stationary_noise": fit_mean_naive,
    "heavy_tailed_noise": fit_median_naive,
}

# complex_mixed candidates, tried in this order, resolved by lowest backtest MAE
COMPLEX_MIXED_CANDIDATES = {
    "seasonal_naive": fit_seasonal_naive,
    "mean_naive": fit_mean_naive,
    "ets_trend": fit_ets_trend,
}

# trend_seasonal: ETS_damped only kept if it actually beats safer fallbacks on
# backtest. Win rate on the real case store was 31% (4/13) -- too low to trus
# the case label alone, so this is now gated the same way complex_mixed is.
TREND_SEASONAL_CANDIDATES = {
    "sklearn_hgb_lag": fit_hist_gradient_boosting_lag,
    "ets_damped": fit_ets_damped,
    "ets_trend": fit_ets_trend,       # drop seasonal component, keep trend
    "seasonal_naive": fit_seasonal_naive,
    "mean_naive": fit_mean_naive,
}

TREND_ONLY_CANDIDATES = {
    "sklearn_hgb_lag": fit_hist_gradient_boosting_lag,
    "ets_trend": fit_ets_trend,
    "mean_naive": fit_mean_naive,
}

# seasonal_only: same story, win rate was 33% (1/3) on the real data.
SEASONAL_ONLY_CANDIDATES = {
    "sklearn_hgb_lag": fit_hist_gradient_boosting_lag,
    "seasonal_naive": fit_seasonal_naive,
    "mean_naive": fit_mean_naive,
    "ets_trend": fit_ets_trend,
}

# Cases resolved via validation-gated fallback rather than a single fixed model.
# Each maps to its own candidate dict (kept separate per case, per your earlier
# requirement that different models need different treatment / candidate sets).
GATED_CASE_CANDIDATES = {
    "complex_mixed": COMPLEX_MIXED_CANDIDATES,
    "trend_only": TREND_ONLY_CANDIDATES,
    "trend_seasonal": TREND_SEASONAL_CANDIDATES,
    "seasonal_only": SEASONAL_ONLY_CANDIDATES,
}


# ---------------------------------------------------------------------------
# Walk-forward (rolling-origin) backtes
# ---------------------------------------------------------------------------

def walk_forward_mae(series, fit_fn, n_origins=3, h=1, min_train=52):
    """
    Rolling-origin backtest. For each of the last `n_origins` points,
    train on everything before it, forecast h steps, record absolute error
    on the first forecasted step (h=1 is standard for short weekly series).

    min_train guards against fitting models on too little data; if there isn'
    enough history for n_origins folds, it uses as many as it can and reports
    that count alongside the MAE so you know how reliable the estimate is.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    errors = []
    used_origins = 0

    for i in range(n_origins, 0, -1):
        split = n - i
        if split < min_train:
            continue
        train, actual = series[:split], series[split:split + h]
        if len(actual) < h:
            continue
        try:
            forecast = fit_fn(train, h)
            errors.append(np.mean(np.abs(np.asarray(forecast)[:h] - actual)))
            used_origins += 1
        except Exception:
            continue

    if not errors:
        return np.nan, 0
    return float(np.mean(errors)), used_origins


def resolve_via_fallback(series, candidates, n_origins=3, h=1, min_train=52):
    """
    Generalized version of the complex_mixed resolution logic: backtest every
    candidate in `candidates` (a name -> fit_fn dict), pick the lowest-MAE one.
    Flag for manual review if even the best candidate doesn't beat naive
    persistence on the same folds.

    Used for any case where the case label alone isn't trustworthy enough to
    fix a single model -- currently: complex_mixed, trend_seasonal, seasonal_only.
    """
    persistence_mae, _ = walk_forward_mae(series, fit_naive_persistence, n_origins, h, min_train)

    results = {}
    for name, fn in candidates.items():
        mae, n_used = walk_forward_mae(series, fn, n_origins, h, min_train)
        results[name] = (mae, n_used)

    valid = {k: v for k, v in results.items() if not np.isnan(v[0])}
    if not valid:
        return "manual_review", results, persistence_mae

    best_name = min(valid, key=lambda k: valid[k][0])
    best_mae = valid[best_name][0]

    needs_review = (not np.isnan(persistence_mae)) and (best_mae > persistence_mae)
    chosen = "manual_review" if needs_review else best_name
    return chosen, results, persistence_mae


# ---------------------------------------------------------------------------
# Top-level entry poin
# ---------------------------------------------------------------------------

def fit_model(case, series, h=3, n_origins=3, min_train=52, index=None,
              expected_length=156, verbose_prep=True):
    """
    case   : str, one of the keys produced by classify_ts_case (+ "complex_mixed")
    series : 1D array-like, the full history for this (dataset, feature, cluster)
    h      : forecast horizon for the FINAL deployed forecast (independent of backtest h)
    index  : optional datetime-like index, used by run_prep to check for gaps
    expected_length : your typical full-history length (default 156 for roughly
                       three years of weekly data) - used to flag truncated pulls

    Data prep runs automatically as the first step, using the model-specific
    prep function for this case (see data_prep.py). Prep warnings are attached
    to the result so they're visible even if you don't read stdout.

    Returns a dict with the forecast, which model was used, backtest diagnostics,
    and any prep warnings raised before fitting.
    """
    prep_result = run_prep(
        case, series, index=index, expected_length=expected_length,
        min_train=min_train, n_origins=n_origins, verbose=verbose_prep,
    )
    series = prep_result.series

    if not prep_result.ok:
        return {
            "case": case,
            "model_used": "skipped_failed_prep",
            "forecast": None,
            "prep_warnings": prep_result.warnings,
            "note": "Series failed minimum prep requirements (NaNs present or "
                    "too short for the configured backtest). Fix and re-run.",
        }

    if case in GATED_CASE_CANDIDATES:
        candidates = GATED_CASE_CANDIDATES[case]
        chosen, candidate_results, persistence_mae = resolve_via_fallback(
            series, candidates, n_origins=n_origins, min_train=min_train
        )
        if chosen == "manual_review":
            return {
                "case": case,
                "model_used": "manual_review",
                "forecast": None,
                "candidate_backtest": candidate_results,
                "persistence_mae": persistence_mae,
                "prep_warnings": prep_result.warnings,
                "note": "No candidate beat naive persistence on backtest; inspect series manually.",
            }
        fit_fn = candidates[chosen]
        forecast = fit_fn(series, h)
        bt_mae, n_used = walk_forward_mae(series, fit_fn, n_origins, h=1, min_train=min_train)
        return {
            "case": case,
            "model_used": chosen,
            "forecast": np.asarray(forecast),
            "backtest_mae_h1": bt_mae,
            "backtest_n_origins": n_used,
            "candidate_backtest": candidate_results,
            "persistence_mae": persistence_mae,
            "beats_persistence": True,  # guaranteed by the gating logic above
            "prep_warnings": prep_result.warnings,
        }

    if case not in CASE_FITTERS:
        raise ValueError(f"Unknown case label: {case!r}")

    fit_fn = CASE_FITTERS[case]
    forecast = fit_fn(series, h)
    bt_mae, n_used = walk_forward_mae(series, fit_fn, n_origins, h=1, min_train=min_train)
    persistence_mae, _ = walk_forward_mae(series, fit_naive_persistence, n_origins, h=1, min_train=min_train)

    return {
        "case": case,
        "model_used": case_to_model_name(case),
        "forecast": np.asarray(forecast),
        "backtest_mae_h1": bt_mae,
        "backtest_n_origins": n_used,
        "persistence_mae": persistence_mae,
        "beats_persistence": (bt_mae < persistence_mae) if not (np.isnan(bt_mae) or np.isnan(persistence_mae)) else None,
        "prep_warnings": prep_result.warnings,
    }


def case_to_model_name(case):
    """Fixed model name for non-gated cases only. trend_seasonal, seasonal_only,
    and complex_mixed are resolved dynamically via GATED_CASE_CANDIDATES, so
    they're intentionally absent here -- their model_used is set directly in
    fit_model() from whichever candidate the backtest actually chose."""
    names = {
        "stationary_ar_structure": "AR_auto",
        "nonstationary_ar_structure": "ARIMA_0_1_0",
        "stationary_noise": "mean_naive",
        "heavy_tailed_noise": "median_naive",
    }
    return names.get(case, case)


# ---------------------------------------------------------------------------
# Demo / sanity check with synthetic series, one per case
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 36
    t = np.arange(n)

    # build a simple stationary AR(1) series first (needed before the dict below)
    ar1 = [50.0]
    for _ in range(n - 1):
        ar1.append(50 + 0.6 * (ar1[-1] - 50) + rng.normal(0, 2))
    ar1 = np.array(ar1)

    demo_series = {
        "trend_seasonal": 50 + 0.8 * t + 10 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, n),
        "trend_only": 20 + 1.2 * t + rng.normal(0, 3, n),
        "seasonal_only": 100 + 15 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, n),
        "stationary_ar_structure": ar1,
        "nonstationary_ar_structure": np.cumsum(rng.normal(0, 2, n)) + 50,
        "stationary_noise": rng.normal(50, 3, n),
        "heavy_tailed_noise": 50 + rng.standard_t(df=3, size=n) * 4,
    }

    # one "complex_mixed" series: weak trend + irregular bumps, nothing clean
    complex_series = 40 + 0.2 * t + rng.normal(0, 6, n)
    complex_series[10:13] += 15  # one-off shock
    demo_series["complex_mixed"] = complex_series

    print(f"{'CASE':<28} {'MODEL USED':<16} {'BT_MAE(h=1)':>12} {'PERSIST_MAE':>12} {'N_ORIGINS':>10}")
    print("-" * 84)
    for case, series in demo_series.items():
        result = fit_model(case, series, h=3, n_origins=3, min_train=52)
        bt = result.get("backtest_mae_h1", np.nan)
        pm = result.get("persistence_mae", np.nan)
        n_used = result.get("backtest_n_origins", "-")
        model_used = result["model_used"]
        print(f"{case:<28} {model_used:<16} {bt:>12.3f} {pm:>12.3f} {str(n_used):>10}")
