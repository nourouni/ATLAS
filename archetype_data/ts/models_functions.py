import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels is noisy with short series
SEASONAL_PERIOD = 52  # weekly data with annual seasonality




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


CASE_FITTERS = {
    "trend_seasonal": fit_ets_damped,
    "trend_only": fit_ets_trend,
    "seasonal_only": fit_seasonal_naive,
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


def resolve_complex_mixed(series, n_origins=3, h=1, min_train=52):
    """
    Backtest all COMPLEX_MIXED_CANDIDATES, pick the lowest-MAE one.
    Flag for manual review if even the best candidate doesn't bea
    naive persistence (last value repeated) on the same folds.
    """
    persistence_mae, _ = walk_forward_mae(series, fit_naive_persistence, n_origins, h, min_train)

    results = {}
    for name, fn in COMPLEX_MIXED_CANDIDATES.items():
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



def fit_model(case, series, h=3, n_origins=3, min_train=52):
    """
    case   : str, one of the keys produced by classify_ts_case (+ "complex_mixed")
    series : 1D array-like, the full history for this (dataset, feature, cluster)
    h      : forecast horizon for the FINAL deployed forecast (independent of backtest h)

    Returns a dict with the forecast, which model was used, and backtest diagnostics.
    """
    series = np.asarray(series, dtype=float)

    if case == "complex_mixed":
        chosen, candidate_results, persistence_mae = resolve_complex_mixed(
            series, n_origins=n_origins, min_train=min_train
        )
        if chosen == "manual_review":
            return {
                "case": case,
                "model_used": "manual_review",
                "forecast": None,
                "candidate_backtest": candidate_results,
                "persistence_mae": persistence_mae,
                "note": "No candidate beat naive persistence on backtest; inspect series manually.",
            }
        fit_fn = COMPLEX_MIXED_CANDIDATES[chosen]
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
    }

def case_to_model_name(case):
    names = {
        "trend_seasonal": "ETS_damped",
        "trend_only": "ETS_trend",
        "seasonal_only": "seasonal_naive",
        "stationary_ar_structure": "AR_auto",
        "nonstationary_ar_structure": "ARIMA_0_1_0",
        "stationary_noise": "mean_naive",
        "heavy_tailed_noise": "median_naive",
    }
    return names.get(case, case)
