import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL


SEASONAL_STRENGTH_THRESHOLD = 0.6
SEASONAL_PERIODS = {
    "monthly_like": 4,
    "quarterly": 13,
    "semiannual": 26,
    "annual": 52,
}


# ---------------------------------------------------------------------------
# STATIONARITY TESTS
# ---------------------------------------------------------------------------

def adf_test(ts: pd.Series, alpha: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for a unit root.
    H0: series has a unit root (non-stationary).
    is_stationary = True if we reject H0 (pvalue < alpha).
    """
    result = adfuller(ts.dropna(), autolag="AIC")
    statistic, pvalue = result[0], result[1]

    return {
        "test": "ADF",
        "statistic": statistic,
        "pvalue": pvalue,
        "is_stationary": pvalue < alpha,
    }


def kpss_test(ts: pd.Series, alpha: float = 0.05) -> dict:
    """
    KPSS test for stationarity around a level.
    H0: series is stationary.
    is_stationary = True if we fail to reject H0 (pvalue > alpha).
    """
    statistic, pvalue, _, _ = kpss(ts.dropna(), regression="c", nlags="auto")

    return {
        "test": "KPSS",
        "statistic": statistic,
        "pvalue": pvalue,
        "is_stationary": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# AUTOCORRELATION TESTS
# ---------------------------------------------------------------------------

def ljung_box_test(ts: pd.Series, lags: int = 52, alpha: float = 0.05) -> dict:
    """
    Ljung-Box test for autocorrelation up to `lags`.
    H0: no autocorrelation up to the given lag.
    no_autocorrelation = True if we fail to reject H0 (pvalue > alpha)
    at the last lag tested.
    """
    result = acorr_ljungbox(ts.dropna(), lags=[lags], return_df=True)
    statistic = result["lb_stat"].iloc[-1]
    pvalue = result["lb_pvalue"].iloc[-1]

    return {
        "test": "Ljung-Box",
        "lags": lags,
        "statistic": statistic,
        "pvalue": pvalue,
        "no_autocorrelation": pvalue > alpha,
    }


def _lag_window_hits(significant_lags: list[int]) -> dict:
    windows = {
        "monthly_like": [4],
        "quarterly": [13],
        "semiannual": [26],
        "annual": [52],
    }
    hits = {}
    for name, centers in windows.items():
        window = set()
        for center in centers:
            window.update(range(max(1, center - 1), center + 2))
        hits[name] = sorted(lag for lag in significant_lags if lag in window)
    return hits


def _false_positive_probability(alpha: float, n_tests: int) -> float:
    if n_tests <= 0:
        return 0.0
    return 1 - (1 - alpha) ** n_tests


def acf_significance(ts: pd.Series, nlags: int = 52, alpha: float = 0.05) -> dict:
    """
    Checks which ACF lags fall outside the approximate confidence band.
    significant_lags: list of lag numbers (1-indexed) that are significant.
    has_autocorrelation = True if any lag is significant.

    Because each lag is checked separately, the expected chance of at least one
    false positive under pure noise is also returned. Use this together with
    Ljung-Box rather than treating one isolated lag as decisive.
    """
    values, confint = acf(ts.dropna(), nlags=nlags, alpha=alpha, fft=True)

    significant_lags = []
    for lag in range(1, nlags + 1):
        lower, upper = confint[lag]
        # significant if the confidence interval does not contain 0
        if lower > 0 or upper < 0:
            significant_lags.append(lag)

    return {
        "test": "ACF significance",
        "nlags": nlags,
        "significant_lags": significant_lags,
        "n_significant_lags": len(significant_lags),
        "lag_window_hits": _lag_window_hits(significant_lags),
        "multiple_lag_false_positive_prob": _false_positive_probability(alpha, nlags),
        "has_autocorrelation": len(significant_lags) > 0,
    }


def pacf_significance(ts: pd.Series, nlags: int = 52, alpha: float = 0.05) -> dict:
    """
    Checks which PACF lags fall outside the approximate confidence band.
    significant_lags: list of lag numbers (1-indexed) that are significant.
    has_partial_autocorrelation = True if any lag is significant.

    Because each lag is checked separately, the expected chance of at least one
    false positive under pure noise is also returned. Use this together with
    Ljung-Box rather than treating one isolated lag as decisive.
    """
    values, confint = pacf(ts.dropna(), nlags=nlags, alpha=alpha, method="ywm")

    significant_lags = []
    for lag in range(1, nlags + 1):
        lower, upper = confint[lag]
        if lower > 0 or upper < 0:
            significant_lags.append(lag)

    return {
        "test": "PACF significance",
        "nlags": nlags,
        "significant_lags": significant_lags,
        "n_significant_lags": len(significant_lags),
        "lag_window_hits": _lag_window_hits(significant_lags),
        "multiple_lag_false_positive_prob": _false_positive_probability(alpha, nlags),
        "has_partial_autocorrelation": len(significant_lags) > 0,
    }


# ---------------------------------------------------------------------------
# NORMALITY TESTS
# ---------------------------------------------------------------------------

def shapiro_test(ts: pd.Series, alpha: float = 0.05) -> dict:
    """
    Shapiro-Wilk test for normality.
    H0: data is normally distributed.
    is_normal = True if we fail to reject H0 (pvalue > alpha).
    """
    statistic, pvalue = stats.shapiro(ts.dropna())

    return {
        "test": "Shapiro-Wilk",
        "statistic": statistic,
        "pvalue": pvalue,
        "is_normal": pvalue > alpha,
    }


def jarque_bera_test(ts: pd.Series, alpha: float = 0.05) -> dict:
    """
    Jarque-Bera test for normality (based on skewness and kurtosis).
    H0: data is normally distributed.
    is_normal = True if we fail to reject H0 (pvalue > alpha).
    """
    statistic, pvalue = stats.jarque_bera(ts.dropna())

    return {
        "test": "Jarque-Bera",
        "statistic": statistic,
        "pvalue": pvalue,
        "is_normal": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# TREND TEST
# ---------------------------------------------------------------------------

def linear_trend_test(ts: pd.Series, alpha: float = 0.05) -> dict:
    """
    Simple linear trend test using scipy.stats.linregress against a
    time index (0, 1, 2, ..., n-1).
    trend_exists = True if the slope is significant (pvalue < alpha).
    """
    ts = ts.dropna()
    x = np.arange(len(ts))
    y = ts.values

    result = stats.linregress(x, y)

    return {
        "test": "Linear trend (OLS)",
        "slope": result.slope,
        "intercept": result.intercept,
        "rvalue": result.rvalue,
        "pvalue": result.pvalue,
        "stderr": result.stderr,
        "trend_exists": result.pvalue < alpha,
    }


# ---------------------------------------------------------------------------
# SEASONALITY / DECOMPOSITION METRICS (STL based)
# ---------------------------------------------------------------------------

def seasonal_strength_stl(
    ts: pd.Series,
    period: int = 52,
    robust: bool = True,
    threshold: float = SEASONAL_STRENGTH_THRESHOLD,
) -> dict:
    """
    Seasonal strength based on STL decomposition, following the
    Hyndman & Athanasopoulos definition:

        F_seasonal = max(0, 1 - Var(residual) / Var(seasonal + residual))

    has_strong_seasonality = True if seasonal_strength > threshold. The
    threshold is a practical rule, not a universal boundary, so the numeric
    strength is always returned and should be interpreted continuously.
    """
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    resid_var = np.var(stl_result.resid)
    seasonal_resid_var = np.var(stl_result.seasonal + stl_result.resid)

    strength = max(0.0, 1 - resid_var / seasonal_resid_var)

    return {
        "test": "STL seasonal strength",
        "period": period,
        "threshold": threshold,
        "seasonal_strength": strength,
        "has_strong_seasonality": strength > threshold,
    }


def seasonal_strength_multi(
    ts: pd.Series,
    periods: dict[str, int] | None = None,
    robust: bool = True,
    threshold: float = SEASONAL_STRENGTH_THRESHOLD,
) -> dict:
    """
    Calculates STL seasonal strength for multiple weekly periods:
    monthly-like (~4 weeks), quarterly (~13), semiannual (~26), and annual
    (~52) by default.
    """
    periods = periods or SEASONAL_PERIODS
    strengths = {}
    flags = {}
    for name, period in periods.items():
        if len(ts.dropna()) < max(8, period * 2):
            strengths[name] = np.nan
            flags[name] = False
            continue
        result = seasonal_strength_stl(ts, period=period, robust=robust, threshold=threshold)
        strengths[name] = result["seasonal_strength"]
        flags[name] = result["has_strong_seasonality"]

    valid_strengths = {k: v for k, v in strengths.items() if not pd.isna(v)}
    strongest_name = max(valid_strengths, key=valid_strengths.get) if valid_strengths else None

    return {
        "test": "STL multi-period seasonal strength",
        "periods": periods,
        "strengths": strengths,
        "flags": flags,
        "threshold": threshold,
        "strongest_period_name": strongest_name,
        "strongest_period": periods.get(strongest_name) if strongest_name else np.nan,
        "strongest_strength": valid_strengths.get(strongest_name, np.nan) if strongest_name else np.nan,
        "has_any_strong_seasonality": any(flags.values()),
    }


def trend_strength_stl(
    ts: pd.Series,
    period: int = 52,
    robust: bool = True,
    threshold: float = SEASONAL_STRENGTH_THRESHOLD,
) -> dict:
    """
    Trend strength based on STL decomposition:

        F_trend = max(0, 1 - Var(residual) / Var(trend + residual))

    has_strong_trend = True if trend_strength > threshold. The threshold is a
    practical rule, not a universal boundary, so the numeric strength is always
    returned and should be interpreted continuously.
    """
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    resid_var = np.var(stl_result.resid)
    trend_resid_var = np.var(stl_result.trend + stl_result.resid)

    strength = max(0.0, 1 - resid_var / trend_resid_var)

    return {
        "test": "STL trend strength",
        "period": period,
        "threshold": threshold,
        "trend_strength": strength,
        "has_strong_trend": strength > threshold,
    }


def movable_holiday_effect_placeholders(ts: pd.Series) -> dict:
    """
    Records that movable holidays need event-calendar regressors rather than
    fixed Gregorian week STL. This intentionally avoids pretending Ramadan or
    Eid recur at the same ISO week.
    """
    index = ts.dropna().index
    return {
        "test": "Movable holiday effect placeholders",
        "requires_event_calendar": True,
        "event_columns_recommended": [
            "is_ramadan_week",
            "weeks_to_ramadan_start",
            "is_eid_al_fitr_week",
            "is_eid_al_adha_week",
            "is_public_holiday_week",
        ],
        "note": (
            "Ramadan and Eid shift across Gregorian weeks; quantify them with "
            "dated event regressors before treating them as ordinary 52-week seasonality."
        ),
        "start": index.min() if len(index) else pd.NaT,
        "end": index.max() if len(index) else pd.NaT,
    }


def seasonal_profile(ts: pd.Series, period: int = 52, robust: bool = True) -> dict:
    """
    Extracts the average seasonal effect per ISO week from the STL seasonal
    component, and identifies the strongest (highest and lowest) weeks.
    """
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()
    seasonal = stl_result.seasonal

    weekly_avg = seasonal.groupby(seasonal.index.isocalendar().week).mean()

    strongest_high_week = int(weekly_avg.idxmax())
    strongest_low_week = int(weekly_avg.idxmin())

    return {
        "test": "STL seasonal profile",
        "weekly_seasonal_effect": weekly_avg.to_dict(),
        "strongest_high_week": strongest_high_week,
        "strongest_low_week": strongest_low_week,
    }


# ---------------------------------------------------------------------------
# RESIDUAL DIAGNOSTICS (STL based)
# ---------------------------------------------------------------------------

def residual_diagnostics(ts: pd.Series, period: int = 52, robust: bool = True,
                          lb_lags: int = 52, alpha: float = 0.05) -> dict:
    """
    Runs STL decomposition and checks the residual component for
    white noise (Ljung-Box) and normality (Shapiro-Wilk).
    """
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()
    residuals = stl_result.resid

    lb_result = acorr_ljungbox(residuals, lags=[lb_lags], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].iloc[-1]

    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)

    return {
        "test": "STL residual diagnostics",
        "residuals": residuals,
        "ljung_box_pvalue": lb_pvalue,
        "shapiro_pvalue": shapiro_pvalue,
        "residuals_are_white_noise": lb_pvalue > alpha,
        "residuals_are_normal": shapiro_pvalue > alpha,
    }
