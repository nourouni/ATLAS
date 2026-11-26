"""
plotting.py

Reusable plotting functions for weekly time series (~3 years typical).
Each function takes a pandas Series with a DatetimeIndex, makes one plot,
saves it as a PNG to save_dir, and returns the matplotlib figure.

No pipelines, no orchestration. Call whatever you need directly
in the notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf
from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

sns.set_theme(style="whitegrid")


def _ensure_dir(save_dir) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _save(fig, save_dir, filename: str):
    save_dir = _ensure_dir(save_dir)
    fig.savefig(save_dir / filename, dpi=300, bbox_inches="tight")


def _safe_lags(ts: pd.Series, requested: int, pacf: bool = False) -> int:
    n = len(ts.dropna())
    upper = max(1, n // 2 - 1) if pacf else max(1, n - 1)
    return max(1, min(requested, upper))


# ---------------------------------------------------------------------------
# BASIC PLOTS
# ---------------------------------------------------------------------------

def plot_time_series(ts: pd.Series, save_dir, title: str = None):
    """Simple line plot of the raw series."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values, marker="o", markersize=3)
    ax.set_title(title or "Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.tight_layout()

    _save(fig, save_dir, "time_series.png")
    return fig


def plot_rolling_stats(ts: pd.Series, save_dir, title: str = None, window: int = 52):
    """Rolling mean and rolling std on top of the raw series."""
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts.values, label="Original", alpha=0.5)
    ax.plot(rolling_mean.index, rolling_mean.values, label=f"Rolling mean ({window})")
    ax.plot(rolling_std.index, rolling_std.values, label=f"Rolling std ({window})")
    ax.set_title(title or "Rolling Mean & Std")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    _save(fig, save_dir, "rolling_stats.png")
    return fig


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def plot_acf(ts: pd.Series, save_dir, title: str = None, lags: int = 52):
    """ACF plot using statsmodels."""
    lags = _safe_lags(ts, lags)
    fig, ax = plt.subplots(figsize=(8, 4))
    sm_plot_acf(ts.dropna(), lags=lags, ax=ax)
    ax.set_title(title or "Autocorrelation (ACF)")
    fig.tight_layout()

    _save(fig, save_dir, "acf.png")
    return fig


def plot_pacf(ts: pd.Series, save_dir, title: str = None, lags: int = 52):
    """PACF plot using statsmodels."""
    lags = _safe_lags(ts, lags, pacf=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sm_plot_pacf(ts.dropna(), lags=lags, ax=ax, method="ywm")
    ax.set_title(title or "Partial Autocorrelation (PACF)")
    fig.tight_layout()

    _save(fig, save_dir, "pacf.png")
    return fig


# ---------------------------------------------------------------------------
# DISTRIBUTIONS
# ---------------------------------------------------------------------------

def plot_distribution(ts: pd.Series, save_dir, title: str = None):
    """Histogram with KDE overlay."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(ts.dropna(), kde=True, ax=ax)
    ax.set_title(title or "Distribution")
    ax.set_xlabel("Value")
    fig.tight_layout()

    _save(fig, save_dir, "distribution.png")
    return fig


def plot_boxplot(ts: pd.Series, save_dir, title: str = None):
    """Simple boxplot of the series values."""
    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(y=ts.dropna(), ax=ax)
    ax.set_title(title or "Boxplot")
    ax.set_ylabel("Value")
    fig.tight_layout()

    _save(fig, save_dir, "boxplot.png")
    return fig


# ---------------------------------------------------------------------------
# STL PLOTS
# ---------------------------------------------------------------------------

def plot_stl(ts: pd.Series, save_dir, title: str = None, period: int = 52, robust: bool = True):
    """STL decomposition: observed, trend, seasonal, residual in subplots."""
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(ts.dropna().index, ts.dropna().values)
    axes[0].set_ylabel("Observed")

    axes[1].plot(stl_result.trend.index, stl_result.trend.values)
    axes[1].set_ylabel("Trend")

    axes[2].plot(stl_result.seasonal.index, stl_result.seasonal.values)
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(stl_result.resid.index, stl_result.resid.values)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    fig.suptitle(title or "STL Decomposition")
    fig.tight_layout()

    _save(fig, save_dir, "stl_decomposition.png")
    return fig


def plot_stl_components_separately(ts: pd.Series, save_dir, title: str = None,
                                    period: int = 52, robust: bool = True):
    """
    STL components plotted as one figure per componen
    (trend, seasonal, residual), returned together as a dict of figures.
    Saved separately as trend.png, seasonal.png, residual.png.
    """
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    figs = {}

    fig_trend, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(stl_result.trend.index, stl_result.trend.values, color="tab:blue")
    ax.set_title(title or "STL Trend Component")
    fig_trend.tight_layout()
    _save(fig_trend, save_dir, "stl_trend.png")
    figs["trend"] = fig_trend

    fig_seasonal, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(stl_result.seasonal.index, stl_result.seasonal.values, color="tab:green")
    ax.set_title(title or "STL Seasonal Component")
    fig_seasonal.tight_layout()
    _save(fig_seasonal, save_dir, "stl_seasonal.png")
    figs["seasonal"] = fig_seasonal

    fig_resid, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(stl_result.resid.index, stl_result.resid.values, color="tab:red")
    ax.set_title(title or "STL Residual Component")
    fig_resid.tight_layout()
    _save(fig_resid, save_dir, "stl_residual.png")
    figs["residual"] = fig_resid

    return figs


# ---------------------------------------------------------------------------
# SEASONALITY VISUALIZATION
# ---------------------------------------------------------------------------

def plot_weekly_seasonality(ts: pd.Series, save_dir, title: str = None):
    """Boxplot of values grouped by ISO week of year."""
    df = ts.dropna().to_frame(name="value")
    df["week"] = df.index.isocalendar().week.astype(int)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x="week", y="value", data=df, ax=ax)
    ax.set_title(title or "Weekly Seasonality")
    ax.set_xlabel("ISO week of year")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    fig.tight_layout()

    _save(fig, save_dir, "weekly_seasonality.png")
    return fig


def plot_monthly_seasonality(ts: pd.Series, save_dir, title: str = None):
    """Compatibility alias; weekly TS should use ISO-week seasonality."""
    return plot_weekly_seasonality(ts, save_dir, title=title)


# ---------------------------------------------------------------------------
# RESIDUAL ANALYSIS
# ---------------------------------------------------------------------------

def plot_residual_acf(ts: pd.Series, save_dir, title: str = None,
                       period: int = 52, robust: bool = True, lags: int = 52):
    """ACF plot of the STL residual component."""
    lags = _safe_lags(ts, lags)
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    fig, ax = plt.subplots(figsize=(8, 4))
    sm_plot_acf(stl_result.resid, lags=lags, ax=ax)
    ax.set_title(title or "Residual ACF")
    fig.tight_layout()

    _save(fig, save_dir, "residual_acf.png")
    return fig


def plot_residual_distribution(ts: pd.Series, save_dir, title: str = None,
                                period: int = 52, robust: bool = True):
    """Histogram + KDE of the STL residual component."""
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(stl_result.resid, kde=True, ax=ax)
    ax.set_title(title or "Residual Distribution")
    ax.set_xlabel("Residual")
    fig.tight_layout()

    _save(fig, save_dir, "residual_distribution.png")
    return fig


def plot_qq_plot(ts: pd.Series, save_dir, title: str = None,
                  period: int = 52, robust: bool = True):
    """QQ plot of the STL residual component against a normal distribution."""
    stl_result = STL(ts.dropna(), period=period, robust=robust).fit()

    fig = sm.qqplot(stl_result.resid, line="45", fit=True)
    fig.set_size_inches(6, 6)
    fig.axes[0].set_title(title or "QQ Plot of Residuals")
    fig.tight_layout()

    _save(fig, save_dir, "qq_plot.png")
    return fig


# ---------------------------------------------------------------------------
# TREND VISUALIZATION
# ---------------------------------------------------------------------------

def plot_trend_regression(ts: pd.Series, save_dir, title: str = None):
    """Scatter of the series with a fitted OLS linear trend line."""
    ts_clean = ts.dropna()
    x = np.arange(len(ts_clean))
    y = ts_clean.values

    result = stats.linregress(x, y)
    fitted = result.intercept + result.slope * x

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_clean.index, y, marker="o", markersize=3, label="Observed", alpha=0.7)
    ax.plot(ts_clean.index, fitted, color="tab:red",
            label=f"Trend (slope={result.slope:.3f}, p={result.pvalue:.3f})")
    ax.set_title(title or "Linear Trend Regression")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    _save(fig, save_dir, "trend_regression.png")
    return fig


# ---------------------------------------------------------------------------
# THESIS QUALITY PLOTS
# ---------------------------------------------------------------------------

def plot_diagnostic_summary(ts: pd.Series, save_dir, title: str = None,
                             period: int = 52, robust: bool = True, lags: int = 52):
    """
    Single combined figure for a quick visual overview:
    raw series, ACF, PACF, and distribution.
    Purely a static visualization, no test logic or automation.
    """
    ts_clean = ts.dropna()
    acf_lags = _safe_lags(ts_clean, lags)
    pacf_lags = _safe_lags(ts_clean, lags, pacf=True)
    stl_result = STL(ts_clean, period=period, robust=robust).fit()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ts_clean.index, ts_clean.values, marker="o", markersize=3)
    axes[0, 0].set_title("Time Series")

    sm_plot_acf(ts_clean, lags=acf_lags, ax=axes[0, 1])
    axes[0, 1].set_title("ACF")

    sm_plot_pacf(ts_clean, lags=pacf_lags, ax=axes[1, 0], method="ywm")
    axes[1, 0].set_title("PACF")

    sns.histplot(ts_clean, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Distribution")

    fig.suptitle(title or "Diagnostic Summary")
    fig.tight_layout()

    _save(fig, save_dir, "diagnostic_summary.png")
    return fig
