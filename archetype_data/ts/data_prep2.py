from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SEASONAL_PERIOD = 52


@dataclass
class PrepResult:
    series: np.ndarray
    ok: bool = True
    warnings: list = field(default_factory=list)
    flagged_outlier_idx: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Universal checks — run on every series regardless of case/model
# ---------------------------------------------------------------------------

def prep_common(series, index=None, expected_length=156, min_train=52, n_origins=3):
    """
    Checks that apply no matter which model will eventually be fit:
      - complete index, no gaps (if a datetime index is supplied)
      - minimum length for the backtest config you intend to use
      - sentinel-like zero/negative flag (informational, not auto-fixed)
      - outlier flag via IQR (informational, not auto-removed)

    Returns the series unchanged (as a clean float array) plus diagnostics.
    Does NOT drop or impute anything — that decision is left to you, since
    at n=36 a wrong auto-fix is often worse than the original issue.
    """
    warnings_list = []
    s = np.asarray(series, dtype=float)

    # --- length / backtest feasibility ---
    if len(s) < expected_length:
        warnings_list.append(
            f"Series has {len(s)} points, expected {expected_length}. "
            f"Confirm this isn't a truncated pull."
        )
    required_for_backtest = min_train + n_origins
    if len(s) < required_for_backtest:
        warnings_list.append(
            f"Series has {len(s)} points but walk-forward backtest needs >= "
            f"{required_for_backtest} (min_train={min_train} + n_origins={n_origins}). "
            f"Backtest will silently under-run or return NaN."
        )

    # --- missing values ---
    n_nan = int(np.isnan(s).sum())
    if n_nan > 0:
        warnings_list.append(
            f"{n_nan} NaN value(s) found. Most models here will fail or silently "
            f"misbehave with NaNs present — resolve before fitting."
        )

    # --- gap check against a supplied datetime index ---
    if index is not None:
        idx = pd.DatetimeIndex(index)
        if len(idx) != len(s):
            warnings_list.append("Index length doesn't match series length.")
        elif not idx.is_monotonic_increasing:
            warnings_list.append("Date index is not sorted ascending - sort by date before fitting.")
        else:
            day_gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
            if np.any(day_gaps > 7):
                missing_weeks = int(np.sum(day_gaps[day_gaps > 7] // 7 - 1))
                warnings_list.append(
                    f"{missing_weeks} weekly period(s) appear to be missing from "
                    f"the date index. Weekly alignment for seasonal_naive / "
                    f"ETS seasonal may be wrong."
                )
            if idx.has_duplicates:
                warnings_list.append("Duplicate weekly date(s) detected in the date index.")

    # --- sentinel zero/negative flag (informational only) ---
    zero_idx = np.where(s == 0)[0].tolist()
    neg_idx = np.where(s < 0)[0].tolist()
    if zero_idx:
        warnings_list.append(
            f"Zero values at index {zero_idx}. Confirm these are real "
            f"(e.g. genuine no-activity week), not a missing-value sentinel."
        )
    if neg_idx:
        warnings_list.append(
            f"Negative values at index {neg_idx}. Confirm this is plausible "
            f"for this feature (e.g. avg_balance overdraft) and not an error."
        )

    # --- outlier flag via IQR (flag only, never auto-removed) ---
    flagged = []
    if len(s) >= 8 and n_nan == 0:
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            flagged = np.where((s < lower) | (s > upper))[0].tolist()
            if flagged:
                warnings_list.append(
                    f"Possible outlier(s) at index {flagged} (value(s) "
                    f"{s[flagged].tolist()}). Flagged only — not removed. "
                    f"Eyeball these before trusting the case label or forecast."
                )

    return PrepResult(
        series=s,
        ok=(n_nan == 0 and len(s) >= required_for_backtest),
        warnings=warnings_list,
        flagged_outlier_idx=flagged,
    )


# ---------------------------------------------------------------------------
# Model-family-specific prep
# Each takes the ALREADY-common-checked series and does only what tha
# model family specifically needs. None of these mutate by default excep
# where noted — they mostly validate and warn.
# ---------------------------------------------------------------------------

def prep_for_naive(series):
    """mean_naive / median_naive: no special prep, these are robust by
    construction. Pass-through, kept for interface symmetry."""
    return PrepResult(series=np.asarray(series, dtype=float))


def prep_for_seasonal_naive(series, period=SEASONAL_PERIOD):
    """
    seasonal_naive needs a *clean phase alignment* — i.e. confirm there's
    at least one full period of history and that the series length is
    consistent with the period (no partial leading/trailing season tha
    would shift which week maps to which).
    """
    s = np.asarray(series, dtype=float)
    warnings_list = []
    if len(s) < period:
        warnings_list.append(
            f"Series shorter than one seasonal period ({period}); "
            f"seasonal_naive will fall back to last-value persistence."
        )
    elif len(s) % period != 0:
        warnings_list.append(
            f"Series length ({len(s)}) is not a multiple of the seasonal "
            f"period ({period}). This is fine for fitting, but double check "
            f"that index[0] actually corresponds to the week you think it does — "
            f"a phase-shifted start will make every seasonal lookup wrong."
        )
    return PrepResult(series=s, warnings=warnings_list)


def prep_for_ets(series, allow_multiplicative=False):
    """
    ETS_trend / ETS_damped: statsmodels' multiplicative seasonal/trend
    options require strictly positive values. Check for that, and flag
    near-zero variance (ETS optimizer can fail to converge on a flat series).
    """
    s = np.asarray(series, dtype=float)
    warnings_list = []

    if allow_multiplicative and np.any(s <= 0):
        warnings_list.append(
            "Series contains zero/negative values — multiplicative ETS "
            "components are invalid here. Use additive trend/seasonal only "
            "(this is already the default in fit_ets_trend/fit_ets_damped)."
        )

    std = np.std(s)
    if std < 1e-6:
        warnings_list.append(
            "Series variance is ~0 (effectively constant). ETS may fail to "
            "converge or return a degenerate fit — consider mean_naive instead."
        )

    return PrepResult(series=s, warnings=warnings_list)


def prep_for_ar(series):
    """
    AR_auto (stationary_ar_structure case): the case label already implies
    stationarity was tested upstream, so we don't re-run ADF/KPSS here.
    What we DO check: degenerate variance (AutoReg fits garbage on a
    constant series) and that there's enough length for at least AR(2)
    plus a few residual degrees of freedom.
    """
    s = np.asarray(series, dtype=float)
    warnings_list = []

    std = np.std(s)
    if std < 1e-6:
        warnings_list.append(
            "Series variance is ~0 — AR coefficients will be degenerate/unstable. "
            "Consider mean_naive instead of AR_auto for this series."
        )
    if len(s) < 8:
        warnings_list.append(
            "Series very short for AR(2) — coefficient estimates will be unstable. "
            "Consider forcing AR(1) only, or mean_naive."
        )

    return PrepResult(series=s, warnings=warnings_list)


def prep_for_arima_rw(series):
    """
    ARIMA(0,1,0) (nonstationary_ar_structure case): we're about to difference
    this series. The one thing worth checking is that it ISN'T already
    stationary — differencing a stationary series adds noise and can flip
    sign patterns in small samples. This is a lightweight ADF check, used
    only as a warning (the case label already made this decision upstream;
    this is a second opinion, not a re-classification).
    """
    s = np.asarray(series, dtype=float)
    warnings_list = []
    try:
        from statsmodels.tsa.stattools import adfuller
        pval = adfuller(s, autolag="AIC")[1]
        if pval < 0.05:
            warnings_list.append(
                f"ADF p-value={pval:.3f} suggests this series may already be "
                f"stationary. Differencing it for ARIMA(0,1,0) could be "
                f"over-differencing — consider AR_auto instead and re-check "
                f"the original case classification for this series."
            )
    except Exception:
        warnings_list.append("ADF re-check failed to run (non-fatal) — proceeding without it.")

    return PrepResult(series=s, warnings=warnings_list)


def prep_for_heavy_tailed(series):
    """
    heavy_tailed_noise (median_naive): outliers are EXPECTED here, not a bug.
    We still flag them via prep_common, but this function exists mainly so
    the pipeline doesn't accidentally suppress that signal — no additional
    cleaning is applied.
    """
    return PrepResult(series=np.asarray(series, dtype=float))


# ---------------------------------------------------------------------------
# Dispatch: case label -> the right model-specific prep function
# ---------------------------------------------------------------------------

PREP_DISPATCH = {
    "trend_seasonal": prep_for_ets,
    "trend_only": prep_for_ets,
    "seasonal_only": prep_for_seasonal_naive,
    "stationary_ar_structure": prep_for_ar,
    "nonstationary_ar_structure": prep_for_arima_rw,
    "stationary_noise": prep_for_naive,
    "heavy_tailed_noise": prep_for_heavy_tailed,
    # complex_mixed runs ALL of seasonal_naive / mean_naive / ets prep since
    # it tries all three candidates — handled separately, see prep_for_complex_mixed
}


def prep_for_complex_mixed(series):
    """
    complex_mixed tries seasonal_naive, mean_naive, and ets_trend candidates,
    so it needs whichever warnings any of those would raise, deduplicated.
    """
    s = np.asarray(series, dtype=float)
    all_warnings = []
    for fn in (prep_for_seasonal_naive, prep_for_naive, prep_for_ets):
        all_warnings.extend(fn(s).warnings)
    # dedupe while preserving order
    seen = set()
    deduped = [w for w in all_warnings if not (w in seen or seen.add(w))]
    return PrepResult(series=s, warnings=deduped)


PREP_DISPATCH["complex_mixed"] = prep_for_complex_mixed


def run_prep(case, series, index=None, expected_length=156, min_train=52, n_origins=3, verbose=True):
    """
    Convenience wrapper: runs prep_common, then the case-specific prep,
    merges warnings, and optionally prints them.
    """
    common = prep_common(series, index=index, expected_length=expected_length,
                          min_train=min_train, n_origins=n_origins)
    specific = PREP_DISPATCH[case](common.series)

    merged_warnings = common.warnings + specific.warnings
    if verbose and merged_warnings:
        print(f"[prep:{case}]")
        for w in merged_warnings:
            print(f"  - {w}")

    return PrepResult(
        series=specific.series,
        ok=common.ok,
        warnings=merged_warnings,
        flagged_outlier_idx=common.flagged_outlier_idx,
    )
