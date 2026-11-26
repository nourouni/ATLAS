"""
Customer engagement propensity from weekly RFM panels.

Reusable modeling logic lives here; the notebook and CLI orchestrate calls and
display saved outputs. Predictors are restricted to cutoff-available fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pl.Config.set_tbl_formatting("ASCII_FULL")

BASE_DIR = Path(__file__).resolve().parents[1]
PANEL_DIR = BASE_DIR / "temporal clustering" / "data" / "panels"
FEATURE_DIR = BASE_DIR / "temporal clustering" / "data" / "features"
OUTPUT_DIR = BASE_DIR / "propensity" / "outputs"

PANEL_PATHS = {
    "all": PANEL_DIR / "weekly_rfm_panel.parquet",
    "retail": PANEL_DIR / "weekly_rfm_panel_retail.parquet",
    "enterprise": PANEL_DIR / "weekly_rfm_panel_enterprise.parquet",
}
TEMPORAL_FEATURE_PATHS = {
    "retail": FEATURE_DIR / "temporal_features_retail.parquet",
    "enterprise": FEATURE_DIR / "temporal_features_enterprise.parquet",
}

TARGET_A_COLUMN = "target_frequency_growth"
TARGET_B_COLUMN = "target_composite_engagement_improves"
TARGET_COLUMN = TARGET_A_COLUMN
TARGET_COLUMNS = (TARGET_A_COLUMN, TARGET_B_COLUMN)
SUSPICIOUS_TERMS = ("future", "lead", "target", "next", "forward", "horizon")


@dataclass(frozen=True)
class SchemaConfig:
    client_id: str = "client_id"
    cutoff_index: str = "week_idx"
    cutoff_start: str = "week_start"
    cutoff_end: str = "week_end"
    segment: str = "segment"
    activity: str = "period_frequency"
    current_frequency: str = "Frequency"
    current_recency: str = "Recency"
    current_monetary: str = "Monetary"


@dataclass(frozen=True)
class PropensityConfig:
    segment: str = "retail"
    lookback_weeks: int = 8
    horizon_weeks: int = 8
    min_relative_growth: float = 0.20
    min_absolute_growth: float = 3.0
    min_future_activity_when_past_zero: float = 3.0
    composite_min_relative_improvement: float = 0.10
    composite_min_absolute_improvement: float = 0.25
    min_precision: float = 0.50
    validation_size: float = 0.20
    test_size: float = 0.20
    random_state: int = 42
    max_clients: int | None = 20_000
    include_static_temporal_features: bool = False
    strict_client_holdout: bool = False
    schema: SchemaConfig = SchemaConfig()


def panel_path(segment: str) -> Path:
    if segment not in PANEL_PATHS:
        raise ValueError(f"Unknown segment {segment!r}. Expected one of: {sorted(PANEL_PATHS)}")
    return PANEL_PATHS[segment]


def temporal_feature_path(segment: str) -> Path | None:
    return TEMPORAL_FEATURE_PATHS.get(segment)


def inspect_parquet_schema(path: str | Path) -> pl.DataFrame:
    schema = pl.scan_parquet(path).collect_schema()
    return pl.DataFrame({"column": list(schema.keys()), "dtype": [str(dtype) for dtype in schema.values()]})


def _existing_columns(path: Path) -> list[str]:
    return list(pl.scan_parquet(path).collect_schema().keys())


def _require_columns(columns: list[str], required: list[str], path: Path) -> None:
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _past_sum_expr(column: str, lookback_weeks: int) -> pl.Expr:
    return pl.sum_horizontal([pl.col(column).shift(step) for step in range(0, lookback_weeks)])


def _future_sum_expr(column: str, horizon_weeks: int) -> pl.Expr:
    return pl.sum_horizontal([pl.col(column).shift(-step) for step in range(1, horizon_weeks + 1)])


def _past_mean_expr(column: str, lookback_weeks: int) -> pl.Expr:
    return pl.mean_horizontal([pl.col(column).shift(step) for step in range(0, lookback_weeks)])


def _future_mean_expr(column: str, horizon_weeks: int) -> pl.Expr:
    return pl.mean_horizontal([pl.col(column).shift(-step) for step in range(1, horizon_weeks + 1)])


def _past_active_ratio_expr(column: str, lookback_weeks: int) -> pl.Expr:
    return pl.mean_horizontal(
        [pl.when(pl.col(column).shift(step) > 0).then(1.0).otherwise(0.0) for step in range(0, lookback_weeks)]
    )


def _future_active_ratio_expr(column: str, horizon_weeks: int) -> pl.Expr:
    return pl.mean_horizontal(
        [pl.when(pl.col(column).shift(-step) > 0).then(1.0).otherwise(0.0) for step in range(1, horizon_weeks + 1)]
    )


def _rolling_mean_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).shift(1).rolling_mean(window_size=window, min_periods=1)


def _rolling_std_expr(column: str, window: int) -> pl.Expr:
    return pl.col(column).shift(1).rolling_std(window_size=window, min_periods=2)


def schema_feature_audit(config: PropensityConfig, panel_file: str | Path | None = None) -> pl.DataFrame:
    source = Path(panel_file) if panel_file is not None else panel_path(config.segment)
    columns = _existing_columns(source)
    schema = config.schema
    mapping = {
        "identifier": [schema.client_id],
        "time": [schema.cutoff_index, schema.cutoff_start, schema.cutoff_end],
        "segment": [schema.segment],
        "current_rfm": [schema.current_recency, schema.current_frequency, schema.current_monetary],
        "target_input": [schema.activity],
        "customer_context": ["employment", "is_tre", "age", "income_decile", "client_group"],
        "raw_date_excluded": ["last_tx_date"],
    }
    rows = [
        {"column": col, "role": role, "source": "weekly_panel"}
        for role, candidates in mapping.items()
        for col in candidates
        if col in columns
    ]
    generated = [
        "past_frequency",
        "future_frequency",
        "past_composite_score",
        "future_composite_score",
        "active_week_ratio",
        "activity_regularity",
        "Frequency_lag_*",
        "Frequency_delta_*",
        "Frequency_ratio_*",
        "Frequency_rolling_mean_*",
        "Frequency_rolling_std_*",
        "Recency_trend_8w",
        "Monetary_trend_8w",
        "inactive_streak",
    ]
    rows.extend({"column": col, "role": "cutoff_safe_generated", "source": "weekly_panel"} for col in generated)
    tf_path = temporal_feature_path(config.segment)
    if tf_path and tf_path.exists():
        for col in _existing_columns(tf_path):
            if col not in {schema.client_id, schema.segment}:
                role = "optional_static_temporal_feature" if config.include_static_temporal_features else "excluded_optional_full_history_temporal_feature"
                rows.append({"column": col, "role": role, "source": "temporal_features"})
    return pl.DataFrame(rows)


def leakage_audit(columns: list[str]) -> pl.DataFrame:
    target_inputs = {
        "future_frequency",
        "absolute_frequency_growth",
        "relative_frequency_growth",
        "future_composite_score",
        "absolute_composite_improvement",
        "relative_composite_improvement",
        *TARGET_COLUMNS,
    }
    rows = []
    for col in columns:
        lowered = col.lower()
        reasons = [term for term in SUSPICIOUS_TERMS if term in lowered]
        if col in target_inputs:
            reasons.append("target_or_target_input")
        if col in {"last_tx_date", "week_start", "week_end"}:
            reasons.append("raw_date_or_cutoff_date")
        if reasons:
            rows.append({"column": col, "excluded": True, "reason": ", ".join(sorted(set(reasons)))})
    return pl.DataFrame(rows) if rows else pl.DataFrame({"column": [], "excluded": [], "reason": []})


def build_engagement_frame(config: PropensityConfig, panel_file: str | Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create cutoff-level predictors plus Target A and Target B target inputs."""
    source = Path(panel_file) if panel_file is not None else panel_path(config.segment)
    if not source.exists():
        raise FileNotFoundError(source)
    schema = config.schema
    columns = _existing_columns(source)
    required = [
        schema.client_id,
        schema.cutoff_index,
        schema.cutoff_start,
        schema.cutoff_end,
        schema.segment,
        schema.activity,
        schema.current_frequency,
        schema.current_recency,
        schema.current_monetary,
    ]
    _require_columns(columns, required, source)
    optional = [col for col in ["Monetary_avg", "employment", "is_tre", "age", "income_decile", "client_group"] if col in columns]

    lf = (
        pl.scan_parquet(source)
        .select(required + optional)
        .with_columns(
            pl.col(schema.client_id).cast(pl.Utf8),
            pl.col(schema.activity).cast(pl.Float64).fill_null(0.0),
            pl.col(schema.current_frequency).cast(pl.Float64).fill_null(0.0),
            pl.col(schema.current_recency).cast(pl.Float64).fill_null(0.0),
            pl.col(schema.current_monetary).cast(pl.Float64).fill_null(0.0),
        )
        .sort([schema.client_id, schema.cutoff_index])
    )
    if config.max_clients is not None:
        lf = lf.join(lf.select(schema.client_id).unique().head(config.max_clients), on=schema.client_id, how="inner")

    lags = (1, 4, 8, 13)
    frame = (
        lf.with_columns(
            pl.int_range(pl.len()).over(schema.client_id).alias("client_week_number"),
            (pl.len().over(schema.client_id) - pl.int_range(pl.len()).over(schema.client_id) - 1).alias("future_weeks_available"),
            _past_sum_expr(schema.activity, config.lookback_weeks).over(schema.client_id).alias("past_frequency"),
            _future_sum_expr(schema.activity, config.horizon_weeks).over(schema.client_id).alias("future_frequency"),
            _past_mean_expr(schema.current_monetary, config.lookback_weeks).over(schema.client_id).alias("past_monetary_mean"),
            _future_mean_expr(schema.current_monetary, config.horizon_weeks).over(schema.client_id).alias("future_monetary_mean"),
            _past_mean_expr(schema.current_recency, config.lookback_weeks).over(schema.client_id).alias("past_recency_mean"),
            _future_mean_expr(schema.current_recency, config.horizon_weeks).over(schema.client_id).alias("future_recency_mean"),
            _past_active_ratio_expr(schema.activity, config.lookback_weeks).over(schema.client_id).alias("past_active_week_ratio"),
            _future_active_ratio_expr(schema.activity, config.horizon_weeks).over(schema.client_id).alias("future_active_week_ratio"),
            *[pl.col(schema.current_frequency).shift(lag).over(schema.client_id).alias(f"Frequency_lag_{lag}w") for lag in lags],
            *[(pl.col(schema.current_frequency) - pl.col(schema.current_frequency).shift(lag)).over(schema.client_id).alias(f"Frequency_delta_{lag}w") for lag in lags],
            *[(pl.col(schema.current_frequency) / (pl.col(schema.current_frequency).shift(lag) + 1e-6)).over(schema.client_id).alias(f"Frequency_ratio_{lag}w") for lag in lags],
            _rolling_mean_expr(schema.activity, 4).over(schema.client_id).alias("Frequency_rolling_mean_4w"),
            _rolling_mean_expr(schema.activity, 8).over(schema.client_id).alias("Frequency_rolling_mean_8w"),
            _rolling_std_expr(schema.activity, 8).over(schema.client_id).alias("Frequency_rolling_std_8w"),
            (pl.col(schema.current_recency) - pl.col(schema.current_recency).shift(8)).over(schema.client_id).alias("Recency_trend_8w"),
            (pl.col(schema.current_monetary) - pl.col(schema.current_monetary).shift(8)).over(schema.client_id).alias("Monetary_trend_8w"),
            (pl.col(schema.activity) > 0).cast(pl.Int8).alias("active_period_flag"),
            (pl.col(schema.activity).shift(1) > 0).cast(pl.Float64).rolling_mean(window_size=8, min_periods=1).over(schema.client_id).alias("active_week_ratio"),
            (1.0 / (1.0 + _rolling_std_expr(schema.activity, 8))).over(schema.client_id).alias("activity_regularity"),
            pl.when(pl.col(schema.activity) > 0).then(0).otherwise(pl.col(schema.activity).eq(0).cum_sum().over(schema.client_id)).alias("inactive_streak"),
        )
        .filter((pl.col("client_week_number") >= config.lookback_weeks - 1) & (pl.col("future_weeks_available") >= config.horizon_weeks))
        .with_columns(
            (pl.col("future_frequency") - pl.col("past_frequency")).alias("absolute_frequency_growth"),
            pl.when(pl.col("past_frequency") > 0)
            .then((pl.col("future_frequency") - pl.col("past_frequency")) / pl.col("past_frequency"))
            .otherwise(None)
            .alias("relative_frequency_growth"),
        )
        .with_columns(
            pl.when(pl.col("past_frequency") == 0)
            .then(pl.col("future_frequency") >= config.min_future_activity_when_past_zero)
            .otherwise((pl.col("future_frequency") >= pl.col("past_frequency") * (1 + config.min_relative_growth)) & (pl.col("absolute_frequency_growth") >= config.min_absolute_growth))
            .fill_null(False)
            .cast(pl.Int8)
            .alias(TARGET_A_COLUMN)
        )
    )
    df = frame.collect()
    return df, schema_feature_audit(config, source), leakage_audit(df.columns)


def add_composite_target(df: pl.DataFrame, train_idx: np.ndarray, config: PropensityConfig) -> pl.DataFrame:
    """Fit composite score transforms on train rows only, then label all rows."""
    components = ["past_frequency", "past_monetary_mean", "past_recency_mean", "past_active_week_ratio"]
    train = df[train_idx].select(components)
    means = {c: float(train.select(pl.mean(c)).item() or 0.0) for c in components}
    stds = {c: float(train.select(pl.std(c)).item() or 1.0) for c in components}
    stds = {c: (v if np.isfinite(v) and v > 1e-9 else 1.0) for c, v in stds.items()}

    def z(col: str, base: str) -> pl.Expr:
        return ((pl.col(col).fill_null(means[base]) - means[base]) / stds[base]).clip(-5, 5)

    scored = (
        df.with_columns(
            (
                z("past_frequency", "past_frequency")
                + z("past_monetary_mean", "past_monetary_mean")
                - z("past_recency_mean", "past_recency_mean")
                + z("past_active_week_ratio", "past_active_week_ratio")
            ).alias("past_composite_score"),
            (
                z("future_frequency", "past_frequency")
                + z("future_monetary_mean", "past_monetary_mean")
                - z("future_recency_mean", "past_recency_mean")
                + z("future_active_week_ratio", "past_active_week_ratio")
            ).alias("future_composite_score"),
        )
        .with_columns(
            (pl.col("future_composite_score") - pl.col("past_composite_score")).alias("absolute_composite_improvement"),
            pl.when(pl.col("past_composite_score").abs() > 1e-6)
            .then((pl.col("future_composite_score") - pl.col("past_composite_score")) / pl.col("past_composite_score").abs())
            .otherwise(None)
            .alias("relative_composite_improvement"),
        )
        .with_columns(
            (
                (pl.col("absolute_composite_improvement") >= config.composite_min_absolute_improvement)
                & (
                    (pl.col("relative_composite_improvement") >= config.composite_min_relative_improvement)
                    | (pl.col("past_composite_score").abs() <= 1e-6)
                )
            )
            .fill_null(False)
            .cast(pl.Int8)
            .alias(TARGET_B_COLUMN)
        )
    )
    scored.attrs = {"composite_transform": {"means": means, "stds": stds}}
    return scored


def _excluded_columns(config: PropensityConfig) -> set[str]:
    schema = config.schema
    return {
        schema.client_id,
        schema.cutoff_start,
        schema.cutoff_end,
        "last_tx_date",
        "client_week_number",
        "future_weeks_available",
        "future_frequency",
        "future_monetary_mean",
        "future_recency_mean",
        "future_active_week_ratio",
        "absolute_frequency_growth",
        "relative_frequency_growth",
        "future_composite_score",
        "absolute_composite_improvement",
        "relative_composite_improvement",
        *TARGET_COLUMNS,
    }


def select_feature_columns(df: pl.DataFrame, config: PropensityConfig) -> tuple[list[str], list[str]]:
    excluded = _excluded_columns(config)
    suspicious = {col for col in df.columns if any(term in col.lower() for term in SUSPICIOUS_TERMS)}
    cols = [col for col in df.columns if col not in excluded and col not in suspicious]
    numeric = [col for col in cols if df.schema[col].is_numeric()]
    categorical = [col for col in cols if df.schema[col] in (pl.String, pl.Boolean, pl.Categorical)]
    return numeric, categorical


def chronological_split(df: pl.DataFrame, config: PropensityConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    schema = config.schema
    weeks = df.select(schema.cutoff_index).unique().sort(schema.cutoff_index).get_column(schema.cutoff_index).to_list()
    n = len(weeks)
    test_n = max(1, int(np.floor(n * config.test_size)))
    val_n = max(1, int(np.floor(n * config.validation_size)))
    train_weeks = set(weeks[: max(1, n - val_n - test_n)])
    val_weeks = set(weeks[max(1, n - val_n - test_n) : n - test_n])
    test_weeks = set(weeks[n - test_n :])
    week_values = df.get_column(schema.cutoff_index).to_list()
    train_mask = np.array([w in train_weeks for w in week_values])
    val_mask = np.array([w in val_weeks for w in week_values])
    test_mask = np.array([w in test_weeks for w in week_values])
    if config.strict_client_holdout:
        test_clients = set(df.filter(test_mask).get_column(schema.client_id).unique().to_list())
        train_mask = np.array([(t and c not in test_clients) for t, c in zip(train_mask, df.get_column(schema.client_id))])
    train_idx, val_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)
    summary = pl.DataFrame([_period_summary(df[idx], name, schema) for idx, name in [(train_idx, "train"), (val_idx, "validation"), (test_idx, "test")]])
    return train_idx, val_idx, test_idx, summary


def _period_summary(df: pl.DataFrame, period: str, schema: SchemaConfig) -> dict[str, Any]:
    if df.height == 0:
        return {"period": period, "rows": 0, "customers": 0, "min_cutoff": None, "max_cutoff": None}
    return {
        "period": period,
        "rows": df.height,
        "customers": df.select(pl.n_unique(schema.client_id)).item(),
        "min_cutoff": df.select(pl.min(schema.cutoff_index)).item(),
        "max_cutoff": df.select(pl.max(schema.cutoff_index)).item(),
    }


def _preprocessor(numeric_idx: list[int], categorical_idx: list[int], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_idx),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_idx),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_model_specs(config: PropensityConfig) -> dict[str, tuple[Any, bool]]:
    specs: dict[str, tuple[Any, bool]] = {
        "dummy_classifier": (DummyClassifier(strategy="prior", random_state=config.random_state), False),
        "logistic_regression": (LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=config.random_state), True),
        "elastic_net_logistic": (LogisticRegression(max_iter=2_000, class_weight="balanced", penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=config.random_state), True),
        "random_forest": (RandomForestClassifier(n_estimators=160, min_samples_leaf=50, class_weight="balanced_subsample", random_state=config.random_state, n_jobs=1), False),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, l2_regularization=0.05, random_state=config.random_state), False),
    }
    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = (XGBClassifier(n_estimators=160, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=config.random_state), False)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = (LGBMClassifier(n_estimators=160, learning_rate=0.05, random_state=config.random_state, verbose=-1), False)
    except Exception:
        pass
    return specs


def _build_pipeline(estimator: Any, feature_cols: list[str], numeric: list[str], categorical: list[str], scale_numeric: bool) -> Pipeline:
    numeric_idx = [feature_cols.index(c) for c in numeric if c in feature_cols]
    categorical_idx = [feature_cols.index(c) for c in categorical if c in feature_cols]
    return Pipeline([("preprocess", _preprocessor(numeric_idx, categorical_idx, scale_numeric)), ("model", clone(estimator))])


def select_feature_sets(df: pl.DataFrame, train_idx: np.ndarray, y_train: np.ndarray, numeric: list[str], categorical: list[str], config: PropensityConfig) -> dict[str, list[str]]:
    feature_cols = numeric + categorical
    if numeric:
        X_num = df[train_idx].select(numeric).to_numpy()
        medians = np.nanmedian(X_num, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        X_num = np.where(np.isnan(X_num), medians, X_num)
        mi = mutual_info_classif(X_num, y_train, random_state=config.random_state)
        ranked_numeric = [c for c, _ in sorted(zip(numeric, mi), key=lambda x: x[1], reverse=True)]
    else:
        X_num = np.empty((len(train_idx), 0))
        ranked_numeric = []
    ranked = ranked_numeric + categorical
    top20 = ranked[:20]
    nonredundant: list[str] = []
    corr_cols = ranked_numeric[: min(len(ranked_numeric), 80)]
    corr_lookup: dict[tuple[str, str], float] = {}
    if corr_cols:
        corr_idx = [numeric.index(col) for col in corr_cols]
        corr_matrix = np.corrcoef(X_num[:, corr_idx], rowvar=False)
        corr_matrix = np.nan_to_num(np.abs(corr_matrix), nan=0.0)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])
        for i, left in enumerate(corr_cols):
            for j, right in enumerate(corr_cols):
                corr_lookup[(left, right)] = float(corr_matrix[i, j])
    for col in ranked_numeric:
        if len(nonredundant) >= 20:
            break
        if all(corr_lookup.get((col, kept), 0.0) < 0.85 for kept in nonredundant):
            nonredundant.append(col)
    nonredundant = (nonredundant + [c for c in categorical if c not in nonredundant])[:20]
    return {
        "all_leakage_safe": feature_cols,
        "top20_relevant": top20,
        "top20_relevant_nonredundant": nonredundant,
    }


def threshold_candidates(y_true: np.ndarray, scores: np.ndarray, config: PropensityConfig) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_f1_threshold = float(thresholds[max(0, int(np.nanargmax(f1[:-1])))] if len(thresholds) else 0.5)
    order = np.sort(scores)[::-1]
    top10_threshold = float(order[max(0, int(np.ceil(len(order) * 0.10)) - 1)]) if len(order) else 0.5
    eligible = thresholds[precision[:-1] >= config.min_precision] if len(thresholds) else np.array([])
    min_precision_threshold = float(eligible[0]) if len(eligible) else 1.0
    return {
        "fixed_0_50": 0.50,
        "validation_best_f1": best_f1_threshold,
        "top_10_percent": top10_threshold,
        "min_precision": min_precision_threshold,
    }


def _topk_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    out: dict[str, float] = {}
    order = np.argsort(-scores)
    for pct in (0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(len(y_true) * pct)))
        precision = float(y_true[order[:k]].mean())
        out[f"top_{int(pct * 100)}pct_precision"] = precision
        out[f"top_{int(pct * 100)}pct_lift"] = float(precision / prevalence) if prevalence > 0 else np.nan
    return ou


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        **_topk_metrics(y_true, scores),
    }


def decile_table(y_true: np.ndarray, scores: np.ndarray, target_name: str, model: str, period: str) -> pl.DataFrame:
    scores = scores.astype(np.float64, copy=False)
    pdf = pl.DataFrame({"y": y_true.astype(np.int8, copy=False), "score": scores}).sort("score", descending=True).with_row_index("rank")
    n = max(1, len(y_true))
    return (
        pdf.with_columns(((pl.col("rank") * 10 / n).floor() + 1).clip(1, 10).cast(pl.Int8).alias("decile"))
        .group_by("decile")
        .agg(pl.len().alias("rows"), pl.mean("y").alias("response_rate"), pl.mean("score").alias("mean_score"), pl.sum("y").alias("positives"))
        .with_columns(pl.lit(target_name).alias("target"), pl.lit(model).alias("model"), pl.lit(period).alias("period"))
        .sort("decile")
    )


def curve_tables(y_true: np.ndarray, scores: np.ndarray, target_name: str, model: str, period: str) -> dict[str, pl.DataFrame]:
    scores = scores.astype(np.float64, copy=False)
    fpr, tpr, roc_t = roc_curve(y_true, scores) if len(np.unique(y_true)) > 1 else (np.array([]), np.array([]), np.array([]))
    precision, recall, pr_t = precision_recall_curve(y_true, scores)
    cal_y, cal_p = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    return {
        "roc": pl.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_t}).with_columns(pl.lit(target_name).alias("target"), pl.lit(model).alias("model"), pl.lit(period).alias("period")),
        "precision_recall": pl.DataFrame({"precision": precision, "recall": recall, "threshold": np.r_[pr_t, np.nan]}).with_columns(pl.lit(target_name).alias("target"), pl.lit(model).alias("model"), pl.lit(period).alias("period")),
        "calibration": pl.DataFrame({"mean_predicted_probability": cal_p, "fraction_of_positives": cal_y}).with_columns(pl.lit(target_name).alias("target"), pl.lit(model).alias("model"), pl.lit(period).alias("period")),
    }


def feature_importance(model: Pipeline, model_name: str, X_holdout: Any, y_holdout: np.ndarray, original_feature_cols: list[str]) -> pl.DataFrame:
    pre = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    try:
        names = list(pre.get_feature_names_out(original_feature_cols))
    except Exception:
        names = original_feature_cols
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        kind = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "model_importance"
    else:
        transformed = pre.transform(X_holdout)
        result = permutation_importance(estimator, transformed, y_holdout, n_repeats=3, random_state=17, scoring="average_precision")
        values = result.importances_mean
        kind = "permutation_pr_auc"
    return pl.DataFrame({"model": model_name, "feature": names[: len(values)], "importance": [float(v) for v in values], "kind": kind}).sort("importance", descending=True)


def performance_through_periods(df: pl.DataFrame, idx: np.ndarray, y: np.ndarray, scores: np.ndarray, target: str, model: str, config: PropensityConfig) -> pl.DataFrame:
    scores = scores.astype(np.float64, copy=False)
    return (
        df[idx].select(config.schema.cutoff_index)
        .with_columns(pl.Series("y", y).cast(pl.Int8), pl.Series("score", scores).cast(pl.Float64))
        .group_by(config.schema.cutoff_index)
        .agg(pl.len().alias("rows"), pl.mean("y").alias("prevalence"), pl.mean("score").alias("mean_score"))
        .with_columns(pl.lit(target).alias("target"), pl.lit(model).alias("model"))
        .sort(config.schema.cutoff_index)
    )


def _safe_concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _write_checkpoint(
    checkpoint_dir: Path | None,
    target_column: str,
    suffix: str,
    rows_val: list[dict[str, Any]],
    rows_test: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    pred_frames: list[pl.DataFrame],
    deciles: list[pl.DataFrame],
    curves: dict[str, list[pl.DataFrame]],
    period_perf: list[pl.DataFrame],
) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prefix = checkpoint_dir / f"{suffix}_{target_column}"
    if rows_val:
        pl.DataFrame(rows_val).write_csv(prefix.with_name(prefix.name + "_validation_partial.csv"))
    if rows_test:
        pl.DataFrame(rows_test).write_csv(prefix.with_name(prefix.name + "_test_partial.csv"))
    if threshold_rows:
        pl.DataFrame(threshold_rows).write_csv(prefix.with_name(prefix.name + "_thresholds_partial.csv"))
    if pred_frames:
        _safe_concat(pred_frames).write_parquet(prefix.with_name(prefix.name + "_predictions_partial.parquet"))
    if deciles:
        _safe_concat(deciles).write_csv(prefix.with_name(prefix.name + "_deciles_partial.csv"))
    for key, frames in curves.items():
        if frames:
            _safe_concat(frames).write_csv(prefix.with_name(prefix.name + f"_{key}_partial.csv"))
    if period_perf:
        _safe_concat(period_perf).write_csv(prefix.with_name(prefix.name + "_period_performance_partial.csv"))


def train_and_evaluate(
    df: pl.DataFrame,
    config: PropensityConfig,
    target_column: str = TARGET_A_COLUMN,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    raw_train_idx, val_idx, test_idx, split_summary = chronological_split(df, config)
    if target_column == TARGET_B_COLUMN and TARGET_B_COLUMN not in df.columns:
        df = add_composite_target(df, raw_train_idx, config)
    train_idx = raw_train_idx
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("Chronological split produced an empty train, validation, or test set.")
    y_all = df.get_column(target_column).to_numpy()
    if np.unique(y_all[train_idx]).shape[0] < 2:
        raise ValueError(f"Training target {target_column} has one class only.")
    numeric, categorical = select_feature_columns(df, config)
    feature_sets = select_feature_sets(df, train_idx, y_all[train_idx], numeric, categorical, config)
    model_specs = _make_model_specs(config)

    rows_val: list[dict[str, Any]] = []
    rows_test: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    pred_frames: list[pl.DataFrame] = []
    deciles: list[pl.DataFrame] = []
    curves = {"roc": [], "precision_recall": [], "calibration": []}
    importances: list[pl.DataFrame] = []
    period_perf: list[pl.DataFrame] = []
    fitted: dict[tuple[str, str], Pipeline] = {}
    train_times: dict[tuple[str, str], float] = {}

    for feature_set, feature_cols in feature_sets.items():
        set_numeric = [c for c in numeric if c in feature_cols]
        set_categorical = [c for c in categorical if c in feature_cols]
        X = df.select(feature_cols).to_numpy()
        for model_name, (estimator, scale_numeric) in model_specs.items():
            print(f"[{target_column}] training {model_name} on {feature_set} ({len(feature_cols)} features)", flush=True)
            pipe = _build_pipeline(estimator, feature_cols, set_numeric, set_categorical, scale_numeric)
            start = perf_counter()
            pipe.fit(X[train_idx], y_all[train_idx])
            train_time = perf_counter() - star
            val_scores = pipe.predict_proba(X[val_idx])[:, 1].astype(np.float64, copy=False)
            test_scores = pipe.predict_proba(X[test_idx])[:, 1].astype(np.float64, copy=False)
            thresholds = threshold_candidates(y_all[val_idx], val_scores, config)
            selected_threshold_name = "validation_best_f1"
            selected_threshold = thresholds[selected_threshold_name]
            for threshold_name, threshold in thresholds.items():
                threshold_rows.append({"target": target_column, "feature_set": feature_set, "model": model_name, "threshold_policy": threshold_name, "threshold": threshold, **_metrics(y_all[val_idx], val_scores, threshold)})
            for period, idx, scores, container in [("validation", val_idx, val_scores, rows_val), ("test", test_idx, test_scores, rows_test)]:
                container.append(
                    {
                        "target": target_column,
                        "period": period,
                        "feature_set": feature_set,
                        "model": model_name,
                        "selected_threshold_policy": selected_threshold_name,
                        "selected_threshold": selected_threshold,
                        "training_time_seconds": train_time,
                        **_metrics(y_all[idx], scores, selected_threshold),
                    }
                )
                deciles.append(decile_table(y_all[idx], scores, target_column, model_name, period).with_columns(pl.lit(feature_set).alias("feature_set")))
                for key, table in curve_tables(y_all[idx], scores, target_column, model_name, period).items():
                    curves[key].append(table.with_columns(pl.lit(feature_set).alias("feature_set")))
            pred_frames.append(
                pl.concat(
                    [
                        df[val_idx]
                        .select(config.schema.client_id, config.schema.cutoff_index, config.schema.segment, pl.col(target_column).alias("actual_target"))
                        .with_columns(
                            pl.lit("validation").alias("period"),
                            pl.Series("propensity_score", val_scores).cast(pl.Float64),
                        ),
                        df[test_idx]
                        .select(config.schema.client_id, config.schema.cutoff_index, config.schema.segment, pl.col(target_column).alias("actual_target"))
                        .with_columns(
                            pl.lit("test").alias("period"),
                            pl.Series("propensity_score", test_scores).cast(pl.Float64),
                        ),
                    ]
                ).with_columns(pl.lit(target_column).alias("target"), pl.lit(feature_set).alias("feature_set"), pl.lit(model_name).alias("model"))
            )
            period_perf.append(performance_through_periods(df, test_idx, y_all[test_idx], test_scores, target_column, model_name, config).with_columns(pl.lit(feature_set).alias("feature_set")))
            fitted[(feature_set, model_name)] = pipe
            train_times[(feature_set, model_name)] = train_time
            _write_checkpoint(
                checkpoint_dir,
                target_column,
                config.segment,
                rows_val,
                rows_test,
                threshold_rows,
                pred_frames,
                deciles,
                curves,
                period_perf,
            )

    validation = pl.DataFrame(rows_val).sort(["pr_auc", "top_10pct_lift", "f1"], descending=[True, True, True])
    test = pl.DataFrame(rows_test).sort(["pr_auc", "top_10pct_lift", "f1"], descending=[True, True, True])
    best_key = (validation.row(0, named=True)["feature_set"], validation.row(0, named=True)["model"])
    best_model = fitted[best_key]
    best_features = feature_sets[best_key[0]]
    best_X = df.select(best_features).to_numpy()
    importances.append(
        feature_importance(best_model, best_key[1], best_X[test_idx], y_all[test_idx], best_features).with_columns(
            pl.lit(target_column).alias("target"),
            pl.lit(best_key[0]).alias("feature_set"),
        )
    )
    return {
        "training_frame": df,
        "target": target_column,
        "feature_sets": _safe_concat([pl.DataFrame({"target": target_column, "feature_set": k, "column": v}) for k, v in feature_sets.items()]),
        "split_summary": split_summary.with_columns(pl.lit(target_column).alias("target")),
        "validation_comparison": validation,
        "test_comparison": test,
        "threshold_comparison": pl.DataFrame(threshold_rows),
        "predictions": _safe_concat(pred_frames),
        "decile_table": _safe_concat(deciles),
        "roc_curves": _safe_concat(curves["roc"]),
        "precision_recall_curves": _safe_concat(curves["precision_recall"]),
        "calibration_curves": _safe_concat(curves["calibration"]),
        "feature_importance": _safe_concat(importances),
        "performance_through_test_periods": _safe_concat(period_perf),
        "selected_model_name": best_key[1],
        "selected_feature_set": best_key[0],
        "selected_model": best_model,
        "selected_feature_columns": best_features,
        "fitted_models": fitted,
    }


def target_prevalence_and_sensitivity(df: pl.DataFrame, config: PropensityConfig) -> pl.DataFrame:
    rows = []
    for rel in [0.10, 0.20, 0.30]:
        for abs_growth in [1.0, 3.0, 5.0]:
            y = np.where(
                df.get_column("past_frequency").to_numpy() == 0,
                df.get_column("future_frequency").to_numpy() >= config.min_future_activity_when_past_zero,
                (df.get_column("future_frequency").to_numpy() >= df.get_column("past_frequency").to_numpy() * (1 + rel))
                & (df.get_column("absolute_frequency_growth").to_numpy() >= abs_growth),
            )
            rows.append({"target": TARGET_A_COLUMN, "relative_threshold": rel, "absolute_threshold": abs_growth, "positive_rate": float(y.mean())})
    if TARGET_B_COLUMN in df.columns:
        rows.append({"target": TARGET_B_COLUMN, "relative_threshold": config.composite_min_relative_improvement, "absolute_threshold": config.composite_min_absolute_improvement, "positive_rate": float(df.select(pl.mean(TARGET_B_COLUMN)).item())})
    return pl.DataFrame(rows)


def run_propensity_experiment(config: PropensityConfig, output_dir: str | Path = OUTPUT_DIR, panel_file: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    frame, schema_audit, leak_audit = build_engagement_frame(config, panel_file=panel_file)
    train_idx, _, _, split_summary = chronological_split(frame, config)
    frame = add_composite_target(frame, train_idx, config)

    suffix = config.segmen
    frame.write_parquet(output / f"engagement_modeling_frame_{suffix}.parquet")
    schema_audit.write_csv(output / f"schema_feature_audit_{suffix}.csv")
    leakage_audit(frame.columns).write_csv(output / f"leakage_audit_{suffix}.csv")
    target_prevalence_and_sensitivity(frame, config).write_csv(output / f"target_prevalence_threshold_sensitivity_{suffix}.csv")
    split_summary.write_csv(output / f"chronological_split_summary_{suffix}.csv")

    results_by_target: dict[str, dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        target_results = train_and_evaluate(frame, config, target, checkpoint_dir=checkpoint_dir)
        results_by_target[target] = target_results
        target_results["validation_comparison"].write_csv(output / f"validation_model_comparison_{suffix}_{target}.csv")
        target_results["test_comparison"].write_csv(output / f"test_model_comparison_{suffix}_{target}.csv")
        target_results["threshold_comparison"].write_csv(output / f"threshold_comparison_{suffix}_{target}.csv")
        target_results["predictions"].write_parquet(output / f"predictions_{suffix}_{target}.parquet")
        target_results["decile_table"].write_csv(output / f"propensity_decile_table_{suffix}_{target}.csv")
        target_results["feature_sets"].write_csv(output / f"selected_feature_lists_{suffix}_{target}.csv")
        target_results["feature_importance"].write_csv(output / f"feature_importance_{suffix}_{target}.csv")
        target_results["performance_through_test_periods"].write_csv(output / f"performance_through_test_periods_{suffix}_{target}.csv")

    combined: dict[str, pl.DataFrame] = {}
    for key in ["feature_sets", "validation_comparison", "test_comparison", "threshold_comparison", "predictions", "decile_table", "roc_curves", "precision_recall_curves", "calibration_curves", "feature_importance", "performance_through_test_periods"]:
        combined[key] = _safe_concat([results_by_target[t][key] for t in TARGET_COLUMNS])
    combined["validation_comparison"].write_csv(output / f"validation_model_comparison_{suffix}.csv")
    combined["test_comparison"].write_csv(output / f"test_model_comparison_{suffix}.csv")
    combined["threshold_comparison"].write_csv(output / f"threshold_comparison_{suffix}.csv")
    combined["predictions"].write_parquet(output / f"predictions_{suffix}.parquet")
    combined["decile_table"].write_csv(output / f"propensity_decile_table_{suffix}.csv")
    combined["roc_curves"].write_csv(output / f"roc_curves_{suffix}.csv")
    combined["precision_recall_curves"].write_csv(output / f"precision_recall_curves_{suffix}.csv")
    combined["calibration_curves"].write_csv(output / f"calibration_curves_{suffix}.csv")
    combined["feature_importance"].write_csv(output / f"feature_importance_{suffix}.csv")
    combined["feature_sets"].write_csv(output / f"selected_feature_lists_{suffix}.csv")
    combined["performance_through_test_periods"].write_csv(output / f"performance_through_test_periods_{suffix}.csv")

    ranked_validation = combined["validation_comparison"].sort(["pr_auc", "top_10pct_lift", "f1"], descending=[True, True, True])
    best_target = ranked_validation.row(0, named=True)["target"]
    best = results_by_target[best_target]
    joblib.dump(best["selected_model"], output / f"best_fitted_pipeline_{suffix}.joblib")

    top_model_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_validation.head(2).iter_rows(named=True), start=1):
        target = row["target"]
        feature_set = row["feature_set"]
        model_name = row["model"]
        model = results_by_target[target]["fitted_models"][(feature_set, model_name)]
        artifact_name = f"best_fitted_pipeline_{suffix}_rank{rank}.joblib"
        joblib.dump(model, output / artifact_name)
        top_model_rows.append(
            {
                "rank": rank,
                "artifact": artifact_name,
                "selected_target": target,
                "selected_feature_set": feature_set,
                "selected_model": model_name,
                "validation_pr_auc": row.get("pr_auc"),
                "validation_top_10pct_lift": row.get("top_10pct_lift"),
                "validation_f1": row.get("f1"),
                "selection_period": "validation",
            }
        )

    pl.DataFrame(
        [
            {
                **{k: v for k, v in asdict(config).items() if k != "schema"},
                "schema": str(asdict(config.schema)),
                "selected_target": best_target,
                "selected_feature_set": best["selected_feature_set"],
                "selected_model": best["selected_model_name"],
                "objective": "Compare frequency-only engagement growth with composite engagement improvement using chronological validation and test periods.",
            }
        ]
    ).write_csv(output / f"selected_model_metadata_{suffix}.csv")
    pl.DataFrame(top_model_rows).write_csv(output / f"selected_top2_model_metadata_{suffix}.csv")

    return {
        "training_frame": frame,
        "schema_audit": schema_audit,
        "leakage_audit": leakage_audit(frame.columns),
        "split_summary": split_summary,
        "target_sensitivity": target_prevalence_and_sensitivity(frame, config),
        "target_results": results_by_target,
        **combined,
        "selected_target": best_target,
        "selected_model_name": best["selected_model_name"],
        "selected_feature_set": best["selected_feature_set"],
        "selected_model": best["selected_model"],
        "selected_top2_models": top_model_rows,
    }


# Backward-compatible alias for older notebook imports.
build_propensity_frame = build_engagement_frame
