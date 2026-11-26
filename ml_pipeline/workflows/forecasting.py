from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd

from ml_pipeline.paths import ARCHETYPE_ROOT, ARTIFACTS_ROOT, FORECAST_METHOD, FORECAST_POINTS
from ml_pipeline.runtime import copy_if_exists, copy_tree_contents, ensure_exists, with_db_transaction


FULL_RERUN_ENV = "ML_PIPELINE_RERUN_FORECASTING"


def promote_forecasting_artifacts() -> None:
    forecast_dir = ARTIFACTS_ROOT / "forecasting"
    method_dir = forecast_dir / "method_outputs" / FORECAST_METHOD / "data"
    for name in [
        "forecast_points.csv",
        "forecast_results.csv",
        "statistical_report.csv",
        "prep_warning_action_items.csv",
        "prep_warning_fix_experiment.csv",
        "case_summary.csv",
        "score_summary.csv",
    ]:
        copy_if_exists(FORECAST_POINTS.parent / name, method_dir / name)
    copy_if_exists(FORECAST_POINTS, forecast_dir / "forecast_points.csv")
    copy_tree_contents(ARCHETYPE_ROOT / "ts" / "model_registry", forecast_dir / "model_registry")
    for name in [
        "pooling_method_workflow.py",
        "forecast_dispatch.py",
        "data_prep.py",
        "evaluation.py",
        "statistical_tests.py",
        "model_selection_export.py",
        "results_analysis.md",
    ]:
        copy_if_exists(ARCHETYPE_ROOT / "ts" / name, forecast_dir / "runtime_sources" / name)


def run(skip_schema: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    ensure_exists(FORECAST_POINTS)

    rerun_requested = os.getenv(FULL_RERUN_ENV, "").strip().lower() in {"1", "true", "yes"}
    if rerun_requested:
        from ml_pipeline.runtime import import_from_dir

        ts_dir = ARCHETYPE_ROOT / "ts"
        workflow = import_from_dir("pooling_method_workflow", ts_dir / "pooling_method_workflow.py", ts_dir)
        workflow.COMPARISON_LABEL_CANDIDATES = [workflow.COMPARISON_LABELS_PATH]
        workflow.ROLLING_LABEL_CANDIDATES = [
            ARCHETYPE_ROOT
            / "temporal clustering"
            / "rolling_temporal_clustering"
            / "data"
            / "rolling_temporal_cluster_labels.parquet",
            ARCHETYPE_ROOT
            / "forecasting comparison"
            / "rolling_temporal_clustering"
            / "data"
            / "rolling_temporal_cluster_labels.parquet",
        ]
        result = workflow.run_full_workflow(FORECAST_METHOD)
        forecast_rows = int(len(result["forecast_points"]))
    else:
        forecast_rows = int(len(pd.read_csv(FORECAST_POINTS)))

    promote_forecasting_artifacts()

    def work(db, conn):
        run_id = db.upsert_model_run(conn, FORECAST_METHOD, "segment_forecast", FORECAST_POINTS)
        return {"forecast_run_id": run_id, "segment_forecasts": db.load_forecasts(conn, FORECAST_POINTS, run_id)}

    counts = with_db_transaction(work, skip_schema)
    counts["forecast_method"] = FORECAST_METHOD
    counts["forecast_rows"] = forecast_rows
    counts["reran_forecasting"] = rerun_requested
    counts["duration_seconds"] = round(time.perf_counter() - start, 3)
    return counts
