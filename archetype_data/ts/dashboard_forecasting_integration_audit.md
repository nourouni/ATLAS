# Dashboard forecasting integration audi

## Previous behavior

The `SegmentForecast` PostgreSQL table currently contains temporal and rolling-temporal pools, including `frequency` and `monetary` targets. The Segments page was resolving pools by population and trailing cluster number, so it could present a temporal pool as if it belonged to a final static behavioral segment.

## Current sources and separation

- Customer page: uses the persisted `customers.forecast_pool` mapping through `get_customer_forecast()`. Existing values are temporal/rolling-temporal pools such as `retail_rolling_temporal_feature_filled_origin24_0`. The UI now labels this as an outlook for similar temporal behaviors and keeps `avg_balance` only.
- Segments page: uses only `archetype_data/ts/method_outputs/static_cluster_pooled/data/forecast_points.csv` and `forecast_results.csv`. It does not fall back to PostgreSQL temporal pools.
- Forecasting page: explains both concepts and shows the static artifact overview separately from the existing temporal-group forecast view.

## Static segment to forecast-pool mapping

The verified static artifact pool mapping is:

| Final segment | Cluster | Static forecast pool |
|---|---:|---|
| `seg_ret_0` / R1 | 0 | `retail_static_2000` |
| `seg_ret_1` / R2 | 1 | `retail_static_2001` |
| `seg_ret_2` / R3 | 2 | `retail_static_2002` |
| `seg_ret_3` / R4 | 3 | `retail_static_2003` |
| `seg_ret_4` / R5 | 4 | `retail_static_2004` |
| `seg_ent_0` / E1 | 0 | `enterprise_static_1000` |
| `seg_ent_1` / E2 | 1 | `enterprise_static_1001` |
| `seg_ent_2` / E3 | 2 | `enterprise_static_1002` |

Only `avg_balance` is exposed for final static segments. Cluster `-1`, retail cluster `2005`, frequency, and monetary forecasts are excluded. The static result metadata flags `retail_static_2004` and `enterprise_static_1001` for manual review; those segments show an honest unavailable state rather than a fallback forecast.

## Artifact paths

- Final static clustering: `archetype_data/static_clustering/retail_clustered.csv`, `enterprise_clustered.csv`.
- Static forecast points: `archetype_data/ts/method_outputs/static_cluster_pooled/data/forecast_points.csv`.
- Static forecast evaluation/results: `archetype_data/ts/method_outputs/static_cluster_pooled/data/forecast_results.csv`.
- Static workflow: `archetype_data/ts/02_static_cluster_pooled_workflow.ipynb`.
- Temporal workflow artifacts remain under the temporal workflow notebooks and model registry beneath `archetype_data/ts`; they are used only for the Customer temporal outlook and existing temporal Forecasting view.

## Customer temporal mapping

A persisted mapping exists in the database: 203,000 customers have a rolling temporal pool and 5,000 customers have no pool. The application does not derive a temporal pool from a static behavioral segment. Customers without a persisted temporal pool receive no misleading individual forecast.

## Files modified

- `backend/banking/dashboard_services.py`
- `backend/banking/templates/banking/dashboard_segments.html`
- `backend/banking/templates/banking/dashboard_forecasting.html`
- `backend/banking/templates/banking/dashboard_customers.html`
- `archetype_data/ts/dashboard_forecasting_integration_audit.md`

## Validation

- Django system check passed.
- Static forecast CSV headers and pool names were inspected.
- PostgreSQL pool/metric inventory was inspected and confirmed temporal-only before the static artifact wiring.
- No `initial_data` path was used.
