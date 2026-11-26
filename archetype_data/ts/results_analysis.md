# TS Results Analysis

Generated from saved CSV outputs in `ts/method_outputs/*/data`.

## Executive Summary

The best current workflow is `rolling_temporal_feature_filled_cluster_pooled`.
It has the strongest final test-horizon performance: 65.0% of series beat persistence, with median `mae / persistence_mae = 0.789`.

The main weakness is not `avg_balance`. Most `avg_balance` pools forecast well. The weak area is `frequency` and `monetary`, especially when the statistical case is `trend_only`.

A key evaluation issue exists: `beats_persistence` appears to reflect the one-step backtest MAE (`backtest_mae_h1`) rather than the final held-out test MAE (`mae`). There are 61 rows where the one-step backtest says the model beats persistence, but the final test MAE is worse than persistence.

## Method Ranking By Final Test Horizon

| method | rows | final test win rate | median mae / persistence mae | one-step win rate | manual review rows |
|---|---:|---:|---:|---:|---:|
| rolling_temporal_feature_filled_cluster_pooled | 60 | 65.0% | 0.789 | 78.0% | 1 |
| dtw_cluster_pooled | 18 | 44.4% | 1.089 | 77.8% | 0 |
| rolling_temporal_feature_cluster_pooled | 24 | 25.0% | 1.313 | 94.7% | 5 |
| static_cluster_pooled | 15 | 20.0% | 1.594 | 40.0% | 0 |
| temporal_feature_cluster_pooled | 42 | 28.6% | 2.272 | 54.8% | 0 |
| temporal_feature_filled_cluster_pooled | 36 | 25.0% | 2.277 | 41.2% | 2 |
| global_pooled | 3 | 33.3% | 11.354 | 100.0% | 0 |

## Target-Level Diagnosis

`avg_balance` is forecastable across most methods. Across all outputs, it beats persistence in 87.3% of rows with median ratio 0.650.

`frequency` is weak. It beats persistence in 51.6% of rows overall, but most non-rolling-filled methods fail on final test. The global pooled `frequency` forecast is especially poor with `mae / persistence_mae = 14.693`.

`monetary` is also weak. It beats persistence in 57.1% of rows overall, but non-rolling-filled temporal feature methods and static/global pooling lose badly on final test.

## Statistical Case Diagnosis

Most series are classified as `trend_only`: 150 out of 198 rows. This case has the weakest final performance, with only 28.0% final test wins and median `mae / persistence_mae = 1.991`.

`trend_seasonal` performs much better: 77.1% final test wins and median ratio 0.651.

`stationary_ar_structure` is acceptable: 60.0% final test wins and median ratio 0.809.

This means the workflow currently works when it detects usable seasonality or AR structure, but the default handling for `trend_only` is too optimistic and often worse than persistence on the final horizon.

## Additional Seasonality And Lag Diagnostics

The statistical workflow now checks STL seasonal strength at multiple weekly periods, not only the annual 52-week cycle:

- monthly-like: 4 weeks
- quarterly: 13 weeks
- semiannual: 26 weeks
- annual: 52 weeks

The numeric strength values are retained alongside practical threshold flags. The `0.6` threshold is stored in the report as a rule-of-thumb cutoff, not treated as a universal boundary.

The ACF/PACF output now records significant lag counts, significant lag lists, named lag-window hits around 4, 13, 26, and 52 weeks, and the approximate probability of at least one false positive when many lags are checked separately. With 52 lags at a 5% level, that probability is about 0.93, so isolated significant lags should be interpreted with the Ljung-Box joint test rather than used alone.

The report also flags that Ramadan, Eid, and public holiday effects need event-calendar regressors. These events shift across Gregorian weeks and should not be forced into a fixed 52-week STL interpretation.

## Specific Red Flags

The rolling outputs do not currently include `case_summary.csv`, even though non-rolling methods do. Missing folders:

- `rolling_temporal_feature_cluster_pooled`
- `rolling_temporal_feature_filled_cluster_pooled`

The largest final-test losses are mostly `trend_only` `frequency` and `monetary` forecasts using `ETS_trend`. Examples:

- `retail_dtw_2000 / frequency`: ratio 38.772
- `retail_temporal_feature_filled_2006 / frequency`: ratio 37.688
- `retail_static_2001 / frequency`: ratio 15.611
- `global_pool / frequency`: ratio 14.693
- `global_pool / monetary`: ratio 11.354

## Recommended Fixes

1. Add final-test persistence comparison as a first-class metric.

   Keep `beats_persistence` if it is meant to represent one-step backtest, but add a separate `test_beats_persistence` and `test_mae_ratio`.

2. Change model dispatch for `trend_only` `frequency` and `monetary`.

   Do not default these to ETS trend without a final-horizon guard. Use a candidate set and pick by rolling-origin validation:

   - persistence / random walk
   - drif
   - damped ETS trend
   - robust median or trimmed-mean naive
   - HGB lag model only when enough lag signal exists

3. Use persistence fallback when validation is weak.

   If `test_mae_ratio` or rolling-origin median ratio is above 1.0, fall back to persistence or damped trend. This should especially apply to `trend_only` `frequency` and `monetary`.

4. Keep `rolling_temporal_feature_filled_cluster_pooled` as the lead workflow.

   It is the only method with acceptable final performance on all three metrics:

   - `avg_balance`: 85.0% final test win rate, median ratio 0.617
   - `frequency`: 60.0% final test win rate, median ratio 0.883
   - `monetary`: 50.0% final test win rate, median ratio 0.926

5. Treat global pooling as a baseline only.

   Global pooling looks good on one-step backtest but fails final horizon for `frequency` and `monetary`. It should not be the production/default forecast for those targets.

6. Generate `case_summary.csv` for rolling workflows.

   This makes the rolling notebooks comparable with the other method folders and makes classification drift visible.

7. Add a model leaderboard by metric and case, not just by method.

   The current aggregate method ranking hides the real issue. The useful leaderboard should answer:

   - best model for `trend_only / frequency`
   - best model for `trend_only / monetary`
   - best model for `trend_seasonal / avg_balance`
   - pools where persistence wins and should remain the forecas

## Practical Next Step

Patch `forecast_dispatch.py` and/or `pooling_method_workflow.py` so each pool tries a small model candidate set, scores candidates on rolling-origin validation, and writes both one-step and final-test persistence ratios. Then regenerate the `ts/method_outputs` folders, starting with `rolling_temporal_feature_filled_cluster_pooled`.
