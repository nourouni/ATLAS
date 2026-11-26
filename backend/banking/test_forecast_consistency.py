from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from . import dashboard_services, services


class PromotedForecastConsistencyTests(SimpleTestCase):
    def setUp(self):
        dashboard_services._static_forecasts_by_pool.cache_clear()

    def tearDown(self):
        dashboard_services._static_forecasts_by_pool.cache_clear()

    def test_dashboard_reads_configured_final_static_forecast_artifact(self):
        expected = Path(dashboard_services.settings.ATLAS_STATIC_FORECAST_POINTS)
        self.assertEqual(dashboard_services.STATIC_FORECAST_POINTS, expected)
        self.assertTrue(expected.exists())
        with expected.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertEqual({row["method"] for row in rows}, {"static_cluster_pooled"})
        self.assertEqual({row["metric"] for row in rows}, {"avg_balance"})
        self.assertEqual({row["pool"] for row in rows}, set(dashboard_services.STATIC_SEGMENT_FORECAST_POOLS.values()))

    def test_dashboard_pool_mapping_matches_final_static_segments(self):
        forecasts = dashboard_services._static_forecasts_by_pool()
        retail = dashboard_services._segment_forecast_for("seg_ret_0", "retail", forecasts)
        enterprise = dashboard_services._segment_forecast_for("seg_ent_0", "enterprise", forecasts)
        self.assertTrue(retail["available"])
        self.assertTrue(enterprise["available"])
        self.assertEqual(retail["pool"], "retail_static_2000")
        self.assertEqual(enterprise["pool"], "enterprise_static_1000")
        self.assertTrue(all(point["forecast"] is not None for point in retail["points"]))
        self.assertTrue(all(point["forecast"] is not None for point in enterprise["points"]))

    def test_negative_segments_are_unavailable(self):
        forecasts = dashboard_services._static_forecasts_by_pool()
        for segment_id, population in (("seg_ret_neg1", "retail"), ("seg_ent_neg1", "enterprise")):
            result = dashboard_services._segment_forecast_for(segment_id, population, forecasts)
            self.assertFalse(result["available"])
            self.assertEqual(result["points"], [])

    def test_unknown_and_broad_activity_segments_are_unavailable(self):
        forecasts = dashboard_services._static_forecasts_by_pool()
        for segment_id, population in (("seg_unknown", "retail"), ("seg_ret_broad_activity", "retail"), ("seg_ent_broad_activity", "enterprise")):
            result = dashboard_services._segment_forecast_for(segment_id, population, forecasts)
            self.assertFalse(result["available"])

    def test_manual_review_segments_are_unavailable(self):
        forecasts = dashboard_services._static_forecasts_by_pool()
        for segment_id, population in (("seg_ret_4", "retail"), ("seg_ent_1", "enterprise")):
            result = dashboard_services._segment_forecast_for(segment_id, population, forecasts)
            self.assertFalse(result["available"])
            self.assertTrue(result["manual_review"])

    def test_missing_static_artifact_is_unavailable(self):
        with patch.object(dashboard_services, "STATIC_FORECAST_POINTS", Path("does-not-exist.csv")):
            self.assertEqual(dashboard_services._static_forecasts_by_pool(), {})

    def test_static_source_does_not_contain_temporal_pools(self):
        forecasts = dashboard_services._static_forecasts_by_pool()
        self.assertFalse(any("rolling" in pool or "temporal" in pool for pool in forecasts))

    def test_forecast_run_resolution_is_deterministic(self):
        manager = services.ModelRun.objects
        with patch.object(manager, "filter") as filter_mock:
            queryset = filter_mock.return_value
            queryset.exclude.return_value.order_by.return_value.first.return_value = object()
            result = services.latest_forecast_run()
        self.assertIsNotNone(result)
        filter_mock.assert_called_once_with(run_name="temporal_feature_filled_cluster_pooled", model_type="segment_forecast")
        queryset.exclude.assert_called_once_with(completed_at__isnull=True)
        queryset.exclude.return_value.order_by.assert_called_once_with("-completed_at", "-run_id")

    def test_missing_forecast_run_is_unavailable(self):
        with patch.object(dashboard_services, "latest_forecast_run", return_value=None):
            result = dashboard_services.forecasting_analytics()
        self.assertEqual(result["forecasts"], [])
        self.assertEqual(result["model_runs"], [])
        self.assertEqual(result["pool_options"], [])
