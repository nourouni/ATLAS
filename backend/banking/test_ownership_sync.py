from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from . import services
from database.sync_ownership_snapshot import load_owned_source


class OwnershipSynchronizationUnitTests(TestCase):
    def _write(self, frame: pl.DataFrame) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "ownership.parquet"
        frame.write_parquet(path)
        return path

    def test_valid_owned_source_is_capped_and_unique(self):
        path = self._write(
            pl.DataFrame(
                {
                    "client_id": [1, 1, 2],
                    "product_id": ["p1", "p2", "p1"],
                    "owned": [True, True, True],
                    "simulation_family": ["a", "b", "a"],
                }
            )
        )
        _, summary = load_owned_source(path)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["customers"], 2)
        self.assertEqual(summary["max_owned_per_customer"], 2)
        self.assertEqual(summary["duplicate_pairs"], 0)

    def test_duplicate_pairs_are_rejected(self):
        path = self._write(
            pl.DataFrame(
                {
                    "client_id": [1, 1],
                    "product_id": ["p1", "p1"],
                    "owned": [True, True],
                    "simulation_family": ["a", "a"],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "duplicate ownership pairs"):
            load_owned_source(path)

    def test_source_above_six_products_is_rejected(self):
        path = self._write(
            pl.DataFrame(
                {
                    "client_id": [1] * 7,
                    "product_id": [f"p{i}" for i in range(7)],
                    "owned": [True] * 7,
                    "simulation_family": ["a"] * 7,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "six-product cap"):
            load_owned_source(path)

    def test_missing_ownership_association_is_unavailable(self):
        with patch.object(services, "latest_recommendation_run", return_value=None):
            self.assertIsNone(services.current_ownership_run())
        with patch.object(services, "current_ownership_run", return_value=None):
            self.assertEqual(services.get_customer_products(1), [])
