from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "archetype_data" / "need_discovery"))

import numpy as np

from group_gap_evidence import (
    GROUPS_OUTPUT,
    INPUT,
    _eligible_rows,
    group_rows,
    run,
    _support_status,
)


def evidence(evidence_id: str, normalized_text: str, *, keep: str = "true", semantic: str = "true") -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "normalized_text": normalized_text,
        "keep_for_need_discovery": keep,
        "semantic_ready": semantic,
        "source_type": "test",
        "normalized_gap_type": "information_gap",
        "existing_need": "",
        "product": "",
        "customer_id_or_case_id": evidence_id,
    }


class GroupGapEvidenceTests(unittest.TestCase):
    def test_current_dataset_is_under_threshold(self):
        before = hashlib.sha256(INPUT.read_bytes()).hexdigest()
        summary = run()
        self.assertFalse(summary["grouping_executed"])
        self.assertEqual(summary["grouping_reason"], "insufficient_evidence")
        self.assertEqual(summary["eligible_evidence_rows"], 3)
        self.assertEqual(before, hashlib.sha256(INPUT.read_bytes()).hexdigest())

    def test_excluded_and_nonsemantic_rows_are_not_eligible(self):
        rows = [
            evidence("a", "bank transfer", keep="false"),
            evidence("b", "bank transfer", semantic="false"),
            evidence("c", "bank transfer"),
        ]
        self.assertEqual([row["evidence_id"] for row in _eligible_rows(rows)], ["c"])

    def test_similar_strings_group_and_unrelated_string_separates(self):
        rows = [evidence("a", "track an international transfer"), evidence("b", "track my transfer status"), evidence("c", "open a savings account")]
        vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
        grouped = group_rows(rows, vectors, similarity_threshold=0.9)
        groups = {row["evidence_id"]: row["semantic_group_id"] for row in grouped}
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])

    def test_singleton_and_support_status(self):
        grouped = group_rows([evidence("only", "a unique banking request")], np.array([[1.0, 0.0]]))
        self.assertEqual(grouped[0]["group_size"], 1)
        self.assertEqual(grouped[0]["support_status"], "insufficient_support")
        self.assertEqual(_support_status(2), "preliminary_support")
        self.assertEqual(_support_status(5), "repeated_support")

    def test_group_ids_are_deterministic(self):
        rows = [evidence("b", "transfer tracking"), evidence("a", "transfer status")]
        vectors = np.array([[1.0, 0.0], [0.99, 0.01]])
        first = group_rows(rows, vectors, similarity_threshold=0.9)
        second = group_rows(rows, vectors, similarity_threshold=0.9)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["semantic_group_id"], "SG0001")

    def test_representative_selection_is_deterministic(self):
        rows = [evidence("a", "transfer one"), evidence("b", "transfer two")]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
        grouped = group_rows(rows, vectors, similarity_threshold=0.9)
        representatives = [row["evidence_id"] for row in grouped if row["is_group_representative"] == "true"]
        self.assertEqual(representatives, ["b"])

    def test_input_files_and_no_synthetic_production_rows(self):
        before = INPUT.read_bytes()
        run()
        self.assertEqual(before, INPUT.read_bytes())
        with GROUPS_OUTPUT.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
