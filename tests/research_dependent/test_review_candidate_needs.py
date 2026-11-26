from __future__ import annotations

import hashlib
import json
import sys
import unittes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "archetype_data" / "need_discovery"))

from review_candidate_needs import (
    TAXONOMY_INPUT,
    apply_review,
    build_summary,
    regenerate_outputs,
    run,
    synchronize_reviews,
    validate_review,
)


def candidate(candidate_id: str = "CN0001") -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "semantic_group_id": "SG0001",
        "proposed_need_label": "A reviewed banking objective",
        "relationship_to_existing_need": "candidate_new_need",
        "closest_existing_need_id": "",
        "closest_existing_need_label": "",
        "group_size": "5",
        "unique_semantic_evidence_count": "5",
        "support_status": "repeated_support",
        "source_type_count": "2",
        "unique_case_or_customer_count": "5",
        "product_count": "1",
        "existing_need_count": "0",
        "summary": "Repeated evidence summary.",
    }


class ReviewCandidateNeedsTests(unittest.TestCase):
    def test_current_zero_candidate_state_creates_valid_empty_report(self):
        summary = run()
        self.assertEqual(summary["candidate_needs_generated"], 0)
        self.assertEqual(summary["pending_review"], 0)
        self.assertEqual(summary["reviewed_candidates"], 0)
        self.assertEqual(summary["accepted"], 0)

    def test_future_candidate_starts_pending(self):
        reviews = synchronize_reviews({"CN0001": candidate()}, [])
        self.assertEqual(reviews[0]["review_status"], "pending")
        self.assertEqual(reviews[0]["review_version"], "0")

    def test_valid_accept_is_reviewed_without_ontology_update(self):
        reviews = synchronize_reviews({"CN0001": candidate()}, [])
        updated, audit = apply_review(
            reviews,
            candidate_id="CN0001",
            decision="accept",
            comment="Repeated evidence supports analyst review.",
            reviewed_by="analyst-1",
            reviewed_at="2026-01-01T00:00:00+00:00",
            taxonomy_ids={"need_024"},
        )
        self.assertEqual(updated[0]["review_status"], "reviewed")
        self.assertEqual(updated[0]["analyst_decision"], "accept")
        self.assertEqual(audit["previous"]["review_status"], "pending")
        self.assertEqual(build_summary({"CN0001": candidate()}, updated)["ontology_updates_applied"], 0)

    def test_valid_reject_merge_observation_and_insufficient(self):
        for decision in ("reject", "keep_under_observation", "insufficient_evidence"):
            reviews = synchronize_reviews({"CN0001": candidate()}, [])
            comment = "Analyst review recorded." if decision != "insufficient_evidence" else "Evidence remains insufficient."
            updated, _ = apply_review(
                reviews,
                candidate_id="CN0001",
                decision=decision,
                comment=comment,
                reviewed_by="analyst",
                reviewed_at="2026-01-01T00:00:00+00:00",
                taxonomy_ids={"need_024"},
            )
            self.assertEqual(updated[0]["analyst_decision"], decision)

        reviews = synchronize_reviews({"CN0001": candidate()}, [])
        updated, _ = apply_review(
            reviews,
            candidate_id="CN0001",
            decision="merge_with_existing",
            comment="Better represented by the existing need.",
            reviewed_by="analyst",
            reviewed_at="2026-01-01T00:00:00+00:00",
            taxonomy_ids={"need_024"},
            merge_need_id="need_024",
        )
        self.assertEqual(updated[0]["closest_existing_need_id"], "need_024")

    def test_merge_invalid_need_id_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_review("CN0001", "merge_with_existing", "comment", {"CN0001"}, {"need_024"}, "need_999")

    def test_unknown_candidate_and_invalid_decision_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_review("CN9999", "accept", "comment", {"CN0001"}, set())
        with self.assertRaises(ValueError):
            validate_review("CN0001", "not_a_decision", "comment", {"CN0001"}, set())

    def test_comments_required_for_material_decisions(self):
        with self.assertRaises(ValueError):
            validate_review("CN0001", "accept", "", {"CN0001"}, set())

    def test_review_update_has_audit_version(self):
        reviews = synchronize_reviews({"CN0001": candidate()}, [])
        first, first_event = apply_review(
            reviews,
            candidate_id="CN0001",
            decision="keep_under_observation",
            comment="Observe more evidence.",
            reviewed_by="analyst",
            reviewed_at="2026-01-01T00:00:00+00:00",
            taxonomy_ids=set(),
        )
        second, second_event = apply_review(
            first,
            candidate_id="CN0001",
            decision="reject",
            comment="Later evidence did not support it.",
            reviewed_by="analyst",
            reviewed_at="2026-01-02T00:00:00+00:00",
            taxonomy_ids=set(),
        )
        self.assertEqual(first[0]["review_version"], "1")
        self.assertEqual(second[0]["review_version"], "2")
        self.assertEqual(second_event["previous"]["analyst_decision"], "keep_under_observation")
        self.assertEqual(first_event["new"]["review_version"], "1")

    def test_current_run_does_not_modify_taxonomy(self):
        before = hashlib.sha256(TAXONOMY_INPUT.read_bytes()).hexdigest()
        run()
        self.assertEqual(before, hashlib.sha256(TAXONOMY_INPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
