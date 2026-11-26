from __future__ import annotations

import json
import tempfile
import unittes
from pathlib import Path

from banking.need_discovery_logging import append_chatbot_gap_event, build_chatbot_gap_even
from archetype_data.recommendation.hybrid_recommender import append_candidate_provenance


class ChatbotGapLoggingTests(unittest.TestCase):
    def test_only_potential_banking_gaps_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chatbot_gap_events.jsonl"
            logged = append_chatbot_gap_event(
                question="How can I track an international transfer?",
                result={"route": "ESCALATE", "intent": "UNKNOWN", "failure_code": "ambiguous_request"},
                request_id="unknown-1",
                path=path,
            )
            self.assertIsNotNone(logged)
            logged = append_chatbot_gap_event(
                question="What documents are needed for this product?",
                result={"route": "PUBLIC_INFO", "intent": "PUBLIC_INFO", "failure_code": "public_evidence_unavailable", "insufficient_evidence": True},
                request_id="public-1",
                path=path,
            )
            self.assertIsNotNone(logged)

            excluded = [
                ("What products do I own?", {"route": "ESCALATE", "failure_code": "authentication_required"}),
                ("Show me another customer's account", {"route": "ESCALATE", "failure_code": "other_customer_access_denied"}),
                ("Hello", {"route": "GREETING", "intent": "UNKNOWN"}),
            ]
            for question, result in excluded:
                self.assertIsNone(
                    append_chatbot_gap_event(question=question, result=result, request_id="excluded", path=path)
                )

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["user_question"], "How can I track an international transfer?")
            self.assertEqual(rows[1]["failure_code"], "public_evidence_unavailable")

    def test_event_builder_is_deterministic_for_supplied_identity(self):
        event = build_chatbot_gap_event(
            question="Can I open a savings account?",
            result={"route": "PUBLIC_INFO", "intent": "PUBLIC_INFO", "failure_code": "public_evidence_unavailable"},
            request_id="case-1",
            event_id="fixed",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(event["event_id"], "fixed")
        self.assertEqual(event["timestamp"], "2026-01-01T00:00:00+00:00")

    def test_recommendation_provenance_serializes_existing_decisions(self):
        checked = [
            {
                "product": "Owned Product",
                "matched_needs": ["need_001: Payments"],
                "already_owned": True,
                "missing_requirements": [],
                "eligible": "ineligible",
                "conflicts": ["already_owned"],
                "alternatives": [],
                "max_owned_to_candidate_affinity": 0.2,
                "segment_product_prior": 0.3,
                "propensity_score": 0.1,
                "need_match": 1,
            },
            {
                "product": "Selected Product",
                "matched_needs": ["need_002: Savings"],
                "already_owned": False,
                "missing_requirements": [],
                "eligible": "eligible",
                "conflicts": [],
                "alternatives": [],
                "max_owned_to_candidate_affinity": 0.4,
                "segment_product_prior": 0.5,
                "propensity_score": 0.9,
                "need_match": 1,
            },
        ]
        ranked = [{**checked[1], "score_components": {"need": 1}, "final_score": 0.9, "rank": 1}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recommendation_candidate_events.jsonl"
            append_candidate_provenance(
                customer_id=7,
                checked_candidates=checked,
                ranked_candidates=ranked,
                selected_candidates=ranked,
                path=path,
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["candidate_status"] for row in rows], ["excluded_owned", "selected"])
        self.assertEqual(rows[0]["hard_conflicts"], [])
        self.assertEqual(rows[1]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
