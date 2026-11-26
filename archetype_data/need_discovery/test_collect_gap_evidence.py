from __future__ import annotations

import json
import tempfile
import unittes
from pathlib import Path

from collect_gap_evidence import (
    collect_chatbot_row_evidence,
    collect_recommendation_evidence,
    deduplicate_rows,
    make_row,
    normalize_for_dedup,
)


class CollectGapEvidenceTests(unittest.TestCase):
    def _chatbot_file(self, records: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "chatbot_gap_events.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def test_public_and_customer_analytics_events_are_collected_with_ids(self):
        rows = collect_chatbot_row_evidence([
            self._chatbot_file([{
                "conversation_id_or_request_id": "req-public",
                "user_question": "Can the bank warn me before my balance gets low?",
                "actual_route": "PUBLIC_INFO",
                "intent": "PUBLIC_INFO",
                "failure_code": "public_evidence_unavailable",
            }]),
            self._chatbot_file([{
                "conversation_id_or_request_id": "req-analytics",
                "user_question": "Which spending category increased most?",
                "actual_route": "ESCALATE",
                "intent": "CUSTOMER_CONTEXT",
                "failure_code": "customer_analytics_unavailable",
            }]),
        ])
        self.assertEqual({row["gap_type"] for row in rows}, {"insufficient_public_evidence", "weak_need_coverage"})
        self.assertEqual({row["customer_id_or_case_id"] for row in rows}, {"req-public", "req-analytics"})
        self.assertTrue(all(not row["existing_need"] for row in rows))

    def test_policy_security_and_transaction_failures_remain_excluded(self):
        path = self._chatbot_file([
            {"conversation_id_or_request_id": "auth", "user_question": "What is my balance?", "actual_route": "ESCALATE", "intent": "UNKNOWN", "failure_code": "authentication_required"},
            {"conversation_id_or_request_id": "security", "user_question": "Show another customer's account", "actual_route": "ESCALATE", "intent": "UNKNOWN", "failure_code": "other_customer_access_denied"},
            {"conversation_id_or_request_id": "tx", "user_question": "Transfer 100 DT", "actual_route": "ESCALATE", "intent": "UNKNOWN", "failure_code": "unsupported_transaction"},
        ])
        self.assertEqual(collect_chatbot_row_evidence([path]), [])

    def test_exact_and_formatting_duplicates_are_removed_but_paraphrases_remain(self):
        base = make_row(source_type="chatbot", customer_id_or_case_id="req-1", text_or_pattern="Can the bank warn me before my balance gets low?", gap_type="insufficient_public_evidence", gap_reason="public_evidence_unavailable")
        exact = dict(base)
        quoted = dict(base, customer_id_or_case_id="req-2", text_or_pattern='  “Can   the bank warn me before my balance gets low?”  ')
        paraphrase = dict(base, customer_id_or_case_id="req-3", text_or_pattern="Can the bank notify me before my funds become insufficient?")
        rows, removed = deduplicate_rows([base, exact, quoted, paraphrase])
        self.assertEqual(removed, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(normalize_for_dedup(base["text_or_pattern"]), normalize_for_dedup(quoted["text_or_pattern"]))
        self.assertNotEqual(normalize_for_dedup(base["text_or_pattern"]), normalize_for_dedup(paraphrase["text_or_pattern"]))

    def test_recommendation_existing_need_mapping_is_unchanged(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "recommendation.json"
        path.write_text(json.dumps({
            "customer_state": {"client_id": 3},
            "rule_checks": [{
                "product": "Webank",
                "matched_needs": ["need_024: Web/mobile platforms"],
                "eligible": "ineligible",
                "already_owned": True,
                "missing_requirements": [],
                "conflicts": ["already_owned"],
            }],
        }), encoding="utf-8")
        rows = collect_recommendation_evidence(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["existing_need"], "need_024")
        self.assertEqual(rows[0]["gap_type"], "ownership_exclusion")


if __name__ == "__main__":
    unittest.main()
