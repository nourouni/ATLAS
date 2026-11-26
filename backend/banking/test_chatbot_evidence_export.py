from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from banking.evaluation.evidence_export import (
    build_evidence_records,
    compare_replay,
    validate_evidence_records,
)
from banking.management.commands.export_chatbot_eval_evidence import Command


def case(case_id: str = "PUB-001") -> dict:
    return {
        "case_id": case_id,
        "question": "What is Carte CIB?",
        "expected_answer_points": ["Carte CIB"],
        "prohibited_claims": ["guaranteed approval"],
    }


def replay_row(case_id: str = "PUB-001", evidence: dict | None = None) -> dict:
    return {
        "case_id": case_id,
        "question": "What is Carte CIB?",
        "authentication_scenario": '{"authenticated": false, "customer_id": null}',
        "expected_semantic_intent": "PUBLIC_INFO",
        "predicted_intent": "PUBLIC_INFO",
        "confidence": 0.9,
        "actual_route": "PUBLIC_INFO",
        "actual_escalated": False,
        "policy_outcome": "ALLOW",
        "failure_code": "",
        "operational_status": "completed_behavioral_case",
        "generation_executed": evidence is not None,
        "final_answer": "Carte CIB is a card.",
        "retrieved_document_ids": ["DISCARDED"],
        "generation_contexts": [{"message": "What is Carte CIB?", "evidence": evidence}] if evidence is not None else [],
    }


class ChatbotEvidenceExportTests(SimpleTestCase):
    def test_rag_passage_is_exact_generation_context_text(self):
        evidence = {
            "public_information": [
                {
                    "answer": "Exact RAG answer supplied to generation.",
                    "citations": [{"citation_id": "S1", "document_id": "Q00001", "source_title": "Title", "source_url": "https://example.test"}],
                }
            ]
        }
        records = build_evidence_records([case()], [replay_row(evidence=evidence)], "run")
        passage = records[0]["exact_retrieved_passages_supplied_to_generation"][0]
        self.assertEqual(passage["text"], "Exact RAG answer supplied to generation.")
        self.assertEqual(passage["document_id"], "Q00001")

    def test_discarded_retrieval_results_are_not_exported(self):
        evidence = {"public_information": [{"answer": "Used text", "citations": [{"document_id": "Q00001"}]}]}
        records = build_evidence_records([case()], [replay_row(evidence=evidence)], "run")
        self.assertEqual(records[0]["retrieved_document_ids"], ["Q00001"])
        self.assertNotIn("DISCARDED", records[0]["retrieved_document_ids"])

    def test_customer_context_is_minimized_and_sanitized(self):
        evidence = {"customer_context": {"customer": {"customer_id": 169887, "segment": "Digital"}, "owned_products": [{"product_name": "Carte CIB"}]}}
        row = replay_row("CUST-001", evidence=evidence)
        row["authentication_scenario"] = '{"authenticated": true, "customer_id": 169887}'
        row["actual_route"] = "CUSTOMER_CONTEXT"
        records = build_evidence_records([case("CUST-001")], [row], "run")
        customer = records[0]["sanitized_customer_context_supplied_to_generation"]
        self.assertEqual(customer["customer"]["customer_id"], "EVAL_CUSTOMER_1")
        self.assertEqual(customer["owned_products"][0]["product_name"], "Carte CIB")

    def test_blocked_case_exports_no_retrieved_context(self):
        row = replay_row("UNAUTH-001", evidence=None)
        row["actual_escalated"] = True
        row["actual_route"] = "ESCALATE"
        row["policy_outcome"] = "UNAUTHENTICATED_BLOCK"
        row["failure_code"] = "authentication_required"
        records = build_evidence_records([case("UNAUTH-001")], [row], "run")
        self.assertEqual(records[0]["evidence_type"], "deterministic_policy")
        self.assertEqual(records[0]["exact_retrieved_passages_supplied_to_generation"], [])
        self.assertEqual(records[0]["sanitized_customer_context_supplied_to_generation"], {})

    def test_duplicate_or_missing_case_ids_fail_validation(self):
        evidence = {"public_information": [{"answer": "Used text", "citations": [{"document_id": "Q00001"}]}]}
        records = build_evidence_records([case()], [replay_row(evidence=evidence)], "run")
        validate_evidence_records([case()], records)
        with self.assertRaises(ValueError):
            validate_evidence_records([case(), case()], records + records)
        with self.assertRaises(ValueError):
            validate_evidence_records([case(), case("PUB-002")], records)

    def test_replay_differences_cannot_silently_replace_original(self):
        original = [{"case_id": "PUB-001", "actual_route": "PUBLIC_INFO", "final_answer": "Original", "retrieved_document_ids": '["Q00001"]'}]
        replay = [replay_row(evidence={"public_information": [{"answer": "Replay", "citations": [{"document_id": "Q00002"}]}]})]
        replay[0]["final_answer"] = "Replay"
        replay[0]["retrieved_document_ids"] = ["Q00002"]
        differences = compare_replay(original, replay)
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["replay_can_validly_support_review_of_original_response"], "no")

    def test_existing_manual_judgments_survive_review_join(self):
        command = Command()
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            review_path = temp_path / "review.csv"
            output_path = temp_path / "ready.csv"
            with review_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id",
                        "question",
                        "expected_semantic_intent",
                        "actual_route",
                        "final_answer",
                        "grounding",
                        "correctness",
                        "completeness",
                        "clarity",
                        "safe_response_or_escalation",
                        "reviewer_notes",
                        "review_status",
                    ],
                )
                writer.writeheader()
                writer.writerow({"case_id": "PUB-001", "question": "q", "grounding": "manual-pass", "correctness": "manual-pass"})
            original = [{"case_id": "PUB-001", "actual_route": "PUBLIC_INFO", "final_answer": "answer", "end_to_end_pass": "True"}]
            record = build_evidence_records(
                [case()],
                [replay_row(evidence={"public_information": [{"answer": "Used", "citations": [{"document_id": "Q00001"}]}]})],
                "run",
            )
            command._write_manual_review_ready(output_path, review_path, original, record, [])
            rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
        self.assertEqual(rows[0]["grounding"], "manual-pass")
        self.assertEqual(rows[0]["correctness"], "manual-pass")\n