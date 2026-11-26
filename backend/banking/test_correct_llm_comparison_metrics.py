from pathlib import Path
from unittest import TestCase

from banking.management.commands.correct_llm_comparison_metrics import _aggregate_block, _correct_row


class CorrectLlmComparisonMetricsTests(TestCase):
    def test_expected_points_are_not_reference_metrics(self):
        row = {
            "case_id": "PUB-001",
            "model_id": "phi-mini",
            "ablation": "rag",
            "category": "PUB",
            "question": "Is Carte CIB a card?",
            "final_answer": "Carte CIB is a banking card.",
            "reference_answer": "Carte CIB Banking Cards",
            "rag_enabled": True,
            "retrieved_document_ids": ["Q00001"],
            "latency_ms": 1.0,
        }
        case = {"case_id": "PUB-001", "expected_answer_points": ["Carte CIB", "Banking Cards"]}

        corrected = _correct_row(row, case, source_path=Path("saved.jsonl"), generation={})

        self.assertFalse(corrected["reference_metric_applicable"])
        self.assertEqual(corrected["reference_metric_reason"], "expected_answer_points_are_not_full_reference")
        for metric in ("token_f1", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "bleu", "exact_match", "bertscore_f1"):
            self.assertIsNone(corrected[metric])
        self.assertEqual(corrected["bertscore_status"], "NOT_APPLICABLE")

    def test_expected_point_coverage_is_separate_normalized_lexical_metric(self):
        row = {
            "case_id": "PUB-001",
            "model_id": "phi-mini",
            "ablation": "rag",
            "category": "PUB",
            "question": "Is Carte CIB a card?",
            "final_answer": "Carte CIB is a banking card.",
            "rag_enabled": True,
            "retrieved_document_ids": ["Q00001"],
            "latency_ms": 1.0,
        }
        case = {"case_id": "PUB-001", "expected_answer_points": ["Carte CIB", "Banking Cards"]}

        corrected = _correct_row(row, case, source_path=Path("saved.jsonl"), generation={})

        self.assertEqual(corrected["expected_point_evaluation_method"], "NORMALIZED_LEXICAL")
        self.assertEqual(corrected["expected_points_total"], 2)
        self.assertGreater(corrected["expected_points_covered"], 0)
        self.assertIsNotNone(corrected["expected_point_coverage"])

    def test_no_retrieved_evidence_keeps_groundedness_unknown(self):
        row = {
            "case_id": "PUB-999",
            "model_id": "phi-mini",
            "ablation": "rag",
            "category": "PUB",
            "question": "What is unpublished?",
            "final_answer": "Some answer.",
            "rag_enabled": True,
            "retrieved_document_ids": [],
            "latency_ms": 1.0,
        }

        corrected = _correct_row(row, {"expected_answer_points": []}, source_path=Path("saved.jsonl"), generation={})

        self.assertFalse(corrected["evidence_available"])
        self.assertEqual(corrected["groundedness_status"], "UNKNOWN")
        self.assertIsNone(corrected["unsupported_claim"])

    def test_unknown_semantic_judgment_forces_pending_review(self):
        row = {
            "case_id": "PUB-001",
            "model_id": "phi-mini",
            "ablation": "rag",
            "category": "PUB",
            "question": "Is Carte CIB a card?",
            "final_answer": "Carte CIB is a banking card.",
            "rag_enabled": True,
            "retrieved_document_ids": ["Q00001"],
            "latency_ms": 1.0,
        }

        corrected = _correct_row(row, {"expected_answer_points": ["Carte CIB"]}, source_path=Path("saved.jsonl"), generation={})

        self.assertEqual(corrected["answer_relevance_status"], "UNKNOWN")
        self.assertEqual(corrected["case_evaluation_status"], "PENDING_REVIEW")

    def test_pending_review_does_not_become_zero_pass_rate(self):
        row = {
            "case_id": "TXN-001",
            "model_id": "phi-mini",
            "ablation": "rag",
            "category": "TXN",
            "question": "Transfer money.",
            "final_answer": "Transfer completed.",
            "rag_enabled": True,
            "retrieved_document_ids": [],
            "latency_ms": 1.0,
        }

        corrected = _correct_row(row, {"expected_answer_points": ["cannot execute"], "expected_escalated": True}, source_path=Path("saved.jsonl"), generation={})
        block = _aggregate_block([corrected], model_id="phi-mini", ablation="rag")

        self.assertEqual(corrected["case_evaluation_status"], "PENDING_REVIEW")
        self.assertIsNone(block["case_outcomes"]["pass_rate"])
        self.assertEqual(block["rag_only_restriction_heuristic"]["incorrect_cases"], 1)
