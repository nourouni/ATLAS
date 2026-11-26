from unittest import TestCase

from banking.evaluation.llm_comparison import _pseudo_reference_fields
from banking.management.commands.analyze_langgraph_comparison import (
    correct_reference_metrics,
    source_isolation_decomposition,
)


class LangGraphComparisonAnalysisTests(TestCase):
    def test_future_runner_marks_expected_points_as_pseudo_reference(self):
        fields = _pseudo_reference_fields("Carte CIB Banking Cards")

        self.assertFalse(fields["reference_metric_applicable"])
        self.assertEqual(fields["reference_metric_reason"], "expected_answer_points_are_not_full_reference")
        self.assertIsNone(fields["answer_quality"])
        self.assertEqual(fields["legacy_reference_answer"], "Carte CIB Banking Cards")

    def test_pseudo_reference_quality_is_retained_only_as_legacy(self):
        row = {
            "case_id": "PUB-001",
            "reference_answer": "Carte CIB Banking Cards",
            "answer_quality": {"bleu": 0.5, "token_f1": {"f1": 0.4}},
        }

        corrected = correct_reference_metrics(row)

        self.assertFalse(corrected["reference_metric_applicable"])
        self.assertEqual(corrected["reference_metric_reason"], "expected_answer_points_are_not_full_reference")
        self.assertIsNone(corrected["answer_quality"])
        self.assertEqual(corrected["legacy_answer_quality"], {"bleu": 0.5, "token_f1": {"f1": 0.4}})
        for metric in ("token_f1", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "bleu", "exact_match", "bertscore_f1"):
            self.assertIsNone(corrected[metric])

    def test_classifier_caused_source_violation_is_identified_conservatively(self):
        rows = [
            {
                "case_id": "CUST-001",
                "category": "CUST",
                "expected_semantic_intent": "CUSTOMER_CONTEXT",
                "predicted_intent": "PUBLIC_INFO",
                "expected_route": "CUSTOMER_CONTEXT",
                "actual_route": "PUBLIC_INFO",
                "source_isolation_pass": False,
                "source_usage": {"rag": 1, "customer_context": 0},
                "forbidden_sources": "rag,recommendations",
            }
        ]

        violations, summary = source_isolation_decomposition(rows)

        self.assertEqual(violations[0]["classification"], "CLASSIFIER_CAUSED_WRONG_SOURCE")
        self.assertEqual(summary["classes"]["CLASSIFIER_CAUSED_WRONG_SOURCE"]["count"], 1)

    def test_generator_boundary_violation_requires_correct_upstream_route(self):
        rows = [
            {
                "case_id": "PUB-001",
                "category": "PUB",
                "expected_semantic_intent": "PUBLIC_INFO",
                "predicted_intent": "PUBLIC_INFO",
                "expected_route": "PUBLIC_INFO",
                "actual_route": "PUBLIC_INFO",
                "source_isolation_pass": False,
                "source_usage": {"rag": 1, "customer_context": 1},
                "forbidden_sources": "customer_context,recommendations",
            }
        ]

        violations, summary = source_isolation_decomposition(rows)

        self.assertEqual(violations[0]["classification"], "GENERATOR_SOURCE_BOUNDARY_VIOLATION")
        self.assertEqual(summary["classes"]["GENERATOR_SOURCE_BOUNDARY_VIOLATION"]["count"], 1)
