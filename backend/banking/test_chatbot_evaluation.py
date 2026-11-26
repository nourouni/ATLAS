from __future__ import annotations

from django.test import SimpleTestCase

from banking.evaluation.metrics import calculate_metrics


class ChatbotEvaluationMetricTests(SimpleTestCase):
    def test_calculates_core_rates_and_prf(self):
        rows = [
            {
                "expected_intent": "PUBLIC_INFO",
                "expected_semantic_intent": "PUBLIC_INFO",
                "predicted_intent": "PUBLIC_INFO",
                "intent_score_applicable": True,
                "expected_route": "PUBLIC_INFO",
                "actual_route": "PUBLIC_INFO",
                "expected_escalated": False,
                "actual_escalated": False,
                "policy_pass": True,
                "source_isolation_pass": True,
                "unsupported_claim_indicator": False,
                "end_to_end_pass": True,
                "latency_ms": 10,
            },
            {
                "expected_intent": "UNKNOWN",
                "expected_semantic_intent": "UNKNOWN",
                "predicted_intent": "PUBLIC_INFO",
                "intent_score_applicable": True,
                "expected_route": "ESCALATE",
                "actual_route": "PUBLIC_INFO",
                "expected_escalated": True,
                "actual_escalated": False,
                "policy_pass": False,
                "source_isolation_pass": False,
                "unsupported_claim_indicator": True,
                "end_to_end_pass": False,
                "latency_ms": 30,
            },
        ]
        metrics = calculate_metrics(rows)
        self.assertEqual(metrics["total_dataset_cases"], 2)
        self.assertEqual(metrics["intent_accuracy"]["value"], 0.5)
        self.assertEqual(metrics["routing_accuracy"]["value"], 0.5)
        self.assertEqual(metrics["authentication_policy_pass_rate"]["value"], 0.5)
        self.assertEqual(metrics["source_isolation_violation_rate"]["value"], 0.5)
        self.assertEqual(metrics["unsupported_claim_indicator_rate"]["value"], 0.5)
        self.assertEqual(metrics["end_to_end_pass_rate"]["value"], 0.5)
        self.assertEqual(metrics["latency_ms"]["mean"], 20)
        self.assertEqual(metrics["per_intent"]["PUBLIC_INFO"]["precision"], 0.5)

    def test_handles_empty_rows(self):
        metrics = calculate_metrics([])
        self.assertEqual(metrics["total_dataset_cases"], 0)
        self.assertEqual(metrics["intent_accuracy"]["value"], 0.0)
        self.assertEqual(metrics["latency_ms"]["mean"], 0.0)
