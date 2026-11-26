from __future__ import annotations

import json
from unittest.mock import patch

from django.test import TestCase

from banking.orchestration.graph import graph
from banking.orchestration.nodes import _public_evidence_relevance, detect_request_mode


def _classifier(intent: str, confidence: float = 0.95) -> str:
    return json.dumps({"intent": intent, "request_mode": "information", "confidence": confidence, "reason": "test"})


class ExploratoryChatbotRegressionTests(TestCase):
    def test_capability_paraphrases_route_to_public_information(self):
        for question in (
            "Does the bank offer alerts when funds are low?",
            "Could warnings arrive before a scheduled payment causes a shortage?",
            "Can transfers be tracked after they are sent?",
            "Can customers receive confirmation when a transfer arrives?",
            "Does the bank offer automatic bill-buffering features?",
            "Can the bank predict when scheduled expenses may leave me short?",
            "I'd like an alert if upcoming bills may leave me short of money.",
            "It would be useful if customers could get warned before scheduled payments leave insufficient funds.",
            "I want to know whether notifications are available when funds are running low.",
            "Is there a way to receive a warning about an upcoming balance shortage?",
            "I'd like to know whether transfer-arrival notifications are available.",
        ):
            self.assertEqual(detect_request_mode(question), "capability")
            with patch("banking.orchestration.nodes._llm_generate", return_value=_classifier("CUSTOMER_CONTEXT")),
                patch("banking.orchestration.nodes.rag_tool") as rag:
                rag.query.return_value = {
                    "answered": True,
                    "answer": "The service provides balance alerts, transfer tracking, predictions, and automatic bill buffering for scheduled payments.",
                    "citations": [{"citation_id": "S1", "source_title": "Alerts, tracking, predictions, and bill buffering"}],
                    "evidence_count": 1,
                }
                result = graph.invoke({"message": question, "authenticated": False})
            self.assertEqual(result["intent"], "PUBLIC_INFO")
            rag.query.assert_called_once()

    def test_action_paraphrases_are_not_capability_questions(self):
        for question in ("Set a low-balance alert for me.", "Transfer 200 DT.", "Reserve 300 DT from my account.", "Notify me whenever my balance drops below 200 DT.", "Set up a transfer-arrival notification for this transfer."):
            self.assertEqual(detect_request_mode(question), "action")
            with patch("banking.orchestration.nodes._llm_generate"), patch("banking.orchestration.nodes.rag_tool") as rag:
                result = graph.invoke({"message": question, "authenticated": True, "customer_id": 1})
            self.assertIn(result["failure_code"], {"unsupported_action", "unsupported_transaction"})
            rag.query.assert_not_called()

    def test_personal_analytics_is_understood_but_not_invented(self):
        for question in ("Which spending category increased most?", "How has my spending changed recently?"):
            self.assertEqual(detect_request_mode(question), "customer_analytics")
            with patch("banking.orchestration.nodes._llm_generate") as llm, patch("banking.orchestration.nodes.rag_tool") as rag:
                result = graph.invoke({"message": question, "authenticated": True, "customer_id": 1})
            self.assertEqual(result["failure_code"], "customer_analytics_unavailable")
            self.assertNotEqual(result["intent"], "UNKNOWN")
            llm.assert_not_called()
            rag.query.assert_not_called()

    def test_irrelevant_public_evidence_is_rejected_before_generation(self):
        relevant, reason = _public_evidence_relevance(
            "Can I monitor subscription payments?",
            {"answer": "This document explains how to report a lost card.", "citations": [{"citation_id": "S1", "source_title": "Lost card procedure"}], "evidence_count": 1},
        )
        self.assertFalse(relevant)
        self.assertIn(reason, {"topic_mismatch", "related_but_not_answering"})

    def test_customer_analytics_requires_a_personal_computation_request(self):
        self.assertEqual(detect_request_mode("Can the bank predict cash shortages?"), "capability")
        self.assertEqual(detect_request_mode("How much did my transport spending change?"), "customer_analytics")
        self.assertEqual(detect_request_mode("Based on my history, how much do I usually spend before payday?"), "customer_analytics")

    def test_answerability_rejects_related_but_non_answering_documents(self):
        cases = (
            ("Can the bank warn me before my balance gets low?", "Atlas Demo Obligataire SICAV investment fund"),
            ("Can I monitor subscription payments?", "Dormant account maintenance information"),
            ("Can I track a transfer after it is sent?", "Generic money-transfer product overview"),
        )
        for query, title in cases:
            relevant, reason = _public_evidence_relevance(
                query,
                {"answer": title, "citations": [{"citation_id": "S1", "source_title": title}], "evidence_count": 1},
            )
            self.assertFalse(relevant, (query, reason))

        with patch("banking.orchestration.nodes._llm_generate", return_value=_classifier("PUBLIC_INFO")) as llm,
            patch("banking.orchestration.nodes.rag_tool") as rag:
            rag.query.return_value = {
                "answered": True,
                "answer": "This document explains how to report a lost card.",
                "citations": [{"citation_id": "S1", "source_title": "Lost card procedure"}],
                "evidence_count": 1,
            }
            result = graph.invoke({"message": "Can I monitor subscription payments?", "authenticated": False})
        self.assertEqual(result["failure_code"], "public_evidence_unavailable")
        self.assertEqual(llm.call_count, 1)  # classifier only; rejected evidence never reaches generation
