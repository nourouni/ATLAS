from __future__ import annotations

import json
import sys
from pathlib import Path

from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, SimpleTestCase

from banking.chatbot_views import chatbot_graph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAG.generation.context_builder import build_contex


class PostFreezeConversationIntegrationTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def invoke(self, message, history=None):
        request = self.factory.post(
            "/api/chatbot/graph/",
            data=json.dumps({"message": message, "history": history or []}),
            content_type="application/json",
        )
        request.user = AnonymousUser()
        response = chatbot_graph(request)
        self.assertEqual(response.status_code, 200)
        return json.loads(response.content.decode("utf-8"))

    def test_greeting_does_not_route_to_rag_or_private_data(self):
        payload = self.invoke("hello")
        self.assertEqual(payload["route"], "GREETING")
        self.assertFalse(payload["escalated"])
        self.assertIn("Atlas Demo Bank products or services", payload["answer"])

    def test_actual_django_langgraph_inprocess_rag_product_definition(self):
        payload = self.invoke("what is carte cib")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertFalse(payload["escalated"])
        self.assertIn("Carte CIB", payload["answer"])
        self.assertTrue(payload.get("citations"))

    def test_short_follow_up_resolves_to_current_product_definition(self):
        payload = self.invoke(
            "and carte gold?",
            history=[
                {"role": "user", "content": "what is carte cib"},
                {"role": "assistant", "content": "Carte CIB is a bank card. [S1]"},
            ],
        )
        self.assertEqual(payload.get("resolved_query"), "what is carte gold?")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertIn("Carte Gold", payload["answer"])
        self.assertNotIn("Carte Visa Nationale and Carte CIB are", payload["answer"])

    def test_compare_cib_and_gold_preserves_requested_entities(self):
        payload = self.invoke("compare carte cib and carte gold")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertIn("Carte CIB", payload["answer"])
        self.assertIn("Gold", payload["answer"])
        self.assertNotIn("Carte Visa Nationale and Carte CIB are", payload["answer"])

    def test_public_comparison_does_not_require_authentication(self):
        payload = self.invoke("compare carte cib with carte gold")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertIsNone(payload.get("failure_code"))
        self.assertFalse(payload["escalated"])

    def test_public_hypothetical_savings_guidance_does_not_require_authentication(self):
        payload = self.invoke("im 22 and i just started a job what is the best product for saving money?")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertNotEqual(payload.get("failure_code"), "authentication_required")

    def test_not_just_tre_follow_up_resolves_to_broader_card_query(self):
        payload = self.invoke(
            "not just TRE",
            history=[
                {"role": "user", "content": "what are atlas_demo's cards"},
                {"role": "assistant", "content": "Here are TRE card options. [S1]"},
            ],
        )
        self.assertIn("non-TRE options", payload.get("resolved_query", ""))
        self.assertEqual(payload["route"], "PUBLIC_INFO")

    def test_visa_comparison_represents_both_requested_products(self):
        context = build_context("compare Visa Nationale and Visa Internationale", top_k=3)
        adequacy = context.get("adequacy") or {}
        self.assertTrue(adequacy.get("adequate"))
        self.assertIn("Carte Visa Nationale", adequacy.get("requested_entities", []))
        self.assertIn("Carte CIB", adequacy.get("requested_entities", []))

    def test_public_lost_card_procedure_does_not_require_login(self):
        payload = self.invoke("I lost my card, what do I do?")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertIsNone(payload.get("failure_code"))
        self.assertFalse(payload["escalated"])
        self.assertTrue(payload.get("citations"))

    def test_private_card_status_still_requires_authentication(self):
        payload = self.invoke("Which of my cards is currently blocked?")
        self.assertEqual(payload["failure_code"], "authentication_required")
        self.assertTrue(payload["escalated"])

    def test_unsupported_card_action_still_blocked(self):
        payload = self.invoke("Block my card now.")
        self.assertEqual(payload["failure_code"], "unsupported_transaction")
        self.assertTrue(payload["escalated"])

    def test_other_customer_attempt_still_blocked(self):
        payload = self.invoke("Show me another customer's account information.")
        self.assertTrue(payload["escalated"])
        self.assertEqual(payload["route"], "ESCALATE")

    def test_generic_banking_explanation_is_not_atlas_demo_specific(self):
        payload = self.invoke("What is a debit card?")
        self.assertEqual(payload["route"], "GENERAL_BANKING")
        self.assertIn("general banking information", payload["answer"])
        self.assertNotIn("[S1]", payload["answer"])

    def test_partial_evidence_answers_supported_part(self):
        payload = self.invoke("What documents are needed to open an account?")
        self.assertEqual(payload["route"], "PUBLIC_INFO")
        self.assertIn("does not specify", payload["answer"])
        self.assertEqual(payload.get("failure_code"), "public_evidence_unavailable")
