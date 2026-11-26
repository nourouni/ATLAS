from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.staticfiles import finders
from django.test import Client, SimpleTestCase, override_settings

from banking.management.commands.create_demo_users import CUSTOMER_USERNAMES


class ChatbotFrontendTests(SimpleTestCase):
    databases = {"default"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from django.core.management import call_command

        cls._temp_dir = tempfile.TemporaryDirectory()
        call_command("create_demo_users", output=str(Path(cls._temp_dir.name) / "accounts.md"), verbosity=0)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def test_demo_page_requires_login(self):
        response = Client().get("/chatbot/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    def test_demo_page_loads_for_authenticated_user_and_points_to_frozen_graph_endpoint(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        response = client.get("/chatbot/")

        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertIn("Atlas Demo Chatbot", content)
        self.assertIn("chatbot_demo.js", content)
        self.assertIn("Customer ID: 187121", content)
        self.assertIn("Logout", content)

    def test_frontend_javascript_posts_message_and_history_to_graph_endpoint(self):
        path = finders.find("banking/chatbot_demo.js")

        self.assertIsNotNone(path)
        script = Path(path).read_text(encoding="utf-8")
        self.assertIn('const endpoint = "/api/chatbot/graph/";', script)
        self.assertIn("JSON.stringify({ message: text, history: history.slice(-6), conversation_state: conversationState })", script)
        self.assertNotIn("customer_id", script)
        self.assertNotIn("innerHTML", script)

    def test_frontend_formats_assistant_responses_safely(self):
        path = finders.find("banking/chatbot_demo.js")

        self.assertIsNotNone(path)
        script = Path(path).read_text(encoding="utf-8")
        self.assertIn("appendInlineFormattedText", script)
        self.assertIn("removeCitationMarkers", script)
        self.assertIn('document.createElement("strong")', script)
        self.assertIn('document.createElement("br")', script)
        self.assertIn("unordered-list", script)
        self.assertIn("ordered-list", script)
        self.assertIn('document.createElement(section.type === "ordered-list" ? "ol" : "ul")', script)
        self.assertIn('section.type === "heading"', script)
        self.assertIn("assistant-sources", script)
        self.assertIn('link.rel = "noopener noreferrer"', script)
        self.assertNotIn("citation-chip", script)
        self.assertNotIn("innerHTML", script)

    def test_graph_endpoint_contract_uses_message_without_browser_customer_id(self):
        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {
                "final_answer": "ok",
                "intent": "UNKNOWN",
                "route": "ESCALATE",
                "failure_code": "unknown_intent",
            }
            response = Client().post(
                "/api/chatbot/graph/",
                data={"message": "Hello", "customer_id": 999},
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        graph_input = mocked_graph.invoke.call_args.args[0]
        self.assertEqual(graph_input["message"], "Hello")
        self.assertIsNone(graph_input["customer_id"])
        self.assertEqual(graph_input["classifier_model"], "qwen2.5:7b-instruct")
        self.assertEqual(graph_input["generator_model"], "qwen2.5:7b-instruct")

    def test_authenticated_graph_uses_profile_customer_mapping(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {"final_answer": "ok", "intent": "CUSTOMER_CONTEXT", "route": "CUSTOMER_CONTEXT"}
            response = client.post(
                "/api/chatbot/graph/",
                data=json.dumps({"message": "what are my products", "customer_id": 999999}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mocked_graph.invoke.call_args.args[0]["customer_id"], user.demo_profile.customer_id)
        self.assertTrue(mocked_graph.invoke.call_args.args[0]["authenticated"])

    @override_settings(DEBUG=True)
    def test_demo_page_keeps_model_selection_bar_hidden(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        response = client.get("/chatbot/")

        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertNotIn("/api/chatbot/dev/models/", content)
        self.assertNotIn("data-dev-model-switcher", content)
        return
        self.assertIn("Phi → Phi", content)
        self.assertIn("Phi → Qwen", content)
        self.assertIn("Qwen → Qwen", content)
        self.assertIn("Classifier: Qwen", content)

    @override_settings(DEBUG=False)
    def test_demo_page_hides_debug_model_switcher_when_debug_false(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        response = client.get("/chatbot/")

        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")
        self.assertNotIn("/api/chatbot/dev/models/", content)
        self.assertNotIn("data-dev-model-switcher", content)

    @override_settings(DEBUG=True)
    def test_dev_model_switch_endpoint_sets_whitelisted_session_models(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client(enforce_csrf_checks=True)
        client.force_login(user)
        client.get("/chatbot/")
        token = client.cookies["csrftoken"].value

        response = client.post(
            "/api/chatbot/dev/models/",
            data=json.dumps({"configuration": "phi_qwen"}),
            content_type="application/json",
            HTTP_X_CSRFTOKEN=token,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["debug_models"]["classifier"], "phi4-mini:latest")
        self.assertEqual(response.json()["debug_models"]["generator"], "qwen2.5:7b-instruct")
        session = client.session
        self.assertEqual(session["dev_classifier_model"], "phi4-mini:latest")
        self.assertEqual(session["dev_generator_model"], "qwen2.5:7b-instruct")

    @override_settings(DEBUG=True)
    def test_dev_model_switch_endpoint_rejects_arbitrary_models(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)

        response = client.post(
            "/api/chatbot/dev/models/",
            data=json.dumps({"configuration": "llama"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "unsupported_model_config")

    @override_settings(DEBUG=False)
    def test_dev_model_switch_endpoint_disabled_when_debug_false(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)

        response = client.post(
            "/api/chatbot/dev/models/",
            data=json.dumps({"configuration": "qwen_qwen"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)

    @override_settings(DEBUG=True)
    def test_graph_uses_session_model_override_and_returns_debug_metadata(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        session = client.session
        session["dev_classifier_model"] = "qwen2.5:7b-instruct"
        session["dev_generator_model"] = "qwen2.5:7b-instruct"
        session.save()

        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {"final_answer": "ok", "intent": "PUBLIC_INFO", "route": "PUBLIC_INFO"}
            response = client.post(
                "/api/chatbot/graph/",
                data=json.dumps({"message": "what is credobliga"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        graph_input = mocked_graph.invoke.call_args.args[0]
        self.assertEqual(graph_input["classifier_model"], "qwen2.5:7b-instruct")
        self.assertEqual(graph_input["generator_model"], "qwen2.5:7b-instruct")
        self.assertEqual(response.json()["debug_models"]["classifier"], "qwen2.5:7b-instruct")

    @override_settings(DEBUG=False)
    def test_graph_ignores_session_model_override_and_hides_debug_metadata_when_debug_false(self):
        from django.contrib.auth import get_user_model

        user = get_user_model().objects.get(username=CUSTOMER_USERNAMES[0])
        client = Client()
        client.force_login(user)
        session = client.session
        session["dev_classifier_model"] = "qwen2.5:7b-instruct"
        session["dev_generator_model"] = "qwen2.5:7b-instruct"
        session.save()

        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {"final_answer": "ok", "intent": "PUBLIC_INFO", "route": "PUBLIC_INFO"}
            response = client.post(
                "/api/chatbot/graph/",
                data=json.dumps({"message": "what is credobliga"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        graph_input = mocked_graph.invoke.call_args.args[0]
        self.assertEqual(graph_input["classifier_model"], "qwen2.5:7b-instruct")
        self.assertEqual(graph_input["generator_model"], "qwen2.5:7b-instruct")
        self.assertNotIn("debug_models", response.json())
