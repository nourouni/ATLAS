from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.db import DatabaseError, connection
from django.test import Client, SimpleTestCase, override_settings

from banking.management.commands.create_demo_users import ADMIN_USERNAME, CUSTOMER_USERNAMES, STAFF_USERNAME
from banking.models import CustomerProduct, DemoUserProfile, PropensityPrediction, Recommendation, SegmentForecas


class SecurityClosureTests(SimpleTestCase):
    databases = {"default"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir = tempfile.TemporaryDirectory()
        call_command("create_demo_users", output=str(Path(cls._temp_dir.name) / "accounts.md"), verbosity=0)
        cls.customer_one = DemoUserProfile.objects.get(user__username=CUSTOMER_USERNAMES[0])
        cls.customer_two = DemoUserProfile.objects.get(user__username=CUSTOMER_USERNAMES[1])
        cls.staff = DemoUserProfile.objects.get(user__username=STAFF_USERNAME)
        cls.admin = DemoUserProfile.objects.get(user__username=ADMIN_USERNAME)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def _client_as(self, profile: DemoUserProfile) -> Client:
        client = Client()
        client.force_login(profile.user)
        return clien

    def test_database_runtime_user_is_not_postgresql_superuser(self):
        with connection.cursor() as cursor:
            cursor.execute("select current_user")
            current_user = cursor.fetchone()[0]
            cursor.execute("select usesuper from pg_user where usename=current_user")
            row = cursor.fetchone()

        self.assertEqual(current_user, "atlas_demo_app")
        self.assertIsNotNone(row)
        self.assertFalse(row[0])

    @override_settings(API_REQUIRE_AUTH=True)
    def test_anonymous_cannot_access_private_customer_context(self):
        response = Client().get(f"/api/chatbot/customers/{self.customer_one.customer_id}/context/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    @override_settings(API_REQUIRE_AUTH=True)
    def test_customer_cannot_obtain_second_customer_context_by_url(self):
        response = self._client_as(self.customer_one).get(f"/api/chatbot/customers/{self.customer_two.customer_id}/context/")

        self.assertEqual(response.status_code, 403)
        self.assertNotContains(response, str(self.customer_two.customer_id), status_code=403)

    def test_customer_body_customer_id_does_not_switch_graph_identity(self):
        client = self._client_as(self.customer_one)
        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {
                "final_answer": "ok",
                "intent": "CUSTOMER_CONTEXT",
                "route": "CUSTOMER_CONTEXT",
            }
            response = client.post(
                "/api/chatbot/graph/",
                data={"message": "Show my products", "customer_id": self.customer_two.customer_id},
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mocked_graph.invoke.call_args.args[0]["customer_id"], self.customer_one.customer_id)

    def test_dashboard_lookup_tampering_values_fail_safely(self):
        client = self._client_as(self.staff)
        for value in ["abc", "1;drop table customers", "<script>alert(1)</script>", "999999999999999999999999"]:
            response = client.get("/dashboard/customers/", {"customer_id": value})
            content = response.content.decode("utf-8")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Customer lookup", content)
            self.assertNotIn("Traceback", content)
            self.assertNotIn("SELECT", content.upper())

    def test_customer_role_cannot_abuse_dashboard_lookup(self):
        response = self._client_as(self.customer_one).get("/dashboard/customers/", {"customer_id": self.customer_two.customer_id})

        self.assertEqual(response.status_code, 403)

    def test_dashboard_templates_escape_dynamic_customer_fields(self):
        injected = "<script>alert('x')</script>"
        with patch("banking.dashboard_services.portfolio_overview", return_value={"summary": {}}), patch(
            "banking.frontend_views.portfolio_overview",
            return_value={
                "summary": {
                    "total_customers": 1,
                    "retail_customers": None,
                    "enterprise_customers": None,
                    "customer_class_available": False,
                    "behavioral_segments": 0,
                    "assigned_behavioral_segments": 0,
                    "distinct_products": 0,
                    "distinct_propensity_products": 0,
                    "total_recommendations": 0,
                    "latest_run": None,
                },
                "segment_distribution": [],
                "product_ownership": [],
                "propensity_overview": [],
                "recommendation_overview": {"by_product": [], "by_segment": []},
                "model_runs": [],
            },
        ), patch(
            "banking.frontend_views.dashboard_context_for_customer_id",
            return_value=(
                {
                    "customer": {
                        "customer_id": 123,
                        "customer_class": injected,
                        "employment": injected,
                        "income_decile": None,
                        "tenure_days": None,
                        "forecast_pool": "pool",
                        "segment_label": injected,
                        "behavioral_segment_id": "S1",
                    },
                    "segment": {"segment_id": "S1", "description": injected},
                    "owned_products": [{"product_name": injected, "family": "", "category": ""}],
                    "propensities": [],
                    "forecasts": [],
                    "recommendations": [{"rank": 1, "product_name": injected, "score_display": "0.1", "reason": injected}],
                },
                None,
            ),
        ):
            response = self._client_as(self.staff).get("/dashboard/customers/", {"customer_id": "123"})

        content = response.content.decode("utf-8")
        self.assertNotIn(injected, content)
        self.assertIn("&lt;script&gt;", content)

    def test_dashboard_database_failure_is_readable(self):
        with patch("banking.frontend_views.portfolio_overview", side_effect=DatabaseError("sensitive sql path")):
            response = self._client_as(self.staff).get("/dashboard/")

        content = response.content.decode("utf-8")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Dashboard data is temporarily unavailable", content)
        self.assertNotIn("sensitive sql path", content)

    def test_dashboard_api_database_failure_is_readable(self):
        with patch("banking.dashboard_views.get_customer_context", side_effect=DatabaseError("raw db exception")):
            response = self._client_as(self.staff).get(f"/api/dashboard/customers/{self.customer_one.customer_id}/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["message"], "Dashboard data is temporarily unavailable.")

    def test_logout_requires_csrf_when_csrf_checks_are_enforced(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.customer_one.user)

        response = client.post("/logout/")

        self.assertEqual(response.status_code, 403)

    def test_unsupported_transaction_endpoint_response_is_escalated(self):
        client = self._client_as(self.customer_one)
        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {
                "final_answer": "I cannot execute transactions. Please contact an advisor.",
                "intent": "UNKNOWN",
                "intent_confidence": 0.91,
                "route": "ESCALATE",
                "failure_code": "unsupported_transaction",
                "blocked": True,
            }
            response = client.post(
                "/api/chatbot/graph/",
                data=json.dumps({"message": "Transfer 500 TND for me"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["escalated"])
        self.assertEqual(payload["failure_code"], "unsupported_transaction")

    def test_human_required_endpoint_response_is_escalated(self):
        client = self._client_as(self.customer_one)
        with patch("banking.chatbot_views.graph") as mocked_graph:
            mocked_graph.invoke.return_value = {
                "final_answer": "An advisor must review this request.",
                "intent": "HUMAN_REQUIRED",
                "intent_confidence": 0.88,
                "route": "ESCALATE",
                "requires_human": True,
            }
            response = client.post(
                "/api/chatbot/graph/",
                data=json.dumps({"message": "Why was my loan rejected?"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["escalated"])

    def test_dashboard_values_match_postgresql_counts_for_demo_customer(self):
        from banking.dashboard_services import customer_dashboard

        context = customer_dashboard(self.customer_one.customer_id)

        self.assertEqual(len(context["owned_products"]), CustomerProduct.objects.filter(customer_id=self.customer_one.customer_id).count())
        self.assertEqual(len(context["propensities"]), PropensityPrediction.objects.filter(customer_id=self.customer_one.customer_id).count())
        self.assertEqual(len(context["recommendations"]), Recommendation.objects.filter(customer_id=self.customer_one.customer_id).count())
        self.assertEqual(
            len(context["forecasts"]),
            SegmentForecast.objects.filter(pool=context["customer"]["forecast_pool"], metric="avg_balance").count(),
        )
