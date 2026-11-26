from __future__ import annotations

import os
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.db import connection
from django.db.models import Coun
from django.test import Client, SimpleTestCase

from .management.commands.create_demo_users import STAFF_USERNAME
from .models import Customer, CustomerProduct, PropensityPrediction
from .models import DemoUserProfile
from .serializers import chatbot_contex
from .services import (
    CustomerNotFoundError,
    get_customer_dashboard_analytics,
    get_customer_context,
    get_customer_products,
    get_customer_profile,
    get_customer_propensities,
    get_customer_segment,
)


class PostgreSQLConnectionTests(SimpleTestCase):
    databases = {"default"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir = tempfile.TemporaryDirectory()
        call_command("create_demo_users", output=str(Path(cls._temp_dir.name) / "accounts.md"), verbosity=0)
        cls.customer_id = os.environ.get("DJANGO_TEST_CUSTOMER_ID")
        if cls.customer_id is None:
            first_customer = Customer.objects.order_by("customer_id").first()
            cls.customer_id = first_customer.customer_id if first_customer else None
        multi_product = (
            CustomerProduct.objects.values("customer_id")
            .annotate(row_count=Count("*"))
            .filter(row_count__gt=1)
            .order_by("customer_id")
            .first()
        )
        cls.multi_product_customer_id = multi_product["customer_id"] if multi_product else None
        multi_propensity = (
            PropensityPrediction.objects.values("customer_id")
            .annotate(row_count=Count("*"))
            .filter(row_count__gt=1)
            .order_by("customer_id")
            .first()
        )
        cls.multi_propensity_customer_id = multi_propensity["customer_id"] if multi_propensity else None

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.client = Client()

    def test_django_can_connect_to_postgresql(self):
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            self.assertEqual(cursor.fetchone()[0], 1)

    def test_health_endpoint_works(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["database"], "ok")

    def test_known_customer_can_be_retrieved(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        profile = get_customer_profile(self.customer_id)
        self.assertEqual(profile["customer_id"], self.customer_id)

    def test_related_customer_data_can_be_retrieved(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        self.assertIsInstance(get_customer_products(self.customer_id), list)
        self.assertIsInstance(get_customer_segment(self.customer_id), (dict, type(None)))
        self.assertIsInstance(get_customer_propensities(self.customer_id), list)
        context = get_customer_context(self.customer_id)
        self.assertIn("owned_products", context)
        self.assertIn("propensities", context)

    def test_customer_context_is_coherent_for_recommendation_customer(self):
        customer = (
            PropensityPrediction.objects.values("customer_id")
            .annotate(row_count=Count("*"))
            .filter(row_count__gt=1)
            .order_by("customer_id")
            .first()
        )
        if customer is None:
            self.skipTest("No customer with propensity outputs is loaded in PostgreSQL.")

        context = get_customer_context(customer["customer_id"])
        self.assertTrue(context["customer"]["behavioral_segment_id"])
        self.assertIsNotNone(context["segment"])
        self.assertTrue(context["propensities"])
        self.assertTrue(context["recommendations"])

        propensity_keys = {
            (item["product_id"], item["rank"], item["run_id"], item["propensity_score"])
            for item in context["propensities"]
        }
        for recommendation in context["recommendations"]:
            key = (
                recommendation["product_id"],
                recommendation["rank"],
                recommendation["run_id"],
                recommendation["score"],
            )
            self.assertIn(key, propensity_keys)

        if context["customer"]["forecast_pool"]:
            self.assertTrue(context["forecasts"])
            self.assertTrue(all(item["metric"] == "avg_balance" for item in context["forecasts"]))

    def test_invalid_customer_id_returns_clean_error(self):
        invalid_id = 999999999999
        with self.assertRaises(CustomerNotFoundError):
            get_customer_profile(invalid_id)

        response = self.client.get(f"/api/customers/{invalid_id}/")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.json())

    def test_chatbot_context_endpoint_returns_limited_fields(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        response = self.client.get(f"/api/chatbot/customers/{self.customer_id}/context/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("customer", payload)
        self.assertIn("owned_products", payload)
        self.assertIn("propensities", payload)
        self.assertIn("recommendations", payload)
        self.assertNotIn("forecasts", payload)
        self.assertNotIn("source_system", payload["customer"])
        if payload["propensities"]:
            self.assertNotIn("run_id", payload["propensities"][0])
            self.assertNotIn("segment_product_prior", payload["propensities"][0])

    def test_chatbot_recommendations_endpoint_handles_missing_outputs(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        response = self.client.get(f"/api/chatbot/customers/{self.customer_id}/recommendations/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["customer_id"], self.customer_id)
        self.assertIsInstance(payload["recommendations"], list)

    def test_dashboard_customer_endpoint_returns_richer_fields(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        self.client.force_login(DemoUserProfile.objects.get(user__username=STAFF_USERNAME).user)
        response = self.client.get(f"/api/dashboard/customers/{self.customer_id}/")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("forecasts", payload)
        self.assertIn("source_system", payload["customer"])
        if payload["propensities"]:
            self.assertIn("run_id", payload["propensities"][0])
            self.assertIn("segment_product_prior", payload["propensities"][0])

    def test_dashboard_analytics_reports_missing_outputs_cleanly(self):
        if self.customer_id is None:
            self.skipTest("No customers are loaded in PostgreSQL.")
        analytics = get_customer_dashboard_analytics(self.customer_id)
        self.assertIn("missing_outputs", analytics)
        self.assertIn("model_runs", analytics)
        for value in analytics["missing_outputs"].values():
            self.assertIsInstance(value, bool)

    def test_multiple_products_can_be_returned(self):
        if self.multi_product_customer_id is None:
            self.skipTest("No customer with multiple products is loaded in PostgreSQL.")
        products = get_customer_products(self.multi_product_customer_id)
        self.assertGreater(len(products), 1)

    def test_multiple_propensities_can_be_returned(self):
        if self.multi_propensity_customer_id is None:
            self.skipTest("No customer with multiple propensities is loaded in PostgreSQL.")
        propensities = get_customer_propensities(self.multi_propensity_customer_id)
        self.assertGreater(len(propensities), 1)

    def test_unknown_customer_returns_clean_errors_for_new_endpoints(self):
        invalid_id = 999999999999
        paths = [
            f"/api/chatbot/customers/{invalid_id}/context/",
            f"/api/chatbot/customers/{invalid_id}/recommendations/",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
            self.assertIn("error", response.json())

        self.client.force_login(DemoUserProfile.objects.get(user__username=STAFF_USERNAME).user)
        dashboard_paths = [
            f"/api/dashboard/customers/{invalid_id}/",
            f"/api/dashboard/customers/{invalid_id}/analytics/",
        ]
        for path in dashboard_paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
            self.assertIn("error", response.json())

    def test_chatbot_serializer_handles_empty_analytical_outputs(self):
        context = {
            "customer": {
                "customer_id": 1,
                "customer_class": "retail",
                "employment": None,
                "income_decile": None,
                "tenure_days": None,
                "source_system": "simulated",
            },
            "segment": None,
            "owned_products": [],
            "propensities": [],
            "forecasts": [],
            "recommendations": [],
        }
        payload = chatbot_context(context)
        self.assertEqual(payload["owned_products"], [])
        self.assertEqual(payload["propensities"], [])
        self.assertEqual(payload["recommendations"], [])
        self.assertNotIn("forecasts", payload)
        self.assertNotIn("source_system", payload["customer"])
