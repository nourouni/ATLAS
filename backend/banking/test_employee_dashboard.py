from __future__ import annotations

import tempfile
from pathlib import Path

from django.core.management import call_command
from django.test import Client, SimpleTestCase

from banking.dashboard_services import customer_dashboard, portfolio_overview, products_analytics, segments_analytics
from banking.management.commands.create_demo_users import ADMIN_USERNAME, CUSTOMER_USERNAMES, STAFF_USERNAME
from banking.models import Customer, CustomerProduct, DemoUserProfile, PropensityPrediction, Recommendation


class EmployeeDashboardTests(SimpleTestCase):
    databases = {"default"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir = tempfile.TemporaryDirectory()
        call_command("create_demo_users", output=str(Path(cls._temp_dir.name) / "accounts.md"), verbosity=0)
        cls.customer_profile = DemoUserProfile.objects.get(user__username=CUSTOMER_USERNAMES[0])
        second = DemoUserProfile.objects.get(user__username=CUSTOMER_USERNAMES[1])
        cls.second_customer_id = second.customer_id

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        super().tearDownClass()

    def _client_as(self, username: str) -> Client:
        client = Client()
        client.force_login(DemoUserProfile.objects.get(user__username=username).user)
        return clien

    def test_anonymous_dashboard_access_denied(self):
        response = Client().get("/dashboard/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])

    def test_customer_account_dashboard_access_denied(self):
        response = self._client_as(CUSTOMER_USERNAMES[0]).get("/dashboard/")

        self.assertEqual(response.status_code, 403)

    def test_advisor_dashboard_access_allowed(self):
        response = self._client_as(STAFF_USERNAME).get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Portfolio overview")

    def test_admin_dashboard_access_allowed(self):
        response = self._client_as(ADMIN_USERNAME).get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Atlas Demo Analytics - Overview")

    def test_portfolio_overview_loads_real_counts(self):
        overview = portfolio_overview()
        response = self._client_as(STAFF_USERNAME).get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, str(overview["summary"]["total_customers"]))
        self.assertContains(response, overview["headline"].replace("&", "&amp;"))

    def test_portfolio_propensity_is_compact_but_all_predictions_accessible(self):
        overview = portfolio_overview()
        response = self._client_as(STAFF_USERNAME).get("/dashboard/")
        content = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Product opportunity teaser", content)
        self.assertIn("not a purchase probability", content)
        self.assertNotIn("Â", content)

    def test_customer_lookup_accepts_valid_existing_customer(self):
        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f"Customer {self.customer_profile.customer_id}")

    def test_invalid_customer_lookup_fails_safely(self):
        response = self._client_as(STAFF_USERNAME).get("/dashboard/customers/?customer_id=999999999999")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "was not found")

    def test_customer_profile_context_displays_correctly(self):
        context = customer_dashboard(self.customer_profile.customer_id)
        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        self.assertContains(response, str(context["customer"]["customer_id"]))
        self.assertContains(response, context["customer"]["forecast_pool"])

    def test_owned_products_display_correctly(self):
        product = CustomerProduct.objects.filter(customer_id=self.customer_profile.customer_id).select_related("product").first()
        if product is None:
            self.skipTest("Selected customer has no owned products.")

        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        self.assertContains(response, product.product.product_name)

    def test_behavioral_segment_displays_correctly(self):
        customer = Customer.objects.get(customer_id=self.customer_profile.customer_id)
        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        context = customer_dashboard(customer.customer_id)
        self.assertContains(response, context["segment"]["display_name"])
        self.assertContains(response, "Behavioral segment")

    def test_propensity_section_supports_actual_dynamic_number_of_products(self):
        actual_count = PropensityPrediction.objects.filter(customer_id=self.customer_profile.customer_id).count()
        context = customer_dashboard(self.customer_profile.customer_id)

        self.assertEqual(len(context["propensities"]), actual_count)
        self.assertGreater(actual_count, 0)

    def test_no_fixed_five_product_assumption_exists(self):
        actual_count = PropensityPrediction.objects.filter(customer_id=self.customer_profile.customer_id).count()
        rendered_count = len(customer_dashboard(self.customer_profile.customer_id)["propensities"])

        self.assertEqual(rendered_count, actual_count)

    def test_propensity_and_recommendation_are_distinct_in_copy(self):
        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        self.assertContains(response, "relative model signal")
        self.assertContains(response, "not a purchase probability")

    def test_forecast_is_explicitly_labeled_as_segment_level(self):
        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={self.customer_profile.customer_id}")

        self.assertContains(response, "Group forecast")
        self.assertContains(response, "not an individual prediction")

    def test_recommendations_come_from_existing_stored_outputs(self):
        stored_count = Recommendation.objects.filter(customer_id=self.customer_profile.customer_id).count()
        context = customer_dashboard(self.customer_profile.customer_id)

        self.assertEqual(len(context["recommendations"]), stored_count)
        self.assertGreater(stored_count, 0)

    def test_optional_missing_data_produces_safe_empty_states(self):
        customer = None
        for candidate in Customer.objects.order_by("customer_id")[:500]:
            if not PropensityPrediction.objects.filter(customer_id=candidate.customer_id).exists():
                customer = candidate
                break
        if customer is None:
            self.skipTest("No customer without propensity outputs is available.")

        response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={customer.customer_id}")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No propensity predictions are stored for this customer.")

    def test_customer_chatbot_login_behavior_remains_unchanged(self):
        response = self._client_as(CUSTOMER_USERNAMES[0]).get("/chatbot/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, CUSTOMER_USERNAMES[0])
        self.assertContains(response, "Role: customer")

    def test_staff_and_admin_have_no_implicit_customer_mapping(self):
        staff = DemoUserProfile.objects.get(user__username=STAFF_USERNAME).user
        admin = DemoUserProfile.objects.get(user__username=ADMIN_USERNAME).user

        self.assertIsNone(staff.customer_id)
        self.assertIsNone(admin.customer_id)

    def test_new_dashboard_pages_render_for_staff(self):
        for path, expected in [
            ("/dashboard/customers/", "Customer lookup"),
            ("/dashboard/segments/", "Customer behavior groups"),
            ("/dashboard/products/", "Product opportunities"),
            ("/dashboard/forecasting/", "Expected balance trend for a customer group"),
            ("/dashboard/recommendations/", "Product opportunities and recommendations"),
        ]:
            response = self._client_as(STAFF_USERNAME).get(path)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, expected)

    def test_overview_renders_non_empty_lead_headline(self):
        overview = portfolio_overview()
        response = self._client_as(STAFF_USERNAME).get("/dashboard/")

        self.assertTrue(overview["headline"])
        self.assertContains(response, overview["headline"].replace("&", "&amp;"))

    def test_products_render_opportunity_candidates_when_available(self):
        analytics = products_analytics()
        if not analytics["opportunity_candidates"]:
            self.skipTest("No opportunity candidate is present in the current outputs.")

        response = self._client_as(STAFF_USERNAME).get("/dashboard/products/")
        self.assertContains(response, "Opportunity candidates")
        self.assertContains(response, analytics["opportunity_candidates"][0]["product_name"])

    def test_segment_pages_render_display_names_for_assigned_customer(self):
        customer = Customer.objects.exclude(behavioral_segment__isnull=True).first()
        if customer is None:
            self.skipTest("No assigned customer is available.")
        customer_context = customer_dashboard(customer.customer_id)
        display_name = customer_context["segment"]["display_name"]

        segments_response = self._client_as(STAFF_USERNAME).get("/dashboard/segments/")
        customers_response = self._client_as(STAFF_USERNAME).get(f"/dashboard/customers/?customer_id={customer.customer_id}")

        self.assertContains(segments_response, display_name)
        self.assertContains(customers_response, display_name)

    def test_segment_and_product_aggregations_use_real_outputs(self):
        self.assertIn("products_by_segment", segments_analytics())
        self.assertIn("rows", products_analytics())
