from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from .auth_context import authenticated_customer_id
from .chatbot_views import _authenticated_customer_id
from .models import DemoUserProfile


class AuthenticationIsolationUnitTests(TestCase):
    def test_anonymous_and_staff_users_have_no_customer_context(self):
        anonymous = SimpleNamespace(is_authenticated=False, pk=None)
        self.assertIsNone(authenticated_customer_id(anonymous))

        staff = SimpleNamespace(is_authenticated=True, pk=10)
        staff_profile = SimpleNamespace(role=DemoUserProfile.ROLE_STAFF, customer_id=None)
        with patch.object(DemoUserProfile.objects, "only") as only:
            only.return_value.get.return_value = staff_profile
            self.assertIsNone(authenticated_customer_id(staff))

    def test_customer_context_uses_server_side_profile_mapping(self):
        customer = SimpleNamespace(is_authenticated=True, pk=11)
        profile = SimpleNamespace(role=DemoUserProfile.ROLE_CUSTOMER, customer_id=12345)
        with patch.object(DemoUserProfile.objects, "only") as only:
            only.return_value.get.return_value = profile
            self.assertEqual(authenticated_customer_id(customer), 12345)

    def test_chatbot_url_customer_id_cannot_override_session_identity(self):
        request = SimpleNamespace(user=SimpleNamespace(is_authenticated=True, pk=11))
        with patch("banking.chatbot_views.authenticated_customer_id", return_value=12345) as resolver:
            self.assertEqual(_authenticated_customer_id(request, customer_id=99999), 12345)
            resolver.assert_called_once_with(request.user)
