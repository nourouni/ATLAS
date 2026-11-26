from __future__ import annotations

from django.apps import AppConfig


class BankingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "banking"

    def ready(self):
        from django.contrib.auth import get_user_model

        User = get_user_model()
        if hasattr(User, "customer_id"):
            return

        def customer_id(user):
            override = getattr(user, "_demo_customer_id_override", None)
            if override is not None:
                return int(override)
            if not getattr(user, "is_authenticated", False):
                return None
            try:
                profile = user.demo_profile
            except Exception:
                return None
            if profile.role != "customer" or profile.customer_id is None:
                return None
            return int(profile.customer_id)

        def set_customer_id(user, value):
            user._demo_customer_id_override = None if value is None else int(value)

        User.add_to_class("customer_id", property(customer_id, set_customer_id))
