from __future__ import annotations

from banking.models import DemoUserProfile


def authenticated_customer_id(user) -> int | None:
    """Return the customer bound to this authenticated Django user."""
    if not getattr(user, "is_authenticated", False) or not getattr(user, "pk", None):
        return None
    try:
        profile = DemoUserProfile.objects.only("role", "customer_id").get(user_id=user.pk)
    except DemoUserProfile.DoesNotExist:
        return None
    if profile.role != DemoUserProfile.ROLE_CUSTOMER or profile.customer_id is None:
        return None
    return int(profile.customer_id)
