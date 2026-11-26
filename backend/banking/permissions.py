from __future__ import annotations

from functools import wraps

from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.views import redirect_to_login


def api_login_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if not settings.API_REQUIRE_AUTH:
            return view_func(request, *args, **kwargs)
        if getattr(request, "user", None) is not None and request.user.is_authenticated:
            return view_func(request, *args, **kwargs)
        return redirect_to_login(request.get_full_path())

    return wrapped


def user_can_access_customer(user, customer_id: int) -> bool:
    if not settings.API_REQUIRE_AUTH:
        return True
    if not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_superuser", False) or getattr(user, "is_staff", False):
        return True
    return getattr(user, "customer_id", None) == int(customer_id)


def forbidden_response() -> JsonResponse:
    return JsonResponse({"error": {"message": "You are not allowed to access this customer."}}, status=403)
