from __future__ import annotations

from django.db import connection
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .permissions import api_login_required
from .services import CustomerNotFoundError, get_customer_context, get_customer_profile


def error_response(message: str, status: int) -> JsonResponse:
    return JsonResponse({"error": {"message": message}}, status=status)


@require_GET
def health(request):
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()
    return JsonResponse({"status": "ok", "database": "ok"})


@require_GET
@api_login_required
def customer_detail(request, customer_id: int):
    try:
        return JsonResponse(get_customer_profile(customer_id))
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)


@require_GET
@api_login_required
def customer_context(request, customer_id: int):
    try:
        return JsonResponse(get_customer_context(customer_id))
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)
