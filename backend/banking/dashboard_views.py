from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.db import DatabaseError
from django.http import HttpResponseForbidden, JsonResponse
from django.views.decorators.http import require_GET

from .dashboard_services import employee_authorized
from .serializers import dashboard_contex
from .services import CustomerNotFoundError, get_customer_context, get_customer_dashboard_analytics
from .views import error_response


def _employee_required(view_func):
    @login_required
    def wrapper(request, *args, **kwargs):
        if not employee_authorized(request.user):
            return HttpResponseForbidden("Dashboard access is available only to advisor/staff and admin demo users.")
        return view_func(request, *args, **kwargs)

    return wrapper


@require_GET
@_employee_required
def customer_detail(request, customer_id: int):
    try:
        return JsonResponse(dashboard_context(get_customer_context(customer_id)))
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)
    except DatabaseError:
        return error_response("Dashboard data is temporarily unavailable.", 503)


@require_GET
@_employee_required
def customer_analytics(request, customer_id: int):
    try:
        return JsonResponse(get_customer_dashboard_analytics(customer_id))
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)
    except DatabaseError:
        return error_response("Dashboard data is temporarily unavailable.", 503)
