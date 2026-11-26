from __future__ import annotations

from django.conf import settings
from django.contrib.auth import authenticate, login, logou
from django.contrib.auth.decorators import login_required
from django.db import DatabaseError
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .chatbot_models import ALLOWED_CHATBOT_CONFIGS, active_model_config
from .auth_context import authenticated_customer_id
from .dashboard_services import (
    dashboard_context_for_customer_id,
    employee_authorized,
    employee_role,
    forecasting_analytics,
    portfolio_overview,
    products_analytics,
    recommendations_analytics,
    segments_analytics,
)


def _safe_next(request):
    next_url = request.POST.get("next") or request.GET.get("next")
    if next_url and url_has_allowed_host_and_scheme(
        next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return next_url
    return reverse("chatbot-demo")


def _demo_role(user):
    if not user.is_authenticated:
        return "anonymous"
    try:
        profile = user.demo_profile
    except Exception:
        if user.is_superuser:
            return "admin"
        if user.is_staff:
            return "staff"
        return "user"
    return profile.role


def _employee_required(view_func):
    @login_required
    def wrapper(request, *args, **kwargs):
        if not employee_authorized(request.user):
            return HttpResponseForbidden("Dashboard access is available only to advisor/staff and admin demo users.")
        return view_func(request, *args, **kwargs)

    return wrapper


@ensure_csrf_cookie
@login_required
@require_GET
def chatbot_demo(request):
    return render(
        request,
        "banking/chatbot_demo.html",
        {
            "demo_role": _demo_role(request.user),
            "can_access_dashboard": employee_authorized(request.user),
            "authenticated_customer_id": authenticated_customer_id(request.user),
            "debug_models_enabled": settings.DEBUG,
            "active_model_config": active_model_config(request) if settings.DEBUG else None,
            "model_config_options": ALLOWED_CHATBOT_CONFIGS if settings.DEBUG else {},
        },
    )


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_home(request):
    try:
        overview = portfolio_overview()
    except DatabaseError:
        overview = None
        dashboard_error = "Dashboard data is temporarily unavailable. Please try again later."
    else:
        dashboard_error = None
    return render(
        request,
        "banking/dashboard_overview.html",
        {
            "page_key": "overview",
            "page_title": "Overview",
            "demo_role": employee_role(request.user),
            "overview": overview,
            "dashboard_error": dashboard_error,
        },
        status=503 if dashboard_error else 200,
    )


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_customers(request):
    try:
        selected_customer, lookup_error = dashboard_context_for_customer_id(request.GET.get("customer_id"))
    except DatabaseError:
        selected_customer = None
        lookup_error = None
        dashboard_error = "Dashboard data is temporarily unavailable. Please try again later."
    else:
        dashboard_error = None
    return render(
        request,
        "banking/dashboard_customers.html",
        {
            "page_key": "customers",
            "page_title": "Customers",
            "demo_role": employee_role(request.user),
            "selected_customer": selected_customer,
            "lookup_customer_id": request.GET.get("customer_id", ""),
            "lookup_error": lookup_error,
            "dashboard_error": dashboard_error,
        },
        status=503 if dashboard_error else 200,
    )


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_segments(request):
    try:
        analytics = segments_analytics()
    except DatabaseError:
        analytics = None
        dashboard_error = "Dashboard data is temporarily unavailable. Please try again later."
    else:
        dashboard_error = None
    return render(request, "banking/dashboard_segments.html", {"page_key": "segments", "page_title": "Segments", "demo_role": employee_role(request.user), "analytics": analytics, "dashboard_error": dashboard_error}, status=503 if dashboard_error else 200)


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_products(request):
    return _dashboard_product_recommendations(request)


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_forecasting(request):
    try:
        analytics = forecasting_analytics(request.GET.get("pool"))
    except DatabaseError:
        analytics = None
        dashboard_error = "Dashboard data is temporarily unavailable. Please try again later."
    else:
        dashboard_error = None
    return render(request, "banking/dashboard_forecasting.html", {"page_key": "forecasting", "page_title": "Forecasting", "demo_role": employee_role(request.user), "analytics": analytics, "dashboard_error": dashboard_error}, status=503 if dashboard_error else 200)


@ensure_csrf_cookie
@require_GET
@_employee_required
def dashboard_recommendations(request):
    return _dashboard_product_recommendations(request)


def _dashboard_product_recommendations(request):
    selected_customer = None
    lookup_error = None
    try:
        raw_customer_id = request.GET.get("customer_id")
        selected_customer, lookup_error = dashboard_context_for_customer_id(raw_customer_id)
        analytics = products_analytics()
    except DatabaseError:
        analytics = None
        dashboard_error = "Dashboard data is temporarily unavailable. Please try again later."
    else:
        dashboard_error = None
    return render(request, "banking/dashboard_product_recommendations.html", {"page_key": "product-recommendations", "page_title": "Product Recommendations", "demo_role": employee_role(request.user), "analytics": analytics, "selected_customer": selected_customer, "lookup_customer_id": request.GET.get("customer_id", ""), "lookup_error": lookup_error, "dashboard_error": dashboard_error}, status=503 if dashboard_error else 200)


@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def demo_login(request):
    if request.user.is_authenticated:
        return redirect(_safe_next(request))

    context = {"next": _safe_next(request)}
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is None:
            context["error"] = "Invalid username or password."
            context["username"] = username
            return render(request, "banking/login.html", context, status=401)

        login(request, user)
        return redirect(_safe_next(request))

    return render(request, "banking/login.html", context)


@require_POST
def demo_logout(request):
    logout(request)
    return redirect(settings.LOGOUT_REDIRECT_URL)
