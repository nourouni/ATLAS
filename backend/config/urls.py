from __future__ import annotations

from django.conf import settings
from django.urls import include, path, re_path
from django.views.static import serve

from banking.frontend_views import (
    chatbot_demo,
    dashboard_customers,
    dashboard_forecasting,
    dashboard_home,
    dashboard_products,
    dashboard_recommendations,
    dashboard_segments,
    demo_login,
    demo_logout,
)

urlpatterns = [
    path("login/", demo_login, name="login"),
    path("logout/", demo_logout, name="logout"),
    path("chatbot/", chatbot_demo, name="chatbot-demo"),
    path("dashboard/", dashboard_home, name="dashboard"),
    path("dashboard/customers/", dashboard_customers, name="dashboard-customers"),
    path("dashboard/segments/", dashboard_segments, name="dashboard-segments"),
    path("dashboard/products/", dashboard_products, name="dashboard-products"),
    path("dashboard/forecasting/", dashboard_forecasting, name="dashboard-forecasting"),
    path("dashboard/recommendations/", dashboard_recommendations, name="dashboard-recommendations"),
    path("api/", include("banking.urls")),
    re_path(
        r"^static/(?P<path>banking/.*)$",
        serve,
        {"document_root": settings.BASE_DIR / "banking" / "static"},
    ),
]
