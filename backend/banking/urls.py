from __future__ import annotations

from django.urls import path

from . import chatbot_views, dashboard_views, rag_views, views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("customers/<int:customer_id>/", views.customer_detail, name="customer-detail"),
    path("customers/<int:customer_id>/context/", views.customer_context, name="customer-context"),
    path(
        "chatbot/customers/<int:customer_id>/context/",
        chatbot_views.customer_context,
        name="chatbot-customer-context",
    ),
    path(
        "chatbot/customers/<int:customer_id>/recommendations/",
        chatbot_views.customer_recommendations,
        name="chatbot-customer-recommendations",
    ),
    path("chatbot/graph/", chatbot_views.chatbot_graph, name="chatbot-graph"),
    path("chatbot/dev/models/", chatbot_views.chatbot_dev_models, name="chatbot-dev-models"),
    path("chatbot/customers/<int:customer_id>/graph/", chatbot_views.chatbot_graph, name="chatbot-customer-graph"),
    path("chatbot/rag/health/", rag_views.rag_health, name="chatbot-rag-health"),
    path("chatbot/rag/", rag_views.rag_query, name="chatbot-rag"),
    path("chatbot/rag/retrieve/", rag_views.rag_retrieve, name="chatbot-rag-retrieve"),
    path(
        "dashboard/customers/<int:customer_id>/",
        dashboard_views.customer_detail,
        name="dashboard-customer-detail",
    ),
    path(
        "dashboard/customers/<int:customer_id>/analytics/",
        dashboard_views.customer_analytics,
        name="dashboard-customer-analytics",
    ),
]
