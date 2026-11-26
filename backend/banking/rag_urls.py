from django.urls import path

from . import rag_views


urlpatterns = [
    path("chatbot/rag/health/", rag_views.rag_health, name="chatbot-rag-health"),
    path("chatbot/rag/", rag_views.rag_query, name="chatbot-rag"),
    path("chatbot/rag/retrieve/", rag_views.rag_retrieve, name="chatbot-rag-retrieve"),
]
