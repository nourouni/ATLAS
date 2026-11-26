import json
from unittest.mock import patch

from django.test import override_settings
from django.test import TestCase

from .rag_tool import InternalRAGTool
from .rag_client import AtlasDemoRAGClient, RAGClientConfig, RAGServiceError


class RAGClientTests(TestCase):
    @override_settings(RAG_SERVICE_URL="http://127.0.0.1:8001", RAG_CONNECT_TIMEOUT=5, RAG_READ_TIMEOUT=30)
    @patch("banking.rag_client.httpx.request")
    def test_client_posts_query_to_fastapi_service(self, mocked_request):
        mocked_request.return_value.status_code = 200
        mocked_request.return_value.json.return_value = {
            "query": "How does Cash Management work?",
            "answered": True,
            "answer": "Answer [S1]",
            "citations": [{"citation_id": "S1", "document_id": "Q00458", "source_title": "Title", "source_url": "https://example.test"}],
            "evidence_count": 1,
            "model": "phi4-mini",
            "generation_latency_ms": 10.5,
        }

        client = AtlasDemoRAGClient()
        payload = client.query(query="How does Cash Management work?", top_k=3, request_id="req-1")

        assert payload["answered"] is True
        mocked_request.assert_called_once()
        _, kwargs = mocked_request.call_args
        assert kwargs["json"] == {"query": "How does Cash Management work?", "top_k": 3}
        assert kwargs["headers"]["X-Request-ID"] == "req-1"

    @patch("banking.rag_client.httpx.request")
    def test_client_raises_clean_error_for_upstream_503(self, mocked_request):
        mocked_request.return_value.status_code = 503
        mocked_request.return_value.json.return_value = {
            "detail": "RAG dependency unavailable: ollama",
            "error_type": "ollama",
        }

        client = AtlasDemoRAGClient(
            RAGClientConfig(
                base_url="http://127.0.0.1:8001",
                connect_timeout_seconds=5,
                read_timeout_seconds=30,
            )
        )

        with self.assertRaises(RAGServiceError) as context:
            client.query(query="How does Cash Management work?")

        assert context.exception.status_code == 503
        assert context.exception.payload["error_type"] == "ollama"

    @patch("banking.rag_client.httpx.request")
    def test_client_raises_clean_error_for_malformed_upstream_response(self, mocked_request):
        mocked_request.return_value.status_code = 200
        mocked_request.return_value.json.return_value = ["not", "a", "dict"]

        client = AtlasDemoRAGClient(
            RAGClientConfig(
                base_url="http://127.0.0.1:8001",
                connect_timeout_seconds=5,
                read_timeout_seconds=30,
            )
        )

        with self.assertRaises(RAGServiceError):
            client.query(query="How does Cash Management work?")


class InternalRAGToolTests(TestCase):
    def test_generator_model_is_forwarded_without_mutating_shared_rag_config(self):
        from .rag_tool import _ensure_project_paths

        _ensure_project_paths()
        from RAG import config as rag_config

        original_model = rag_config.LLM_MODEL
        rag_config.LLM_MODEL = "shared-default-model"
        tool = InternalRAGTool()

        try:
            with patch("rag_api.service.rag_service.query") as mocked_query:
                mocked_query.return_value = {"answered": True, "answer": "ok"}
                tool.query(query="first", generator_model="phi4-mini:latest")
                tool.query(query="second", generator_model="qwen2.5:7b-instruct")
            assert rag_config.LLM_MODEL == "shared-default-model"
        finally:
            rag_config.LLM_MODEL = original_model

        assert mocked_query.call_args_list[0].kwargs["model"] == "phi4-mini:latest"
        assert mocked_query.call_args_list[1].kwargs["model"] == "qwen2.5:7b-instruct"


@override_settings(API_REQUIRE_AUTH=False)
class RAGViewTests(TestCase):
    @patch("banking.rag_views.rag_client")
    def test_query_view_returns_answerable_response(self, mocked_rag_client):
        mocked_rag_client.query.return_value = {
            "query": "How does Cash Management work?",
            "answered": True,
            "answer": "Answer [S1]",
            "citations": [{"citation_id": "S1", "document_id": "Q00458", "source_title": "Title", "source_url": "https://example.test"}],
            "evidence_count": 1,
            "model": "phi4-mini",
            "generation_latency_ms": 10.5,
        }

        response = self.client.post(
            "/api/rag/query/",
            json.dumps({"query": "How does Cash Management work?", "top_k": 3}),
            content_type="application/json",
        )

        assert response.status_code == 404

        response = self.client.post(
            "/api/chatbot/rag/",
            json.dumps({"query": "How does Cash Management work?", "top_k": 3}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json()["answered"] is True

    @patch("banking.rag_views.rag_client")
    def test_query_view_returns_unanswered_response_without_error(self, mocked_rag_client):
        mocked_rag_client.query.return_value = {
            "query": "What is Carte CIB?",
            "answered": False,
            "answer": "I could not find verified Atlas Demo Bank information supporting an answer to this question.",
            "reason": "no_verified_documents_for_product: Carte CIB",
            "citations": [],
            "evidence_count": 0,
        }

        response = self.client.post(
            "/api/chatbot/rag/",
            json.dumps({"query": "What is Carte CIB?"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        assert response.json()["answered"] is False
        assert response.json()["citations"] == []

    def test_blank_query_is_rejected(self):
        response = self.client.post(
            "/api/chatbot/rag/",
            json.dumps({"query": "   "}),
            content_type="application/json",
        )

        assert response.status_code == 400

    @patch("banking.rag_views.rag_client")
    def test_upstream_failure_returns_503(self, mocked_rag_client):
        mocked_rag_client.query.side_effect = RAGServiceError(
            "RAG dependency unavailable: ollama",
            status_code=503,
            payload={"error_type": "ollama"},
        )

        response = self.client.post(
            "/api/chatbot/rag/",
            json.dumps({"query": "How does Cash Management work?"}),
            content_type="application/json",
        )

        assert response.status_code == 503
        assert response.json()["error"] == "ollama"

    @patch("banking.rag_views.rag_client")
    def test_timeout_returns_503(self, mocked_rag_client):
        mocked_rag_client.query.side_effect = RAGServiceError("RAG service timed out")

        response = self.client.post(
            "/api/chatbot/rag/",
            json.dumps({"query": "How does Cash Management work?"}),
            content_type="application/json",
        )

        assert response.status_code == 503
        assert response.json()["error"] == "rag_service_unavailable"
