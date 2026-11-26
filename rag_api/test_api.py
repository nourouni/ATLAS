from fastapi.testclient import TestClien

from rag_api.main import app
from rag_api.service import RAGDependencyError


class StubService:
    def health(self):
        return {
            "status": "ok",
            "chroma": True,
            "collection": "atlas_demo_verified_knowledge",
            "collection_count": 33,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "retriever": "hybrid_rrf",
            "ollama": True,
            "llm_model": "phi4-mini",
            "details": {},
        }

    def query(self, *, query, top_k=None, product=None, category=None, request_id=None):
        if "Carte CIB" in query:
            return {
                "query": query,
                "answered": False,
                "answer": "I could not find verified Atlas Demo Bank information supporting an answer to this question.",
                "reason": "no_verified_documents_for_product: Carte CIB",
                "citations": [],
                "evidence_count": 0,
                "model": "phi4-mini",
                "generation_latency_ms": None,
            }
        return {
            "query": query,
            "answered": True,
            "answer": "Cash Management access is configured through verified Atlas Demo evidence. [S1]",
            "citations": [
                {
                    "citation_id": "S1",
                    "document_id": "Q00458",
                    "source_title": "Atlas Demo Cash Management | Atlas Demo Bank",
                    "source_url": "https://example.test/cash-management",
                }
            ],
            "evidence_count": 1,
            "model": "phi4-mini",
            "generation_latency_ms": 1.2,
        }

    def retrieve(self, *, query, top_k=None, product=None, category=None):
        return {
            "query": query,
            "retriever": "hybrid_rrf",
            "results": [
                {
                    "rank": 1,
                    "document_id": "Q00458",
                    "product": "Atlas Demo Cash Management",
                    "category": "Corporate Banking",
                    "question_type": "how_it_works",
                    "content": "Verified content",
                    "source_urls": ["https://example.test/cash-management"],
                    "source_titles": ["Atlas Demo Cash Management | Atlas Demo Bank"],
                    "verification_status": "verified",
                    "confidence": "high",
                }
            ],
            "evidence_count": 1,
            "reason": None,
        }


def test_health(monkeypatch):
    monkeypatch.setattr("rag_api.main.rag_service", StubService())
    client = TestClient(app)

    response = client.get("/rag/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chroma"] is True
    assert body["ollama"] is True


def test_answerable_query(monkeypatch):
    monkeypatch.setattr("rag_api.main.rag_service", StubService())
    client = TestClient(app)

    response = client.post("/rag/query", json={"query": "How does Cash Management work?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["citations"]
    assert body["citations"][0]["citation_id"] == "S1"
    assert body["evidence_count"] == 1


def test_unsupported_product_returns_unanswered(monkeypatch):
    monkeypatch.setattr("rag_api.main.rag_service", StubService())
    client = TestClient(app)

    response = client.post("/rag/query", json={"query": "What is Carte CIB?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["citations"] == []
    assert body["reason"] == "no_verified_documents_for_product: Carte CIB"


def test_blank_query_rejected(monkeypatch):
    monkeypatch.setattr("rag_api.main.rag_service", StubService())
    client = TestClient(app)

    response = client.post("/rag/query", json={"query": "   "})

    assert response.status_code == 422


def test_dependency_failure_returns_503(monkeypatch):
    class FailingService(StubService):
        def query(self, **kwargs):
            raise RAGDependencyError("Ollama generation failed", dependency="ollama")

    monkeypatch.setattr("rag_api.main.rag_service", FailingService())
    client = TestClient(app)

    response = client.post("/rag/query", json={"query": "How does Cash Management work?", "top_k": 3})

    assert response.status_code == 503
    assert response.json()["error_type"] == "ollama"


def test_retrieve_returns_verified_documents(monkeypatch):
    monkeypatch.setattr("rag_api.main.rag_service", StubService())
    client = TestClient(app)

    response = client.post("/rag/retrieve", json={"query": "How does Cash Management work?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["retriever"] == "hybrid_rrf"
    assert body["results"][0]["verification_status"] == "verified"\n