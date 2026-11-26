from __future__ import annotations

import hashlib
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from RAG import config
except ModuleNotFoundError:
    import config


class RAGDependencyError(RuntimeError):
    """Raised when a backing RAG dependency is unavailable."""

    def __init__(self, message: str, *, dependency: str):
        super().__init__(message)
        self.dependency = dependency


def query_fingerprint(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def _public_query_response(result: dict[str, Any]) -> dict[str, Any]:
    citations = result.get("citations") or []
    answered = bool(result.get("answered"))
    return {
        "query": result.get("query", ""),
        "answered": answered,
        "answer": result.get("answer", ""),
        "reason": result.get("reason") if not answered else None,
        "citations": citations,
        "evidence_count": int(result.get("evidence_count") or len(citations) or 0),
        "model": result.get("model") or config.LLM_MODEL,
        "generation_latency_ms": result.get("generation_latency_ms"),
    }


class RAGService:
    def __init__(self) -> None:
        self._answer_generator = None
        self._retriever_module = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._answer_generator = importlib.import_module("RAG.generation.answer_generator")
        self._retriever_module = importlib.import_module("RAG.retrieval.retriever")
        self._retriever_module.get_retriever()
        self._initialized = True

    @property
    def answer_generator(self):
        if self._answer_generator is None:
            self.initialize()
        return self._answer_generator

    @property
    def retriever_module(self):
        if self._retriever_module is None:
            self.initialize()
        return self._retriever_module

    def query(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
        request_id: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        fingerprint = query_fingerprint(query)
        try:
            result = self.answer_generator.generate_answer(
                query=query,
                top_k=top_k or getattr(config, "CONTEXT_TOP_K", config.TOP_K),
                product=product,
                category=category,
                model=model,
            )
        except RuntimeError as exc:
            message = str(exc)
            selected_model = model or config.LLM_MODEL
            dependency = "ollama" if "Ollama" in message or selected_model in message else "rag"
            LOGGER.exception(
                "rag_query_dependency_error request_id=%s query_hash=%s query_len=%s dependency=%s",
                request_id,
                fingerprint,
                len(query),
                dependency,
            )
            raise RAGDependencyError(message, dependency=dependency) from exc
        except Exception:
            LOGGER.exception(
                "rag_query_unexpected_error request_id=%s query_hash=%s query_len=%s",
                request_id,
                fingerprint,
                len(query),
            )
            raise

        public = _public_query_response(result)
        LOGGER.info(
            "rag_query request_id=%s query_hash=%s query_len=%s answered=%s evidence_count=%s model=%s latency_ms=%.2f",
            request_id,
            fingerprint,
            len(query),
            public["answered"],
            public["evidence_count"],
            public["model"],
            (time.perf_counter() - start) * 1000,
        )
        return public

    def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.retriever_module.retrieve(
                query=query,
                top_k=top_k or getattr(config, "TOP_K", 5),
                product=product,
                category=category,
            )
        except RuntimeError as exc:
            raise RAGDependencyError(str(exc), dependency="retriever") from exc
        return {
            "query": result.get("query", query),
            "retriever": result.get("retriever_type", config.RETRIEVER_TYPE),
            "results": result.get("results", []),
            "evidence_count": len(result.get("results", [])),
            "reason": result.get("no_results_reason"),
        }

    def health(self) -> dict[str, Any]:
        details: dict[str, str] = {}
        chroma_ok = False
        ollama_ok = False
        collection_count: int | None = None

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
            collection = client.get_collection(config.CHROMA_COLLECTION)
            collection_count = int(collection.count())
            chroma_ok = collection_count > 0
            if not chroma_ok:
                details["chroma"] = "collection is empty"
        except Exception as exc:
            details["chroma"] = exc.__class__.__name__

        try:
            import ollama

            client = ollama.Client(host=getattr(config, "OLLAMA_HOST", None), timeout=5)
            models_response = client.list()
            installed = {
                item.get("name") or item.get("model")
                for item in getattr(models_response, "models", None) or models_response.get("models", [])
            }
            ollama_ok = any(
                model_name == config.LLM_MODEL or str(model_name).startswith(f"{config.LLM_MODEL}:")
                for model_name in installed
            )
            if not ollama_ok:
                details["ollama"] = f"model `{config.LLM_MODEL}` is not installed"
        except Exception as exc:
            details["ollama"] = exc.__class__.__name__

        status = "ok" if chroma_ok and ollama_ok else "degraded"
        return {
            "status": status,
            "chroma": chroma_ok,
            "collection": config.CHROMA_COLLECTION,
            "collection_count": collection_count,
            "embedding_model": config.EMBEDDING_MODEL,
            "retriever": config.RETRIEVER_TYPE,
            "ollama": ollama_ok,
            "llm_model": config.LLM_MODEL,
            "details": details,
        }


rag_service = RAGService()
