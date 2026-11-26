from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from django.conf import settings


LOGGER = logging.getLogger(__name__)


class RAGToolError(RuntimeError):
    def __init__(self, message: str, *, dependency: str | None = None):
        super().__init__(message)
        self.dependency = dependency


def _ensure_project_paths() -> None:
    repo_root = Path(getattr(settings, "REPO_ROOT", Path(__file__).resolve().parents[2]))
    rag_dir = repo_root / "RAG"
    for path in (repo_root, rag_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


class InternalRAGTool:
    def query(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
        request_id: str | None = None,
        generator_model: str | None = None,
    ) -> dict[str, Any]:
        _ensure_project_paths()
        try:
            from rag_api.service import RAGDependencyError, rag_service

            return rag_service.query(
                query=query,
                top_k=top_k,
                product=product,
                category=category,
                request_id=request_id,
                model=generator_model or getattr(settings, "CHATBOT_GENERATOR_MODEL", getattr(settings, "CHATBOT_OLLAMA_MODEL", None)),
            )
        except RAGDependencyError as exc:
            raise RAGToolError(str(exc), dependency=exc.dependency) from exc
        except Exception:
            LOGGER.exception("internal_rag_tool_error request_id=%s", request_id)
            raise

    def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
        generator_model: str | None = None,
    ) -> dict[str, Any]:
        _ensure_project_paths()
        try:
            from rag_api.service import RAGDependencyError, rag_service

            return rag_service.retrieve(query=query, top_k=top_k, product=product, category=category)
        except RAGDependencyError as exc:
            raise RAGToolError(str(exc), dependency=exc.dependency) from exc

    def health(self) -> dict[str, Any]:
        _ensure_project_paths()
        try:
            from rag_api.service import rag_service

            return rag_service.health()
        except Exception as exc:
            raise RAGToolError(str(exc), dependency="rag") from exc


rag_tool = InternalRAGTool()
