from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from django.conf import settings


LOGGER = logging.getLogger(__name__)


class RAGServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass(frozen=True)
class RAGClientConfig:
    base_url: str
    connect_timeout_seconds: floa
    read_timeout_seconds: floa


def get_rag_client_config() -> RAGClientConfig:
    return RAGClientConfig(
        base_url=getattr(settings, "RAG_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/"),
        connect_timeout_seconds=float(getattr(settings, "RAG_CONNECT_TIMEOUT", 5)),
        read_timeout_seconds=float(getattr(settings, "RAG_READ_TIMEOUT", 130)),
    )


class AtlasDemoRAGClient:
    def __init__(self, config: RAGClientConfig | None = None):
        self.config = config or get_rag_client_config()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/rag/health")

    def query(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if product is not None:
            payload["product"] = produc
        if category is not None:
            payload["category"] = category
        return self._request("POST", "/rag/query", json=payload, request_id=request_id)

    def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        product: str | None = None,
        category: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if product is not None:
            payload["product"] = produc
        if category is not None:
            payload["category"] = category
        return self._request("POST", "/rag/retrieve", json=payload, request_id=request_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {"Accept": "application/json"}
        if request_id:
            headers["X-Request-ID"] = request_id

        try:
            timeout = httpx.Timeout(
                self.config.read_timeout_seconds,
                connect=self.config.connect_timeout_seconds,
            )
            response = httpx.request(
                method,
                url,
                json=json,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise RAGServiceError("RAG service timed out") from exc
        except httpx.RequestError as exc:
            raise RAGServiceError("RAG service is unavailable") from exc

        payload = self._decode_json(response)
        if response.status_code >= 400:
            raise RAGServiceError(
                payload.get("detail") or "RAG service returned an error",
                status_code=response.status_code,
                payload=payload,
            )
        return payload

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGServiceError(
                "RAG service returned a non-JSON response",
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise RAGServiceError(
                "RAG service returned an unexpected response shape",
                status_code=response.status_code,
            )
        return payload


rag_client = AtlasDemoRAGClient()
