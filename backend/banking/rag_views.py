from __future__ import annotations

import logging
import json
import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exemp
from django.views.decorators.http import require_GET, require_POST

from .permissions import api_login_required
from .rag_client import RAGServiceError, rag_clien


LOGGER = logging.getLogger(__name__)


def _json_body(request) -> tuple[dict, JsonResponse | None]:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}, JsonResponse({"error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return {}, JsonResponse({"error": "invalid_json"}, status=400)
    return payload, None


def _validated_query_payload(request) -> tuple[dict, JsonResponse | None]:
    payload, error = _json_body(request)
    if error:
        return {}, error

    query = " ".join(str(payload.get("query", "")).split())
    if not query:
        return {}, JsonResponse({"error": "invalid_query", "message": "query must not be blank"}, status=400)
    if len(query) > 2000:
        return {}, JsonResponse({"error": "invalid_query", "message": "query is too long"}, status=400)

    top_k = payload.get("top_k")
    if top_k is not None:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return {}, JsonResponse({"error": "invalid_top_k", "message": "top_k must be an integer"}, status=400)
        if top_k < 1 or top_k > 10:
            return {}, JsonResponse({"error": "invalid_top_k", "message": "top_k must be between 1 and 10"}, status=400)

    data = {"query": query}
    if top_k is not None:
        data["top_k"] = top_k

    for key in ("product", "category"):
        value = payload.get(key)
        if value is None:
            continue
        normalized = " ".join(str(value).split())
        if len(normalized) > 200:
            return {}, JsonResponse({"error": f"invalid_{key}", "message": f"{key} is too long"}, status=400)
        data[key] = normalized or None

    return data, None


def _request_id(request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _rag_error_response(exc: RAGServiceError, request_id: str) -> JsonResponse:
    LOGGER.warning(
        "rag_service_error request_id=%s status_code=%s error_type=%s",
        request_id,
        exc.status_code,
        exc.payload.get("error_type"),
    )
    upstream_status = exc.status_code or 503
    response_status = 503 if upstream_status >= 500 else 400
    return JsonResponse(
        {
            "error": exc.payload.get("error_type") or "rag_service_unavailable",
            "message": "RAG service is temporarily unavailable" if response_status == 503 else str(exc),
            "request_id": request_id,
        },
        status=response_status,
    )


@require_GET
@api_login_required
def rag_health(request):
    request_id = _request_id(request)
    try:
        payload = rag_client.health()
    except RAGServiceError as exc:
        return _rag_error_response(exc, request_id)
    response = JsonResponse(payload)
    response["X-Request-ID"] = request_id
    return response


@csrf_exemp
@require_POST
@api_login_required
def rag_query(request):
    request_id = _request_id(request)
    data, error = _validated_query_payload(request)
    if error:
        return error

    try:
        payload = rag_client.query(
            query=data["query"],
            top_k=data.get("top_k"),
            product=data.get("product"),
            category=data.get("category"),
            request_id=request_id,
        )
    except RAGServiceError as exc:
        return _rag_error_response(exc, request_id)

    response = JsonResponse(payload, status=200)
    response["X-Request-ID"] = request_id
    return response


@csrf_exemp
@require_POST
@api_login_required
def rag_retrieve(request):
    request_id = _request_id(request)
    data, error = _validated_query_payload(request)
    if error:
        return error

    try:
        payload = rag_client.retrieve(
            query=data["query"],
            top_k=data.get("top_k"),
            product=data.get("product"),
            category=data.get("category"),
            request_id=request_id,
        )
    except RAGServiceError as exc:
        return _rag_error_response(exc, request_id)

    response = JsonResponse(payload, status=200)
    response["X-Request-ID"] = request_id
    return response
