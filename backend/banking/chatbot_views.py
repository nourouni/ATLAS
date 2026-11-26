from __future__ import annotations

import json
import logging
import uuid

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exemp
from django.views.decorators.http import require_GET
from django.views.decorators.http import require_POST

from .chatbot_models import active_model_config, apply_dev_model_config
from .auth_context import authenticated_customer_id
from .need_discovery_logging import append_chatbot_gap_even
from .orchestration.graph import graph
from .permissions import api_login_required, forbidden_response, user_can_access_customer
from .serializers import chatbot_context, chatbot_recommendation
from .services import CustomerNotFoundError, get_customer_context, get_customer_recommendations, get_customer_profile
from .views import error_response


LOGGER = logging.getLogger(__name__)


@require_GET
@api_login_required
def customer_context(request, customer_id: int):
    if not user_can_access_customer(request.user, customer_id):
        return forbidden_response()
    try:
        return JsonResponse(chatbot_context(get_customer_context(customer_id)))
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)


def _request_id(request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _json_body(request) -> tuple[dict, JsonResponse | None]:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}, JsonResponse({"error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return {}, JsonResponse({"error": "invalid_json"}, status=400)
    return payload, None


def _authenticated_customer_id(request, customer_id: int | None = None) -> int | None:
    # The optional URL argument is deliberately ignored. Identity comes only
    # from the authenticated Django session's server-side profile mapping.
    return authenticated_customer_id(getattr(request, "user", None))


@csrf_exemp
@require_POST
@api_login_required
def chatbot_graph(request, customer_id: int | None = None):
    payload, error = _json_body(request)
    if error:
        return error

    request_id = _request_id(request)
    authenticated = bool(getattr(request, "user", None) and request.user.is_authenticated)
    resolved_customer_id = _authenticated_customer_id(request, customer_id)
    LOGGER.info(
        "chatbot_graph_context request_id=%s authenticated=%s customer_id=%s",
        request_id,
        authenticated,
        resolved_customer_id,
    )
    debug_models = active_model_config(request)
    result = graph.invoke(
        {
            "message": payload.get("message") or payload.get("query") or "",
            "history": payload.get("history") if isinstance(payload.get("history"), list) else [],
            "conversation_state": payload.get("conversation_state") if isinstance(payload.get("conversation_state"), dict) else {},
            "customer_id": resolved_customer_id,
            "authenticated": authenticated,
            "request_id": request_id,
            "classifier_model": debug_models["classifier"],
            "generator_model": debug_models["generator"],
        }
    )
    try:
        append_chatbot_gap_event(
            question=payload.get("message") or payload.get("query") or "",
            result=result,
            request_id=request_id,
        )
    except Exception:
        # Prospective reporting must never change chatbot availability.
        LOGGER.exception("chatbot_gap_event_write_failed request_id=%s", request_id)
    LOGGER.info(
        "chatbot_graph request_id=%s intent=%s confidence=%s route=%s authenticated=%s rag_used=%s customer_context_used=%s recommendation_used=%s escalated=%s failure_code=%s",
        request_id,
        result.get("intent"),
        result.get("intent_confidence"),
        result.get("route"),
        authenticated,
        "yes" if result.get("rag_context") else "no",
        "yes" if result.get("customer_context") else "no",
        "yes" if result.get("recommendation_context") else "no",
        "yes" if result.get("blocked") or result.get("requires_human") or result.get("insufficient_evidence") or result.get("failure_code") else "no",
        result.get("failure_code"),
    )
    response_payload = {
        "answer": result.get("final_answer"),
        "intent": result.get("intent"),
        "request_mode": result.get("request_mode"),
        "confidence": result.get("intent_confidence"),
        "route": result.get("route"),
        "escalated": bool(result.get("blocked") or result.get("requires_human") or result.get("insufficient_evidence") or result.get("failure_code")),
        "failure_code": result.get("failure_code"),
        "request_id": request_id,
    }
    if result.get("resolved_query") and result.get("resolved_query") != result.get("original_message"):
        response_payload["resolved_query"] = result.get("resolved_query")
    if result.get("conversation_state"):
        response_payload["conversation_state"] = result.get("conversation_state")
    if result.get("rag_context"):
        response_payload["citations"] = result["rag_context"][0].get("citations", [])
    if settings.DEBUG:
        response_payload["debug_models"] = {
            "classifier": debug_models["classifier"],
            "generator": debug_models["generator"],
        }
    response = JsonResponse(response_payload, status=200)
    response["X-Request-ID"] = request_id
    return response


@require_POST
@api_login_required
def chatbot_dev_models(request):
    if not settings.DEBUG:
        return JsonResponse({"error": "dev_model_switch_disabled"}, status=404)
    payload, error = _json_body(request)
    if error:
        return error
    try:
        models = apply_dev_model_config(request, str(payload.get("configuration") or ""))
    except ValueError:
        return JsonResponse({"error": "unsupported_model_config"}, status=400)
    except PermissionError:
        return JsonResponse({"error": "dev_model_switch_disabled"}, status=404)
    return JsonResponse({"ok": True, "debug_models": models})


@require_GET
@api_login_required
def customer_recommendations(request, customer_id: int):
    if not user_can_access_customer(request.user, customer_id):
        return forbidden_response()
    try:
        get_customer_profile(customer_id)
        recommendations = [chatbot_recommendation(item) for item in get_customer_recommendations(customer_id)]
        return JsonResponse({"customer_id": customer_id, "recommendations": recommendations})
    except CustomerNotFoundError as exc:
        return error_response(str(exc), 404)
