from __future__ import annotations

from typing import Any

from django.conf import settings


ALLOWED_CHATBOT_MODELS = {
    "phi": "phi4-mini:latest",
    "qwen": "qwen2.5:7b-instruct",
}

ALLOWED_CHATBOT_CONFIGS = {
    "phi_phi": ("phi", "phi", "Phi → Phi"),
    "phi_qwen": ("phi", "qwen", "Phi → Qwen"),
    "qwen_qwen": ("qwen", "qwen", "Qwen → Qwen"),
}


def _model_key(model_name: str | None) -> str:
    for key, value in ALLOWED_CHATBOT_MODELS.items():
        if value == model_name:
            return key
    return "phi"


def _session_value(request_or_context: Any, key: str) -> str | None:
    if not getattr(settings, "DEBUG", False) or request_or_context is None:
        return None
    if isinstance(request_or_context, dict):
        value = request_or_context.get(key)
    else:
        value = getattr(request_or_context, "session", {}).get(key)
    if value in ALLOWED_CHATBOT_MODELS.values():
        return value
    return None


def get_chatbot_classifier_model(request_or_context: Any = None) -> str:
    return (
        _session_value(request_or_context, "dev_classifier_model")
        or getattr(settings, "CHATBOT_CLASSIFIER_MODEL", None)
        or getattr(settings, "CHATBOT_OLLAMA_MODEL", None)
        or getattr(settings, "RAG_LLM_MODEL", "phi4-mini:latest")
    )


def get_chatbot_generator_model(request_or_context: Any = None) -> str:
    return (
        _session_value(request_or_context, "dev_generator_model")
        or getattr(settings, "CHATBOT_GENERATOR_MODEL", None)
        or getattr(settings, "CHATBOT_OLLAMA_MODEL", None)
        or getattr(settings, "RAG_LLM_MODEL", "phi4-mini:latest")
    )


def active_model_config(request_or_context: Any = None) -> dict[str, str]:
    classifier = get_chatbot_classifier_model(request_or_context)
    generator = get_chatbot_generator_model(request_or_context)
    classifier_key = _model_key(classifier)
    generator_key = _model_key(generator)
    return {
        "classifier": classifier,
        "generator": generator,
        "classifier_label": classifier_key.title(),
        "generator_label": generator_key.title(),
        "config_key": f"{classifier_key}_{generator_key}",
    }


def apply_dev_model_config(request, config_key: str) -> dict[str, str]:
    if not getattr(settings, "DEBUG", False):
        raise PermissionError("dev_model_switch_disabled")
    if config_key not in ALLOWED_CHATBOT_CONFIGS:
        raise ValueError("unsupported_model_config")
    classifier_key, generator_key, _label = ALLOWED_CHATBOT_CONFIGS[config_key]
    request.session["dev_classifier_model"] = ALLOWED_CHATBOT_MODELS[classifier_key]
    request.session["dev_generator_model"] = ALLOWED_CHATBOT_MODELS[generator_key]
    request.session.modified = True
    return active_model_config(request)
