from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.paren
REPO_ROOT = BASE_DIR.paren

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "database" / ".env")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "banking",
]

TEST_RUNNER = "config.test_runner.ExistingDatabaseTestRunner"
API_REQUIRE_AUTH = os.environ.get("DJANGO_API_REQUIRE_AUTH", "false").lower() in {"1", "true", "yes"}
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://127.0.0.1:8001")
RAG_CONNECT_TIMEOUT = float(os.environ.get("RAG_CONNECT_TIMEOUT", "5"))
RAG_READ_TIMEOUT = float(os.environ.get("RAG_READ_TIMEOUT", "130"))
CHATBOT_INTENT_CONFIDENCE_THRESHOLD = float(os.environ.get("CHATBOT_INTENT_CONFIDENCE_THRESHOLD", "0.70"))
CHATBOT_OLLAMA_HOST = os.environ.get("CHATBOT_OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
CHATBOT_OLLAMA_MODEL = os.environ.get("CHATBOT_OLLAMA_MODEL", "qwen2.5:7b-instruct")
CHATBOT_CLASSIFIER_MODEL = os.environ.get("CHATBOT_CLASSIFIER_MODEL", CHATBOT_OLLAMA_MODEL)
CHATBOT_GENERATOR_MODEL = os.environ.get("CHATBOT_GENERATOR_MODEL", CHATBOT_OLLAMA_MODEL)
CHATBOT_OLLAMA_TIMEOUT = float(os.environ.get("CHATBOT_OLLAMA_TIMEOUT", "120"))
ATLAS_STATIC_FORECAST_POINTS = os.environ.get(
    "ATLAS_STATIC_FORECAST_POINTS",
    str(REPO_ROOT / "demo" / "forecast_points.csv"),
)

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "NAME": os.environ.get("POSTGRES_DB", ""),
        "USER": os.environ.get("POSTGRES_USER", ""),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", ""),
        "OPTIONS": {
            "sslmode": os.environ.get("POSTGRES_SSLMODE", "prefer"),
        },
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "static/"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/chatbot/"
LOGOUT_REDIRECT_URL = "/login/"
