from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_api.schemas import (
    ErrorResponse,
    HealthResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrieveResponse,
)
from rag_api.service import RAGDependencyError, rag_service

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = int(os.getenv("RAG_API_MAX_REQUEST_BYTES", "16384"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_service.initialize()
    except Exception:
        LOGGER.exception("rag_startup_initialization_failed")
    yield


app = FastAPI(
    title="Atlas Demo Internal RAG API",
    version="0.1.0",
    lifespan=lifespan,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("RAG_API_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-request-id"],
    )


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "detail": "request body is too large",
                "error_type": "request_too_large",
                "request_id": request_id,
            },
        )

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(RAGDependencyError)
async def rag_dependency_error_handler(request: Request, exc: RAGDependencyError):
    request_id = getattr(request.state, "request_id", None)
    LOGGER.warning(
        "rag_dependency_error request_id=%s dependency=%s error_type=%s",
        request_id,
        exc.dependency,
        exc.__class__.__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": f"RAG dependency unavailable: {exc.dependency}",
            "error_type": exc.dependency,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    LOGGER.exception("rag_unexpected_error request_id=%s error_type=%s", request_id, exc.__class__.__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal RAG service error",
            "error_type": "internal_error",
            "request_id": request_id,
        },
    )


@app.get("/rag/health", response_model=HealthResponse)
def health() -> dict:
    return rag_service.health()


@app.post("/rag/query", response_model=RAGQueryResponse)
def query_rag(payload: RAGQueryRequest, request: Request) -> dict:
    return rag_service.query(
        query=payload.query,
        top_k=payload.top_k,
        product=payload.product,
        category=payload.category,
        request_id=getattr(request.state, "request_id", None),
        model=payload.model,
    )


@app.post("/rag/retrieve", response_model=RetrieveResponse)
def retrieve_rag(payload: RAGQueryRequest) -> dict:
    return rag_service.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        product=payload.product,
        category=payload.category,
    )
