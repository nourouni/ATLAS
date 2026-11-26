from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from RAG import config
except ModuleNotFoundError:
    import config


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default_factory=lambda: int(getattr(config, "CONTEXT_TOP_K", config.TOP_K)), ge=1, le=10)
    product: str | None = Field(default=None, max_length=200)
    category: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=200)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        query = " ".join(value.split())
        if not query:
            raise ValueError("query must not be blank")
        return query

    @field_validator("product", "category", "model")
    @classmethod
    def optional_filters_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class Citation(BaseModel):
    citation_id: str
    document_id: str | None = None
    source_title: str = ""
    source_url: str = ""


class RAGQueryResponse(BaseModel):
    query: str
    answered: bool
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_count: int = 0
    model: str | None = None
    generation_latency_ms: float | None = None
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    chroma: bool
    collection: str
    collection_count: int | None = None
    embedding_model: str
    retriever: str
    ollama: bool
    llm_model: str
    details: dict[str, str] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    rank: in
    document_id: str | None = None
    product: str = ""
    category: str = ""
    question_type: str = ""
    content: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    verification_status: str = ""
    confidence: str = ""


class RetrieveResponse(BaseModel):
    query: str
    retriever: str
    results: list[RetrievedDocument]
    evidence_count: in
    reason: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
    request_id: str | None = None
    extra: dict[str, Any] | None = None
