"""Lightweight, non-mutating readiness checks for the active RAG service."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from RAG import config
from RAG.retrieval.common import load_verified_documents


def resolve_paths() -> dict[str, Any]:
    evidence = Path(config.VERIFIED_KNOWLEDGE_PATH)
    vectorstore = Path(config.CHROMA_PATH)
    report = Path(config.INDEX_BUILD_REPORT_PATH)
    return {
        "endpoint": os.environ.get("RAG_SERVICE_URL", "http://127.0.0.1:8001"),
        "evidence_path": evidence,
        "vectorstore_path": vectorstore,
        "collection": config.CHROMA_COLLECTION,
        "index_report_path": report,
        "embedding_model": config.EMBEDDING_MODEL,
        "evidence_exists": evidence.exists(),
        "vectorstore_exists": vectorstore.exists(),
        "index_report_exists": report.exists(),
    }


def check_readiness(*, check_collection: bool = True) -> dict[str, Any]:
    paths = resolve_paths()
    errors: list[str] = []
    warnings: list[str] = []
    evidence = paths["evidence_path"]
    vectorstore = paths["vectorstore_path"]
    report_path = paths["index_report_path"]

    if not evidence.exists():
        errors.append(f"RAG evidence source not found: {evidence}")
    if not vectorstore.exists():
        errors.append(f"RAG vector store not found: {vectorstore}")
    if not report_path.exists():
        errors.append(f"RAG index report not found: {report_path}")

    source_count = None
    if evidence.exists():
        try:
            source_count = len(load_verified_documents(evidence))
        except Exception as exc:
            errors.append(f"RAG evidence source could not be read: {exc}")

    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"RAG index report could not be read: {exc}")

    indexed_count = report.get("collection_count") or report.get("indexed_documents")
    if source_count is not None and indexed_count is not None and int(source_count) != int(indexed_count):
        errors.append(
            f"RAG vector store/source mismatch: source has {source_count} production documents, "
            f"index report has {indexed_count}"
        )

    collection_count = None
    if check_collection and not errors and vectorstore.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(vectorstore))
            collection = client.get_collection(config.CHROMA_COLLECTION)
            collection_count = int(collection.count())
            if source_count is not None and collection_count != source_count:
                errors.append(
                    f"RAG collection/source mismatch: collection has {collection_count}, source has {source_count}"
                )
        except Exception as exc:
            errors.append(f"RAG vector store is not readable: {exc}")

    return {
        "status": "READY" if not errors else "REBUILD_OR_FIX_REQUIRED",
        "paths": {key: str(value) if isinstance(value, Path) else value for key, value in paths.items()},
        "source_document_count": source_count,
        "indexed_document_count": indexed_count,
        "collection_count": collection_count,
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    result = check_readiness()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "READY" else 1)
