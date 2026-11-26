import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from rag import config
except ModuleNotFoundError:
    import config

try:
    from RAG.retrieval.common import load_verified_documents
except ModuleNotFoundError:
    from retrieval.common import load_verified_documents


OFFICIAL_DOMAINS = {
    "example.org",
    "example.org",
    "example.org",
    "example.org",
}

PRODUCTION_STATUSES = {
    "verified",
    "partially_verified",
    "official_information_not_public",
    "human_intervention",
}


def load_config():
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "chroma_path": Path(config.CHROMA_PATH),
        "collection_name": config.CHROMA_COLLECTION,
        "verified_knowledge_path": Path(config.VERIFIED_KNOWLEDGE_PATH),
        "report_path": Path(config.INDEX_BUILD_REPORT_PATH),
    }


def require_dependencies():
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: chromadb. Install it with `python -m pip install chromadb`."
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it with "
            "`python -m pip install sentence-transformers`."
        ) from exc

    return chromadb, SentenceTransformer


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def is_official_url(url):
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    return host in OFFICIAL_DOMAINS or host.endswith(".example.org") or host.endswith(".example.org")


def load_production_documents(path):
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            document["_line_number"] = line_number
            if clean(document.get("verification_status")) in PRODUCTION_STATUSES:
                documents.append(document)
    return documents


def validate_document(document, seen_ids):
    reasons = []
    document_id = clean(document.get("id"))
    if not document_id:
        reasons.append("missing document id")
    elif document_id in seen_ids:
        reasons.append("duplicate document id")

    if clean(document.get("verification_status")) not in PRODUCTION_STATUSES:
        reasons.append("verification_status is not production-safe")

    if not clean(document.get("content")):
        reasons.append("empty content")

    source_urls = document.get("source_urls")
    if not isinstance(source_urls, list):
        reasons.append("source_urls is not a list")

    try:
        json.dumps(prepare_metadata(document), ensure_ascii=False)
    except TypeError as exc:
        reasons.append(f"metadata is not JSON serializable: {exc}")

    return reasons


def build_embedding_text(document):
    parts = [
        f"Product: {clean(document.get('product'))}",
        f"Category: {clean(document.get('category'))}",
    ]
    if clean(document.get("subcategory")):
        parts.append(f"Subcategory: {clean(document.get('subcategory'))}")
    parts.extend([
        f"Question: {clean(document.get('question'))}",
        f"Answer: {clean(document.get('content'))}",
    ])
    return "\n".join(part for part in parts if not part.endswith(": "))


def prepare_metadata(document):
    source_urls = document.get("source_urls") or []
    source_titles = document.get("source_titles") or []
    return {
        "product": clean(document.get("product")),
        "question_id": clean(document.get("id")),
        "category": clean(document.get("category")),
        "subcategory": clean(document.get("subcategory")) or None,
        "question_type": clean(document.get("question_type")),
        "document_type": clean(document.get("document_type")),
        "verification_status": clean(document.get("verification_status")),
        "confidence": clean(document.get("confidence")),
        "verification_notes": clean(document.get("verification_notes")),
        "source_urls": json.dumps(source_urls, ensure_ascii=False),
        "source_titles": json.dumps(source_titles, ensure_ascii=False),
    }


def load_embedding_model(model_name, SentenceTransformer):
    return SentenceTransformer(model_name)


def get_chroma_collection(chromadb, chroma_path, collection_name):
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def get_collection_ids(collection):
    result = collection.get(include=[])
    return set(result.get("ids", []))


def sync_stale_documents(collection, source_ids):
    existing_ids = get_collection_ids(collection)
    stale_ids = sorted(existing_ids - source_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return stale_ids


def embed_documents(model, texts):
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).tolist()


def upsert_documents(collection, documents, texts, embeddings, metadatas):
    if not documents:
        return 0
    collection.upsert(
        ids=[document["id"] for document in documents],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(documents)


def validate_collection(chromadb, chroma_path, collection_name, expected_ids):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)
    count = collection.count()
    stored = collection.get(ids=sorted(expected_ids)[:5], include=["metadatas", "documents"])
    stored_ids = stored.get("ids", [])
    duplicate_check = collection.get(include=[])
    all_ids = duplicate_check.get("ids", [])
    duplicate_ids = sorted({item for item in all_ids if all_ids.count(item) > 1})

    errors = []
    if count != len(expected_ids):
        errors.append(f"collection count {count} does not match expected {len(expected_ids)}")
    if set(stored_ids) != set(sorted(expected_ids)[:5]):
        errors.append("sample ID retrieval did not return all requested records")
    for metadata in stored.get("metadatas", []):
        if metadata.get("verification_status") not in PRODUCTION_STATUSES:
            errors.append("sample metadata has non-production-safe status")
        if not metadata.get("source_urls"):
            errors.append("sample metadata is missing source_urls")
    if duplicate_ids:
        errors.append("duplicate IDs detected: " + ", ".join(duplicate_ids))

    return {
        "count": count,
        "sample_records": [
            {
                "id": document_id,
                "metadata": metadata,
            }
            for document_id, metadata in zip(stored_ids, stored.get("metadatas", []))
        ],
        "errors": errors,
    }


def write_build_report(report_path, report):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    settings = load_config()
    chromadb, SentenceTransformer = require_dependencies()

    raw_documents = load_verified_documents(settings["verified_knowledge_path"])
    valid_documents = []
    skipped = []
    seen_ids = set()

    for document in raw_documents:
        reasons = validate_document(document, seen_ids)
        if reasons:
            skipped.append({
                "id": document.get("id"),
                "line_number": document.get("_line_number"),
                "reasons": reasons,
            })
            continue
        seen_ids.add(document["id"])
        valid_documents.append(document)

    source_ids = {document["id"] for document in valid_documents}
    _, collection = get_chroma_collection(
        chromadb,
        settings["chroma_path"],
        settings["collection_name"],
    )
    stale_ids = sync_stale_documents(collection, source_ids)

    model = load_embedding_model(settings["embedding_model"], SentenceTransformer)
    texts = [build_embedding_text(document) for document in valid_documents]
    metadatas = [prepare_metadata(document) for document in valid_documents]
    embeddings = embed_documents(model, texts)
    upserted = upsert_documents(collection, valid_documents, texts, embeddings, metadatas)

    integrity = validate_collection(
        chromadb,
        settings["chroma_path"],
        settings["collection_name"],
        source_ids,
    )
    if integrity["errors"]:
        raise RuntimeError("Chroma integrity check failed: " + "; ".join(integrity["errors"]))

    report = {
        "embedding_model": settings["embedding_model"],
        "source_documents": len(raw_documents),
        "indexed_documents": len(valid_documents),
        "skipped_documents": len(skipped),
        "deleted_stale_documents": len(stale_ids),
        "collection_count": integrity["count"],
        "chroma_path": str(settings["chroma_path"]),
        "collection": settings["collection_name"],
        "stale_document_ids_deleted": stale_ids,
        "skipped": skipped,
        "sample_records": integrity["sample_records"],
    }
    write_build_report(settings["report_path"], report)

    print(f"Embedding model: {settings['embedding_model']}")
    print(f"Production-safe documents loaded: {len(raw_documents)}")
    print(f"Documents embedded: {len(valid_documents)}")
    print(f"Documents upserted: {upserted}")
    print(f"Invalid documents skipped: {len(skipped)}")
    print(f"Stale documents removed: {len(stale_ids)}")
    print(f"Final Chroma collection size: {integrity['count']}")
    print(f"Persistence path: {settings['chroma_path']}")
    print("Sample stored records:")
    for record in integrity["sample_records"]:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
