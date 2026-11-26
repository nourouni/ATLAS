import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from RAG import config
except ModuleNotFoundError:
    from rag import config


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PRODUCTION_STATUSES = {
    "verified",
    "partially_verified",
    "official_information_not_public",
    "human_intervention",
}


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def _json_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_verified_documents(path=None):
    source = Path(path or config.VERIFIED_KNOWLEDGE_PATH)
    if not source.exists():
        raise FileNotFoundError(f"RAG evidence source not found: {source}")
    if source.suffix.lower() == ".csv":
        documents = []
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                status = clean(row.get("verification_status"))
                if status not in PRODUCTION_STATUSES:
                    continue
                documents.append({
                    "id": clean(row.get("question_id")),
                    "document_type": "production_qa",
                    "product": clean(row.get("related_product")),
                    "category": clean(row.get("category")),
                    "subcategory": clean(row.get("subcategory")),
                    "question_type": clean(row.get("question_type")),
                    "question": clean(row.get("question")),
                    "answer": clean(row.get("answer")),
                    "content": clean(row.get("answer") or row.get("verification_notes")),
                    "source_urls": _json_list(row.get("source_urls")),
                    "source_titles": _json_list(row.get("source_titles")),
                    "verification_status": status,
                    "confidence": clean(row.get("confidence")),
                    "verification_notes": clean(row.get("verification_notes")),
                })
        return [item for item in documents if item["id"] and item["content"]]
    documents = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            if clean(document.get("verification_status")) in PRODUCTION_STATUSES:
                documents.append(document)
    return documents


def build_retrieval_text(document):
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


def bge_query(query, model_name=None):
    model = model_name or config.EMBEDDING_MODEL
    if model.startswith("BAAI/bge-"):
        return BGE_QUERY_PREFIX + query
    return query


def tokenize(text):
    text = clean(text).lower()
    text = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s'-]", " ", text)
    return [token for token in text.split() if token]


def make_result(document, score, text=None, metadata=None):
    return {
        "document_id": document["id"],
        "document": text if text is not None else build_retrieval_text(document),
        "product": clean(document.get("product")),
        "score": float(score),
        "metadata": metadata or {
            "product": clean(document.get("product")),
            "question_id": clean(document.get("id")),
            "category": clean(document.get("category")),
            "subcategory": clean(document.get("subcategory")) or None,
            "question_type": clean(document.get("question_type")),
            "document_type": clean(document.get("document_type")),
            "verification_status": clean(document.get("verification_status")),
            "confidence": clean(document.get("confidence")),
            "verification_notes": clean(document.get("verification_notes")),
            "source_urls": json.dumps(document.get("source_urls") or [], ensure_ascii=False),
            "source_titles": json.dumps(document.get("source_titles") or [], ensure_ascii=False),
        },
    }
