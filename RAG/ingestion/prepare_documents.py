import csv
import as
import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "RAG" / "data"

QA_CSV_CANDIDATES = [
    DATA_DIR / "atlas_demo_chatbot_grounded_answers_verified.csv",
    DATA_DIR / "atlas_demo_chatbot_grounded_answers_production.csv",
    DATA_DIR / "atlas_demo_chatbot_grounded_answers_cleaned.csv",
    ROOT / "atlas_demo_chatbot_grounded_answers_cleaned.csv",
]
KB_CANDIDATES = [
    DATA_DIR / "kb_extended_verified_batch13.py",
    DATA_DIR / "kb_extended.py",
    ROOT / "kb_extended.py",
]

OUTPUT_JSONL = DATA_DIR / "verified_knowledge.jsonl"
OUTPUT_REJECTIONS = DATA_DIR / "rejected_documents.csv"

PRODUCTION_STATUSES = {
    "verified",
    "partially_verified",
    "official_information_not_public",
    "human_intervention",
}
EXCLUDED_STATUSES = {"unverified", "needs_manual_review"}

OFFICIAL_DOMAINS = {
    "example.org",
    "example.org",
    "example.org",
    "example.org",
}


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(str(p) for p in paths))


def normalize_text(value):
    if value is None:
        return None
    text = str(value)
    if not text.strip():
        return None

    text = text.replace("�", "é")
    try:
        if "Ã" in text:
            text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        pass

    replacements = {
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã‰": "É",
        "r�sidents": "résidents",
        "r�sidanat": "résidanat",
        "Tr�s": "Très",
        "Bar�me": "Barème",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def load_canonical_product_names(kb_path):
    names = {}
    try:
        spec = importlib.util.spec_from_file_location("kb_extended_for_ingestion", kb_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for collection_name in ("PRODUCTS", "EXTENDED_PRODUCTS"):
            collection = getattr(module, collection_name, {})
            for raw_name in collection:
                cleaned = normalize_text(raw_name)
                if cleaned:
                    names[cleaned.lower()] = cleaned
    except Exception as exc:
        print(f"WARNING: Could not execute KB module {kb_path}: {exc}. Falling back to AST product extraction.")
        tree = ast.parse(kb_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id in {"PRODUCTS", "EXTENDED_PRODUCTS"} for target in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key in node.value.keys:
                if isinstance(key, ast.Constant):
                    cleaned = normalize_text(key.value)
                    if cleaned:
                        names[cleaned.lower()] = cleaned
    return names


def canonicalize_name(value, canonical_names):
    cleaned = normalize_text(value)
    if cleaned is None:
        return None
    return canonical_names.get(cleaned.lower(), cleaned)


def parse_json_list(value):
    cleaned = normalize_text(value)
    if cleaned is None:
        return []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [normalize_text(item) for item in parsed if normalize_text(item)]
    except json.JSONDecodeError:
        pass
    return [normalize_text(match) for match in re.findall(r"https?://[^,\]\s\"]+", cleaned)]


def normalize_url(value):
    text = normalize_text(value)
    if text is None:
        return None
    urls = re.findall(r"https?://[^\s)\]]+", text)
    return urls[-1] if urls else tex


def is_official_url(url):
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    return host in OFFICIAL_DOMAINS or host.endswith(".example.org") or host.endswith(".example.org")


def normalize_sources(url_value, title_value):
    urls = []
    for url in parse_json_list(url_value):
        normalized_url = normalize_url(url)
        if normalized_url and normalized_url not in urls:
            urls.append(normalized_url)

    titles = []
    for title in parse_json_list(title_value):
        if title and title not in titles:
            titles.append(title)

    return urls, titles


def status_key(value):
    return (normalize_text(value) or "").lower()


def row_get(row, field):
    target = field.lower().lstrip("\ufeff")
    for key, value in row.items():
        if str(key).lower().lstrip("\ufeff") == target:
            return value
    return None


def load_production_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    production = [row for row in rows if status_key(row_get(row, "verification_status")) in PRODUCTION_STATUSES]
    excluded = [row for row in rows if status_key(row_get(row, "verification_status")) in EXCLUDED_STATUSES]
    unknown = [
        row for row in rows
        if status_key(row_get(row, "verification_status")) not in PRODUCTION_STATUSES | EXCLUDED_STATUSES
    ]
    return rows, production, excluded, unknown


def build_document(row, canonical_products, canonical_categories):
    source_urls, source_titles = normalize_sources(row_get(row, "source_urls"), row_get(row, "source_titles"))
    question_id = normalize_text(row_get(row, "question_id"))
    category = canonicalize_name(row_get(row, "category"), canonical_categories)
    product = canonicalize_name(row_get(row, "related_product"), canonical_products)
    subcategory = normalize_text(row_get(row, "subcategory"))
    answer = normalize_text(row_get(row, "answer"))
    verification_notes = normalize_text(row_get(row, "verification_notes") or row_get(row, "verification_note"))
    status = status_key(row_get(row, "verification_status"))
    content = answer or verification_notes

    return {
        "id": question_id,
        "document_type": "production_qa",
        "product": product,
        "category": category,
        "subcategory": subcategory,
        "question_type": normalize_text(row_get(row, "question_type")),
        "question": normalize_text(row_get(row, "question")),
        "answer": answer,
        "content": content,
        "source_urls": source_urls,
        "source_titles": source_titles,
        "verification_status": status,
        "confidence": normalize_text(row_get(row, "confidence")),
        "verification_notes": verification_notes,
    }


def validate_document(document, seen_ids):
    reasons = []
    doc_id = document.get("id")
    if not doc_id:
        reasons.append("missing id")
    elif doc_id in seen_ids:
        reasons.append("duplicate id")

    if not document.get("content"):
        reasons.append("empty content")

    if document.get("verification_status") not in PRODUCTION_STATUSES:
        reasons.append("verification_status is not production-safe")

    official_urls = [url for url in document.get("source_urls", []) if is_official_url(url)]
    if official_urls:
        document["source_urls"] = official_urls

    product = document.get("product")
    if not product:
        category = (document.get("category") or "").lower()
        if "general" not in category and "lost" not in category:
            reasons.append("blank related product for non-general row")

    return reasons


def write_jsonl(documents, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def write_rejections(rejections, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "question", "reason_rejected", "verification_status"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejections)


def main():
    csv_path = first_existing(QA_CSV_CANDIDATES)
    kb_path = first_existing(KB_CANDIDATES)

    canonical_products = load_canonical_product_names(kb_path)
    rows, production_rows, excluded_rows, unknown_rows = load_production_rows(csv_path)
    canonical_categories = {
        normalize_text(row_get(row, "category")).lower(): normalize_text(row_get(row, "category"))
        for row in rows
        if normalize_text(row_get(row, "category"))
    }

    documents = []
    rejections = []
    seen_ids = set()

    for row in production_rows:
        document = build_document(row, canonical_products, canonical_categories)
        reasons = validate_document(document, seen_ids)
        if reasons:
            rejections.append({
                "question_id": normalize_text(row_get(row, "question_id")) or "",
                "question": normalize_text(row_get(row, "question")) or "",
                "reason_rejected": "; ".join(reasons),
                "verification_status": normalize_text(row_get(row, "verification_status")) or "",
            })
            continue
        seen_ids.add(document["id"])
        documents.append(document)

    write_jsonl(documents, OUTPUT_JSONL)
    write_rejections(rejections, OUTPUT_REJECTIONS)

    status_counts = {status: 0 for status in sorted(PRODUCTION_STATUSES | EXCLUDED_STATUSES)}
    for row in rows:
        status = status_key(row_get(row, "verification_status"))
        if status in status_counts:
            status_counts[status] += 1

    print(f"Total input rows: {len(rows)}")
    print(f"Verified count: {status_counts['verified']}")
    print(f"Partially verified count: {status_counts['partially_verified']}")
    print(f"Official information not public count: {status_counts['official_information_not_public']}")
    print(f"Human intervention count: {status_counts['human_intervention']}")
    print(f"Excluded unverified count: {status_counts['unverified']}")
    print(f"Excluded needs manual review count: {status_counts['needs_manual_review']}")
    print(f"Final JSONL document count: {len(documents)}")
    print(f"Documents rejected: {len(rejections)}")
    if unknown_rows:
        unknown_statuses = sorted({status_key(row_get(row, "verification_status")) or "<blank>" for row in unknown_rows})
        print("WARNING: Unknown verification statuses found: " + ", ".join(unknown_statuses))
    print(f"Unique products represented: {len({doc.get('product') for doc in documents if doc.get('product')})}")


if __name__ == "__main__":
    main()
