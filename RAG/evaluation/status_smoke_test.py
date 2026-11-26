import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAG.retrieval import retriever


STATUSES = [
    "verified",
    "partially_verified",
    "official_information_not_public",
    "human_intervention",
]


def load_examples():
    path = ROOT / "RAG" / "data" / "verified_knowledge.jsonl"
    examples = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            document = json.loads(line)
            examples.setdefault(document.get("verification_status"), document)
    return examples


def main():
    examples = load_examples()
    for status in STATUSES:
        document = examples.get(status)
        if not document:
            print(f"missing example for status={status}")
            continue
        response = retriever.retrieve(document["question"], top_k=3)
        first = response.get("results", [{}])[0]
        print(json.dumps({
            "query": document["question"],
            "question_id": first.get("document_id"),
            "retrieval_score": first.get("score"),
            "verification_status": first.get("verification_status"),
            "confidence": first.get("confidence"),
            "answer": first.get("content"),
            "sources": first.get("source_urls"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
