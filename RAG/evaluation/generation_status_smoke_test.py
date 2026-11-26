import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAG.generation.answer_generator import generate_answer
from RAG.generation.context_builder import build_contex


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


def summarize_evidence(context):
    return [
        {
            "question_id": item.get("document_id"),
            "retrieval_score": item.get("retrieval_score"),
            "verification_status": item.get("verification_status"),
            "confidence": item.get("confidence"),
            "sources": item.get("source_urls"),
        }
        for item in context.get("evidence", [])
    ]


def main():
    examples = load_examples()
    for status in STATUSES:
        document = examples.get(status)
        if not document:
            print(json.dumps({"status": status, "error": "missing example"}, ensure_ascii=False))
            continue

        query = document["question"]
        context = build_context(query=query, top_k=3)
        response = generate_answer(query=query, top_k=3)
        print(json.dumps({
            "query": query,
            "retrieved": summarize_evidence(context),
            "final_generated_answer": response.get("answer"),
            "answered": response.get("answered"),
            "citations": response.get("citations"),
            "llm_called": response.get("llm_called"),
            "deterministic_fallback_used": response.get("deterministic_fallback_used"),
            "fallback_reason": response.get("fallback_reason"),
            "reason": response.get("reason"),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
