import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdic
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "rag"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from rag import config
    from rag.generation.answer_generator import extract_citation_ids, generate_answer
except ModuleNotFoundError:
    import config
    from generation.answer_generator import extract_citation_ids, generate_answer


EVAL_DIR = ROOT / "rag" / "evaluation"
QUERY_FILE = EVAL_DIR / "rag_evaluation_queries.json"
RESULTS_CSV = EVAL_DIR / "rag_evaluation_results.csv"
SUMMARY_CSV = EVAL_DIR / "rag_evaluation_summary.csv"
REPORT_MD = EVAL_DIR / "rag_evaluation_report.md"
VERIFIED_JSONL = ROOT / "rag" / "data" / "verified_knowledge.jsonl"

HUMAN_REVIEW_FIELDS = [
    "correctness_rating",
    "groundedness_rating",
    "answer_relevance_rating",
    "citation_correctness_rating",
    "partial_answer_quality_rating",
    "hallucination_present",
    "review_notes",
]


def load_queries():
    return json.loads(QUERY_FILE.read_text(encoding="utf-8"))


def load_verified_docs():
    docs = []
    with VERIFIED_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def numeric_tokens(text):
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|dt|tnd|days?|jours?|mois|ans?)?\b", (text or "").lower()))


def evidence_text_for_docs(doc_ids, docs_by_id):
    return " ".join((docs_by_id.get(doc_id) or {}).get("content", "") for doc_id in doc_ids)


def rank_of_expected(retrieved_ids, expected_ids):
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            return index
    return None


def citation_validation(answer, citations):
    used = extract_citation_ids(answer)
    provided = {item.get("citation_id") for item in citations}
    invalid = sorted(used - provided)
    return used, invalid


def classify_generation_failure(response):
    reason = response.get("reason", "") or ""
    answer = response.get("answer", "") or ""
    if "Missing Ollama Python client" in reason or "Ollama generation failed" in reason:
        return "generation_system_failure"
    if reason == "invalid_or_missing_citations" or "found verified evidence" in answer.lower():
        return "grounded_generation_failure"
    return ""


def evaluate_case(case, docs_by_id):
    response = generate_answer(case["query"])
    expected_ids = case.get("expected_document_ids") or []
    retrieved_ids = response.get("evidence_document_ids") or []
    citations = response.get("citations") or []
    used_citations, invalid_citations = citation_validation(response.get("answer", ""), citations)
    rank = rank_of_expected(retrieved_ids, expected_ids)
    expected_found = rank is not None

    supplied_evidence_text = evidence_text_for_docs(retrieved_ids, docs_by_id)
    answer_numbers = numeric_tokens(response.get("answer", ""))
    evidence_numbers = numeric_tokens(supplied_evidence_text)
    unsupported_number_risk = bool(answer_numbers - evidence_numbers)

    expected_product = case.get("expected_product") or ""
    retrieved_products = [
        (docs_by_id.get(doc_id) or {}).get("product", "")
        for doc_id in retrieved_ids
    ]
    wrong_product_retrieval = bool(
        expected_produc
        and response.get("has_evidence")
        and retrieved_products
        and expected_product not in retrieved_products
        and expected_ids
    )

    unexpected_answer_when_no_evidence = bool(
        not response.get("has_evidence")
        and response.get("answered")
    )
    generation_failure_type = classify_generation_failure(response)
    generation_failure = bool(generation_failure_type)
    invalid_citation = bool(invalid_citations or (response.get("answered") and not response.get("citation_validation_passed", False)))

    return {
        "query": case["query"],
        "evaluation_category": case.get("evaluation_category", ""),
        "question_type": case.get("question_type", ""),
        "expected_product": expected_product,
        "expected_document_ids": json.dumps(expected_ids, ensure_ascii=False),
        "has_evidence": response.get("has_evidence", False),
        "retrieved_document_ids": json.dumps(retrieved_ids, ensure_ascii=False),
        "expected_document_found": expected_found if expected_ids else "",
        "rank_of_expected_document": rank or "",
        "retrieved_product": json.dumps(retrieved_products, ensure_ascii=False),
        "answered": response.get("answered", False),
        "answer": response.get("answer", ""),
        "citations_used": json.dumps(sorted(used_citations), ensure_ascii=False),
        "citation_validation_passed": response.get("citation_validation_passed", len(invalid_citations) == 0),
        "evidence_count": response.get("evidence_count", len(retrieved_ids)),
        "generation_latency_ms": response.get("generation_latency_ms", 0.0),
        "context_latency_ms": response.get("context_latency_ms", 0.0),
        "total_rag_latency_ms": response.get("total_rag_latency_ms", 0.0),
        "unsupported_number_risk": unsupported_number_risk,
        "invalid_citation": invalid_citation,
        "unexpected_answer_when_no_evidence": unexpected_answer_when_no_evidence,
        "generation_failure": generation_failure,
        "generation_failure_type": generation_failure_type,
        "wrong_product_retrieval": wrong_product_retrieval,
        "llm_called": response.get("llm_called", False),
        "reason": response.get("reason", ""),
        "reason_for_flag": reason_for_flag({
            "unsupported_number_risk": unsupported_number_risk,
            "invalid_citation": invalid_citation,
            "unexpected_answer_when_no_evidence": unexpected_answer_when_no_evidence,
            "generation_failure": generation_failure,
            "generation_failure_type": generation_failure_type,
            "wrong_product_retrieval": wrong_product_retrieval,
        }),
        **{field: "" for field in HUMAN_REVIEW_FIELDS},
    }


def reason_for_flag(row):
    reasons = []
    if row.get("unexpected_answer_when_no_evidence"):
        reasons.append("no-evidence routing failure")
    if row.get("wrong_product_retrieval"):
        reasons.append("genuine retrieval/product-detection failure")
    if row.get("unsupported_number_risk"):
        reasons.append("unsupported numeric claim heuristic")
    if row.get("invalid_citation"):
        reasons.append("citation integrity failure")
    if row.get("generation_failure"):
        failure_type = row.get("generation_failure_type")
        if failure_type == "generation_system_failure":
            reasons.append("operational generation failure")
        elif failure_type == "grounded_generation_failure":
            reasons.append("grounded generation/citation failure")
        else:
            reasons.append("generation failure")
    return "; ".join(reasons)


def summarize_rows(rows, label, scope):
    expected_rows = [row for row in rows if row["expected_document_ids"] != "[]"]
    no_evidence_expected_rows = [
        row for row in rows
        if row["expected_document_ids"] == "[]"
        and row["evaluation_category"] in {"unanswerable", "high_risk", "adversarial"}
    ]
    answered_rows = [row for row in rows if row["answered"]]
    llm_rows = [row for row in rows if row["llm_called"]]
    successful_llm_rows = [row for row in llm_rows if not row["generation_failure"] and not row["invalid_citation"]]
    latencies = [float(row["total_rag_latency_ms"] or 0.0) for row in rows]
    context_latencies = [float(row["context_latency_ms"] or 0.0) for row in rows]
    generation_latencies = [float(row["generation_latency_ms"] or 0.0) for row in rows if row["generation_latency_ms"]]
    ranks = [int(row["rank_of_expected_document"]) for row in expected_rows if row["rank_of_expected_document"]]

    def hit_at(k):
        if not expected_rows:
            return ""
        return sum(bool(row["rank_of_expected_document"]) and int(row["rank_of_expected_document"]) <= k for row in expected_rows) / len(expected_rows)

    def product_hit_at(k):
        candidates = [row for row in expected_rows if row["expected_product"]]
        if not candidates:
            return ""
        hits = 0
        for row in candidates:
            products = json.loads(row["retrieved_product"]) if row["retrieved_product"] else []
            if row["expected_product"] in products[:k]:
                hits += 1
        return hits / len(candidates)

    return {
        "scope": scope,
        "label": label,
        "query_count": len(rows),
        "retrieval_hit_1": hit_at(1),
        "retrieval_hit_3": hit_at(3),
        "retrieval_mrr": (sum(1 / rank for rank in ranks) / len(expected_rows)) if expected_rows else "",
        "product_hit_1": product_hit_at(1),
        "product_hit_3": product_hit_at(3),
        "overall_answer_rate": (len(answered_rows) / len(rows)) if rows else 0.0,
        "llm_calls": len(llm_rows),
        "successful_llm_calls": len(successful_llm_rows),
        "llm_generation_success_rate": (len(successful_llm_rows) / len(llm_rows)) if llm_rows else "",
        "safe_no_evidence_refusals": sum((not row["answered"] and not row["llm_called"]) for row in no_evidence_expected_rows),
        "safe_partial_answers": sum(row["answered"] and not row["generation_failure"] for row in rows if row["evaluation_category"] == "partially_answerable"),
        "generation_success_rate": (len(successful_llm_rows) / len(llm_rows)) if llm_rows else "",
        "citation_validity_rate": sum(not row["invalid_citation"] for row in rows) / len(rows) if rows else 0.0,
        "correct_no_evidence_refusal_rate": (
            sum((not row["answered"] and not row["llm_called"] and row["citations_used"] == "[]") for row in no_evidence_expected_rows) / len(no_evidence_expected_rows)
            if no_evidence_expected_rows else ""
        ),
        "critical_safety_failures": sum(
            row["invalid_citation"]
            or row["unexpected_answer_when_no_evidence"]
            or (row["generation_failure"] and row.get("generation_failure_type") != "generation_system_failure")
            or row["wrong_product_retrieval"]
            for row in rows
        ),
        "generation_system_failures": sum(row.get("generation_failure_type") == "generation_system_failure" for row in rows),
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
        "retrieval_context_median_ms": statistics.median(context_latencies) if context_latencies else 0.0,
        "generation_median_ms": statistics.median(generation_latencies) if generation_latencies else 0.0,
        "manual_review_flags": sum(row["unsupported_number_risk"] for row in rows),
    }


def build_summaries(rows):
    summaries = [summarize_rows(rows, "overall", "overall")]
    for category, group in sorted(group_rows(rows, "evaluation_category").items()):
        summaries.append(summarize_rows(group, category, "evaluation_category"))
    for question_type, group in sorted(group_rows(rows, "question_type").items()):
        summaries.append(summarize_rows(group, question_type, "question_type"))
    return summaries


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return grouped


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(rows, summaries, corpus_count):
    overall = summaries[0]
    category_counts = Counter(row["evaluation_category"] for row in rows)
    flagged = [
        row for row in rows
        if row["unsupported_number_risk"]
        or row["invalid_citation"]
        or row["unexpected_answer_when_no_evidence"]
        or row["generation_failure"]
        or row["wrong_product_retrieval"]
    ]

    lines = [
        "# Atlas Demo RAG End-to-End Evaluation Report",
        "",
        "## Architecture Evaluated",
        "- Embedding: `BAAI/bge-base-en-v1.5`",
        f"- Retriever: `{config.RETRIEVER_TYPE}`",
        f"- Generator: `{config.LLM_MODEL}` via `{config.LLM_PROVIDER}`",
        f"- Temperature: `{config.LLM_TEMPERATURE}`",
        f"- Context top_k: `{getattr(config, 'CONTEXT_TOP_K', 3)}`",
        f"- Evaluation date: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Corpus Size",
        f"- Verified documents: {corpus_count}",
        "",
        "## Evaluation Dataset Composition",
    ]
    for category, count in category_counts.items():
        lines.append(f"- {category}: {count}")

    lines.extend([
        "",
        "## Retrieval Performance",
        f"- Hit@1: {overall['retrieval_hit_1']}",
        f"- Hit@3: {overall['retrieval_hit_3']}",
        f"- MRR: {overall['retrieval_mrr']}",
        f"- Product Hit@1: {overall['product_hit_1']}",
        f"- Product Hit@3: {overall['product_hit_3']}",
        "",
        "## Generation And Citation Performance",
        f"- Overall answer rate: {overall['overall_answer_rate']:.3f}",
        f"- LLM calls: {overall['llm_calls']}",
        f"- Successful LLM calls: {overall['successful_llm_calls']}",
        f"- LLM generation success rate: {overall['llm_generation_success_rate']}",
        f"- Safe partial answers: {overall['safe_partial_answers']}",
        f"- Citation validity rate: {overall['citation_validity_rate']:.3f}",
        f"- Critical safety failures: {overall['critical_safety_failures']}",
        f"- Generation system failures: {overall['generation_system_failures']}",
        "",
        "## No-Evidence Safety",
        f"- Correct no-evidence refusal rate: {overall['correct_no_evidence_refusal_rate']}",
        "",
        "## Adversarial, French, Paraphrase, And Noisy Query Performance",
        "See `rag_evaluation_summary.csv` for category-level summaries. Do not draw strong conclusions from categories with very small sample sizes.",
        "",
        "## Latency",
        f"- Retrieval/context median: {overall['retrieval_context_median_ms']:.2f} ms",
        f"- Generation median: {overall['generation_median_ms']:.2f} ms",
        f"- Mean: {overall['mean_latency_ms']:.2f} ms",
        f"- Median: {overall['median_latency_ms']:.2f} ms",
        f"- P95: {overall['p95_latency_ms']:.2f} ms",
        "",
        "## Automatically Flagged Cases",
    ])
    if not flagged:
        lines.append("- No automatic critical flags were raised.")
    for row in flagged[:20]:
        flags = [
            name for name in [
                "unsupported_number_risk",
                "invalid_citation",
                "unexpected_answer_when_no_evidence",
                "generation_failure",
                "wrong_product_retrieval",
            ]
            if row[name]
        ]
        lines.append(f"- `{row['query']}`: {', '.join(flags)} ({row.get('reason_for_flag', '')})")

    lines.extend([
        "",
        "## Human Review",
        "Human-review rating fields use a 1-5 scale: 1 = poor, 2 = weak, 3 = acceptable, 4 = good, 5 = excellent. Subjective fields are intentionally left blank for manual scoring.",
        "",
        "## Limitations",
        "The verified corpus is still small, so this evaluation is preliminary. Automated checks cannot fully prove semantic correctness or citation correctness; human review is required before any production claim.",
        "",
        "## Conclusion",
        "This framework evaluates the selected BGE Base + Hybrid RRF + verified-only context + Phi-4 Mini pipeline as one system. Rerun it after expanding verification and rebuilding the Chroma index.",
    ])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summaries, corpus_count):
    overall = summaries[0]
    print("Atlas Demo RAG End-to-End Evaluation")
    print("==================================")
    print(f"Corpus documents: {corpus_count}")
    print(f"Evaluation queries: {overall['query_count']}")
    print()
    print(f"Retrieval Hit@1: {overall['retrieval_hit_1']}")
    print(f"Retrieval Hit@3: {overall['retrieval_hit_3']}")
    print(f"Product Hit@3: {overall['product_hit_3']}")
    print()
    print(f"Generation success: {overall['generation_success_rate']:.3f}")
    print(f"LLM calls: {overall['llm_calls']}")
    print(f"Successful LLM calls: {overall['successful_llm_calls']}")
    print(f"LLM generation success: {overall['llm_generation_success_rate']}")
    print(f"Safe no-evidence refusals: {overall['safe_no_evidence_refusals']}")
    print(f"Safe partial answers: {overall['safe_partial_answers']}")
    print(f"Citation validity: {overall['citation_validity_rate']:.3f}")
    print(f"Correct no-evidence refusals: {overall['correct_no_evidence_refusal_rate']}")
    adversarial = next((row for row in summaries if row["label"] == "adversarial"), None)
    print(f"Adversarial failures: {adversarial['critical_safety_failures'] if adversarial else 0}")
    print(f"Critical safety failures: {overall['critical_safety_failures']}")
    print(f"Generation system failures: {overall['generation_system_failures']}")
    print()
    print(f"Mean RAG latency: {overall['mean_latency_ms']:.2f} ms")
    print(f"Median RAG latency: {overall['median_latency_ms']:.2f} ms")
    print(f"P95 RAG latency: {overall['p95_latency_ms']:.2f} ms")
    print()
    print(f"Queries requiring manual review: {overall['manual_review_flags']}")


def main():
    queries = load_queries()
    docs = load_verified_docs()
    docs_by_id = {doc["id"]: doc for doc in docs}
    rows = [evaluate_case(case, docs_by_id) for case in queries]
    summaries = build_summaries(rows)

    result_fields = [
        "query",
        "evaluation_category",
        "question_type",
        "expected_product",
        "expected_document_ids",
        "has_evidence",
        "retrieved_document_ids",
        "expected_document_found",
        "rank_of_expected_document",
        "retrieved_product",
        "answered",
        "answer",
        "citations_used",
        "citation_validation_passed",
        "llm_called",
        "reason",
        "evidence_count",
        "generation_latency_ms",
        "context_latency_ms",
        "total_rag_latency_ms",
        "unsupported_number_risk",
        "invalid_citation",
        "unexpected_answer_when_no_evidence",
        "generation_failure",
        "generation_failure_type",
        "wrong_product_retrieval",
        "reason_for_flag",
        *HUMAN_REVIEW_FIELDS,
    ]
    summary_fields = [
        "scope",
        "label",
        "query_count",
        "retrieval_hit_1",
        "retrieval_hit_3",
        "retrieval_mrr",
        "product_hit_1",
        "product_hit_3",
        "overall_answer_rate",
        "llm_calls",
        "successful_llm_calls",
        "llm_generation_success_rate",
        "safe_no_evidence_refusals",
        "safe_partial_answers",
        "generation_success_rate",
        "citation_validity_rate",
        "correct_no_evidence_refusal_rate",
        "critical_safety_failures",
        "generation_system_failures",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "retrieval_context_median_ms",
        "generation_median_ms",
        "manual_review_flags",
    ]

    write_csv(RESULTS_CSV, rows, result_fields)
    write_csv(SUMMARY_CSV, summaries, summary_fields)
    write_report(rows, summaries, len(docs))
    print_summary(summaries, len(docs))


if __name__ == "__main__":
    main()
