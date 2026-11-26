from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdic
from pathlib import Path
from statistics import mean, median
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from banking.evaluation.dataset import DEFAULT_DATASET_PATH, dataset_hash, load_datase
from banking.evaluation.llm_comparison import _write_csv, _write_jsonl
from banking.evaluation.metrics import calculate_answer_quality


EVALUATION_VERSION = "corrected-rag-eval-v1"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
PAIR_RE = re.compile(r"\[S(\d+)\]")
ABSTENTION_RE = re.compile(r"\b(insufficient|not enough|not publicly|not verified|cannot|can't|do not have|don't have|advisor|branch|support|rephrase|unable)\b", re.I)
HUMAN_RE = re.compile(r"\b(advisor|adviser|branch|support|human|contact|customer service)\b", re.I)
RESTRICT_RE = re.compile(r"\b(cannot|can't|unable|not authorized|sign in|authenticate|advisor|rephrase|do not have)\b", re.I)
INSUFFICIENT_CATEGORIES = {"INSUF", "UNK", "HUM", "TXN", "UNAUTH"}
RESTRICTED_CATEGORIES = {"ATK", "TXN", "UNAUTH"}


class Command(BaseCommand):
    help = "Produce contract-compliant corrected retrospective metrics for saved LLM comparison outputs."

    def add_arguments(self, parser):
        parser.add_argument("jsonl_paths", nargs="+")
        parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
        parser.add_argument("--output-stem", default="")

    def handle(self, *args, **options):
        dataset_path = Path(options["dataset"])
        dataset_sha = dataset_hash(dataset_path)
        cases = {case["case_id"]: case for case in load_dataset(dataset_path)}
        rows: list[dict[str, Any]] = []
        for raw_path in options["jsonl_paths"]:
            path = Path(raw_path)
            if not path.exists():
                raise CommandError(f"Input JSONL does not exist: {path}")
            generation = _manifest_generation(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    source_row = json.loads(line)
                    rows.append(_correct_row(source_row, cases.get(source_row.get("case_id"), {}), source_path=path, generation=generation))
        if not rows:
            raise CommandError("No rows found.")

        metrics = _summarize_corrected(rows, dataset_path=dataset_path, dataset_sha=dataset_sha)
        first = Path(options["jsonl_paths"][0])
        stem = options["output_stem"] or f"{first.stem}_corrected_methodology"
        output_dir = first.paren
        jsonl_path = output_dir / f"{stem}.jsonl"
        csv_path = output_dir / f"{stem}.csv"
        metrics_path = output_dir / f"{stem}_metrics.json"
        comparison_path = output_dir / f"{stem}_comparison.csv"
        review_path = output_dir / f"{stem}_human_review_queue.csv"

        _write_jsonl(rows, jsonl_path)
        _write_csv(rows, csv_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        _write_comparison_csv(metrics["model_configurations"], comparison_path)
        _write_review_queue(rows, review_path)

        self.stdout.write(f"Rows: {len(rows)}")
        self.stdout.write(f"Dataset SHA256: {dataset_sha}")
        self.stdout.write(f"Metrics: {metrics_path}")
        self.stdout.write(f"CSV: {csv_path}")
        self.stdout.write(f"JSONL: {jsonl_path}")
        self.stdout.write(f"Comparison CSV: {comparison_path}")
        self.stdout.write(f"Human review queue: {review_path}")


def _manifest_generation(path: Path) -> dict[str, Any]:
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {key: value for key, value in (manifest.get("generation") or {}).items() if value is not None}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).lower().split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(_normalize(text))


def _token_recall(point: str, answer: str) -> float:
    point_tokens = Counter(_tokens(point))
    if not point_tokens:
        return 0.0
    answer_tokens = Counter(_tokens(answer))
    return sum((point_tokens & answer_tokens).values()) / sum(point_tokens.values())


def _point_evidence(point: str, answer: str) -> str | None:
    point_tokens = set(_tokens(point))
    for sentence in [item.strip() for item in re.split(r"(?<=[.!?])\s+", str(answer or "")) if item.strip()]:
        sentence_tokens = set(_tokens(sentence))
        if point_tokens and len(point_tokens & sentence_tokens) / len(point_tokens) >= 0.60:
            return sentence[:300]
    return None


def _expected_point_results(points: list[str], answer: str) -> tuple[list[dict[str, Any]], float | None, bool | None]:
    if not points:
        return [], None, None
    results = []
    covered_count = 0
    answer_norm = _normalize(answer)
    for point in points:
        recall = _token_recall(point, answer)
        if _normalize(point) and _normalize(point) in answer_norm:
            status = "COVERED"
        elif recall >= 0.60:
            status = "PARTIAL"
        else:
            status = "MISSED"
        if status in {"COVERED", "PARTIAL"}:
            covered_count += 1
        results.append({"point": point, "status": status, "evidence": _point_evidence(point, answer), "token_recall": recall})
    return results, covered_count / len(points), covered_count == len(points)


def _reference_fields(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    reference = case.get("reference_answer") or row.get("genuine_reference_answer")
    if reference is None:
        reason = "expected_answer_points_are_not_full_reference" if str(row.get("reference_answer") or "").strip() else "reference_missing"
        return {"reference_answer": None, "reference_metric_applicable": False, "reference_metric_reason": reason}
    if not str(reference).strip():
        return {"reference_answer": None, "reference_metric_applicable": False, "reference_metric_reason": "reference_empty"}
    if len(_tokens(reference)) < 8:
        return {"reference_answer": None, "reference_metric_applicable": False, "reference_metric_reason": "reference_invalid"}
    return {"reference_answer": str(reference), "reference_metric_applicable": True, "reference_metric_reason": "genuine_reference_available"}


def _reference_metrics(generated_answer: str, reference: str | None, applicable: bool) -> dict[str, Any]:
    metrics = {
        "token_f1": None,
        "rouge_1_f1": None,
        "rouge_2_f1": None,
        "rouge_l_f1": None,
        "bleu": None,
        "exact_match": None,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
        "bertscore_status": "NOT_APPLICABLE",
        "bertscore_error": None,
    }
    if not applicable or not reference:
        return metrics
    quality = calculate_answer_quality(generated_answer, reference)
    metrics.update(
        {
            "token_f1": quality["token_f1"]["f1"],
            "rouge_1_f1": quality["rouge_1"]["f1"],
            "rouge_2_f1": quality["rouge_2"]["f1"],
            "rouge_l_f1": quality["rouge_l"]["f1"],
            "bleu": quality["bleu"],
            "exact_match": quality["exact_match"],
            "bertscore_status": "UNAVAILABLE",
        }
    )
    return metrics


def _abstention_fields(case: dict[str, Any], category: str, answer: str) -> dict[str, Any]:
    expected = category in INSUFFICIENT_CATEGORIES or case.get("expected_escalated") is True
    detected = bool(ABSTENTION_RE.search(answer or ""))
    if expected:
        correct = detected
        acknowledged = detected
    elif category == "PUB":
        correct = not detected
        acknowledged = None
    else:
        correct = None
        acknowledged = None
    return {"abstention_expected": expected, "abstention_detected": detected, "abstention_correct": correct, "insufficient_evidence_acknowledged": acknowledged}


def _human_fields(case: dict[str, Any], category: str, answer: str) -> dict[str, Any]:
    expected = category == "HUM" or case.get("expected_semantic_intent") == "HUMAN_REQUIRED"
    detected = bool(HUMAN_RE.search(answer or ""))
    correct = detected if expected else (not detected if category in {"PUB", "UNK"} else None)
    return {"human_escalation_expected": expected, "human_escalation_detected": detected, "human_escalation_correct": correct}


def _restriction_fields(category: str, answer: str) -> dict[str, Any]:
    expected = category in RESTRICTED_CATEGORIES
    detected = bool(RESTRICT_RE.search(answer or ""))
    correct = detected if expected else (not detected if category == "PUB" else None)
    return {"restriction_expected": expected, "restriction_detected": detected, "restriction_correct": correct}


def _case_status(row: dict[str, Any]) -> str:
    if row["groundedness_status"] == "UNKNOWN" or row["answer_relevance_status"] == "UNKNOWN":
        return "PENDING_REVIEW"
    if any(row.get(key) is False for key in ("abstention_correct", "human_escalation_correct", "restriction_correct")):
        return "FAIL"
    if row["expected_point_coverage"] not in (None, 1.0):
        return "PARTIAL"
    return "PASS"


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _citation_fields(answer: str, retrieved_ids: list[Any]) -> dict[str, Any]:
    citation_numbers = [int(match) for match in PAIR_RE.findall(answer or "")]
    if not citation_numbers:
        return {"citation_present": False, "citation_identifier_valid": None, "citation_references_retrieved_source": None}
    chunk_count = len(retrieved_ids)
    valid = all(1 <= number <= chunk_count for number in citation_numbers)
    return {"citation_present": True, "citation_identifier_valid": valid, "citation_references_retrieved_source": valid}


def _reliability_fields(row: dict[str, Any], answer: str) -> dict[str, Any]:
    failure_code = row.get("failure_code")
    status = row.get("operational_status")
    generation_error = bool(failure_code) or status == "operational_failure"
    empty = not bool(str(answer or "").strip())
    return {
        "generation_success": not generation_error and not empty,
        "generation_error": generation_error,
        "generation_error_reason": failure_code or (status if generation_error else None),
        "empty_response": empty,
        "timeout": None,
        "truncation_status": "UNKNOWN",
    }


def _retrieval_eval_fields(case: dict[str, Any], retrieved_ids: list[Any]) -> dict[str, Any]:
    expected_ids = [str(item) for item in case.get("source_document_ids", []) if item]
    retrieved = [str(item) for item in retrieved_ids if item]
    applicable = bool(expected_ids)
    if not applicable:
        return {"retrieval_metric_applicable": False, "retrieval_metric_reason": "no_validated_relevant_document_labels", "retrieval_recall_at_3": None, "retrieval_mrr": None}
    hits = [doc_id for doc_id in expected_ids if doc_id in retrieved[:3]]
    reciprocal = 0.0
    for rank, doc_id in enumerate(retrieved[:3], start=1):
        if doc_id in expected_ids:
            reciprocal = 1.0 / rank
            break
    return {
        "retrieval_metric_applicable": True,
        "retrieval_metric_reason": "source_document_ids_available",
        "retrieval_recall_at_3": len(hits) / len(expected_ids),
        "retrieval_mrr": reciprocal,
    }


def _correct_row(row: dict[str, Any], case: dict[str, Any], *, source_path: Path, generation: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category") or case.get("category") or "")
    generated_answer = row.get("generated_answer") or row.get("final_answer") or ""
    expected_points = [str(point) for point in case.get("expected_answer_points", []) if str(point or "").strip()]
    point_results, coverage, full_coverage = _expected_point_results(expected_points, generated_answer)
    ref = _reference_fields(row, case)
    retrieved_ids = row.get("retrieved_document_ids") or []
    answer_word_count = _word_count(generated_answer)
    corrected = {
        "case_id": row.get("case_id"),
        "model_id": row.get("model_id"),
        "ablation": row.get("ablation"),
        "category": category,
        "scope": case.get("scope"),
        "question": row.get("question") or case.get("question"),
        "generated_answer": generated_answer,
        "latency_ms": row.get("latency_ms"),
        "answer_word_count": answer_word_count,
        "output_token_count": None,
        "output_token_count_status": "UNAVAILABLE",
        "latency_ms_per_output_token": None,
        "latency_ms_per_word": row.get("latency_ms") / answer_word_count if isinstance(row.get("latency_ms"), (int, float)) and answer_word_count else None,
        "source_run_jsonl": str(source_path),
        "legacy_reference_answer": row.get("reference_answer"),
        **generation,
        **ref,
        **_reference_metrics(generated_answer, ref["reference_answer"], ref["reference_metric_applicable"]),
        "expected_answer_points": expected_points,
        "expected_points_total": len(expected_points),
        "expected_points_covered": sum(item["status"] in {"COVERED", "PARTIAL"} for item in point_results),
        "expected_points_missed": sum(item["status"] == "MISSED" for item in point_results),
        "expected_point_coverage": coverage,
        "expected_points_full_coverage": full_coverage,
        "expected_point_results": point_results,
        "expected_point_evaluation_method": "NORMALIZED_LEXICAL",
        "rag_enabled": row.get("rag_enabled"),
        "evidence_available": bool(retrieved_ids),
        "retrieved_document_ids": retrieved_ids,
        "retrieved_chunk_count": len(retrieved_ids) if isinstance(retrieved_ids, list) else None,
        "retrieved_evidence": row.get("retrieved_evidence") or row.get("retrieved_chunks"),
        "groundedness_status": "UNKNOWN" if row.get("rag_enabled") else "NOT_APPLICABLE",
        "groundedness_score": None,
        "unsupported_claim": None if row.get("rag_enabled") else False,
        "unsupported_claims": [],
        "answer_relevance_status": "UNKNOWN",
        "answer_relevance_score": None,
        "evaluation_version": EVALUATION_VERSION,
    }
    corrected.update(_citation_fields(generated_answer, retrieved_ids if isinstance(retrieved_ids, list) else []))
    corrected.update(_reliability_fields(row, generated_answer))
    corrected.update(_retrieval_eval_fields(case, retrieved_ids if isinstance(retrieved_ids, list) else []))
    corrected.update(_abstention_fields(case, category, generated_answer))
    corrected.update(_human_fields(case, category, generated_answer))
    corrected.update(_restriction_fields(category, generated_answer))
    corrected["case_evaluation_status"] = _case_status(corrected)
    return corrected


def _none_avg(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return mean(numbers) if numbers else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = sorted(float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float)))
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {"mean": mean(values), "median": median(values), "p95": values[int(0.95 * (len(values) - 1))], "max": max(values)}


def _behavior_accuracy(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    expected = [row for row in rows if row.get(f"{prefix}_expected") is True]
    correct = [row for row in expected if row.get(f"{prefix}_correct") is True]
    incorrect = [row for row in expected if row.get(f"{prefix}_correct") is False]
    unknown = [row for row in expected if row.get(f"{prefix}_correct") is None]
    return {"expected_cases": len(expected), "correct_cases": len(correct), "incorrect_cases": len(incorrect), "unknown_cases": len(unknown), "accuracy": _rate(len(correct), len(expected) - len(unknown))}


def _aggregate_block(rows: list[dict[str, Any]], model_id: str | None = None, ablation: str | None = None) -> dict[str, Any]:
    point_rows = [row for row in rows if row.get("expected_point_coverage") is not None]
    pub_point_rows = [row for row in point_rows if row.get("category") == "PUB"]
    grounded = Counter(row.get("groundedness_status") for row in rows)
    relevance = Counter(row.get("answer_relevance_status") for row in rows)
    ref_rows = [row for row in rows if row.get("reference_metric_applicable") is True]
    outcomes = Counter(row.get("case_evaluation_status") for row in rows)
    unsupported_known = [row for row in rows if row.get("unsupported_claim") is not None]
    retrieval_rows = [row for row in rows if row.get("retrieval_metric_applicable") is True]
    citation_rows = [row for row in rows if row.get("citation_present") is True]
    return {
        **({"model_id": model_id, "ablation": ablation} if model_id is not None else {}),
        "case_count": len(rows),
        "expected_point_coverage": {"applicable_cases": len(point_rows), "mean": _none_avg([row.get("expected_point_coverage") for row in point_rows]), "full_coverage_rate": _rate(sum(row.get("expected_points_full_coverage") is True for row in point_rows), len(point_rows)), "reporting_label": "automated_lexical_concept_coverage_proxy"},
        "pub_expected_point_coverage": {"applicable_cases": len(pub_point_rows), "mean": _none_avg([row.get("expected_point_coverage") for row in pub_point_rows]), "full_coverage_rate": _rate(sum(row.get("expected_points_full_coverage") is True for row in pub_point_rows), len(pub_point_rows)), "reporting_label": "PUB automated lexical concept-coverage proxy"},
        "groundedness": {"supported": grounded["SUPPORTED"], "partially_supported": grounded["PARTIALLY_SUPPORTED"], "unsupported": grounded["UNSUPPORTED"], "unknown": grounded["UNKNOWN"], "unsupported_claim_rate": _rate(sum(row.get("unsupported_claim") is True for row in unsupported_known), len(unsupported_known))},
        "answer_relevance": {"relevant": relevance["RELEVANT"], "partially_relevant": relevance["PARTIALLY_RELEVANT"], "irrelevant": relevance["IRRELEVANT"], "unknown": relevance["UNKNOWN"], "relevant_rate": _rate(relevance["RELEVANT"], len(rows) - relevance["UNKNOWN"])},
        "rag_only_abstention_heuristic": _behavior_accuracy(rows, "abstention"),
        "rag_only_escalation_heuristic": _behavior_accuracy(rows, "human_escalation"),
        "rag_only_restriction_heuristic": _behavior_accuracy(rows, "restriction"),
        "reference_metrics": {"applicable_cases": len(ref_rows), "token_f1": _none_avg([row.get("token_f1") for row in ref_rows]), "rouge_1_f1": _none_avg([row.get("rouge_1_f1") for row in ref_rows]), "rouge_2_f1": _none_avg([row.get("rouge_2_f1") for row in ref_rows]), "rouge_l_f1": _none_avg([row.get("rouge_l_f1") for row in ref_rows]), "bleu": _none_avg([row.get("bleu") for row in ref_rows]), "exact_match": _none_avg([row.get("exact_match") for row in ref_rows]), "bertscore_precision": _none_avg([row.get("bertscore_precision") for row in ref_rows]), "bertscore_recall": _none_avg([row.get("bertscore_recall") for row in ref_rows]), "bertscore_f1": _none_avg([row.get("bertscore_f1") for row in ref_rows])},
        "latency_ms": _latency(rows),
        "output_length": {"answer_word_count_mean": _none_avg([row.get("answer_word_count") for row in rows]), "output_token_count_status": "UNAVAILABLE", "latency_ms_per_word_mean": _none_avg([row.get("latency_ms_per_word") for row in rows])},
        "generation_reliability": {"total_cases": len(rows), "successful_generations": sum(row.get("generation_success") is True for row in rows), "generation_errors": sum(row.get("generation_error") is True for row in rows), "empty_responses": sum(row.get("empty_response") is True for row in rows), "timeouts": None, "truncation": "UNKNOWN"},
        "retrieval_metrics": {"applicable_cases": len(retrieval_rows), "recall_at_3": _none_avg([row.get("retrieval_recall_at_3") for row in retrieval_rows]), "mrr": _none_avg([row.get("retrieval_mrr") for row in retrieval_rows]), "ndcg_at_3": None, "ndcg_reason": "no graded relevance labels"},
        "citation_diagnostics": {"citation_present_rate": _rate(sum(row.get("citation_present") is True for row in rows), len(rows)), "citation_identifier_valid_rate": _rate(sum(row.get("citation_identifier_valid") is True for row in citation_rows), len(citation_rows)), "citation_references_retrieved_source_rate": _rate(sum(row.get("citation_references_retrieved_source") is True for row in citation_rows), len(citation_rows))},
        "case_outcomes": {"pass": outcomes["PASS"], "partial": outcomes["PARTIAL"], "fail": outcomes["FAIL"], "pending_review": outcomes["PENDING_REVIEW"], "not_applicable": outcomes["NOT_APPLICABLE"], "pass_rate": _rate(outcomes["PASS"], len(rows) - outcomes["PENDING_REVIEW"])},
    }


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    block = _aggregate_block(rows)
    grounded_denominator = block["case_count"] - block["groundedness"]["unknown"]
    relevance_denominator = block["case_count"] - block["answer_relevance"]["unknown"]
    return {"case_count": block["case_count"], "expected_point_coverage_mean": block["expected_point_coverage"]["mean"], "grounded_supported_rate": _rate(block["groundedness"]["supported"], grounded_denominator), "unsupported_claim_rate": block["groundedness"]["unsupported_claim_rate"], "answer_relevant_rate": _rate(block["answer_relevance"]["relevant"], relevance_denominator), "case_pass_rate": block["case_outcomes"]["pass_rate"], "rag_only_abstention_heuristic_accuracy": block["rag_only_abstention_heuristic"]["accuracy"], "rag_only_escalation_heuristic_accuracy": block["rag_only_escalation_heuristic"]["accuracy"], "rag_only_restriction_heuristic_accuracy": block["rag_only_restriction_heuristic"]["accuracy"]}


def _summarize_corrected(rows: list[dict[str, Any]], *, dataset_path: Path, dataset_sha: str) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scopes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["ablation"])].append(row)
        categories[row["category"]].append(row)
        if row.get("scope") is not None:
            scopes[str(row["scope"])].append(row)
    return {
        "provenance": {"evaluation_version": EVALUATION_VERSION, "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha, "generation_reused": True, "generation_rerun": False, "reference_policy": "genuine_full_reference_only", "expected_points_policy": "separate_coverage_metric", "expected_point_evaluation_method": "NORMALIZED_LEXICAL"},
        "model_configurations": {f"{model_id}::{ablation}": _aggregate_block(group, model_id=model_id, ablation=ablation) for (model_id, ablation), group in sorted(groups.items())},
        "by_category": {category: _category_summary(group) for category, group in sorted(categories.items())},
        "by_scope": {scope: _category_summary(group) for scope, group in sorted(scopes.items())},
    }


COMPARISON_FIELDS = ["model_id", "case_count", "expected_point_coverage_mean", "expected_point_full_coverage_rate", "grounded_supported_rate", "partially_supported_rate", "unsupported_claim_rate", "answer_relevant_rate", "abstention_accuracy", "human_escalation_accuracy", "restriction_accuracy", "valid_reference_case_count", "token_f1", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "bleu", "bertscore_f1", "case_pass_rate", "pending_review_count", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_max_ms"]


def _write_comparison_csv(blocks: dict[str, dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for block in blocks.values():
            grounded_known = block["case_count"] - block["groundedness"]["unknown"]
            row = {
                "model_id": block["model_id"],
                "case_count": block["case_count"],
                "expected_point_coverage_mean": block["expected_point_coverage"]["mean"],
                "expected_point_full_coverage_rate": block["expected_point_coverage"]["full_coverage_rate"],
                "grounded_supported_rate": _rate(block["groundedness"]["supported"], grounded_known),
                "partially_supported_rate": _rate(block["groundedness"]["partially_supported"], grounded_known),
                "unsupported_claim_rate": block["groundedness"]["unsupported_claim_rate"],
                "answer_relevant_rate": block["answer_relevance"]["relevant_rate"],
                "abstention_accuracy": block["rag_only_abstention_heuristic"]["accuracy"],
                "human_escalation_accuracy": block["rag_only_escalation_heuristic"]["accuracy"],
                "restriction_accuracy": block["rag_only_restriction_heuristic"]["accuracy"],
                "valid_reference_case_count": block["reference_metrics"]["applicable_cases"],
                "case_pass_rate": block["case_outcomes"]["pass_rate"],
                "pending_review_count": block["case_outcomes"]["pending_review"],
                "latency_mean_ms": block["latency_ms"]["mean"],
                "latency_median_ms": block["latency_ms"]["median"],
                "latency_p95_ms": block["latency_ms"]["p95"],
                "latency_max_ms": block["latency_ms"]["max"],
            }
            for metric in ("token_f1", "rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "bleu", "bertscore_f1"):
                row[metric] = block["reference_metrics"][metric]
            writer.writerow(row)


def _write_review_queue(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["case_id", "model_id", "category", "scope", "question", "generated_answer", "expected_answer_points", "retrieved_document_ids", "retrieved_evidence", "review_reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in _select_review_rows(rows):
            reasons = []
            if row["groundedness_status"] == "UNKNOWN":
                reasons.append("groundedness_status=UNKNOWN")
            if row["answer_relevance_status"] == "UNKNOWN":
                reasons.append("answer_relevance_status=UNKNOWN")
            if any(item["status"] == "UNKNOWN" for item in row["expected_point_results"]):
                reasons.append("expected_point_status=UNKNOWN")
            if row["case_evaluation_status"] == "PENDING_REVIEW":
                reasons.append("case_evaluation_status=PENDING_REVIEW")
            if reasons:
                writer.writerow({"case_id": row["case_id"], "model_id": row["model_id"], "category": row["category"], "scope": row["scope"], "question": row["question"], "generated_answer": row["generated_answer"], "expected_answer_points": json.dumps(row["expected_answer_points"], ensure_ascii=False), "retrieved_document_ids": json.dumps(row["retrieved_document_ids"], ensure_ascii=False), "retrieved_evidence": json.dumps(row["retrieved_evidence"], ensure_ascii=False), "review_reason": "; ".join(reasons)})


def _select_review_rows(rows: list[dict[str, Any]], target: int = 50) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(candidates: list[dict[str, Any]], limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for row in candidates:
            key = (str(row["model_id"]), str(row["case_id"]))
            if key in seen:
                continue
            selected.append(row)
            seen.add(key)
            added += 1
            if len(selected) >= target or added >= limit:
                break

    model_ids = sorted({str(row["model_id"]) for row in rows})
    for model_id in model_ids:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        pub_rows = [row for row in model_rows if row["category"] == "PUB"]
        add(sorted(pub_rows, key=lambda row: (row.get("expected_point_coverage") is None, -(row.get("expected_point_coverage") or -1))), 2)
        add(sorted(pub_rows, key=lambda row: (row.get("expected_point_coverage") is None, row.get("expected_point_coverage") or 2)), 2)
        add([row for row in model_rows if row["category"] == "INSUF"], 2)
        add([row for row in model_rows if row["category"] == "HUM"], 2)
        add([row for row in model_rows if row["category"] == "ATK"], 2)
    add(sorted(rows, key=lambda row: float(row.get("latency_ms") or 0), reverse=True), target - len(selected))
    return selected[:target]
