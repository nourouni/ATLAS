from __future__ import annotations

from collections import Counter, defaultdic
import math
import re
from statistics import mean, median
from typing import Any


INTENTS = ["PUBLIC_INFO", "CUSTOMER_CONTEXT", "RECOMMENDATION", "RECOMMENDATION_EXPLANATION", "HUMAN_REQUIRED", "UNKNOWN"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _rate(numerator: int, denominator: int) -> dict[str, float | int]:
    return {"value": numerator / denominator if denominator else 0.0, "numerator": numerator, "denominator": denominator}


def _plain_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _plain_rate(tp, tp + fp)
    recall = _plain_rate(tp, tp + fn)
    f1 = _plain_rate(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted_counts = Counter(predicted)
    expected_counts = Counter(expected)
    overlap = sum((predicted_counts & expected_counts).values())
    precision = _plain_rate(overlap, len(predicted))
    recall = _plain_rate(overlap, len(expected))
    f1 = _plain_rate(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(max(0, len(tokens) - n + 1)))


def rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    predicted = _ngrams(_tokens(prediction), n)
    expected = _ngrams(_tokens(reference), n)
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((predicted & expected).values())
    precision = _plain_rate(overlap, sum(predicted.values()))
    recall = _plain_rate(overlap, sum(expected.values()))
    f1 = _plain_rate(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            current.append(previous[idx - 1] + 1 if left_token == right_token else max(previous[idx], current[-1]))
        previous = curren
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = _lcs_length(predicted, expected)
    precision = _plain_rate(overlap, len(predicted))
    recall = _plain_rate(overlap, len(expected))
    f1 = _plain_rate(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(reference)
    if not predicted or not expected:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(predicted, n)
        ref_ngrams = _ngrams(expected, n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precisions.append((overlap + 1) / (sum(pred_ngrams.values()) + 1))
    brevity = 1.0 if len(predicted) > len(expected) else math.exp(1 - len(expected) / len(predicted))
    return brevity * math.exp(sum(math.log(score) for score in precisions) / max_n)


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(_tokens(prediction)) == " ".join(_tokens(reference)))


def calculate_answer_quality(prediction: str, reference: str) -> dict[str, Any]:
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_1": rouge_n(prediction, reference, 1),
        "rouge_2": rouge_n(prediction, reference, 2),
        "rouge_l": rouge_l(prediction, reference),
        "bleu": bleu(prediction, reference),
    }


def flatten_answer_quality(row: dict[str, Any]) -> dict[str, Any]:
    quality = row.get("answer_quality")
    if not isinstance(quality, dict):
        return row
    row["exact_match"] = quality.get("exact_match")
    row["token_f1"] = _nested_float(quality, ("token_f1", "f1"))
    row["rouge_1_f1"] = _nested_float(quality, ("rouge_1", "f1"))
    row["rouge_2_f1"] = _nested_float(quality, ("rouge_2", "f1"))
    row["rouge_l_f1"] = _nested_float(quality, ("rouge_l", "f1"))
    row["bleu"] = quality.get("bleu")
    bertscore = quality.get("bertscore")
    if isinstance(bertscore, dict):
        row["bertscore_precision"] = bertscore.get("precision")
        row["bertscore_recall"] = bertscore.get("recall")
        row["bertscore_f1"] = bertscore.get("f1")
    return row


def add_bertscore(
    rows: list[dict[str, Any]],
    *,
    lang: str = "en",
    model_type: str | None = None,
    idf: bool = False,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    try:
        import importlib.metadata as metadata

        from bert_score import score
    except Exception as exc:  # pragma: no cover - exercised only when optional dependency is absent.
        raise RuntimeError("Install the optional 'bert-score' package to compute BERTScore.") from exc

    pairs = [
        (idx, str(row.get("final_answer") or ""), str(row.get("reference_answer") or ""))
        for idx, row in enumerate(rows)
        if row.get("reference_answer")
    ]
    if not pairs:
        return {"applicable": False, "reason": "no reference answers available"}

    _, predictions, references = zip(*pairs)
    precision, recall, f1 = score(
        list(predictions),
        list(references),
        lang=lang,
        model_type=model_type,
        idf=idf,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )
    for pair_idx, (row_idx, _, _) in enumerate(pairs):
        quality = rows[row_idx].setdefault("answer_quality", {})
        quality["bertscore"] = {
            "precision": float(precision[pair_idx]),
            "recall": float(recall[pair_idx]),
            "f1": float(f1[pair_idx]),
        }
        flatten_answer_quality(rows[row_idx])

    return {
        "applicable": True,
        "case_count": len(pairs),
        "implementation": "bert-score",
        "package_version": metadata.version("bert-score"),
        "lang": lang,
        "model_type": model_type or "bert-score default for language",
        "idf": idf,
        "batch_size": batch_size,
        "device": device or "bert-score default",
    }


def _nested_float(item: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = item
    for key in path:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return float(value) if isinstance(value, (int, float)) else None


def aggregate_answer_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [row.get("answer_quality") for row in rows if isinstance(row.get("answer_quality"), dict)]
    if not metric_rows:
        return {"applicable": False, "reason": "no reference answers available"}

    def avg(path: tuple[str, ...]) -> float:
        values = []
        for item in metric_rows:
            value: Any = item
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            if isinstance(value, (int, float)):
                values.append(float(value))
        return mean(values) if values else 0.0

    def avg_optional(path: tuple[str, ...]) -> float | None:
        values = []
        for item in metric_rows:
            value: Any = item
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            if isinstance(value, (int, float)):
                values.append(float(value))
        return mean(values) if values else None

    return {
        "applicable": True,
        "case_count": len(metric_rows),
        "exact_match": avg(("exact_match",)),
        "token_f1": avg(("token_f1", "f1")),
        "rouge_1_f1": avg(("rouge_1", "f1")),
        "rouge_2_f1": avg(("rouge_2", "f1")),
        "rouge_l_f1": avg(("rouge_l", "f1")),
        "bleu": avg(("bleu",)),
        "bertscore_precision": avg_optional(("bertscore", "precision")),
        "bertscore_recall": avg_optional(("bertscore", "recall")),
        "bertscore_f1": avg_optional(("bertscore", "f1")),
    }


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    scorable = [row for row in rows if row.get("intent_score_applicable") is True]
    intent_correct = sum(row.get("expected_semantic_intent") == row.get("predicted_intent") for row in scorable)
    route_correct = sum(row.get("expected_route") == row.get("actual_route") for row in rows)
    e2e_pass = sum(bool(row.get("end_to_end_pass")) for row in rows)
    policy_pass = sum(bool(row.get("policy_pass")) for row in rows)
    isolation_fail = sum(not bool(row.get("source_isolation_pass")) for row in rows)
    unsupported_claims = sum(bool(row.get("unsupported_claim_indicator")) for row in rows)

    by_intent: dict[str, dict[str, float]] = {}
    for intent in INTENTS:
        tp = sum(row.get("expected_semantic_intent") == intent and row.get("predicted_intent") == intent for row in scorable)
        fp = sum(row.get("expected_semantic_intent") != intent and row.get("predicted_intent") == intent for row in scorable)
        fn = sum(row.get("expected_semantic_intent") == intent and row.get("predicted_intent") != intent for row in scorable)
        by_intent[intent] = _prf(tp, fp, fn)

    confusion = {expected: {predicted: 0 for predicted in INTENTS} for expected in INTENTS}
    for row in scorable:
        expected = row.get("expected_semantic_intent")
        predicted = row.get("predicted_intent")
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

    esc_tp = sum(row.get("expected_escalated") is True and row.get("actual_escalated") is True for row in rows)
    esc_fp = sum(row.get("expected_escalated") is False and row.get("actual_escalated") is True for row in rows)
    esc_fn = sum(row.get("expected_escalated") is True and row.get("actual_escalated") is False for row in rows)

    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") not in (None, "")]
    route_latencies: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("latency_ms") is not None:
            route_latencies[str(row.get("actual_route"))].append(row["latency_ms"])

    status_counts = Counter(str(row.get("operational_status") or "unknown") for row in rows)
    dependency_failures = Counter(str(row.get("failure_code")) for row in rows if row.get("operational_status") == "operational_failure")
    by_category = {}
    for category in sorted(set(str(row.get("category")) for row in rows)):
        cat_rows = [row for row in rows if str(row.get("category")) == category]
        by_category[category] = {
            "case_count": len(cat_rows),
            "completed": sum(row.get("operational_status") == "completed_behavioral_case" for row in cat_rows),
            "operational_failures": sum(row.get("operational_status") == "operational_failure" for row in cat_rows),
            "policy_blocked": sum(row.get("operational_status") == "policy_blocked" for row in cat_rows),
            "intent_scorable": sum(row.get("intent_score_applicable") is True for row in cat_rows),
            "routing_accuracy": _rate(sum(row.get("routing_pass") is True for row in cat_rows), len(cat_rows)),
            "end_to_end_pass_rate": _rate(sum(row.get("end_to_end_pass") is True for row in cat_rows), len(cat_rows)),
            "answer_quality": aggregate_answer_quality(cat_rows),
        }

    return {
        "total_dataset_cases": total,
        "cases_attempted": total,
        "completed_cases": status_counts["completed_behavioral_case"],
        "operational_failures": status_counts["operational_failure"],
        "skipped_cases": status_counts["skipped"],
        "policy_blocked_cases": status_counts["policy_blocked"],
        "cases_awaiting_human_review": sum(row.get("human_review_status") == "pending" for row in rows),
        "intent_scorable_cases": len(scorable),
        "intent_accuracy": _rate(intent_correct, len(scorable)),
        "routing_accuracy": _rate(route_correct, total),
        "per_intent": by_intent,
        "confusion_matrix": confusion,
        "escalation": _prf(esc_tp, esc_fp, esc_fn),
        "authentication_policy_pass_rate": _rate(policy_pass, total),
        "source_isolation_violation_rate": _rate(isolation_fail, total),
        "source_isolation_violation_count": isolation_fail,
        "unsupported_claim_indicator_rate": _rate(unsupported_claims, total),
        "end_to_end_pass_rate": _rate(e2e_pass, total),
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": _p95(latencies),
            "max": max(latencies) if latencies else 0.0,
            "by_route": {
                route: {
                    "count": len(values),
                    "mean": mean(values),
                    "median": median(values),
                    "p95": _p95(values),
                    "max": max(values),
                }
                for route, values in route_latencies.items()
            },
        },
        "answer_quality": aggregate_answer_quality(rows),
        "dependency_failure_counts": dict(dependency_failures),
        "by_category": by_category,
        "predicted_intent_counts": dict(Counter(str(row.get("predicted_intent")) for row in rows)),
        "operational_status_counts": dict(status_counts),
    }
