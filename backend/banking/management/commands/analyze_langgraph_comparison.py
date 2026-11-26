from __future__ import annotations

import csv
import json
from collections import Counter, defaultdic
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand

from banking.evaluation.metrics import INTENTS, calculate_metrics


REFERENCE_REASON = "expected_answer_points_are_not_full_reference"
REFERENCE_METRICS = (
    "token_f1",
    "rouge_1_f1",
    "rouge_2_f1",
    "rouge_l_f1",
    "bleu",
    "exact_match",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _route_for_intent(intent: str | None) -> str | None:
    return {
        "PUBLIC_INFO": "PUBLIC_INFO",
        "CUSTOMER_CONTEXT": "CUSTOMER_CONTEXT",
        "RECOMMENDATION": "RECOMMENDATION",
        "RECOMMENDATION_EXPLANATION": "RECOMMENDATION",
        "HUMAN_REQUIRED": "ESCALATE",
        "UNKNOWN": "ESCALATE",
    }.get(str(intent or ""))


def _parse_csvish(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _forbidden_source_used(row: dict[str, Any]) -> bool:
    usage = row.get("source_usage")
    if not isinstance(usage, dict):
        return False
    for source in _parse_csvish(row.get("forbidden_sources")):
        if usage.get(source):
            return True
    return False


def correct_reference_metrics(row: dict[str, Any]) -> dict[str, Any]:
    corrected = dict(row)
    legacy = corrected.get("legacy_answer_quality") or corrected.get("answer_quality")
    corrected["legacy_answer_quality"] = legacy
    corrected["legacy_reference_answer"] = corrected.get("legacy_reference_answer") or corrected.get("reference_answer")
    corrected["reference_metric_applicable"] = False
    corrected["reference_metric_reason"] = REFERENCE_REASON if corrected.get("legacy_reference_answer") else "reference_missing"
    corrected["answer_quality"] = None
    corrected["metric_applicability"] = REFERENCE_REASON
    for metric in REFERENCE_METRICS:
        corrected[metric] = None
    corrected["bertscore_status"] = "NOT_APPLICABLE"
    return corrected


def intent_diagnostics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scorable = [row for row in rows if row.get("intent_score_applicable") is True]
    pair_counts = Counter((row.get("expected_semantic_intent") or "", row.get("predicted_intent") or "") for row in scorable)
    pair_rows = [
        {"expected_intent": expected, "predicted_intent": predicted, "count": count}
        for (expected, predicted), count in sorted(pair_counts.items())
    ]
    stats = []
    for intent in INTENTS:
        tp = sum(row.get("expected_semantic_intent") == intent and row.get("predicted_intent") == intent for row in scorable)
        fp = sum(row.get("expected_semantic_intent") != intent and row.get("predicted_intent") == intent for row in scorable)
        fn = sum(row.get("expected_semantic_intent") == intent and row.get("predicted_intent") != intent for row in scorable)
        destinations = Counter(row.get("predicted_intent") or "" for row in scorable if row.get("expected_semantic_intent") == intent and row.get("predicted_intent") != intent)
        values = _prf(tp, fp, fn)
        stats.append(
            {
                "intent": intent,
                "expected_count": tp + fn,
                "predicted_count": tp + fp,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "most_common_misclassification_destinations": "; ".join(f"{name}:{count}" for name, count in destinations.most_common()),
            }
        )
    return stats, pair_rows


def source_isolation_decomposition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("source_isolation_pass") is not False:
            continue
        expected = row.get("expected_semantic_intent")
        predicted = row.get("predicted_intent")
        actual_route = row.get("actual_route")
        expected_route = row.get("expected_route")
        predicted_route = _route_for_intent(predicted)
        reason = "UNKNOWN"
        if predicted != expected and predicted_route and actual_route == predicted_route and actual_route != expected_route:
            reason = "CLASSIFIER_CAUSED_WRONG_SOURCE"
        elif predicted == expected and actual_route != expected_route:
            reason = "ROUTER_CAUSED_WRONG_SOURCE"
        elif predicted == expected and actual_route == expected_route and _forbidden_source_used(row):
            reason = "GENERATOR_SOURCE_BOUNDARY_VIOLATION"
        elif not _forbidden_source_used(row):
            reason = "EVALUATION_ARTIFACT_OR_FALSE_POSITIVE"
        counts[reason] += 1
        out.append(
            {
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "expected_intent": expected,
                "predicted_intent": predicted,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "source_usage": json.dumps(row.get("source_usage"), ensure_ascii=False),
                "classification": reason,
            }
        )
    total = len(rows)
    summary = {
        "total_cases": total,
        "violation_count": len(out),
        "violation_rate": _rate(len(out), total),
        "classes": {
            name: {"count": count, "rate_of_all_cases": _rate(count, total), "rate_of_violations": _rate(count, len(out))}
            for name, count in sorted(counts.items())
        },
    }
    return out, summary


class Command(BaseCommand):
    help = "Retrospectively correct LangGraph pseudo-reference metrics and write intent/source diagnostics."

    def add_arguments(self, parser):
        parser.add_argument("jsonl", type=Path)
        parser.add_argument("--output-stem", default="")

    def handle(self, *args, **options):
        source_path: Path = options["jsonl"]
        output_stem = options["output_stem"] or f"{source_path.stem}_corrected_langgraph"
        output_dir = source_path.paren
        rows = [correct_reference_metrics(row) for row in _read_jsonl(source_path)]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[f"{row.get('model_id')}::{row.get('ablation')}"].append(row)
        metrics = {}
        for key, group_rows in sorted(grouped.items()):
            block = calculate_metrics(group_rows)
            block["answer_quality"] = {"applicable": False, "reason": REFERENCE_REASON}
            for category in block.get("by_category", {}).values():
                category["answer_quality"] = {"applicable": False, "reason": REFERENCE_REASON}
            intent_rows, pair_rows = intent_diagnostics(group_rows)
            violation_rows, violation_summary = source_isolation_decomposition(group_rows)
            block["intent_diagnostics"] = intent_rows
            block["intent_confusion_pairs"] = pair_rows
            block["source_isolation_decomposition"] = violation_summary
            metrics[key] = block
        _write_json(output_dir / f"{output_stem}_metrics.json", metrics)
        _write_csv(output_dir / f"{output_stem}.csv", rows, sorted({key for row in rows for key in row}))
        all_intents = [item for group in metrics.values() for item in group["intent_diagnostics"]]
        all_pairs = [item for group in metrics.values() for item in group["intent_confusion_pairs"]]
        all_violations = []
        for group_rows in grouped.values():
            violations, _ = source_isolation_decomposition(group_rows)
            all_violations.extend(violations)
        _write_csv(output_dir / f"{output_stem}_intent_diagnostics.csv", all_intents, ["intent", "expected_count", "predicted_count", "true_positive", "false_positive", "false_negative", "precision", "recall", "f1", "most_common_misclassification_destinations"])
        _write_csv(output_dir / f"{output_stem}_intent_confusion_pairs.csv", all_pairs, ["expected_intent", "predicted_intent", "count"])
        _write_csv(output_dir / f"{output_stem}_source_isolation_decomposition.csv", all_violations, ["case_id", "category", "expected_intent", "predicted_intent", "expected_route", "actual_route", "source_usage", "classification"])
        self.stdout.write(f"Rows: {len(rows)}")
        self.stdout.write(f"Metrics: {output_dir / f'{output_stem}_metrics.json'}")
