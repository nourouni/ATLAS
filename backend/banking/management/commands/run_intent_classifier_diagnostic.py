from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdic
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from banking.evaluation.dataset import DEFAULT_DATASET_PATH, dataset_hash, load_datase
from banking.evaluation.llm_comparison import GenerationConfig, MODEL_REGISTRY, ModelSpec, _generate
from banking.evaluation.metrics import INTENTS
from banking.orchestration.nodes import IntentResult, _safe_json
from banking.orchestration.prompts import INTENT_PROMPT


DEFAULT_QLORA_RUN = Path("artifacts/llm_evaluation/llm_comparison_20260908T130108.jsonl")
DEFAULT_PHI_RUN = Path("artifacts/llm_evaluation/llm_comparison_20260908T163910.jsonl")
PRIORITY_INTENTS = ["CUSTOMER_CONTEXT", "RECOMMENDATION", "RECOMMENDATION_EXPLANATION", "HUMAN_REQUIRED", "UNKNOWN", "PUBLIC_INFO"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _parse_intent(text: str) -> tuple[str, float | None, str]:
    payload = _safe_json(text)
    parsed = IntentResult.model_validate(payload) if hasattr(IntentResult, "model_validate") else IntentResult.parse_obj(payload)
    return parsed.intent, parsed.confidence, parsed.reason


def _history_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows}


def _transition(old_correct: bool | None, new_correct: bool) -> str:
    if old_correct is True and new_correct:
        return "previously_correct_still_correct"
    if old_correct is True and not new_correct:
        return "previously_correct_now_wrong"
    if old_correct is False and new_correct:
        return "previously_wrong_now_correct"
    if old_correct is False and not new_correct:
        return "previously_wrong_still_wrong"
    return "not_previously_scorable"


def select_cases(cases: list[dict[str, Any]], phi_rows: list[dict[str, Any]], qlora_rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    case_map = {case["case_id"]: case for case in cases}
    histories = [_history_by_case(phi_rows), _history_by_case(qlora_rows)]
    selected: list[str] = []

    def add(case_id: str) -> None:
        if case_id in case_map and case_id not in selected:
            selected.append(case_id)

    for intent in PRIORITY_INTENTS:
        for history in histories:
            rows = [
                row for row in history.values()
                if row.get("intent_score_applicable") is True
                and row.get("expected_semantic_intent") == inten
                and row.get("predicted_intent") != row.get("expected_semantic_intent")
            ]
            for row in sorted(rows, key=lambda item: item["case_id"])[:5]:
                add(row["case_id"])

    for intent in PRIORITY_INTENTS:
        rows = [
            row for history in histories for row in history.values()
            if row.get("intent_score_applicable") is True
            and row.get("expected_semantic_intent") == inten
            and row.get("predicted_intent") == row.get("expected_semantic_intent")
        ]
        for row in sorted(rows, key=lambda item: item["case_id"])[:2]:
            add(row["case_id"])

    if len(selected) < size:
        for case in cases:
            if case.get("expected_semantic_intent") in PRIORITY_INTENTS:
                add(case["case_id"])
            if len(selected) >= size:
                break
    return [case_map[case_id] for case_id in selected[:size]]


def evaluate_model(model: ModelSpec, cases: list[dict[str, Any]], generation: GenerationConfig, history: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        raw = ""
        predicted = ""
        confidence = None
        reason = ""
        status = "ok"
        failure = ""
        try:
            raw = _generate(model, INTENT_PROMPT.format(message=case["question"]), generation)
            predicted, confidence, reason = _parse_intent(raw)
        except json.JSONDecodeError as exc:
            status = "malformed_json"
            failure = str(exc)
        except ValueError as exc:
            status = "malformed_json"
            failure = str(exc)
        except Exception as exc:
            status = "runtime_failure"
            failure = str(exc)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        expected = case.get("expected_semantic_intent") or case.get("expected_intent")
        old_row = history.get(case["case_id"], {})
        old_predicted = old_row.get("predicted_intent") or ""
        old_correct = None
        if old_row.get("intent_score_applicable") is True:
            old_correct = old_predicted == expected
        new_correct = status == "ok" and predicted == expected
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case.get("category", case["case_id"].split("-")[0]),
                "question": case["question"],
                "expected_intent": expected,
                "predicted_intent": predicted,
                "confidence": confidence,
                "reason": reason,
                "status": status,
                "failure": failure,
                "latency_ms": latency_ms,
                "correct": new_correct,
                "previous_predicted_intent": old_predicted,
                "previous_correct": old_correct,
                "transition": _transition(old_correct, new_correct),
                "raw_response": raw,
            }
        )
    return rows, summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "ok"]
    correct = sum(row["correct"] for row in valid)
    confusion = {expected: {predicted: 0 for predicted in INTENTS} for expected in INTENTS}
    for row in valid:
        expected = row["expected_intent"]
        predicted = row["predicted_intent"]
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1
    per_intent = {}
    for intent in INTENTS:
        tp = sum(row["expected_intent"] == intent and row["predicted_intent"] == intent for row in valid)
        fp = sum(row["expected_intent"] != intent and row["predicted_intent"] == intent for row in valid)
        fn = sum(row["expected_intent"] == intent and row["predicted_intent"] != intent for row in valid)
        per_intent[intent] = _prf(tp, fp, fn)
    return {
        "case_count": len(rows),
        "valid_predictions": len(valid),
        "accuracy": _rate(correct, len(valid)),
        "malformed_json_count": sum(row["status"] == "malformed_json" for row in rows),
        "runtime_failure_count": sum(row["status"] == "runtime_failure" for row in rows),
        "latency_ms": {
            "mean": sum(row["latency_ms"] for row in rows) / len(rows) if rows else None,
            "median": sorted(row["latency_ms"] for row in rows)[len(rows) // 2] if rows else None,
            "max": max((row["latency_ms"] for row in rows), default=None),
        },
        "per_intent": per_intent,
        "confusion_matrix": confusion,
        "transitions": dict(Counter(row["transition"] for row in rows)),
    }


class Command(BaseCommand):
    help = "Run a classifier-only targeted diagnostic on frozen evaluation cases."

    def add_arguments(self, parser):
        parser.add_argument("--models", default="phi-mini,mistral-banking-qlora")
        parser.add_argument("--size", type=int, default=30)
        parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
        parser.add_argument("--phi-run", type=Path, default=DEFAULT_PHI_RUN)
        parser.add_argument("--qlora-run", type=Path, default=DEFAULT_QLORA_RUN)
        parser.add_argument("--output-stem", default="intent_classifier_diagnostic_20260908")
        parser.add_argument("--ollama-host", default="")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=128)
        parser.add_argument("--top-k", type=int, default=3)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        model_ids = [item.strip() for item in options["models"].split(",") if item.strip()]
        unknown = sorted(set(model_ids) - set(MODEL_REGISTRY))
        if unknown:
            raise CommandError(f"Unknown model id(s): {', '.join(unknown)}")
        dataset_path: Path = options["dataset"]
        cases = load_dataset(dataset_path)
        phi_rows = _read_jsonl(options["phi_run"])
        qlora_rows = _read_jsonl(options["qlora_run"])
        selected = select_cases(cases, phi_rows, qlora_rows, options["size"])
        histories = {"phi-mini": _history_by_case(phi_rows), "mistral-banking-qlora": _history_by_case(qlora_rows)}
        generation = GenerationConfig(temperature=options["temperature"], max_tokens=options["max_tokens"], top_k=options["top_k"], seed=options["seed"])
        if options["ollama_host"]:
            settings.CHATBOT_OLLAMA_HOST = options["ollama_host"]
        output_dir = Path("artifacts/llm_evaluation")
        all_rows = []
        metrics = {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_hash(dataset_path),
            "targeted_subset": {
                "size": len(selected),
                "case_ids": [case["case_id"] for case in selected],
                "category_distribution": dict(Counter(case.get("category", case["case_id"].split("-")[0]) for case in selected)),
                "intent_distribution": dict(Counter(case.get("expected_semantic_intent") or case.get("expected_intent") for case in selected)),
            },
            "generation": {
                "temperature": generation.temperature,
                "max_tokens": generation.max_tokens,
                "top_k": generation.top_k,
                "seed": generation.seed,
            },
            "models": {},
        }
        for model_id in model_ids:
            model = MODEL_REGISTRY[model_id]
            if model_id == "phi-mini":
                model = ModelSpec(model.model_id, model.display_name, "phi4-mini:latest", notes=model.notes, backend=model.backend)
            rows, summary = evaluate_model(model, selected, generation, histories.get(model_id, {}))
            for row in rows:
                row["model_id"] = model_id
                row["model_backend"] = model.backend
                row["inference_model"] = model.ollama_model or model.base_model
            all_rows.extend(rows)
            metrics["models"][model_id] = summary
        stem = options["output_stem"]
        with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["model_id", "case_id", "category", "question", "expected_intent", "predicted_intent", "confidence", "reason", "status", "failure", "latency_ms", "correct", "previous_predicted_intent", "previous_correct", "transition", "inference_model", "raw_response"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.stdout.write(f"Rows: {len(all_rows)}")
        self.stdout.write(f"Metrics: {output_dir / f'{stem}_metrics.json'}")
