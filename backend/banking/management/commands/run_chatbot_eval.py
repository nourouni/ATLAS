from __future__ import annotations

import csv
import json
from pathlib import Path
from django.conf import settings
from django.utils import timezone

from django.core.management.base import BaseCommand

from banking.evaluation.dataset import ARTIFACT_DIR, DEFAULT_DATASET_PATH, DEFAULT_METADATA_PATH, DEFAULT_REVIEW_PATH, dataset_hash, freeze_dataset_metadata, load_datase
from banking.evaluation.metrics import calculate_metrics
from banking.evaluation.runner import evaluate_case, validate_one_result_per_case, write_human_review


class Command(BaseCommand):
    help = "Run deterministic chatbot evaluation and write results, metrics, and human-review CSV."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
        parser.add_argument("--results", default=str(ARTIFACT_DIR / "chatbot_eval_results.csv"))
        parser.add_argument("--metrics", default=str(ARTIFACT_DIR / "chatbot_eval_metrics.json"))
        parser.add_argument("--review", default=str(DEFAULT_REVIEW_PATH))
        parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--clean-baseline", action="store_true")

    def handle(self, *args, **options):
        dataset_path = Path(options["dataset"])
        result_path = Path(options["results"])
        metrics_path = Path(options["metrics"])
        review_path = Path(options["review"])
        metadata_path = Path(options["metadata"])
        metadata = freeze_dataset_metadata(dataset_path=dataset_path, metadata_path=metadata_path)
        frozen_hash = metadata["sha256"]
        cases = load_dataset(dataset_path)
        if options["limit"] is not None:
            cases = cases[: options["limit"]]

        fields = self._fields()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        completed = self._read_existing_results(result_path) if result_path.exists() and not options["clean_baseline"] else {}
        rows = [completed[case["case_id"]] for case in cases if case["case_id"] in completed]

        with result_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(self._serialize(row, fields))
            for case in cases:
                if case["case_id"] in completed:
                    continue
                row = evaluate_case(case)
                rows.append(row)
                writer.writerow(self._serialize(row, fields))
                handle.flush()

        validate_one_result_per_case(cases, rows)
        if dataset_hash(dataset_path) != frozen_hash:
            raise ValueError("Dataset hash changed during evaluation.")
        metrics = calculate_metrics(rows)
        metrics["dataset"] = metadata
        metrics["execution_configuration"] = {
            "execution_timestamp": timezone.now().isoformat(),
            "ollama_model": getattr(settings, "CHATBOT_OLLAMA_MODEL", ""),
            "intent_confidence_threshold": getattr(settings, "CHATBOT_INTENT_CONFIDENCE_THRESHOLD", ""),
            "rag_endpoint": getattr(settings, "RAG_SERVICE_URL", ""),
            "prompt_identifiers": ["INTENT_PROMPT", "generation_prompt"],
            "application_mode": "django-requestfactory-evaluation",
            "runner_version": "chatbot-eval-runner-v1",
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_human_review(rows, review_path)
        self.stdout.write(f"Rows: {len(rows)}")
        self.stdout.write(f"Intent accuracy: {metrics['intent_accuracy']['numerator']}/{metrics['intent_accuracy']['denominator']} = {metrics['intent_accuracy']['value']:.3f}")
        self.stdout.write(f"Routing accuracy: {metrics['routing_accuracy']['numerator']}/{metrics['routing_accuracy']['denominator']} = {metrics['routing_accuracy']['value']:.3f}")
        self.stdout.write(f"End-to-end pass rate: {metrics['end_to_end_pass_rate']['numerator']}/{metrics['end_to_end_pass_rate']['denominator']} = {metrics['end_to_end_pass_rate']['value']:.3f}")

    def _fields(self):
        return [
            "case_id", "category", "question", "authentication_scenario",
            "expected_semantic_intent", "classification_executed", "intent_score_applicable",
            "predicted_intent", "confidence", "expected_route", "actual_route",
            "expected_escalated", "actual_escalated", "policy_outcome", "failure_code",
            "allowed_sources", "forbidden_sources", "source_usage", "retrieved_document_ids",
            "final_answer", "latency_ms", "intent_pass", "policy_pass", "routing_pass",
            "escalation_pass", "source_isolation_pass", "unsupported_claim_indicator",
            "answer_points_pass", "end_to_end_pass", "operational_status",
            "failure_reason", "human_review_status",
        ]

    def _serialize(self, row, fields):
        serializable = row.copy()
        if isinstance(serializable.get("source_usage"), dict):
            serializable["source_usage"] = json.dumps(serializable["source_usage"], sort_keys=True)
        if isinstance(serializable.get("retrieved_document_ids"), list):
            serializable["retrieved_document_ids"] = json.dumps(serializable["retrieved_document_ids"])
        return {key: serializable.get(key, "") for key in fields}

    def _read_existing_results(self, path: Path):
        rows = {}
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_id = row.get("case_id")
                if not case_id or case_id in rows:
                    continue
                parsed = dict(row)
                for key in ("classification_executed", "intent_score_applicable", "expected_escalated", "actual_escalated", "policy_pass", "routing_pass", "escalation_pass", "source_isolation_pass", "unsupported_claim_indicator", "answer_points_pass", "end_to_end_pass"):
                    if parsed.get(key) in {"True", "False"}:
                        parsed[key] = parsed[key] == "True"
                try:
                    parsed["latency_ms"] = float(parsed["latency_ms"]) if parsed.get("latency_ms") else None
                    parsed["source_usage"] = json.loads(parsed.get("source_usage") or "{}")
                    parsed["retrieved_document_ids"] = json.loads(parsed.get("retrieved_document_ids") or "[]")
                except json.JSONDecodeError:
                    continue
                rows[case_id] = parsed
        return rows
