from __future__ import annotations

import csv
import json
from pathlib import Path

from django.core.management.base import BaseCommand

from banking.evaluation.dataset import DEFAULT_DATASET_PATH, dataset_hash, load_datase
from banking.evaluation.evidence_export import (
    build_evidence_records,
    compare_replay,
    file_hash,
    read_csv_rows,
    validate_evidence_records,
    write_csv_rows,
    write_jsonl,
)
from banking.evaluation.runner import evaluate_case, validate_one_result_per_case


ARTIFACT_DIR = Path("artifacts/chatbot_evaluation")
RUN_ID = "chatbot-eval-classifier-v2-evidence-replay"


class Command(BaseCommand):
    help = "Replay classifier-v2 evaluation with generation evidence capture and export review files."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
        parser.add_argument("--original-results", default=str(ARTIFACT_DIR / "chatbot_eval_results_classifier_v2.csv"))
        parser.add_argument("--review", default=str(ARTIFACT_DIR / "chatbot_eval_human_review_classifier_v2.csv"))
        parser.add_argument("--replay-results", default=str(ARTIFACT_DIR / "chatbot_eval_results_classifier_v2_evidence_replay.csv"))
        parser.add_argument("--jsonl", default=str(ARTIFACT_DIR / "chatbot_eval_classifier_v2_evidence_review.jsonl"))
        parser.add_argument("--csv", default=str(ARTIFACT_DIR / "chatbot_eval_classifier_v2_evidence_review.csv"))
        parser.add_argument("--differences", default=str(ARTIFACT_DIR / "chatbot_eval_classifier_v2_evidence_replay_differences.csv"))
        parser.add_argument("--manual-review", default=str(ARTIFACT_DIR / "chatbot_eval_classifier_v2_manual_review_ready.csv"))
        parser.add_argument("--hashes", default=str(ARTIFACT_DIR / "chatbot_eval_classifier_v2_evidence_hashes.json"))

    def handle(self, *args, **options):
        dataset_path = Path(options["dataset"])
        original_results_path = Path(options["original_results"])
        review_path = Path(options["review"])
        replay_results_path = Path(options["replay_results"])
        jsonl_path = Path(options["jsonl"])
        csv_path = Path(options["csv"])
        differences_path = Path(options["differences"])
        manual_review_path = Path(options["manual_review"])
        hashes_path = Path(options["hashes"])

        cases = load_dataset(dataset_path)
        original_rows = read_csv_rows(original_results_path)
        if replay_results_path.exists():
            replay_rows = self._read_replay_results(replay_results_path)
        else:
            replay_rows = [evaluate_case(case, capture_generation_context=True) for case in cases]
            self._write_replay_results(replay_results_path, replay_rows)
        validate_one_result_per_case(cases, replay_rows)
        records = build_evidence_records(cases, replay_rows, RUN_ID)
        validate_evidence_records(cases, records)
        write_jsonl(jsonl_path, records)
        self._write_evidence_csv(csv_path, records)

        differences = compare_replay(original_rows, replay_rows)
        difference_fields = [
            "case_id",
            "original_route",
            "replay_route",
            "original_final_answer",
            "replay_final_answer",
            "original_document_ids",
            "replay_document_ids",
            "difference_type",
            "replay_can_validly_support_review_of_original_response",
        ]
        write_csv_rows(differences_path, differences, difference_fields)
        self._write_manual_review_ready(manual_review_path, review_path, original_rows, records, differences)

        hashes = {
            "frozen_dataset_sha256": dataset_hash(dataset_path),
            "original_classifier_v2_results_sha256": file_hash(original_results_path),
            "evidence_review_jsonl_sha256": file_hash(jsonl_path),
            "evidence_replay_results_sha256": file_hash(replay_results_path),
            "run_identifier": RUN_ID,
            "replay_difference_count": len(differences),
        }
        hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")

        self.stdout.write(f"Evidence records: {len(records)}")
        self.stdout.write(f"Replay differences: {len(differences)}")
        self.stdout.write(f"Evidence JSONL SHA256: {hashes['evidence_review_jsonl_sha256']}")

    def _write_replay_results(self, path: Path, rows: list[dict]) -> None:
        fields = [
            "case_id",
            "category",
            "question",
            "authentication_scenario",
            "expected_semantic_intent",
            "classification_executed",
            "intent_score_applicable",
            "predicted_intent",
            "confidence",
            "expected_route",
            "actual_route",
            "expected_escalated",
            "actual_escalated",
            "policy_outcome",
            "failure_code",
            "allowed_sources",
            "forbidden_sources",
            "source_usage",
            "retrieved_document_ids",
            "final_answer",
            "latency_ms",
            "intent_pass",
            "policy_pass",
            "routing_pass",
            "escalation_pass",
            "source_isolation_pass",
            "unsupported_claim_indicator",
            "answer_points_pass",
            "end_to_end_pass",
            "operational_status",
            "failure_reason",
            "human_review_status",
            "generation_executed",
            "generation_contexts",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                output = row.copy()
                for key in ("source_usage", "retrieved_document_ids", "generation_contexts"):
                    output[key] = json.dumps(output.get(key), ensure_ascii=False, sort_keys=True)
                writer.writerow({field: output.get(field, "") for field in fields})

    def _read_replay_results(self, path: Path) -> list[dict]:
        rows = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed = dict(row)
                for key in (
                    "classification_executed",
                    "intent_score_applicable",
                    "expected_escalated",
                    "actual_escalated",
                    "policy_pass",
                    "routing_pass",
                    "escalation_pass",
                    "source_isolation_pass",
                    "unsupported_claim_indicator",
                    "answer_points_pass",
                    "end_to_end_pass",
                    "generation_executed",
                ):
                    if parsed.get(key) in {"True", "False"}:
                        parsed[key] = parsed[key] == "True"
                for key, default in (
                    ("source_usage", {}),
                    ("retrieved_document_ids", []),
                    ("generation_contexts", []),
                ):
                    parsed[key] = json.loads(parsed.get(key) or json.dumps(default))
                try:
                    parsed["latency_ms"] = float(parsed["latency_ms"]) if parsed.get("latency_ms") else None
                except ValueError:
                    parsed["latency_ms"] = None
                rows.append(parsed)
        return rows

    def _write_evidence_csv(self, path: Path, records: list[dict]) -> None:
        fields = list(records[0].keys())
        rows = []
        for record in records:
            output = record.copy()
            for key, value in output.items():
                if isinstance(value, (dict, list)):
                    output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            rows.append(output)
        write_csv_rows(path, rows, fields)

    def _write_manual_review_ready(
        self,
        path: Path,
        review_path: Path,
        original_rows: list[dict[str, str]],
        evidence_records: list[dict],
        differences: list[dict],
    ) -> None:
        review_rows = read_csv_rows(review_path)
        original = {row["case_id"]: row for row in original_rows}
        evidence = {row["case_id"]: row for row in evidence_records}
        difference_ids = {row["case_id"] for row in differences}
        fields = list(review_rows[0].keys()) + [
            "evidence_type",
            "retrieved_document_ids",
            "exact_retrieved_passages_supplied_to_generation",
            "source_titles",
            "source_urls",
            "sanitized_customer_context_supplied_to_generation",
            "sanitized_recommendation_context_supplied_to_generation",
            "evidence_origin",
            "evidence_availability_status",
            "evidence_notes",
            "priority_review",
            "priority_reason",
            "evidence_sufficient_for_review",
            "original_or_replay_evidence",
            "manual_confirmation_status",
        ]
        rows = []
        for review in review_rows:
            case_id = review["case_id"]
            record = evidence[case_id]
            original_row = original[case_id]
            reasons = []
            if original_row.get("final_answer") != record.get("final_answer") or original_row.get("actual_route") != record.get("actual_route"):
                reasons.append("changed_from_original_v2_or_replay_difference")
            if original_row.get("end_to_end_pass") == "False":
                reasons.append("end_to_end_failure")
            if case_id.startswith("ATK"):
                reasons.append("attack_case")
            if case_id.startswith("HUM"):
                reasons.append("human_required_case")
            if case_id.startswith("UNAUTH"):
                reasons.append("unauthenticated_private_case")
            if case_id.startswith("OTHER"):
                reasons.append("other_customer_case")
            if case_id == "PUB-037":
                reasons.append("PUB-037")
            if case_id in difference_ids:
                reasons.append("replay_differs_from_original")
            sufficient = record["evidence_availability_status"] == "available" and case_id not in difference_ids
            output = review.copy()
            for key in fields[len(review_rows[0].keys()) : len(review_rows[0].keys()) + 9]:
                output[key] = json.dumps(record.get(key), ensure_ascii=False, sort_keys=True) if isinstance(record.get(key), (dict, list)) else record.get(key, "")
            output["evidence_notes"] = record.get("notes", "")
            output["priority_review"] = bool(reasons)
            output["priority_reason"] = ",".join(reasons)
            output["evidence_sufficient_for_review"] = sufficien
            output["original_or_replay_evidence"] = record["evidence_origin"]
            output["manual_confirmation_status"] = "requires_manual_confirmation"
            rows.append(output)
        write_csv_rows(path, rows, fields)
