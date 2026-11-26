from __future__ import annotations

import csv
import copy
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser, User
from django.test import RequestFactory

from banking.chatbot_views import chatbot_graph
from banking.orchestration import nodes
from banking.orchestration.prompts import generation_prompt as real_generation_promp

from .dataset import DEFAULT_REVIEW_PATH, load_datase
from .metrics import calculate_metrics


OPERATIONAL_FAILURE_CODES = {
    "public_evidence_unavailable",
    "customer_data_unavailable",
    "recommendation_unavailable",
    "recommendation_explanation_unavailable",
    "generation_unavailable",
    "retrieval_failed",
}


def _user(customer_id: int | None):
    if customer_id is None:
        return AnonymousUser()
    user = User(username=f"eval-{customer_id}")
    user.customer_id = customer_id
    return user


def _document_ids(payload: dict[str, Any]) -> list[str]:
    return [
        item.get("document_id") or item.get("citation_id")
        for item in payload.get("citations", [])
        if item.get("document_id") or item.get("citation_id")
    ]


def _source_isolation(case: dict[str, Any], usage: dict[str, int]) -> bool:
    source_map = {"rag": "rag", "customer_context": "customer_context", "recommendations": "recommendations"}
    for source in case["forbidden_sources"]:
        key = source_map.get(source)
        if key and usage.get(key, 0) > 0:
            return False
    for source in case["allowed_sources"]:
        key = source_map.get(source)
        if key and usage.get(key, 0) == 0 and case["expected_route"] != "ESCALATE":
            return False
    return True


def _answer_points_pass(case: dict[str, Any], answer: str) -> bool:
    points = [point for point in case.get("expected_answer_points", []) if point]
    if not points:
        return True
    text = answer.lower()
    return any(str(point).lower() in text for point in points)


def _unsupported_claim(case: dict[str, Any], answer: str) -> bool:
    text = answer.lower()
    return any(str(claim).lower() in text for claim in case.get("prohibited_claims", []) if claim)


def evaluate_case(case: dict[str, Any], *, capture_generation_context: bool = False) -> dict[str, Any]:
    factory = RequestFactory()
    auth = case["authentication"]
    body = {"message": case["question"]}
    if auth.get("body_customer_id") is not None:
        body["customer_id"] = auth["body_customer_id"]
    request = factory.post("/api/chatbot/graph/", data=json.dumps(body), content_type="application/json", HTTP_X_REQUEST_ID=f"eval-{case['case_id']}")
    request.user = _user(auth.get("customer_id") if auth.get("authenticated") else None)

    usage = {"rag": 0, "customer_context": 0, "recommendations": 0, "ollama": 0}
    real_rag = nodes.rag_tool.query
    real_profile = nodes.services.get_customer_profile
    real_segment = nodes.services.get_customer_segmen
    real_products = nodes.services.get_customer_products
    real_recs = nodes.services.get_customer_recommendations
    real_llm = nodes._llm_generate
    captured_contexts: list[dict[str, Any]] = []

    def rag_wrap(*args, **kwargs):
        usage["rag"] += 1
        return real_rag(*args, **kwargs)

    def profile_wrap(*args, **kwargs):
        usage["customer_context"] += 1
        return real_profile(*args, **kwargs)

    def segment_wrap(*args, **kwargs):
        return real_segment(*args, **kwargs)

    def products_wrap(*args, **kwargs):
        return real_products(*args, **kwargs)

    def recs_wrap(*args, **kwargs):
        usage["recommendations"] += 1
        return real_recs(*args, **kwargs)

    def llm_wrap(*args, **kwargs):
        usage["ollama"] += 1
        return real_llm(*args, **kwargs)

    def generation_prompt_wrap(message: str, evidence: dict[str, Any]) -> str:
        captured_contexts.append({"message": message, "evidence": copy.deepcopy(evidence)})
        return real_generation_prompt(message, evidence)

    started = time.perf_counter()
    patches = [
        patch("banking.orchestration.nodes.rag_tool.query", side_effect=rag_wrap),
        patch("banking.orchestration.nodes.services.get_customer_profile", side_effect=profile_wrap),
        patch("banking.orchestration.nodes.services.get_customer_segment", side_effect=segment_wrap),
        patch("banking.orchestration.nodes.services.get_customer_products", side_effect=products_wrap),
        patch("banking.orchestration.nodes.services.get_customer_recommendations", side_effect=recs_wrap),
        patch("banking.orchestration.nodes._llm_generate", side_effect=llm_wrap),
    ]
    if capture_generation_context:
        patches.append(patch("banking.orchestration.nodes.generation_prompt", side_effect=generation_prompt_wrap))

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        if capture_generation_context:
            with patches[6]:
                response = chatbot_graph(request, customer_id=auth.get("path_customer_id"))
        else:
            response = chatbot_graph(request, customer_id=auth.get("path_customer_id"))
    latency_ms = (time.perf_counter() - started) * 1000
    payload = json.loads(response.content.decode("utf-8"))
    answer = payload.get("answer") or ""

    failure_code = payload.get("failure_code") or ""
    classification_executed = usage["ollama"] > 0
    operational_status = "completed_behavioral_case"
    policy_outcome = "ALLOW"
    if failure_code == "authentication_required":
        operational_status = "policy_blocked"
        policy_outcome = "UNAUTHENTICATED_BLOCK"
    elif failure_code == "unsupported_transaction":
        operational_status = "policy_blocked"
        policy_outcome = "UNSUPPORTED_TRANSACTION_BLOCK"
    elif failure_code == "human_required":
        operational_status = "policy_blocked"
        policy_outcome = "HUMAN_REQUIRED"
    elif failure_code == "unknown_intent":
        policy_outcome = "UNKNOWN_OR_LOW_CONFIDENCE"
    elif failure_code in OPERATIONAL_FAILURE_CODES:
        operational_status = "operational_failure"

    intent_score_applicable = classification_executed and operational_status != "operational_failure"
    expected_semantic_intent = case.get("expected_semantic_intent") or case.get("expected_intent")
    predicted_intent = payload.get("intent") if intent_score_applicable else ""
    source_isolation_pass = _source_isolation(case, usage)
    policy_pass = True
    if not auth.get("authenticated") and case["expected_route"] == "ESCALATE":
        policy_pass = usage["customer_context"] == 0 and usage["recommendations"] == 0
    unsupported_claim = _unsupported_claim(case, answer)
    routing_pass = payload.get("route") == case["expected_route"]
    escalation_pass = payload.get("escalated") == case["expected_escalated"]
    intent_pass = ""
    if intent_score_applicable:
        intent_pass = predicted_intent == expected_semantic_intent or (expected_semantic_intent == "UNKNOWN" and payload.get("route") == "ESCALATE")
    answer_points_pass = _answer_points_pass(case, answer) if not payload.get("escalated") else True
    intent_ok = True if intent_pass == "" else bool(intent_pass)
    end_to_end_pass = all([response.status_code == 200, intent_ok, routing_pass, escalation_pass, source_isolation_pass, policy_pass, not unsupported_claim, answer_points_pass, operational_status != "operational_failure"])

    row = {
        "case_id": case["case_id"],
        "category": case.get("category", case["case_id"].split("-")[0]),
        "question": case["question"],
        "authentication_scenario": json.dumps(case["authentication"], sort_keys=True),
        "expected_semantic_intent": expected_semantic_intent,
        "classification_executed": classification_executed,
        "intent_score_applicable": intent_score_applicable,
        "predicted_intent": predicted_intent,
        "confidence": payload.get("confidence") if intent_score_applicable else "",
        "expected_route": case["expected_route"],
        "actual_route": payload.get("route"),
        "expected_escalated": case["expected_escalated"],
        "actual_escalated": payload.get("escalated"),
        "policy_outcome": policy_outcome,
        "failure_code": failure_code,
        "allowed_sources": ",".join(case["allowed_sources"]),
        "forbidden_sources": ",".join(case["forbidden_sources"]),
        "source_usage": usage,
        "retrieved_document_ids": _document_ids(payload),
        "final_answer": answer,
        "latency_ms": round(latency_ms, 2),
        "intent_pass": intent_pass,
        "policy_pass": policy_pass,
        "routing_pass": routing_pass,
        "escalation_pass": escalation_pass,
        "source_isolation_pass": source_isolation_pass,
        "unsupported_claim_indicator": unsupported_claim,
        "answer_points_pass": answer_points_pass,
        "end_to_end_pass": end_to_end_pass,
        "operational_status": operational_status,
        "failure_reason": "" if end_to_end_pass else (failure_code or "automatic_check_failed"),
        "human_review_status": "pending",
    }
    if capture_generation_context:
        row["generation_contexts"] = captured_contexts
        row["generation_executed"] = bool(captured_contexts)
    return row


def write_human_review(rows: list[dict[str, Any]], path: Path = DEFAULT_REVIEW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for previous in csv.DictReader(handle):
                if previous.get("case_id"):
                    existing[previous["case_id"]] = previous
    fields = ["case_id", "question", "expected_semantic_intent", "actual_route", "final_answer", "grounding", "correctness", "completeness", "clarity", "safe_response_or_escalation", "reviewer_notes", "review_status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            previous = existing.get(row["case_id"], {})
            review_status = previous.get("review_status") or row.get("human_review_status") or "pending"
            output = {key: row.get(key, "") for key in fields}
            for key in ("grounding", "correctness", "completeness", "clarity", "safe_response_or_escalation"):
                output[key] = previous.get(key, "pending") if review_status != "complete" else previous.get(key, output.get(key, ""))
            output["reviewer_notes"] = previous.get("reviewer_notes", "")
            output["review_status"] = review_status
            writer.writerow(output)


def run_evaluation(dataset_path: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = load_dataset(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    rows = [evaluate_case(case) for case in cases]
    return rows, calculate_metrics(rows)


def validate_one_result_per_case(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    case_ids = [case["case_id"] for case in cases]
    result_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Dataset contains duplicate case_id values.")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Results contain duplicate case_id values.")
    missing = sorted(set(case_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(case_ids))
    if missing or extra:
        raise ValueError(f"Result coverage mismatch: missing={missing} extra={extra}")
