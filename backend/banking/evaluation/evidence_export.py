from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ACCOUNT_NUMBER_RE = re.compile(r"\b\d{12,}\b")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return defaul
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return defaul


def customer_aliases(rows: list[dict[str, Any]]) -> dict[str, str]:
    ids: list[str] = []
    for row in rows:
        auth = _json_loads(row.get("authentication_scenario"), {})
        for key in ("customer_id", "body_customer_id", "path_customer_id"):
            value = auth.get(key)
            if value is not None and str(value) not in ids:
                ids.append(str(value))
    return {value: f"EVAL_CUSTOMER_{index + 1}" for index, value in enumerate(sorted(ids))}


def sanitize_value(value: Any, aliases: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item, aliases) for key, item in value.items() if not SECRET_RE.search(str(key))}
    if isinstance(value, list):
        return [sanitize_value(item, aliases) for item in value]
    if isinstance(value, int):
        return aliases.get(str(value), value)
    if isinstance(value, str):
        sanitized = value
        for raw, alias in aliases.items():
            sanitized = re.sub(rf"\b{re.escape(raw)}\b", alias, sanitized)
        return ACCOUNT_NUMBER_RE.sub("REDACTED_ACCOUNT_NUMBER", sanitized)
    return value


def generation_context(row: dict[str, Any]) -> dict[str, Any]:
    contexts = row.get("generation_contexts") or []
    if not contexts:
        return {}
    return contexts[-1].get("evidence") or {}


def evidence_type(row: dict[str, Any], context: dict[str, Any]) -> str:
    if row.get("operational_status") == "operational_failure":
        return "operational_failure"
    if row.get("generation_executed") is not True:
        return "deterministic_policy" if row.get("actual_escalated") is True else "none"
    if "public_information" in context:
        return "rag"
    if "customer_context" in context:
        return "customer_context"
    if "recommendation_context" in context:
        return "recommendation_explanation" if row.get("actual_route") == "RECOMMENDATION_EXPLANATION" else "recommendation"
    return "none"


def public_evidence(context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    passages: list[dict[str, Any]] = []
    ids: list[str] = []
    titles: list[str] = []
    urls: list[str] = []
    for group_index, item in enumerate(context.get("public_information") or [], start=1):
        citations = item.get("citations") or []
        answer = item.get("answer")
        if answer:
            passages.append(
                {
                    "order": len(passages) + 1,
                    "document_id": citations[0].get("document_id") or citations[0].get("citation_id") if citations else "",
                    "rank": group_index,
                    "text": answer,
                    "source_title": citations[0].get("source_title", "") if citations else "",
                    "source_url": citations[0].get("source_url", "") if citations else "",
                    "retrieval_score": "",
                    "note": "RAG generated answer supplied to chatbot generation; raw retrieval chunks were not supplied by this orchestration layer.",
                }
            )
        for citation in citations:
            document_id = citation.get("document_id") or citation.get("citation_id")
            if document_id:
                ids.append(document_id)
            if citation.get("source_title"):
                titles.append(citation["source_title"])
            if citation.get("source_url"):
                urls.append(citation["source_url"])
    return passages, ids, titles, urls


def build_evidence_records(cases: list[dict[str, Any]], replay_rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    aliases = customer_aliases(replay_rows)
    by_case = {row["case_id"]: row for row in replay_rows}
    records: list[dict[str, Any]] = []
    for case in cases:
        row = by_case[case["case_id"]]
        context = generation_context(row)
        kind = evidence_type(row, context)
        passages, passage_ids, titles, urls = public_evidence(context)
        sanitized_customer = sanitize_value(context.get("customer_context") or {}, aliases)
        sanitized_recommendation = sanitize_value(context.get("recommendation_context") or {}, aliases)
        retrieved_ids = passage_ids or row.get("retrieved_document_ids") or []
        notes = []
        if kind == "rag" and not passages:
            notes.append("RAG generation executed but no public evidence text was captured.")
        if kind == "rag" and passages:
            notes.append("Evidence is from evidence-capture replay, not original classifier-v2 run.")
        if kind in {"customer_context", "recommendation", "recommendation_explanation"}:
            notes.append("Sanitized chatbot-safe context captured immediately before generation prompt formatting.")
        if kind == "deterministic_policy":
            notes.append("Blocked or escalated before generation; grounding against external evidence is not applicable.")
        if kind == "operational_failure":
            notes.append("Operational failure; missing evidence was not fabricated.")
        records.append(
            {
                "case_id": case["case_id"],
                "evaluation_run_identifier": run_id,
                "question": case["question"],
                "expected_semantic_intent": row.get("expected_semantic_intent"),
                "predicted_intent": row.get("predicted_intent"),
                "confidence": row.get("confidence"),
                "actual_route": row.get("actual_route"),
                "authentication_scenario": sanitize_value(_json_loads(row.get("authentication_scenario"), {}), aliases),
                "policy_outcome": row.get("policy_outcome"),
                "failure_code": row.get("failure_code"),
                "operational_status": row.get("operational_status"),
                "generation_executed": row.get("generation_executed") is True,
                "final_answer": sanitize_value(row.get("final_answer") or "", aliases),
                "expected_answer_points": case.get("expected_answer_points", []),
                "prohibited_claims": case.get("prohibited_claims", []),
                "evidence_type": kind,
                "retrieved_document_ids": retrieved_ids,
                "exact_retrieved_passages_supplied_to_generation": passages,
                "source_titles": titles,
                "source_urls": urls,
                "sanitized_customer_context_supplied_to_generation": sanitized_customer,
                "sanitized_recommendation_context_supplied_to_generation": sanitized_recommendation,
                "evidence_origin": "evidence-capture replay",
                "evidence_availability_status": (
                    "not_assessable"
                    if kind == "operational_failure"
                    else "available"
                    if contex
                    else "not_applicable"
                    if kind == "deterministic_policy"
                    else "not_assessable"
                ),
                "notes": " ".join(notes),
            }
        )
    return records


def compare_replay(original_rows: list[dict[str, str]], replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original = {row["case_id"]: row for row in original_rows}
    differences: list[dict[str, Any]] = []
    for replay in replay_rows:
        old = original[replay["case_id"]]
        old_ids = _json_loads(old.get("retrieved_document_ids"), [])
        new_ids = replay.get("retrieved_document_ids") or []
        diff_types = []
        if old.get("actual_route") != replay.get("actual_route"):
            diff_types.append("route")
        if old.get("final_answer") != replay.get("final_answer"):
            diff_types.append("final_answer")
        if old_ids != new_ids:
            diff_types.append("document_ids")
        if diff_types:
            differences.append(
                {
                    "case_id": replay["case_id"],
                    "original_route": old.get("actual_route"),
                    "replay_route": replay.get("actual_route"),
                    "original_final_answer": old.get("final_answer"),
                    "replay_final_answer": replay.get("final_answer"),
                    "original_document_ids": json.dumps(old_ids, ensure_ascii=False),
                    "replay_document_ids": json.dumps(new_ids, ensure_ascii=False),
                    "difference_type": ",".join(diff_types),
                    "replay_can_validly_support_review_of_original_response": "yes" if diff_types == ["document_ids"] else "no",
                }
            )
    return differences


def validate_evidence_records(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    expected = {case["case_id"] for case in cases}
    actual = [record["case_id"] for record in records]
    if len(actual) != len(set(actual)):
        raise ValueError("Evidence export contains duplicate case_id values.")
    if set(actual) != expected:
        raise ValueError(f"Evidence export case mismatch: missing={sorted(expected - set(actual))} extra={sorted(set(actual) - expected)}")
    for record in records:
        json.dumps(record, ensure_ascii=False)
        if _has_secret_like_key(record):
            raise ValueError(f"Potential secret-like field in evidence export for {record['case_id']}")
        if record["evidence_type"] == "rag" and record["generation_executed"] and not record["exact_retrieved_passages_supplied_to_generation"]:
            raise ValueError(f"Missing RAG evidence text for {record['case_id']}")
        if record["evidence_type"] == "customer_context" and not record["sanitized_customer_context_supplied_to_generation"]:
            raise ValueError(f"Missing customer context for {record['case_id']}")
        if record["evidence_type"] in {"recommendation", "recommendation_explanation"} and not record["sanitized_recommendation_context_supplied_to_generation"]:
            raise ValueError(f"Missing recommendation context for {record['case_id']}")
        if record["evidence_type"] == "deterministic_policy":
            if record["exact_retrieved_passages_supplied_to_generation"]:
                raise ValueError(f"Blocked case exported retrieved passages for {record['case_id']}")
            if record["sanitized_customer_context_supplied_to_generation"] or record["sanitized_recommendation_context_supplied_to_generation"]:
                raise ValueError(f"Blocked case exported protected context for {record['case_id']}")


def _has_secret_like_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_RE.search(str(key)):
                return True
            if _has_secret_like_key(item):
                return True
    if isinstance(value, list):
        return any(_has_secret_like_key(item) for item in value)
    return False
