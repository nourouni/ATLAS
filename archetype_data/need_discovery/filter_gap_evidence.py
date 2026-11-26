"""Deterministic Step 2 filtering for normalized need-gap evidence.

This module does not infer new needs. It only separates potentially useful
filtered gap evidence from operational/security/low-value evidence and adds
conservative normalization fields for a later step.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "need_discovery" / "gap_evidence.csv"
FILTERED_OUTPUT = ROOT / "artifacts" / "need_discovery" / "filtered_gap_evidence.csv"
EXCLUDED_OUTPUT = ROOT / "artifacts" / "need_discovery" / "excluded_gap_evidence.csv"
SUMMARY_OUTPUT = ROOT / "artifacts" / "need_discovery" / "filtering_summary.json"
MINIMUM_SEMANTIC_ROWS_THRESHOLD = 20

BASE_FIELDS = [
    "evidence_id",
    "source_type",
    "customer_id_or_case_id",
    "text_or_pattern",
    "normalized_text",
    "existing_need",
    "product",
    "original_gap_type",
    "normalized_gap_type",
    "gap_reason",
    "keep_for_need_discovery",
    "filter_reason",
    "timestamp",
    "semantic_ready",
    "normalized_text_frequency",
    "normalized_gap_type_frequency",
    "existing_need_frequency",
    "product_frequency",
]


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(text(item) for item in value if text(item))
    return str(value).strip()


def normalize_text(value: str) -> str:
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"^\s*(user|customer|query|question)\s*:\s*", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"([!?.,;:])\1+", r"\1", value)
    return value


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(term in lowered for term in terms)


def _is_generic_public_failure(row: dict[str, str]) -> bool:
    raw = normalize_text(row.get("text_or_pattern", ""))
    reason = normalize_text(row.get("gap_reason", ""))
    case_id = row.get("customer_id_or_case_id", "")
    combined = f"{raw} {reason}".lower()
    generic = (
        not raw
        or "public information request; retrieval did not provide usable public evidence" in combined
        or raw.lower() in {"public evidence unavailable", "public_evidence_unavailable"}
    )
    # The available methodology artifact explicitly describes PUB-034 and
    # PUB-042 as public-evidence dependency failures, not preserved questions.
    documented_dependency_case = case_id in {"PUB-034", "PUB-042"} and "methodology report" in combined
    return generic or documented_dependency_case


def _structured_recommendation_text(row: dict[str, str]) -> str:
    need = row.get("existing_need", "")
    product = row.get("product", "")
    original = row.get("original_gap_type", "") or row.get("gap_type", "")
    raw = normalize_text(row.get("text_or_pattern", ""))
    reason = normalize_text(row.get("gap_reason", ""))
    pieces: list[str] = []
    if need:
        pieces.append(f"Need: {need}.")
    if product:
        pieces.append(f"Product: {product}.")
    if original == "missing_requirement":
        requirement = raw or reason
        pieces.append(f"Gap: missing requirement {requirement}.")
    elif original:
        pieces.append(f"Gap: {original.replace('_', ' ')}.")
    if pieces:
        return " ".join(pieces)
    return raw


def _semantic_ready(normalized: str, row: dict[str, str]) -> bool:
    lowered = normalized.strip().lower()
    if not lowered or len(lowered) < 12:
        return False
    if lowered in {"unknown", "human_required", "public evidence unavailable", "public_evidence_unavailable"}:
        return False
    if row.get("source_type") == "chatbot":
        return bool(
            _contains_any(
                normalized,
                (
                    "account",
                    "bank",
                    "banking",
                    "card",
                    "cash",
                    "credit",
                    "debit",
                    "fee",
                    "loan",
                    "mortgage",
                    "payment",
                    "saving",
                    "transfer",
                    "transaction",
                    "product",
                    "service",
                    "atlas_demo",
                ),
            )
        )
    if row.get("source_type") == "recommendation":
        return bool(row.get("product") and (row.get("existing_need") or row.get("original_gap_type")))
    return False


def classify_row(row: dict[str, str]) -> tuple[str, bool, str, str, bool]:
    """Return normalized type, keep flag, reason, normalized text, semantic flag."""
    original = text(row.get("gap_type"))
    source = text(row.get("source_type")).lower()
    raw = normalize_text(text(row.get("text_or_pattern")))
    reason = normalize_text(text(row.get("gap_reason")))
    combined = f"{original} {raw} {reason}".lower()

    if original == "ownership_exclusion":
        normalized_type = "ownership_exclusion"
        keep = False
        filter_reason = "Ownership exclusion is normally correct recommender behavior and is retained only as low-value supporting evidence."
        normalized = _structured_recommendation_text(row)
    elif original == "missing_requirement":
        normalized_type = "conditional_access_gap"
        keep = True
        filter_reason = "Kept as an access/eligibility issue; a missing prerequisite is not itself an unmet need."
        normalized = _structured_recommendation_text(row)
    elif original == "insufficient_public_evidence":
        normalized = raw
        if source == "chatbot" and not _is_generic_public_failure(row):
            normalized_type = "information_gap"
            keep = True
            filter_reason = "Legitimate public-information evidence gap with preserved question text."
        else:
            normalized_type = "operational_failure"
            keep = False
            filter_reason = "Public-evidence dependency failure is documented, but the original question is unavailable."
    elif original == "unknown_intent":
        normalized = raw
        if _contains_any(combined, ("authentication", "unauthorized", "security", "prompt injection", "other customer")):
            normalized_type = "authentication_or_policy"
            keep = False
            filter_reason = "Authentication, authorization, or security-policy evidence is excluded from need discovery."
        elif _contains_any(combined, ("unsupported transaction", "unsupported action")):
            normalized_type = "unsupported_action"
            keep = False
            filter_reason = "Unsupported transactional capability alone is not need-gap evidence."
        elif _semantic_ready(normalized, row):
            normalized_type = "unresolved_banking_request"
            keep = True
            filter_reason = "Meaningful unresolved banking request retained for later grouping."
        else:
            normalized_type = "insufficient_semantic_content"
            keep = False
            filter_reason = "Unknown/ambiguous evidence does not preserve enough banking meaning."
    elif original == "human_required":
        normalized = raw
        if _semantic_ready(normalized, row) and not _contains_any(combined, ("fraud", "approval", "rejected", "freeze", "appeal")):
            normalized_type = "unresolved_banking_request"
            keep = True
            filter_reason = "Meaningful service-related human-required request retained."
        else:
            normalized_type = "operational_failure"
            keep = False
            filter_reason = "Human-required case is operational/decision-related or lacks semantic banking content."
    elif original == "eligibility_failed":
        normalized = _structured_recommendation_text(row)
        normalized_type = "eligibility_gap"
        keep = True
        filter_reason = "Candidate eligibility failure retained as structured access evidence."
    elif original == "no_product_for_need":
        normalized = _structured_recommendation_text(row)
        normalized_type = "missing_product_coverage"
        keep = True
        filter_reason = "Existing need has no recorded mapped product candidate."
    elif original == "weak_need_coverage":
        normalized = _structured_recommendation_text(row) or raw
        normalized_type = "weak_need_coverage"
        keep = True
        filter_reason = "Existing weak-coverage evidence retained."
    elif original in {"hard_conflict"}:
        normalized = _structured_recommendation_text(row) or raw
        normalized_type = "eligibility_gap"
        keep = True
        filter_reason = "Recorded hard conflict retained as candidate eligibility evidence."
    elif _contains_any(combined, ("authentication", "unauthorized", "security", "prompt injection", "other customer")):
        normalized = raw
        normalized_type = "authentication_or_policy"
        keep = False
        filter_reason = "Authentication, authorization, or security-policy evidence is excluded from need discovery."
    elif _contains_any(combined, ("unsupported transaction", "unsupported action")):
        normalized = raw
        normalized_type = "unsupported_action"
        keep = False
        filter_reason = "Unsupported transactional capability alone is not need-gap evidence."
    elif _contains_any(combined, ("malformed", "empty", "greeting", "chitchat")):
        normalized = raw
        normalized_type = "malformed_or_non_banking"
        keep = False
        filter_reason = "Malformed, empty, greeting, or non-banking content is excluded."
    elif _contains_any(combined, ("retrieval_failed", "network", "server", "internal exception", "temporary")):
        normalized = raw
        normalized_type = "operational_failure"
        keep = False
        filter_reason = "Technical/backend failure is excluded from need evidence."
    else:
        normalized = raw or _structured_recommendation_text(row)
        normalized_type = "insufficient_semantic_content"
        keep = False
        filter_reason = "Evidence does not fit a useful deterministic need-discovery category."

    semantic_ready = _semantic_ready(normalized, row)
    if not normalized:
        semantic_ready = False
    return normalized_type, keep, filter_reason, normalized, semantic_ready


def _output_fields(input_fields: list[str]) -> list[str]:
    extras = [field for field in input_fields if field not in BASE_FIELDS]
    return BASE_FIELDS + extras


def _read_input() -> tuple[list[dict[str, str]], list[str], str]:
    before = INPUT.read_bytes()
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields, before.hex()


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate(input_rows: list[dict[str, str]], kept: list[dict[str, Any]], excluded: list[dict[str, Any]], input_bytes_hex: str) -> None:
    if INPUT.read_bytes().hex() != input_bytes_hex:
        raise AssertionError("Input gap_evidence.csv was modified")
    input_ids = [text(row.get("evidence_id")) for row in input_rows]
    kept_ids = [text(row.get("evidence_id")) for row in kept]
    excluded_ids = [text(row.get("evidence_id")) for row in excluded]
    if len(input_ids) != len(set(input_ids)):
        raise AssertionError("Input evidence_id is not unique")
    if len(kept_ids) != len(set(kept_ids)) or len(excluded_ids) != len(set(excluded_ids)):
        raise AssertionError("Output evidence_id is not unique within an output")
    if set(kept_ids) & set(excluded_ids):
        raise AssertionError("An evidence_id appears in both outputs")
    if set(input_ids) != set(kept_ids) | set(excluded_ids):
        raise AssertionError("Input and output evidence IDs do not match exactly")
    for row in kept + excluded:
        if not row.get("normalized_gap_type"):
            raise AssertionError("normalized_gap_type is empty")
        if not row.get("filter_reason"):
            raise AssertionError("filter_reason is empty")
        if row.get("semantic_ready") not in {"true", "false"}:
            raise AssertionError("semantic_ready is not populated")


def run() -> dict[str, Any]:
    rows, input_fields, input_bytes_hex = _read_input()
    prepared: list[dict[str, Any]] = []
    for row in rows:
        normalized_type, keep, filter_reason, normalized, semantic_ready = classify_row(row)
        output = dict(row)
        output.update(
            {
                "text_or_pattern": text(row.get("text_or_pattern")),
                "normalized_text": normalized,
                "existing_need": text(row.get("existing_need")),
                "product": text(row.get("product")),
                "original_gap_type": text(row.get("gap_type")),
                "normalized_gap_type": normalized_type,
                "gap_reason": text(row.get("gap_reason")),
                "keep_for_need_discovery": "true" if keep else "false",
                "filter_reason": filter_reason,
                "timestamp": text(row.get("timestamp")),
                "semantic_ready": "true" if semantic_ready else "false",
            }
        )
        prepared.append(output)

    normalized_text_counts = Counter(row["normalized_text"] for row in prepared)
    normalized_type_counts = Counter(row["normalized_gap_type"] for row in prepared)
    need_counts = Counter(row["existing_need"] for row in prepared if row["existing_need"])
    product_counts = Counter(row["product"] for row in prepared if row["product"])
    for row in prepared:
        row["normalized_text_frequency"] = str(normalized_text_counts[row["normalized_text"]])
        row["normalized_gap_type_frequency"] = str(normalized_type_counts[row["normalized_gap_type"]])
        row["existing_need_frequency"] = str(need_counts.get(row["existing_need"], 0))
        row["product_frequency"] = str(product_counts.get(row["product"], 0))

    kept = [row for row in prepared if row["keep_for_need_discovery"] == "true"]
    excluded = [row for row in prepared if row["keep_for_need_discovery"] == "false"]
    fields = _output_fields(input_fields)
    FILTERED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write(FILTERED_OUTPUT, kept, fields)
    _write(EXCLUDED_OUTPUT, excluded, fields)
    _validate(rows, kept, excluded, input_bytes_hex)

    semantic_ready_kept = sum(row["semantic_ready"] == "true" for row in kept)
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "excluded_rows": len(excluded),
        "keep_rate": len(kept) / len(rows) if rows else 0.0,
        "semantic_ready_rows": sum(row["semantic_ready"] == "true" for row in prepared),
        "semantic_not_ready_rows": sum(row["semantic_ready"] == "false" for row in prepared),
        "semantic_ready_kept_rows": semantic_ready_kept,
        "semantic_grouping_ready": semantic_ready_kept >= MINIMUM_SEMANTIC_ROWS_THRESHOLD,
        "minimum_semantic_rows_threshold": MINIMUM_SEMANTIC_ROWS_THRESHOLD,
        "counts_by_original_gap_type": dict(sorted(Counter(row["original_gap_type"] for row in prepared).items())),
        "counts_by_normalized_gap_type": dict(sorted(normalized_type_counts.items())),
        "counts_by_filter_reason": dict(sorted(Counter(row["filter_reason"] for row in prepared).items())),
        "counts_by_source_type": dict(sorted(Counter(row["source_type"] for row in prepared).items())),
        "unique_normalized_texts": len({row["normalized_text"] for row in prepared if row["normalized_text"]}),
        "unique_existing_needs": len({row["existing_need"] for row in prepared if row["existing_need"]}),
        "unique_products": len({row["product"] for row in prepared if row["product"]}),
        "warning": (
            "Current evidence volume is insufficient for meaningful semantic clustering. "
            "Pipeline outputs are structural/preparatory only."
            if semantic_ready_kept < MINIMUM_SEMANTIC_ROWS_THRESHOLD
            else "Evidence volume passed the configured operational readiness threshold."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(f"input_rows={summary['input_rows']}")
    print(f"kept_rows={summary['kept_rows']}")
    print(f"excluded_rows={summary['excluded_rows']}")
    print(f"semantic_ready_kept_rows={summary['semantic_ready_kept_rows']}")
    print(f"semantic_grouping_ready={summary['semantic_grouping_ready']}")
    print(f"counts_by_normalized_gap_type={summary['counts_by_normalized_gap_type']}")
    print(f"filtered_output={FILTERED_OUTPUT}")
    print(f"excluded_output={EXCLUDED_OUTPUT}")
    print(f"summary_output={SUMMARY_OUTPUT}")
    print("validation=passed")


if __name__ == "__main__":
    main()
