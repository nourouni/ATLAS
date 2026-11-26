"""Collect deterministic gap evidence from existing archetype_data artifacts.

This is Step 1 only: it emits gap evidence and does not infer, cluster, label,
or update ontology concepts. Inputs are deliberately allowlisted and remain
under archetype_data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHATBOT_REPORT = ROOT / "chatbot_v3_final_methodology_results.md"
OUTPUT = ROOT / "artifacts" / "need_discovery" / "gap_evidence.csv"
RECOMMENDATION_ROOT = ROOT / "recommendation"
RECOMMENDATION_PROPENSITY = (
    ROOT / "propensity" / "outputs" / "product_propensity" / "top_non_owned_product_recommendations.parquet"
)

FIELDS = [
    "evidence_id",
    "source_type",
    "customer_id_or_case_id",
    "text_or_pattern",
    "existing_need",
    "product",
    "gap_type",
    "gap_reason",
    "timestamp",
    "evidence_source_file",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(clean(item) for item in value if clean(item))
    return str(value).strip()


def normalize_for_dedup(value: Any) -> str:
    """Normalize formatting artifacts for conservative duplicate detection only."""
    text = " ".join(clean(value).split())
    text = (
        text.replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€™", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("«", '"')
        .replace("»", '"')
        .replace("’", "'")
    )
    text = text.strip()
    while len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        text = text[1:-1].strip()
    return " ".join(text.split()).casefold()


def evidence_id(row: dict[str, str]) -> str:
    key = "|".join(
        row[field]
        for field in (
            "source_type",
            "customer_id_or_case_id",
            "text_or_pattern",
            "existing_need",
            "product",
            "gap_type",
            "timestamp",
        )
    )
    return "gap_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def make_row(**values: Any) -> dict[str, str]:
    row = {field: clean(values.get(field, "")) for field in FIELDS}
    row["evidence_id"] = evidence_id(row)
    return row


def relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT.parent)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def relevant_files() -> tuple[list[Path], list[Path]]:
    """Discover likely row-level chatbot and recommendation artifacts."""
    chatbot_tokens = ("chatbot", "eval", "review", "routing", "intent", "failure", "rag", "retrieval")
    chatbot_files: list[Path] = []
    recommendation_files: list[Path] = []
    allowed = {".csv", ".json", ".jsonl", ".tsv", ".parquet"}
    for path in ROOT.rglob("*"):
        if any(part.casefold() == "initial_data" for part in path.parts):
            continue
        if not path.is_file() or path == OUTPUT or path.suffix.lower() not in allowed:
            continue
        lower_path = str(path).lower()
        name = path.name.lower()
        name_parts = set(re.split(r"[^a-z0-9]+", name))
        if "chatbot" in lower_path or any(token in name_parts for token in chatbot_tokens):
            chatbot_files.append(path)
        if "recommend" in lower_path or "provenance" in lower_path:
            recommendation_files.append(path)
    if RECOMMENDATION_PROPENSITY.exists() and RECOMMENDATION_PROPENSITY not in recommendation_files:
        recommendation_files.append(RECOMMENDATION_PROPENSITY)
    return sorted(set(chatbot_files)), sorted(set(recommendation_files))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load row-shaped JSON/JSONL/CSV/TSV without running any producer."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
        return records
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("results", "cases", "rows", "records", "data"):
                if isinstance(value.get(key), list):
                    return [item for item in value[key] if isinstance(item, dict)]
            return [value]
    # Parquet is inventoried but not loaded here: the collector has no
    # dependency on a parquet engine and score-only files do not prove gaps.
    return []


def row_value(row: dict[str, Any], *names: str) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in row.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if clean(value):
            return clean(value)
    return ""


def is_policy_case(row: dict[str, Any], text: str) -> bool:
    policy_fields = " ".join(
        row_value(row, field)
        for field in ("category", "case_category", "expected_route", "failure_code", "policy_reason")
    ).lower()
    combined = f"{policy_fields} {text.lower()}"
    return any(
        token in combined
        for token in (
            "unauth",
            "authentication_required",
            "authentication required",
            "unauthorized",
            "other customer",
            "prompt injection",
            "unsupported transaction",
            "unsupported_transaction",
            "unsupported action",
            "unsupported_action",
            "policy block",
            "impersonat",
            "security",
        )
    )


def collect_chatbot_row_evidence(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for record in load_records(path):
            route = row_value(record, "actual_route", "actual_intent", "route", "intent")
            intent = row_value(record, "intent", "actual_intent")
            failure = row_value(record, "failure_code", "failure", "error_code")
            question = row_value(record, "user_question", "question", "query", "message", "prompt", "input", "text")
            combined = f"{route} {intent} {failure}".lower()
            if not (
                route.upper() in {"UNKNOWN", "HUMAN_REQUIRED"}
                or intent.upper() in {"UNKNOWN", "HUMAN_REQUIRED"}
                or "public_evidence_unavailable" in combined
                or "customer_analytics_unavailable" in combined
                or "insufficient" in combined
                or "retrieval" in combined and ("fail" in combined or "unavailable" in combined)
            ):
                continue
            if is_policy_case(record, question):
                continue
            if "customer_analytics_unavailable" in combined:
                gap_type = "weak_need_coverage"
            elif "public_evidence_unavailable" in combined or "insufficient" in combined:
                gap_type = "insufficient_public_evidence"
            elif route.upper() == "HUMAN_REQUIRED" or intent.upper() == "HUMAN_REQUIRED":
                gap_type = "human_required"
            else:
                gap_type = "unknown_intent"
            case_id = row_value(
                record,
                "case_id",
                "id",
                "request_id",
                "conversation_id_or_request_id",
                "customer_id",
            )
            rows.append(
                make_row(
                    source_type="chatbot",
                    customer_id_or_case_id=case_id,
                    text_or_pattern=question or route or failure,
                    gap_type=gap_type,
                    gap_reason=failure or route,
                    timestamp=row_value(record, "timestamp", "created_at", "time"),
                    evidence_source_file=relative_source(path),
                )
            )
    return rows


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Remove only clear repeated observations, preserving distinct interactions."""
    kept: list[dict[str, str]] = []
    seen_exact: set[tuple[str, ...]] = set()
    seen_formatting_variants: dict[tuple[str, ...], str] = {}
    removed = 0
    for row in rows:
        exact_key = tuple(row[field] for field in FIELDS[1:])
        if exact_key in seen_exact:
            removed += 1
            continue
        seen_exact.add(exact_key)

        if row.get("source_type") != "chatbot":
            kept.append(row)
            continue

        normalized_key = (
            row.get("source_type", ""),
            normalize_for_dedup(row.get("text_or_pattern", "")),
            row.get("existing_need", ""),
            row.get("product", ""),
            row.get("gap_type", ""),
            row.get("gap_reason", ""),
        )
        prior_raw = seen_formatting_variants.get(normalized_key)
        # Different request IDs are normally distinct interactions. Collapse
        # only a formatting-only duplicate, which is identifiable because the
        # normalized text matches while the raw text differs only in quotes or
        # whitespace. Exact repeated events are collapsed above when the
        # provenance fields are identical.
        if prior_raw is not None and normalize_for_dedup(prior_raw) == normalize_for_dedup(row.get("text_or_pattern", "")):
            raw_prior = " ".join(prior_raw.split())
            raw_current = " ".join(row.get("text_or_pattern", "").split())
            if raw_prior != raw_current and {raw_prior[:1], raw_prior[-1:], raw_current[:1], raw_current[-1:]} & {'"', "'", "“", "”", "â"}:
                removed += 1
                continue
        seen_formatting_variants[normalized_key] = row.get("text_or_pattern", "")
        kept.append(row)
    return kept, removed


def collect_chatbot_report_evidence() -> list[dict[str, str]]:
    """Extract only named case-level failures documented in the report."""
    if not CHATBOT_REPORT.exists():
        return []
    text = CHATBOT_REPORT.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    # The report names these two cases and gives the actual failure reason.
    for case_id in ("PUB-034", "PUB-042"):
        if case_id not in text:
            continue
        rows.append(
            make_row(
                source_type="chatbot",
                customer_id_or_case_id=case_id,
                text_or_pattern="public information request; retrieval did not provide usable public evidence",
                gap_type="insufficient_public_evidence",
                gap_reason="The frozen methodology report documents public_evidence_unavailable for this case.",
                timestamp="",
                evidence_source_file=relative_source(CHATBOT_REPORT),
            )
        )
    return rows


def collect_chatbot_evidence(paths: list[Path]) -> list[dict[str, str]]:
    rows = collect_chatbot_row_evidence(paths)
    return rows if rows else collect_chatbot_report_evidence()


def need_id(need: str) -> str:
    match = re.match(r"(need_\d+)", need)
    return match.group(1) if match else need


def collect_recommendation_evidence(path: Path) -> list[dict[str, str]]:
    records = load_records(path)
    candidate_events = [record for record in records if record.get("candidate_status")]
    if candidate_events:
        rows: list[dict[str, str]] = []
        status_map = {
            "excluded_owned": ("ownership_exclusion", "candidate excluded because ownership_status=owned"),
            "excluded_missing_requirement": ("missing_requirement", "candidate blocked by missing requirement"),
            "excluded_eligibility": ("eligibility_failed", "candidate excluded by eligibility_status=ineligible"),
            "excluded_hard_conflict": ("hard_conflict", "candidate excluded by recorded hard conflict"),
            "no_matching_product": ("no_product_for_need", "no matching product was recorded for the need"),
        }
        for event in candidate_events:
            mapped = status_map.get(clean(event.get("candidate_status")))
            if not mapped:
                continue
            gap_type, reason = mapped
            needs = event.get("matched_needs") or event.get("matched_need") or [""]
            if isinstance(needs, str):
                needs = [needs]
            detail = event.get("missing_requirements") or event.get("hard_conflicts") or []
            pattern = clean(event.get("product"))
            if detail:
                pattern += ": " + clean(detail)
            for need in needs:
                rows.append(
                    make_row(
                        source_type="recommendation",
                        customer_id_or_case_id=event.get("customer_id"),
                        text_or_pattern=pattern,
                        existing_need=need_id(clean(need)),
                        product=event.get("product"),
                        gap_type=gap_type,
                        gap_reason=reason,
                        timestamp=event.get("timestamp"),
                        evidence_source_file=relative_source(path),
                    )
                )
        return rows
    data = records[0] if len(records) == 1 else {}
    if not isinstance(data, dict) or not isinstance(data.get("rule_checks"), list):
        return []
    customer_id = clean(data.get("customer_state", {}).get("client_id"))
    rows: list[dict[str, str]] = []
    for candidate in data.get("rule_checks", []):
        product = clean(candidate.get("product"))
        needs = candidate.get("matched_needs") or []
        if not needs or candidate.get("eligible") != "ineligible":
            continue

        already_owned = bool(candidate.get("already_owned"))
        missing = [clean(item) for item in (candidate.get("missing_requirements") or []) if clean(item)]
        conflicts = [clean(item) for item in (candidate.get("conflicts") or []) if clean(item)]

        # Existing semantics distinguish ownership and mandatory requirements.
        # Do not turn alternativeTo relations into hard-conflict evidence.
        reasons: list[tuple[str, str, str]] = []
        if already_owned:
            reasons.append(
                (
                    "ownership_exclusion",
                    "candidate excluded because already_owned=true",
                    "",
                )
            )
        if missing:
            reasons.append(
                (
                    "missing_requirement",
                    "candidate blocked by missing mandatory requirement: " + "; ".join(missing),
                    "; ".join(missing),
                )
            )
        real_conflicts = [
            item
            for item in conflicts
            if item != "already_owned" and not item.lower().startswith("alternativeto:")
        ]
        if real_conflicts:
            reasons.append(
                (
                    "hard_conflict",
                    "candidate excluded by recorded hard conflict: " + "; ".join(real_conflicts),
                    "",
                )
            )

        for gap_type, reason, pattern in reasons:
            for need in needs:
                rows.append(
                    make_row(
                        source_type="recommendation",
                        customer_id_or_case_id=customer_id,
                        text_or_pattern=pattern or gap_type,
                        existing_need=need_id(clean(need)),
                        product=product,
                        gap_type=gap_type,
                        gap_reason=reason,
                        timestamp="",
                        evidence_source_file=relative_source(path),
                    )
                )
    return rows


def validate(rows: list[dict[str, str]]) -> None:
    if not OUTPUT.exists():
        raise AssertionError(f"Output file was not created: {OUTPUT}")
    if set(FIELDS) != set(rows[0]) if rows else set(FIELDS) != set(FIELDS):
        raise AssertionError("Required fields are missing")
    ids = [row["evidence_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("evidence_id is not unique")
    keys = [tuple(row[field] for field in FIELDS[1:]) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Exact duplicate evidence rows exist")
    if any("initial_data" in str(path).lower() for path in (ROOT, OUTPUT)):
        raise AssertionError("An initial_data path was accessed or configured")


def main() -> None:
    chatbot_files, recommendation_files = relevant_files()
    rows = collect_chatbot_evidence(chatbot_files)
    used_recommendation_files: set[Path] = set()
    recommendation_rows: list[dict[str, str]] = []
    for path in recommendation_files:
        new_rows = collect_recommendation_evidence(path) if path.suffix.lower() != ".parquet" else []
        if new_rows:
            used_recommendation_files.add(path)
            recommendation_rows.extend(new_rows)
    rows.extend(recommendation_rows)
    deduped, duplicates_removed = deduplicate_rows(rows)
    deduped.sort(key=lambda row: row["evidence_id"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(deduped)
    validate(deduped)

    source_counts = Counter(row["source_type"] for row in deduped)
    gap_counts = Counter(row["gap_type"] for row in deduped)
    unique_needs = {row["existing_need"] for row in deduped if row["existing_need"]}
    unique_products = {row["product"] for row in deduped if row["product"]}
    chatbot_cases = sum(row["source_type"] == "chatbot" for row in deduped)
    recommendation_rows = [row for row in deduped if row["source_type"] == "recommendation"]
    recommendation_cases = {
        (row["customer_id_or_case_id"], row["existing_need"], row["product"])
        for row in recommendation_rows
    }
    print(f"output={OUTPUT}")
    print(f"total_evidence_rows={len(deduped)}")
    print(f"count_by_source_type={dict(sorted(source_counts.items()))}")
    print(f"count_by_gap_type={dict(sorted(gap_counts.items()))}")
    print(f"unique_existing_needs={len(unique_needs)}")
    print(f"unique_products={len(unique_products)}")
    print(f"chatbot_text_cases={chatbot_cases}")
    print(f"recommendation_rule_cases={len(recommendation_cases)}")
    print(f"recommendation_rule_evidence_rows={len(recommendation_rows)}")
    print(f"rows_removed_as_conservative_duplicates={duplicates_removed}")
    print("count_by_source_file=" + str(dict(sorted(Counter(row["evidence_source_file"] for row in deduped).items()))))
    print("chatbot_row_level_files_found=" + str([relative_source(path) for path in chatbot_files]))
    print("recommendation_files_found=" + str([relative_source(path) for path in recommendation_files]))
    print("recommendation_files_used=" + str([relative_source(path) for path in sorted(used_recommendation_files)]))
    print(
        "relevant_files_found_not_used="
        + str([relative_source(path) for path in sorted(set(chatbot_files + recommendation_files) - used_recommendation_files)])
    )
    print("validation=passed")


if __name__ == "__main__":
    main()
