"""Step 5: analyst validation and final reporting for candidate needs.

Analyst decisions are explicit, auditable, and never update the ontology.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "need_discovery"
CANDIDATES_INPUT = ARTIFACT_ROOT / "candidate_needs.csv"
CANDIDATE_EVIDENCE_INPUT = ARTIFACT_ROOT / "candidate_need_evidence.csv"
FILTERING_SUMMARY_INPUT = ARTIFACT_ROOT / "filtering_summary.json"
GROUP_SUMMARY_INPUT = ARTIFACT_ROOT / "semantic_group_summary.json"
LABELING_SUMMARY_INPUT = ARTIFACT_ROOT / "candidate_need_labeling_summary.json"
FILTERED_EVIDENCE_INPUT = ARTIFACT_ROOT / "filtered_gap_evidence.csv"
GROUPS_INPUT = ARTIFACT_ROOT / "semantic_evidence_groups.csv"

REVIEWS_OUTPUT = ARTIFACT_ROOT / "candidate_need_reviews.csv"
VALIDATED_OUTPUT = ARTIFACT_ROOT / "validated_need_candidates.csv"
REPORT_OUTPUT = ARTIFACT_ROOT / "new_need_report.md"
SUMMARY_OUTPUT = ARTIFACT_ROOT / "need_discovery_summary.json"
HISTORY_OUTPUT = ARTIFACT_ROOT / "candidate_need_review_history.jsonl"
TAXONOMY_INPUT = ROOT / "recommendation" / "ontology_inspection_outputs" / "product_need_examples.csv"

DECISIONS = {"accept", "reject", "merge_with_existing", "keep_under_observation", "insufficient_evidence"}
REVIEW_FIELDS = [
    "candidate_id", "semantic_group_id", "proposed_need_label", "relationship_to_existing_need",
    "closest_existing_need_id", "closest_existing_need_label", "analyst_decision", "analyst_comment",
    "reviewed_by", "reviewed_at", "review_status", "review_version",
]
VALIDATED_FIELDS = [
    "candidate_id", "semantic_group_id", "proposed_need_label", "llm_relationship",
    "closest_existing_need_id", "closest_existing_need_label", "analyst_decision", "analyst_comment",
    "reviewed_by", "reviewed_at", "review_status", "group_size", "unique_semantic_evidence_count",
    "support_status", "source_type_count", "unique_case_or_customer_count", "product_count",
    "existing_need_count", "evidence_ids", "ontology_update_status", "recommended_next_action",
]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def existing_need_ids(path: Path = TAXONOMY_INPUT) -> set[str]:
    return {
        _text(row.get("need_id") or row.get("need"))
        for row in _read_csv(path)
        if _text(row.get("need_id") or row.get("need"))
    }


def _candidate_index() -> dict[str, dict[str, str]]:
    return {_text(row.get("candidate_id")): row for row in _read_csv(CANDIDATES_INPUT) if _text(row.get("candidate_id"))}


def synchronize_reviews(candidates: dict[str, dict[str, str]], reviews: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {_text(row.get("candidate_id")): dict(row) for row in reviews if _text(row.get("candidate_id")) in candidates}
    for candidate_id in sorted(candidates):
        if candidate_id not in by_id:
            candidate = candidates[candidate_id]
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "semantic_group_id": _text(candidate.get("semantic_group_id")),
                "proposed_need_label": _text(candidate.get("proposed_need_label")),
                "relationship_to_existing_need": _text(candidate.get("relationship_to_existing_need")),
                "closest_existing_need_id": _text(candidate.get("closest_existing_need_id")),
                "closest_existing_need_label": _text(candidate.get("closest_existing_need_label")),
                "analyst_decision": "",
                "analyst_comment": "",
                "reviewed_by": "",
                "reviewed_at": "",
                "review_status": "pending",
                "review_version": "0",
            }
    return [by_id[candidate_id] for candidate_id in sorted(by_id)]


def validate_review(
    candidate_id: str,
    decision: str,
    comment: str,
    candidate_ids: set[str],
    taxonomy_ids: set[str],
    merge_need_id: str = "",
) -> str:
    if candidate_id not in candidate_ids:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
    if decision not in DECISIONS:
        raise ValueError(f"Invalid analyst decision: {decision}")
    if decision in {"accept", "reject", "merge_with_existing"} and not _text(comment):
        raise ValueError(f"A comment is required for decision: {decision}")
    if decision == "merge_with_existing":
        if merge_need_id and merge_need_id not in taxonomy_ids:
            raise ValueError(f"Invalid existing need ID for merge: {merge_need_id}")
        if not merge_need_id:
            raise ValueError("merge_with_existing requires a valid existing need ID")
    return merge_need_id


def apply_review(
    reviews: list[dict[str, str]],
    *,
    candidate_id: str,
    decision: str,
    comment: str,
    reviewed_by: str,
    reviewed_at: str,
    taxonomy_ids: set[str],
    merge_need_id: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidate_ids = {_text(row.get("candidate_id")) for row in reviews}
    chosen_merge_id = validate_review(candidate_id, decision, comment, candidate_ids, taxonomy_ids, merge_need_id)
    updated: list[dict[str, str]] = []
    audit_event: dict[str, Any] | None = None
    for row in reviews:
        current = dict(row)
        if current.get("candidate_id") == candidate_id:
            previous = dict(current)
            if chosen_merge_id:
                current["closest_existing_need_id"] = chosen_merge_id
            current.update(
                {
                    "analyst_decision": decision,
                    "analyst_comment": _text(comment),
                    "reviewed_by": _text(reviewed_by) or "analyst",
                    "reviewed_at": reviewed_at,
                    "review_status": "reviewed",
                    "review_version": str(int(current.get("review_version") or 0) + 1),
                }
            )
            audit_event = {"event_type": "review_update", "timestamp": reviewed_at, "candidate_id": candidate_id, "previous": previous, "new": current}
        updated.append(current)
    if audit_event is None:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
    return updated, audit_even


def _candidate_evidence() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for row in _read_csv(CANDIDATE_EVIDENCE_INPUT):
        candidate_id = _text(row.get("candidate_id"))
        evidence_id = _text(row.get("evidence_id"))
        if candidate_id and evidence_id:
            mapping.setdefault(candidate_id, []).append(evidence_id)
    return {key: sorted(set(value)) for key, value in mapping.items()}


def _validated_rows(candidates: dict[str, dict[str, str]], reviews: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence = _candidate_evidence()
    rows: list[dict[str, Any]] = []
    for review in reviews:
        if review.get("review_status") != "reviewed":
            continue
        candidate = candidates.get(review.get("candidate_id", ""), {})
        rows.append(
            {
                "candidate_id": review.get("candidate_id", ""),
                "semantic_group_id": review.get("semantic_group_id", ""),
                "proposed_need_label": review.get("proposed_need_label", ""),
                "llm_relationship": review.get("relationship_to_existing_need", ""),
                "closest_existing_need_id": review.get("closest_existing_need_id", ""),
                "closest_existing_need_label": review.get("closest_existing_need_label", ""),
                "analyst_decision": review.get("analyst_decision", ""),
                "analyst_comment": review.get("analyst_comment", ""),
                "reviewed_by": review.get("reviewed_by", ""),
                "reviewed_at": review.get("reviewed_at", ""),
                "review_status": review.get("review_status", ""),
                "group_size": candidate.get("group_size", ""),
                "unique_semantic_evidence_count": candidate.get("unique_semantic_evidence_count", ""),
                "support_status": candidate.get("support_status", ""),
                "source_type_count": candidate.get("source_type_count", ""),
                "unique_case_or_customer_count": candidate.get("unique_case_or_customer_count", ""),
                "product_count": candidate.get("product_count", ""),
                "existing_need_count": candidate.get("existing_need_count", ""),
                "evidence_ids": ";".join(evidence.get(review.get("candidate_id", ""), [])),
                "ontology_update_status": "not_applied",
                "recommended_next_action": "manual_ontology_review" if review.get("analyst_decision") == "accept" else "retain_for_pipeline_audit",
            }
        )
    return rows


def _pipeline_values() -> dict[str, Any]:
    filtering = json.loads(FILTERING_SUMMARY_INPUT.read_text(encoding="utf-8")) if FILTERING_SUMMARY_INPUT.exists() else {}
    grouping = json.loads(GROUP_SUMMARY_INPUT.read_text(encoding="utf-8")) if GROUP_SUMMARY_INPUT.exists() else {}
    labeling = json.loads(LABELING_SUMMARY_INPUT.read_text(encoding="utf-8")) if LABELING_SUMMARY_INPUT.exists() else {}
    return {
        "raw_evidence_rows": filtering.get("input_rows", 0),
        "filtered_evidence_rows": filtering.get("kept_rows", 0),
        "semantic_ready_rows": filtering.get("semantic_ready_kept_rows", 0),
        "semantic_grouping_ready": bool(grouping.get("semantic_grouping_ready", False)),
        "semantic_grouping_executed": bool(grouping.get("grouping_executed", False)),
        "candidate_labeling_executed": bool(labeling.get("labeling_executed", False)),
        "candidate_needs_generated": labeling.get("candidate_needs_generated", 0),
    }


def build_summary(candidates: dict[str, dict[str, str]], reviews: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(
        row.get("analyst_decision", "")
        for row in reviews
        if row.get("review_status") == "reviewed" and row.get("analyst_decision")
    )
    values = _pipeline_values()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **values,
        "pending_review": sum(row.get("review_status") == "pending" for row in reviews),
        "reviewed_candidates": sum(row.get("review_status") == "reviewed" for row in reviews),
        "accepted": counts.get("accept", 0),
        "rejected": counts.get("reject", 0),
        "merged_with_existing": counts.get("merge_with_existing", 0),
        "keep_under_observation": counts.get("keep_under_observation", 0),
        "insufficient_evidence": counts.get("insufficient_evidence", 0),
        "ontology_updates_applied": 0,
        "step4_artifact_sha256": _hash(LABELING_SUMMARY_INPUT),
        "candidate_count": len(candidates),
        "review_artifact_sha256": _hash(REVIEWS_OUTPUT),
        "pipeline_version": "step5_analyst_validation_v1",
    }


def build_report(candidates: dict[str, dict[str, str]], reviews: list[dict[str, str]], summary: dict[str, Any]) -> str:
    lines = [
        "# New Need Discovery Report",
        "",
        "## Pipeline status",
        "",
        f"- Evidence collection: {summary['raw_evidence_rows']} raw rows.",
        f"- Deterministic filtering: {summary['filtered_evidence_rows']} rows retained for discovery.",
        f"- Semantic grouping: {'executed' if summary['semantic_grouping_executed'] else 'not executed'}.",
        f"- Candidate labeling: {'executed' if summary['candidate_labeling_executed'] else 'not executed'}.",
        f"- Analyst validation: {summary['reviewed_candidates']} reviewed, {summary['pending_review']} pending.",
        "",
        "No candidate need is automatically approved or added to the ontology.",
        "",
        "## Review categories",
        "",
        f"- Accepted candidate new needs: {summary['accepted']}",
        f"- Merged with existing needs: {summary['merged_with_existing']}",
        f"- Rejected candidates: {summary['rejected']}",
        f"- Under observation: {summary['keep_under_observation']}",
        f"- Insufficient evidence: {summary['insufficient_evidence']}",
        "",
    ]
    if not candidates:
        lines.extend(
            [
                "## Current state",
                "",
                "No candidate needs are currently available for analyst validation because semantic grouping has not yet executed due to insufficient evidence volume.",
                "Evidence collection is active prospectively. Future qualifying evidence will flow through the same Step 1–Step 5 pipeline.",
                "",
            ]
        )
    else:
        lines.extend(["## Candidate review details", ""])
        for review in reviews:
            candidate = candidates.get(review.get("candidate_id", ""), {})
            if review.get("review_status") != "reviewed":
                continue
            lines.extend(
                [
                    f"### {review.get('candidate_id')}: {review.get('proposed_need_label') or '(unlabeled)'}",
                    "",
                    f"- Analyst decision: `{review.get('analyst_decision')}`",
                    f"- Evidence support: group `{review.get('semantic_group_id')}`, {candidate.get('group_size', 0)} rows, support `{candidate.get('support_status', '')}`.",
                    f"- Affected source-type count: {candidate.get('source_type_count', 0)}.",
                    f"- Closest existing need: {review.get('closest_existing_need_id') or 'none'} {review.get('closest_existing_need_label') or ''}".strip(),
                    f"- Why proposed: {candidate.get('summary') or 'See Step 4 candidate artifact.'}",
                    f"- Analyst comment: {review.get('analyst_comment') or '(none)'}",
                    "- Ontology update status: `not_applied`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Governance boundary",
            "",
            "Accepted candidates remain analyst proposals. Ontology updates applied: 0. Any future ontology change requires a separate manual review step.",
            "",
        ]
    )
    return "\n".join(lines)


def regenerate_outputs(candidates: dict[str, dict[str, str]], reviews: list[dict[str, str]]) -> dict[str, Any]:
    validated = _validated_rows(candidates, reviews)
    _write_csv(REVIEWS_OUTPUT, REVIEW_FIELDS, reviews)
    _write_csv(VALIDATED_OUTPUT, VALIDATED_FIELDS, validated)
    summary = build_summary(candidates, reviews)
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_OUTPUT.write_text(build_report(candidates, reviews, summary), encoding="utf-8")
    return summary


def _append_history(event: dict[str, Any]) -> None:
    HISTORY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def run() -> dict[str, Any]:
    """Synchronize pending reviews and regenerate Step 5 artifacts."""
    candidates = _candidate_index()
    reviews = synchronize_reviews(candidates, _read_csv(REVIEWS_OUTPUT))
    return regenerate_outputs(candidates, reviews)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review candidate needs without applying ontology updates.")
    parser.add_argument("--list-pending", action="store_true", help="List candidates whose review_status is pending.")
    parser.add_argument("--candidate", help="Candidate ID to review.")
    parser.add_argument("--decision", choices=sorted(DECISIONS))
    parser.add_argument("--comment", default="")
    parser.add_argument("--reviewed-by", default="analyst")
    parser.add_argument("--merge-need-id", default="")
    args = parser.parse_args(argv)

    candidates = _candidate_index()
    reviews = synchronize_reviews(candidates, _read_csv(REVIEWS_OUTPUT))
    taxonomy_ids = existing_need_ids()

    if args.candidate or args.decision:
        if not args.candidate or not args.decision:
            parser.error("--candidate and --decision must be supplied together")
        try:
            merge_id = args.merge_need_id
            if args.decision == "merge_with_existing" and not merge_id:
                existing = next(row for row in reviews if row.get("candidate_id") == args.candidate)
                merge_id = _text(existing.get("closest_existing_need_id"))
            reviews, audit_event = apply_review(
                reviews,
                candidate_id=args.candidate,
                decision=args.decision,
                comment=args.comment,
                reviewed_by=args.reviewed_by,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
                taxonomy_ids=taxonomy_ids,
                merge_need_id=merge_id,
            )
        except (StopIteration, ValueError) as exc:
            parser.error(str(exc))
        _append_history(audit_event)

    summary = regenerate_outputs(candidates, reviews)
    if args.list_pending:
        pending = [row for row in reviews if row.get("review_status") == "pending"]
        print(f"pending_candidates={len(pending)}")
        for row in pending:
            print(f"{row.get('candidate_id')}: {row.get('proposed_need_label')}")
    else:
        print(f"pending_review={summary['pending_review']}")
        print(f"reviewed_candidates={summary['reviewed_candidates']}")
        print(f"accepted={summary['accepted']}")
        print(f"rejected={summary['rejected']}")
        print(f"merged_with_existing={summary['merged_with_existing']}")
        print(f"keep_under_observation={summary['keep_under_observation']}")
        print(f"insufficient_evidence={summary['insufficient_evidence']}")
    print(f"reviews_output={REVIEWS_OUTPUT}")
    print(f"validated_output={VALIDATED_OUTPUT}")
    print(f"report_output={REPORT_OUTPUT}")
    print(f"summary_output={SUMMARY_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
