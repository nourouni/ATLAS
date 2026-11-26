"""Business-facing presentation metadata for canonical behavioral segments.

This module changes only how canonical behavioral IDs are presented.  It does
not assign customers, rename IDs, or infer profiles from cluster numbers.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any


ACTIVE_BEHAVIORAL_SEGMENT_IDS = frozenset(
    {
        "seg_ret_neg1",
        "seg_ret_0",
        "seg_ret_1",
        "seg_ret_2",
        "seg_ret_3",
        "seg_ret_4",
        "seg_ent_neg1",
        "seg_ent_0",
        "seg_ent_1",
        "seg_ent_2",
    }
)

_PRESENTATION = {
    "seg_ret_neg1": {
        "display_name": "Broad-Channel Retail",
        "description": "A small retail group with unusually broad channel and product use that does not form one stable pattern.",
        "strengths": ["Broad channel usage", "Wide product ownership"],
        "watch_outs": ["Customer-level evidence is more useful than the group label"],
    },
    "seg_ret_0": {
        "display_name": "Frequent Everyday Users",
        "description": "These customers transact frequently and show regular everyday activity, alongside low balances during much of the observed period.",
        "strengths": ["Frequent account activity", "Regular everyday usage"],
        "watch_outs": ["Balances are low during much of the observed period"],
    },
    "seg_ret_1": {
        "display_name": "Established Multi-Product",
        "description": "These customers have the highest relative balances among regular retail groups, broad channel use, and comparatively wide product ownership.",
        "strengths": ["Broad product usage", "Multi-channel access"],
        "watch_outs": ["Balance variability remains relatively high"],
    },
    "seg_ret_2": {
        "display_name": "Active Digital Retail",
        "description": "These customers combine high transaction and monetary activity with strong digital usage and limited reliance on branches.",
        "strengths": ["High activity", "Frequent digital usage"],
        "watch_outs": ["Digital preference does not by itself establish product eligibility"],
    },
    "seg_ret_3": {
        "display_name": "Digital Frequent Users",
        "description": "These customers transact very frequently through digital channels, while their relative monetary activity and balances are lower than other regular retail groups.",
        "strengths": ["Frequent digital activity", "Low reliance on assisted channels"],
        "watch_outs": ["High activity does not imply high monetary value"],
    },
    "seg_ret_4": {
        "display_name": "ATM-Focused Light Users",
        "description": "These customers show lower transaction activity, strong ATM reliance, and the lowest digital usage among regular retail groups.",
        "strengths": ["Consistent account access pattern", "Clear ATM usage"],
        "watch_outs": ["Digital usage is comparatively limited"],
    },
    "seg_ent_neg1": {
        "display_name": "Broad-Activity Enterprise",
        "description": "A small enterprise group with activity patterns that differ structurally from the regular enterprise groups and do not form one stable profile.",
        "strengths": ["Distinct activity pattern"],
        "watch_outs": ["Customer-level evidence is more useful than the group label"],
    },
    "seg_ent_0": {
        "display_name": "High-Value Digital Enterprises",
        "description": "These enterprises show the strongest relative activity, balances, and digital transaction usage among regular enterprise groups.",
        "strengths": ["High business activity", "Strong digital usage"],
        "watch_outs": ["Activity and balances alone do not establish a financing or investment need"],
    },
    "seg_ent_1": {
        "display_name": "Mixed-Channel Enterprises",
        "description": "These enterprises show moderate activity with mixed channel use and the lowest relative balances among regular enterprise groups.",
        "strengths": ["Flexible channel usage", "Regular business activity"],
        "watch_outs": ["Deposit and cash-flow patterns vary over time"],
    },
    "seg_ent_2": {
        "display_name": "Branch-Oriented Core Enterprises",
        "description": "These enterprises have lower relative activity, moderate balances, and stronger branch and ATM use than other regular enterprise groups.",
        "strengths": ["Broad assisted-channel access", "Established core activity"],
        "watch_outs": ["Digital usage is comparatively lower"],
    },
}


def _mapping_path() -> Path:
    # Resolve from this module rather than Django settings so the helper is
    # also usable by focused presentation tests.
    return Path(__file__).resolve().parents[2] / "artifacts" / "mappings" / "behavioral_segments.csv"


@lru_cache(maxsize=1)
def behavioral_segment_definitions() -> dict[str, dict[str, Any]]:
    """Return presentation-ready definitions for active canonical IDs only."""
    path = _mapping_path()
    source_rows: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = {
                row["segment_id"]: row
                for row in csv.DictReader(handle)
                if row.get("segment_id") in ACTIVE_BEHAVIORAL_SEGMENT_IDS
            }

    result: dict[str, dict[str, Any]] = {}
    for segment_id in ACTIVE_BEHAVIORAL_SEGMENT_IDS:
        source = dict(source_rows.get(segment_id) or {})
        presentation = _PRESENTATION[segment_id]
        result[segment_id] = {
            **source,
            "segment_id": segment_id,
            "technical_name": source.get("segment_name") or segment_id,
            "source_description": source.get("description") or "",
            "display_name": presentation["display_name"],
            "description": presentation["description"],
            "strengths": list(presentation["strengths"]),
            "watch_outs": list(presentation["watch_outs"]),
        }
    return resul


def resolve_behavioral_segment(segment_id: str | None, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a canonical ID without silently mapping unknown IDs."""
    segment_id = str(segment_id or "")
    definition = behavioral_segment_definitions().get(segment_id)
    if definition:
        return {**(fallback or {}), **definition}
    resolved = dict(fallback or {})
    resolved.update(
        {
            "segment_id": segment_id,
            "display_name": "Unclassified behavior",
            "description": "No verified behavioral profile is available for this segment.",
            "strengths": [],
            "watch_outs": [],
        }
    )
    return resolved


def is_active_behavioral_segment(segment_id: str | None) -> bool:
    return str(segment_id or "") in ACTIVE_BEHAVIORAL_SEGMENT_IDS
