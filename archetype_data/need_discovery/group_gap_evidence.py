"""Step 3: deterministic semantic representation and evidence grouping.

This produces semantic evidence groups only. It never labels a group as a new
need and never modifies recommendation or ontology artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter, defaultdic
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "need_discovery" / "filtered_gap_evidence.csv"
FILTERING_SUMMARY = ROOT / "artifacts" / "need_discovery" / "filtering_summary.json"
GROUPS_OUTPUT = ROOT / "artifacts" / "need_discovery" / "semantic_evidence_groups.csv"
SUMMARY_OUTPUT = ROOT / "artifacts" / "need_discovery" / "semantic_group_summary.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.82
MINIMUM_ROWS_THRESHOLD = 20

GROUP_FIELDS = [
    "evidence_id",
    "semantic_group_id",
    "normalized_text",
    "source_type",
    "normalized_gap_type",
    "existing_need",
    "product",
    "group_size",
    "similarity_to_group_centroid",
    "is_group_representative",
    "support_status",
    "unique_semantic_evidence_count",
    "unique_source_type_count",
    "unique_customer_or_case_count",
    "unique_product_count",
    "unique_existing_need_count",
    "gap_type_distribution",
]


def _read_rows() -> tuple[list[dict[str, str]], list[str]]:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("keep_for_need_discovery", "").lower() == "true"
        and row.get("semantic_ready", "").lower() == "true"
        and row.get("normalized_text", "").strip()
    ]


def _support_status(size: int) -> str:
    if size == 1:
        return "insufficient_support"
    if size <= 4:
        return "preliminary_support"
    return "repeated_support"


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    return vectors / norms


def group_rows(
    rows: list[dict[str, str]],
    vectors: np.ndarray,
    *,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Group already-embedded rows; exposed for deterministic unit tests."""
    if not rows:
        return []
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if len(rows) != len(vectors):
        raise ValueError("rows and vectors must have the same length")

    order = sorted(range(len(rows)), key=lambda index: rows[index].get("evidence_id", ""))
    ordered_rows = [rows[index] for index in order]
    ordered_vectors = _normalize_vectors(np.asarray(vectors, dtype=float)[order])
    if len(ordered_rows) == 1:
        labels = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - similarity_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(ordered_vectors)

    members: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[int(label)].append(index)
    group_order = sorted(members, key=lambda label: min(ordered_rows[index]["evidence_id"] for index in members[label]))
    group_ids = {label: f"SG{position:04d}" for position, label in enumerate(group_order, start=1)}

    output: list[dict[str, Any]] = []
    for label in group_order:
        indices = members[label]
        group_vectors = ordered_vectors[indices]
        centroid = _normalize_vectors(np.mean(group_vectors, axis=0, keepdims=True))[0]
        similarities = group_vectors @ centroid
        representative_local = max(
            range(len(indices)),
            key=lambda local: (float(similarities[local]), ordered_rows[indices[local]]["evidence_id"]),
        )
        representative_index = indices[representative_local]
        group_rows_data = [ordered_rows[index] for index in indices]
        unique_texts = {row.get("normalized_text", "") for row in group_rows_data if row.get("normalized_text", "")}
        unique_sources = {row.get("source_type", "") for row in group_rows_data if row.get("source_type", "")}
        unique_cases = {row.get("customer_id_or_case_id", "") for row in group_rows_data if row.get("customer_id_or_case_id", "")}
        unique_products = {row.get("product", "") for row in group_rows_data if row.get("product", "")}
        unique_needs = {row.get("existing_need", "") for row in group_rows_data if row.get("existing_need", "")}
        type_counts = Counter(row.get("normalized_gap_type", "") for row in group_rows_data)
        type_counts.pop("", None)
        for local, index in enumerate(indices):
            row = ordered_rows[index]
            output.append(
                {
                    "evidence_id": row.get("evidence_id", ""),
                    "semantic_group_id": group_ids[label],
                    "normalized_text": row.get("normalized_text", ""),
                    "source_type": row.get("source_type", ""),
                    "normalized_gap_type": row.get("normalized_gap_type", ""),
                    "existing_need": row.get("existing_need", ""),
                    "product": row.get("product", ""),
                    "group_size": len(indices),
                    "similarity_to_group_centroid": round(float(similarities[local]), 8),
                    "is_group_representative": "true" if index == representative_index else "false",
                    "support_status": _support_status(len(indices)),
                    "unique_semantic_evidence_count": len(unique_texts),
                    "unique_source_type_count": len(unique_sources),
                    "unique_customer_or_case_count": len(unique_cases),
                    "unique_product_count": len(unique_products),
                    "unique_existing_need_count": len(unique_needs),
                    "gap_type_distribution": json.dumps(dict(sorted(type_counts.items())), ensure_ascii=False, sort_keys=True),
                }
            )
    return sorted(output, key=lambda row: (row["semantic_group_id"], row["evidence_id"]))


def _load_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Step 3 requires sentence-transformers when the readiness gate opens. "
            "Install the declared dependency in the project environment, then rerun."
        ) from exc
    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Local embedding model {EMBEDDING_MODEL!r} is not available in the local model cache. "
            "Populate the cache explicitly, then rerun; no unrelated fallback is used."
        ) from exc


def _write_groups(rows: list[dict[str, Any]]) -> None:
    with GROUPS_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    rows, _ = _read_rows()
    filtering_summary = json.loads(FILTERING_SUMMARY.read_text(encoding="utf-8"))
    eligible = _eligible_rows(rows)
    minimum_threshold = int(filtering_summary.get("minimum_semantic_rows_threshold", MINIMUM_ROWS_THRESHOLD))
    grouping_ready = bool(filtering_summary.get("semantic_grouping_ready")) and len(eligible) >= minimum_threshold
    input_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    GROUPS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not grouping_ready:
        _write_groups([])
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "eligible_evidence_rows": len(eligible),
            "semantic_grouping_ready": False,
            "grouping_executed": False,
            "grouping_reason": "insufficient_evidence",
            "minimum_rows_threshold": minimum_threshold,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": None,
            "embeddings_generated": False,
            "embedding_artifact": None,
            "clustering_method": "agglomerative_cosine",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "number_of_groups": None,
            "singleton_groups": None,
            "largest_group_size": None,
            "input_filtered_csv_sha256": input_hash,
            "library_versions": {"python": platform.python_version(), "scikit_learn": sklearn_version},
            "statement": "No candidate needs were discovered. Current evidence is below the operational grouping threshold.",
        }
        SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    model = _load_embedding_model()
    texts = [row["normalized_text"] for row in sorted(eligible, key=lambda row: row["evidence_id"])]
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=float)
    grouped = group_rows(sorted(eligible, key=lambda row: row["evidence_id"]), vectors, similarity_threshold=SIMILARITY_THRESHOLD)
    _write_groups(grouped)
    group_sizes = Counter(row["semantic_group_id"] for row in grouped)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eligible_evidence_rows": len(eligible),
        "semantic_grouping_ready": True,
        "grouping_executed": True,
        "grouping_reason": "ready",
        "minimum_rows_threshold": minimum_threshold,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(vectors.shape[1]),
        "embeddings_generated": True,
        "embedding_artifact": None,
        "clustering_method": "agglomerative_cosine",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "number_of_groups": len(group_sizes),
        "singleton_groups": sum(size == 1 for size in group_sizes.values()),
        "largest_group_size": max(group_sizes.values(), default=0),
        "input_filtered_csv_sha256": input_hash,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn_version},
        "statement": "Semantic evidence groups were generated; groups are not validated new needs.",
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(f"eligible_evidence_rows={summary['eligible_evidence_rows']}")
    print(f"semantic_grouping_ready={summary['semantic_grouping_ready']}")
    print(f"grouping_executed={summary['grouping_executed']}")
    print(f"grouping_reason={summary['grouping_reason']}")
    print(f"number_of_groups={summary['number_of_groups']}")
    print(f"groups_output={GROUPS_OUTPUT}")
    print(f"summary_output={SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
