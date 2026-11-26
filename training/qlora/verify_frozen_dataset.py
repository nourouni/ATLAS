from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "llm_training" / "frozen_v1" / "dataset_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    return None


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.exists():
        print(f"FAIL: missing manifest {MANIFEST}")
        return 1
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = ["atlas_demo_training_candidates.jsonl", "atlas_demo_training_rejected.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl"]
    frozen_dir = MANIFEST.paren
    for name in required:
        if not (frozen_dir / name).exists():
            fail(errors, f"missing required file: {name}")
    for item in manifest.get("frozen_files", []):
        path = ROOT / item["path"]
        if not path.exists():
            fail(errors, f"missing frozen file: {item['path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        actual_rows = row_count(path)
        if actual_size != item.get("size_bytes"):
            fail(errors, f"size mismatch for {item['path']}: {actual_size} != {item.get('size_bytes')}")
        if actual_hash != item.get("sha256"):
            fail(errors, f"hash mismatch for {item['path']}")
        if item.get("row_count") is not None and actual_rows != item.get("row_count"):
            fail(errors, f"row count mismatch for {item['path']}: {actual_rows} != {item.get('row_count')}")
    counts = manifest.get("counts", {})
    split_counts = counts.get("splits", {})
    for split, expected in split_counts.items():
        actual = row_count(frozen_dir / f"{split}.jsonl")
        if actual != expected:
            fail(errors, f"split count mismatch for {split}: {actual} != {expected}")
    checklist = manifest.get("quality_results", {})
    for key in ["source_id_leakage", "frozen_eval_leakage", "meta_instruction_leakage", "ungrounded_comparisons", "ontology_trust_violations"]:
        if checklist.get(key) != 0:
            fail(errors, f"{key} expected 0, found {checklist.get(key)}")
    if errors:
        print("FROZEN DATASET VERIFICATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("FROZEN DATASET VERIFICATION: PASS")
    print(json.dumps({"manifest": str(MANIFEST), "files_checked": len(manifest.get("frozen_files", [])), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
