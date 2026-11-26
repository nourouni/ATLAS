from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "llm_training" / "frozen_v1"
ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    roles = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"invalid role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        roles.append(role)
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("each example needs at least one user and one assistant message")
    if roles[-1] != "assistant":
        raise ValueError("final message must be assistant")
    previous = None
    for role in roles:
        if role == "system":
            if previous is not None:
                raise ValueError("system messages are only allowed at the beginning")
        elif role == previous and role != "system":
            raise ValueError("consecutive user or assistant messages are not allowed")
        previous = role
    return messages


def fallback_chat_template(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content'].strip()}")
    return "\n\n".join(parts).strip()


def format_file(src: Path, dst: Path, split: str, dry_run: bool) -> dict[str, Any]:
    count = 0
    tasks: Counter[str] = Counter()
    scopes: Counter[str] = Counter()
    errors = []
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = None if dry_run else dst.open("w", encoding="utf-8")
    try:
        with src.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    msgs = validate_messages(row.get("messages"))
                except ValueError as exc:
                    errors.append({"line": line_no, "error": str(exc)})
                    continue
                formatted = {
                    "id": row.get("example_id"),
                    "split": split,
                    "text": fallback_chat_template(msgs),
                    "messages": msgs,
                    "metadata": {
                        "task": row.get("task"),
                        "scope": row.get("scope"),
                        "entities": row.get("entities", []),
                        "source_group_id": row.get("source_group_id"),
                        "provenance": row.get("provenance", {}),
                        "quality_flags": row.get("quality_flags", []),
                    },
                }
                if out:
                    out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                count += 1
                tasks[row.get("task", "")] += 1
                scopes[row.get("scope", "")] += 1
    finally:
        if out:
            out.close()
    return {"split": split, "rows": count, "errors": errors, "task_distribution": dict(tasks), "scope_distribution": dict(scopes)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir = dataset_dir / "formatted"
    report = {"dataset_dir": str(dataset_dir), "dry_run": args.dry_run, "splits": {}}
    for split in ["train", "validation", "test"]:
        src = dataset_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}_formatted.jsonl"
        report["splits"][split] = format_file(src, dst, split, args.dry_run)
    report["status"] = "PASS" if not any(v["errors"] for v in report["splits"].values()) else "FAIL"
    if not args.dry_run:
        (out_dir / "formatting_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
