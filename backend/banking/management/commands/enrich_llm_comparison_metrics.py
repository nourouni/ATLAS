from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from banking.evaluation.llm_comparison import _summarize, _write_csv, _write_jsonl
from banking.evaluation.metrics import add_bertscore, calculate_answer_quality, flatten_answer_quality


class Command(BaseCommand):
    help = "Retrospectively add BERTScore and flat answer-level metrics to an existing LLM comparison JSONL."

    def add_arguments(self, parser):
        parser.add_argument("jsonl_path")
        parser.add_argument("--output-stem", default="")
        parser.add_argument("--lang", default="en")
        parser.add_argument("--model-type", default="distilbert-base-uncased")
        parser.add_argument("--idf", action="store_true")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--device", default="")

    def handle(self, *args, **options):
        source_path = Path(options["jsonl_path"])
        if not source_path.exists():
            raise CommandError(f"Input JSONL does not exist: {source_path}")

        rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            reference = row.get("reference_answer") or ""
            if reference:
                row["answer_quality"] = calculate_answer_quality(row.get("final_answer") or "", reference)
            flatten_answer_quality(row)

        bertscore_config = add_bertscore(
            rows,
            lang=options["lang"],
            model_type=options["model_type"] or None,
            idf=options["idf"],
            batch_size=options["batch_size"],
            device=options["device"] or None,
        )
        metrics = _summarize(rows)
        metrics["_bertscore_config"] = bertscore_config

        stem = options["output_stem"] or f"{source_path.stem}_with_bertscore"
        jsonl_path = source_path.with_name(f"{stem}.jsonl")
        csv_path = source_path.with_name(f"{stem}.csv")
        metrics_path = source_path.with_name(f"{stem}_metrics.json")

        _write_jsonl(rows, jsonl_path)
        _write_csv(rows, csv_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

        self.stdout.write(f"Rows: {len(rows)}")
        self.stdout.write(f"BERTScore: {json.dumps(bertscore_config, sort_keys=True)}")
        self.stdout.write(f"Metrics: {metrics_path}")
        self.stdout.write(f"CSV: {csv_path}")
        self.stdout.write(f"JSONL: {jsonl_path}")
