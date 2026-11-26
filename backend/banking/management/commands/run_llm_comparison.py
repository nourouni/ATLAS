from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from banking.evaluation.llm_comparison import (
    EXPERIMENTS,
    LLM_EVALUATION_DIR,
    MODEL_REGISTRY,
    GenerationConfig,
    run_llm_comparison,
)


class Command(BaseCommand):
    help = "Run controlled LLM/RAG/LangGraph comparison ablations without retraining models."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default="")
        parser.add_argument("--output-dir", default=str(LLM_EVALUATION_DIR))
        parser.add_argument("--models", default="phi-mini,mistral-banking-base,mistral-banking-qlora,qwen2.5-7b,llama3.1-8b")
        parser.add_argument("--ablations", default="rag,langgraph")
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=512)
        parser.add_argument("--top-k", type=int, default=3)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--allow-cpu", action="store_true", help="Debug override only. By default this command refuses to run without a visible GPU.")

    def handle(self, *args, **options):
        model_ids = _parse_csv(options["models"])
        ablations = _parse_csv(options["ablations"])
        unknown_models = sorted(set(model_ids) - set(MODEL_REGISTRY))
        unknown_ablations = sorted(set(ablations) - set(EXPERIMENTS))
        if unknown_models:
            raise CommandError(f"Unknown model id(s): {', '.join(unknown_models)}. Available: {', '.join(MODEL_REGISTRY)}")
        if unknown_ablations:
            raise CommandError(f"Unknown ablation(s): {', '.join(unknown_ablations)}. Available: {', '.join(EXPERIMENTS)}")

        kwargs = {
            "model_ids": model_ids,
            "ablations": ablations,
            "output_dir": Path(options["output_dir"]),
            "limit": options["limit"],
            "generation": GenerationConfig(
                temperature=options["temperature"],
                max_tokens=options["max_tokens"],
                top_k=options["top_k"],
                seed=options["seed"],
            ),
            "require_gpu": not options["allow_cpu"],
        }
        if options["dataset"]:
            kwargs["dataset_path"] = Path(options["dataset"])

        result = run_llm_comparison(**kwargs)
        self.stdout.write(f"Rows: {len(result['rows'])}")
        self.stdout.write(f"Manifest: {result['paths']['manifest']}")
        self.stdout.write(f"Metrics: {result['paths']['metrics']}")
        self.stdout.write(f"CSV: {result['paths']['csv']}")
        self.stdout.write(f"JSONL: {result['paths']['jsonl']}")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
