from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import ollama
from django.conf import settings
from django.utils import timezone

from banking.orchestration import nodes
from banking.orchestration.prompts import generation_promp

from .dataset import DEFAULT_DATASET_PATH, dataset_hash, load_datase
from .metrics import calculate_metrics
from .runner import evaluate_case


LLM_EVALUATION_DIR = settings.REPO_ROOT / "artifacts" / "llm_evaluation"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    ollama_model: str
    adapter_path: str = ""
    quantization: str = ""
    notes: str = ""
    backend: str = "ollama"
    base_model: str = ""


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    top_k: int = 3
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    ablation: str
    rag_enabled: bool
    langgraph_enabled: bool


MODEL_REGISTRY = {
    "phi-mini": ModelSpec("phi-mini", "Phi-mini", "phi4-mini", notes="Lightweight model already used by the project chatbot."),
    "mistral-banking-base": ModelSpec(
        "mistral-banking-base",
        "Mistral-7B-Banking-v2",
        "",
        quantization="4-bit NF4 via bitsandbytes",
        notes="Banking/customer-support domain base model.",
        backend="transformers",
        base_model="bitext/Mistral-7B-Banking-v2",
    ),
    "mistral-banking-qlora": ModelSpec(
        "mistral-banking-qlora",
        "Mistral-7B-Banking-v2 + Atlas Demo QLoRA",
        "",
        adapter_path=str(settings.REPO_ROOT / "artifacts" / "llm_training" / "adapters" / "atlas_demo-qlora-v1"),
        quantization="4-bit NF4 base model with QLoRA adapter via PEFT",
        notes="Existing Atlas Demo adapter loaded over bitext/Mistral-7B-Banking-v2. This command does not train or overwrite it.",
        backend="transformers",
        base_model="bitext/Mistral-7B-Banking-v2",
    ),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", "Qwen 2.5 7B Instruct", "qwen2.5:7b-instruct", notes="General-purpose similarly sized baseline."),
    "llama3.1-8b": ModelSpec("llama3.1-8b", "Llama 3.1 8B", "llama3.1:8b", notes="General-purpose deployment-scale baseline."),
}


_TRANSFORMERS_GENERATORS: dict[str, Any] = {}


EXPERIMENTS = {
    "llm": ExperimentConfig("llm", rag_enabled=False, langgraph_enabled=False),
    "rag": ExperimentConfig("rag", rag_enabled=True, langgraph_enabled=False),
    "langgraph": ExperimentConfig("langgraph", rag_enabled=True, langgraph_enabled=True),
}


def _ollama_generate(model: str, prompt: str, config: GenerationConfig) -> str:
    client = ollama.Client(host=getattr(settings, "CHATBOT_OLLAMA_HOST", None), timeout=float(getattr(settings, "CHATBOT_OLLAMA_TIMEOUT", 120)))
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "seed": config.seed,
        },
    )
    return str(response["message"]["content"]).strip()


class TransformersGenerator:
    def __init__(self, model: ModelSpec):
        self.model = model
        self.model_name = model.base_model or model.ollama_model
        self.adapter_path = model.adapter_path
        self._tokenizer = None
        self._model = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import importlib.metadata as metadata

            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("Transformers/PEFT Mistral inference requires torch, transformers, bitsandbytes, and peft.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Transformers/PEFT Mistral inference.")

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map="auto",
        )

        adapter_metadata = {}
        if self.adapter_path:
            adapter_config_path = Path(self.adapter_path) / "adapter_config.json"
            if not adapter_config_path.exists():
                raise RuntimeError(f"Adapter config not found: {adapter_config_path}")
            adapter_metadata = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            adapter_base = adapter_metadata.get("base_model_name_or_path")
            if adapter_base != self.model_name:
                raise RuntimeError(f"Adapter base mismatch: expected {self.model_name}, found {adapter_base}")
            loaded_model = PeftModel.from_pretrained(loaded_model, self.adapter_path)

        loaded_model.eval()
        self._tokenizer = tokenizer
        self._model = loaded_model
        self.metadata = {
            "backend": "transformers",
            "base_model": self.model_name,
            "adapter_path": self.adapter_path,
            "adapter_base_model": adapter_metadata.get("base_model_name_or_path") if adapter_metadata else "",
            "active_adapter": getattr(loaded_model, "active_adapter", ""),
            "device": str(next(loaded_model.parameters()).device),
            "cuda_device": torch.cuda.get_device_name(0),
            "quantization": "4-bit NF4 bitsandbytes",
            "package_versions": {
                "torch": metadata.version("torch"),
                "transformers": metadata.version("transformers"),
                "peft": metadata.version("peft"),
                "bitsandbytes": metadata.version("bitsandbytes"),
            },
        }

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self.load()
        import torch

        tokenizer = self._tokenizer
        model = self._model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        generation_kwargs = {
            "max_new_tokens": config.max_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if config.temperature > 0:
            generation_kwargs["temperature"] = config.temperature
        if config.temperature > 0 and config.top_k > 0:
            generation_kwargs["top_k"] = config.top_k
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _generate(model: ModelSpec, prompt: str, config: GenerationConfig) -> str:
    if model.backend == "ollama":
        return _ollama_generate(model.ollama_model, prompt, config)
    if model.backend == "transformers":
        generator = _TRANSFORMERS_GENERATORS.get(model.model_id)
        if generator is None:
            generator = TransformersGenerator(model)
            _TRANSFORMERS_GENERATORS[model.model_id] = generator
        return generator.generate(prompt, config)
    raise RuntimeError(f"Unsupported model backend: {model.backend}")


def gpu_available() -> tuple[bool, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return False, "No CUDA device detected and nvidia-smi is unavailable."
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode == 0 and names:
        return True, names[0]
    return False, (completed.stderr or "nvidia-smi did not report a GPU.").strip()


def _reference_answer(case: dict[str, Any]) -> str:
    points = [str(point) for point in case.get("expected_answer_points", []) if point]
    return " ".join(points)


def _pseudo_reference_fields(reference: str) -> dict[str, Any]:
    return {
        "reference_answer": reference,
        "legacy_reference_answer": reference,
        "reference_metric_applicable": False,
        "reference_metric_reason": "expected_answer_points_are_not_full_reference" if reference else "reference_missing",
        "answer_quality": None,
        "metric_applicability": "expected_answer_points_are_not_full_reference",
    }


def _evidence_for_case(case: dict[str, Any], rag_enabled: bool) -> dict[str, Any]:
    if not rag_enabled:
        return {}
    intent = case.get("expected_semantic_intent") or case.get("expected_intent")
    if intent == "PUBLIC_INFO":
        return {"public_information": [nodes._public_context({"message": case["question"], "request_id": f"llm-eval-{case['case_id']}"})]}
    return {}


def _run_llm_ablation(case: dict[str, Any], model: ModelSpec, experiment: ExperimentConfig, generation: GenerationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    evidence = _evidence_for_case(case, experiment.rag_enabled)
    prompt = generation_prompt(case["question"], evidence)
    answer = _generate(model, prompt, generation)
    latency_ms = (time.perf_counter() - started) * 1000
    reference = _reference_answer(case)
    return {
        "case_id": case["case_id"],
        "category": case.get("category", case["case_id"].split("-")[0]),
        "question": case["question"],
        "model_id": model.model_id,
        "ablation": experiment.ablation,
        "rag_enabled": experiment.rag_enabled,
        "langgraph_enabled": experiment.langgraph_enabled,
        **_pseudo_reference_fields(reference),
        "final_answer": answer,
        "retrieved_document_ids": [item.get("document_id") for item in evidence.get("public_information", [{}])[0].get("citations", [])] if evidence else [],
        "latency_ms": round(latency_ms, 2),
    }


def _run_langgraph_case(case: dict[str, Any], model: ModelSpec, generation: GenerationConfig) -> dict[str, Any]:
    def llm_generate(prompt: str) -> str:
        return _generate(model, prompt, generation)

    with patch("banking.orchestration.nodes._llm_generate", side_effect=llm_generate):
        row = evaluate_case(case)
    reference = _reference_answer(case)
    row.update(
        {
            "model_id": model.model_id,
            "ablation": "langgraph",
            "rag_enabled": True,
            "langgraph_enabled": True,
            **_pseudo_reference_fields(reference),
        }
    )
    return row


def run_llm_comparison(
    *,
    model_ids: list[str],
    ablations: list[str],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = LLM_EVALUATION_DIR,
    limit: int | None = None,
    generation: GenerationConfig | None = None,
    require_gpu: bool = True,
) -> dict[str, Any]:
    generation = generation or GenerationConfig()
    gpu_ok, gpu_detail = gpu_available()
    if require_gpu and not gpu_ok:
        raise RuntimeError(f"GPU is required for this evaluation. {gpu_detail}")
    cases = load_dataset(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model = MODEL_REGISTRY[model_id]
        for ablation in ablations:
            experiment = EXPERIMENTS[ablation]
            runner: Callable[[dict[str, Any]], dict[str, Any]]
            if experiment.langgraph_enabled:
                runner = lambda case, model=model: _run_langgraph_case(case, model, generation)
            else:
                runner = lambda case, model=model, experiment=experiment: _run_llm_ablation(case, model, experiment, generation)
            for case in cases:
                rows.append(runner(case))

    timestamp = timezone.now().strftime("%Y%m%dT%H%M%S")
    stem = f"llm_comparison_{timestamp}"
    jsonl_path = output_dir / f"{stem}.jsonl"
    csv_path = output_dir / f"{stem}.csv"
    metrics_path = output_dir / f"{stem}_metrics.json"
    manifest_path = output_dir / f"{stem}_manifest.json"
    _write_jsonl(rows, jsonl_path)
    _write_csv(rows, csv_path)

    metrics = _summarize(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "timestamp": timezone.now().isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash(dataset_path),
        "case_count": len(cases),
        "models": [asdict(MODEL_REGISTRY[model_id]) for model_id in model_ids],
        "ablations": [asdict(EXPERIMENTS[ablation]) for ablation in ablations],
        "generation": asdict(generation),
        "hardware": {"gpu_required": require_gpu, "gpu_available": gpu_ok, "gpu_detail": gpu_detail},
        "runtime_models": {
            model_id: getattr(_TRANSFORMERS_GENERATORS.get(model_id), "metadata", {})
            for model_id in model_ids
            if model_id in _TRANSFORMERS_GENERATORS
        },
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path), "metrics": str(metrics_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": rows, "metrics": metrics, "manifest": manifest, "paths": {"jsonl": jsonl_path, "csv": csv_path, "metrics": metrics_path, "manifest": manifest_path}}


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['model_id']}::{row['ablation']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, group in groups.items():
        group_metrics = calculate_metrics(group)
        if group and "expected_route" in group[0]:
            summary[key] = group_metrics
        else:
            summary[key] = {
                "case_count": len(group),
                "answer_quality": group_metrics["answer_quality"],
                "latency_ms": group_metrics["latency_ms"],
                "by_category": {
                    category: {
                        "case_count": values["case_count"],
                        "answer_quality": values["answer_quality"],
                    }
                    for category, values in group_metrics["by_category"].items()
                },
            }
    return summary
