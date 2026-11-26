from __future__ import annotations

import csv
import datetime as d
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("qlora_config.yaml")
MANIFEST_PATH = ROOT / "artifacts" / "llm_training" / "frozen_v1" / "dataset_manifest.json"


def run_local(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def package_versions() -> dict[str, str | None]:
    versions = {"python": sys.version.replace("\n", " ")}
    for name in ["torch", "transformers", "bitsandbytes", "accelerate", "peft", "trl", "datasets"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def parameter_counts(model) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_pct": round(trainable / total * 100, 4)}


def build_tokenizer_fn(tokenizer, max_seq_length: int):
    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        else:
            full_text = row["text"]
            prompt_text = "\n\n".join(f"{m['role'].upper()}:\n{m['content'].strip()}" for m in messages[:-1]).strip()
        encoded = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding="max_length")
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
        labels = encoded["input_ids"].copy()
        mask_upto = min(len(prompt_ids), len(labels))
        labels[:mask_upto] = [-100] * mask_upto
        labels = [label if attn else -100 for label, attn in zip(labels, encoded["attention_mask"])]
        encoded["labels"] = labels
        return encoded

    return tokenize


def write_history_csv(log_history: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({key for row in log_history for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_history)


def write_report(path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# Atlas Demo QLoRA v1 Training Report",
        "",
        f"Status: {metadata['status']}",
        f"Run timestamp: {metadata['timestamp_utc']}",
        f"Base model: `{metadata['base_model']}`",
        f"Frozen dataset: `{metadata['dataset_dir']}`",
        f"Manifest: `{metadata['manifest_path']}`",
        f"Output directory: `{metadata['output_dir']}`",
        "",
        "## Dataset",
        "",
        f"- Train rows: {metadata['dataset_counts']['train']}",
        f"- Validation rows: {metadata['dataset_counts']['validation']}",
        f"- Test rows: {metadata['dataset_counts']['test']} (not loaded for training)",
        "",
        "## Configuration",
        "",
        f"- Max sequence length: {metadata['config']['max_seq_length']}",
        f"- Epochs: {metadata['config']['num_train_epochs']}",
        f"- Batch size: {metadata['config']['per_device_train_batch_size']}",
        f"- Gradient accumulation: {metadata['config']['gradient_accumulation_steps']}",
        f"- Learning rate: {metadata['config']['learning_rate']}",
        f"- LoRA: r={metadata['config']['lora_r']}, alpha={metadata['config']['lora_alpha']}, dropout={metadata['config']['lora_dropout']}",
        f"- Target modules: {', '.join(metadata['config']['target_modules'])}",
        "",
        "## Environment",
        "",
        f"- GPU: {metadata['hardware'].get('gpu')}",
        f"- VRAM GB: {metadata['hardware'].get('gpu_vram_gb')}",
        f"- Compute dtype: {metadata['compute_dtype']}",
        f"- Quantization: 4-bit NF4, double quantization enabled",
        "",
        "## Results",
        "",
        f"- Best validation loss: {metadata['metrics'].get('best_validation_loss')}",
        f"- Validation perplexity: {metadata['metrics'].get('validation_perplexity')}",
        f"- Best checkpoint: `{metadata['metrics'].get('best_checkpoint')}`",
        f"- Final adapter: `{metadata['output_dir']}`",
        f"- Peak GPU memory GB: {metadata['metrics'].get('peak_gpu_memory_gb')}",
        "",
        "## Limitations",
        "",
        "- This is one controlled QLoRA run over frozen_v1, not a model search.",
        "- Held-out test evaluation must be run separately after training completes.",
        "- RAG and PostgreSQL remain the factual authorities for production-style behavior.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output_dir = ROOT / config["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing adapter directory: {output_dir}")

    run_local(["training/qlora/verify_frozen_dataset.py"])
    run_local(["training/qlora/format_dataset.py", "--dry-run"])

    from datasets import Datase
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to run full 7B QLoRA training on CPU.")

    formatted_dir = ROOT / config["dataset_dir"] / "formatted"
    train_rows = load_jsonl(formatted_dir / "train_formatted.jsonl")
    validation_rows = load_jsonl(formatted_dir / "validation_formatted.jsonl")
    test_count = sum(1 for line in (ROOT / config["dataset_dir"] / "test.jsonl").open("r", encoding="utf-8") if line.strip())

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["gradient_checkpointing"])
    lora_config = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"], target_modules=config["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    counts = parameter_counts(model)
    print(json.dumps({"parameters": counts, "gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "compute_dtype": str(dtype), "quantization_loaded": bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model.base_model.model, "is_loaded_in_4bit", False))}, indent=2))

    tokenize = build_tokenizer_fn(tokenizer, config["max_seq_length"])
    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0].keys()))
    eval_dataset = Dataset.from_list(validation_rows).map(tokenize, remove_columns=list(validation_rows[0].keys()))
    estimated_steps_per_epoch = math.ceil(len(train_rows) / (config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]))
    estimated_total_steps = int(estimated_steps_per_epoch * config["num_train_epochs"])
    warmup_steps = max(1, int(estimated_total_steps * config["warmup_ratio"]))

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_steps=warmup_steps,
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(dtype is torch.float16),
        bf16=(dtype is torch.bfloat16),
        report_to="none",
        remove_unused_columns=False,
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.copy2(CONFIG_PATH, output_dir / "training_config.yaml")
    write_history_csv(trainer.state.log_history, output_dir / "training_history.csv")

    validation_loss = eval_metrics.get("eval_loss")
    metadata = {
        "status": "PASS",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_model": config["base_model"],
        "dataset_dir": config["dataset_dir"],
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "output_dir": config["output_dir"],
        "config": config,
        "dataset_counts": {"train": len(train_rows), "validation": len(validation_rows), "test": test_count},
        "test_split_used": False,
        "package_versions": package_versions(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)},
        "compute_dtype": str(dtype),
        "parameters": counts,
        "metrics": {
            **result.metrics,
            **eval_metrics,
            "estimated_optimization_steps": estimated_total_steps,
            "warmup_steps": warmup_steps,
            "best_validation_loss": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "final_checkpoint": str(output_dir),
            "validation_perplexity": math.exp(validation_loss) if validation_loss is not None and validation_loss < 20 else None,
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        },
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "TRAINING_REPORT.md", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
