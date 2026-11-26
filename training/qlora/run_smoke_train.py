from __future__ import annotations

import datetime as d
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("qlora_config.yaml")
SMOKE_CONFIG_PATH = Path(__file__).with_name("qlora_smoke_config.yaml")
SMOKE_DIR = ROOT / "artifacts" / "llm_training" / "adapters" / "smoke-test"
REQUIRED_PACKAGES = ["torch", "transformers", "peft", "bitsandbytes", "datasets", "trl", "yaml"]


def run_local(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def environment_summary() -> dict:
    packages = {name: bool(importlib.util.find_spec(name)) for name in REQUIRED_PACKAGES}
    summary = {
        "python": sys.version.replace("\n", " "),
        "packages": packages,
        "cuda_available": False,
        "gpu": None,
        "gpu_vram_gb": None,
        "torch_version": None,
        "transformers_version": None,
        "peft_version": None,
        "bitsandbytes_version": None,
    }
    if packages["torch"]:
        import torch

        summary["torch_version"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["gpu"] = torch.cuda.get_device_name(0)
            summary["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    for package, key in [("transformers", "transformers_version"), ("peft", "peft_version"), ("bitsandbytes", "bitsandbytes_version")]:
        if packages.get(package):
            module = __import__(package)
            summary[key] = getattr(module, "__version__", "unknown")
    return summary


def write_report(metadata: dict) -> None:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "smoke_test_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# QLoRA Smoke Test Report",
        "",
        f"Timestamp: {metadata['timestamp']}",
        f"Model: `{metadata['model']}`",
        f"Dataset: `{metadata['dataset_path']}`",
        f"Examples used: {metadata['max_examples']}",
        f"Max sequence length: {metadata['max_seq_length']}",
        f"LoRA config: r={metadata['lora']['r']}, alpha={metadata['lora']['alpha']}, dropout={metadata['lora']['dropout']}",
        f"CUDA available: {metadata['environment']['cuda_available']}",
        f"GPU: {metadata['environment']['gpu'] or 'none'}",
        f"Quantization loaded: {metadata['quantization_loaded']}",
        f"LoRA attached: {metadata['lora_attached']}",
        f"Trainable parameters: {metadata['trainable_parameters']}",
        f"Optimization steps completed: {metadata['optimization_steps_completed']}",
        f"Final smoke loss: {metadata['final_smoke_loss']}",
        f"Test split touched: {metadata['test_split_used']}",
        f"Full training run: {metadata['full_training_run']}",
        f"Status: {metadata['status']}",
        "",
        "## Commands",
        "",
        *[f"- `{command}`" for command in metadata["commands"]],
        "",
        "## Warnings / Errors",
        "",
        *[f"- {item}" for item in metadata["warnings_errors"]],
    ]
    (SMOKE_DIR / "SMOKE_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    smoke_config = {
        "max_examples": 20,
        "max_steps": 2,
        "max_seq_length": 512,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "output_dir": "artifacts/llm_training/adapters/smoke-test",
    }
    SMOKE_CONFIG_PATH.write_text(yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8")
    env = environment_summary()
    metadata = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": config["base_model"],
        "dataset_path": str(ROOT / config["dataset_dir"] / "formatted" / "train_formatted.jsonl"),
        "max_examples": smoke_config["max_examples"],
        "max_seq_length": smoke_config["max_seq_length"],
        "lora": {"r": config["lora_r"], "alpha": config["lora_alpha"], "dropout": config["lora_dropout"], "target_modules": config["target_modules"]},
        "environment": env,
        "quantization_loaded": False,
        "lora_attached": False,
        "trainable_parameters": None,
        "optimization_steps_completed": 0,
        "final_smoke_loss": None,
        "output_files_created": [],
        "commands": [
            "python training/qlora/verify_frozen_dataset.py",
            "python training/qlora/format_dataset.py --dry-run",
            "python training/qlora/run_smoke_train.py",
        ],
        "test_split_used": False,
        "full_training_run": False,
        "warnings_errors": [],
        "status": "FAIL",
    }
    missing = [name for name, installed in env["packages"].items() if not installed]
    if missing:
        metadata["warnings_errors"].append(f"Missing required packages: {', '.join(missing)}")
        metadata["warnings_errors"].append("Install with: python -m pip install -r training/qlora/requirements.txt")
        write_report(metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 1
    if not env["cuda_available"]:
        metadata["warnings_errors"].append("CUDA is unavailable; refusing to run 7B QLoRA smoke training on CPU.")
        write_report(metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 1

    run_local(["training/qlora/verify_frozen_dataset.py"])
    run_local(["training/qlora/format_dataset.py", "--dry-run"])

    from datasets import Datase
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    import torch

    data_path = ROOT / config["dataset_dir"] / "formatted" / "train_formatted.jsonl"
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= smoke_config["max_examples"]:
                break
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map="auto")
    metadata["quantization_loaded"] = True
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    metadata["lora_attached"] = True
    metadata["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def tokenize(row: dict) -> dict:
        encoded = tokenizer(row["text"], truncation=True, max_length=smoke_config["max_seq_length"], padding="max_length")
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_list(rows).map(tokenize, remove_columns=list(rows[0].keys()))
    args = TrainingArguments(
        output_dir=str(SMOKE_DIR),
        max_steps=smoke_config["max_steps"],
        per_device_train_batch_size=smoke_config["per_device_train_batch_size"],
        gradient_accumulation_steps=smoke_config["gradient_accumulation_steps"],
        logging_steps=1,
        save_steps=smoke_config["max_steps"],
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    result = trainer.train()
    metadata["optimization_steps_completed"] = smoke_config["max_steps"]
    metadata["final_smoke_loss"] = result.metrics.get("train_loss")
    trainer.save_model(str(SMOKE_DIR))
    tokenizer.save_pretrained(str(SMOKE_DIR))
    metadata["output_files_created"] = [str(p.relative_to(ROOT)) for p in SMOKE_DIR.iterdir()]
    metadata["status"] = "PASS"
    write_report(metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
