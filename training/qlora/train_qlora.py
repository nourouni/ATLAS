from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("qlora_config.yaml")


def run_local(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> int:
    print("WARNING: held-out test split must not be used for tuning.")
    print("WARNING: frozen chatbot evaluation must not be used for training.")
    print("WARNING: adapter is thesis/demo research only, not production banking software.")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    run_local(["training/qlora/verify_frozen_dataset.py"])
    formatted = ROOT / config["dataset_dir"] / "formatted" / "train_formatted.jsonl"
    if not formatted.exists():
        run_local(["training/qlora/format_dataset.py"])

    from datasets import load_datase
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        TrainingArguments,
    )
    from trl import SFTTrainer
    import torch

    dataset_dir = ROOT / config["dataset_dir"] / "formatted"
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(dataset_dir / "train_formatted.jsonl"),
            "validation": str(dataset_dir / "validation_formatted.jsonl"),
        },
    )
    dtype = torch.bfloat16 if config["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map="auto")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        seed=config["seed"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = result.metrics | trainer.evaluate()
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
