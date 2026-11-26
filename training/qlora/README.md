# Atlas Demo QLoRA Scaffolding

This folder prepares QLoRA adapter training for `bitext/Mistral-7B-Banking-v2` using the frozen Atlas Demo dataset in `artifacts/llm_training/frozen_v1/`.

The frozen dataset is a research/thesis artifact built from verified public/project information. It is not official Atlas Demo Bank training data. Freezing preserves hashes, row counts, split membership, and audit evidence so later QLoRA runs are reproducible.

Architecture role:

```tex
RAG = Atlas Demo factual authority
PostgreSQL = private customer facts
LangGraph/Django = control and security
QLoRA adapter = banking/customer-service reasoning and conversation behavior
```

## Verify Datase

```bash
python training/qlora/verify_frozen_dataset.py
```

## Format Datase

```bash
python training/qlora/format_dataset.py
```

For validation only:

```bash
python training/qlora/format_dataset.py --dry-run
```

## Smoke Training

Do not run unless explicitly requested. It is intended to verify model loading, dataset formatting, LoRA attachment, and one or two optimization steps.

```bash
python training/qlora/run_smoke_train.py
```

## Full Training

Do not tune on the test split. Do not use the frozen chatbot evaluation as training data.

```bash
python training/qlora/train_qlora.py
```

Adapter target:

```tex
artifacts/llm_training/adapters/atlas_demo-mistral-banking-qlora-v1/
```

If GPU memory is limited, consider reducing `max_seq_length`, increasing `gradient_accumulation_steps`, or using a smaller smoke run before full training. Keep defaults unchanged for the frozen v1 plan unless hardware requires an explicit documented change.

## Evaluation

Validation evaluation:

```bash
python training/qlora/evaluate_adapter.py --split validation
```

Final test evaluation should be run once after training is frozen:

```bash
python training/qlora/evaluate_adapter.py --split test --confirm-test-eval
```

The evaluation script does not use the frozen chatbot evaluation dataset.
