This repository trains and evaluates an LLM-based C to Rust translator.

## What to run
- `trainer.py`: Fine-tunes the baseline language model with LoRA on `final_dataset.jsonl`. Outputs go under `training_results/`.
- `evaluator.py`: Cross-validates (baseline vs proposed prompt/training) on `final_dataset.jsonl`. Outputs go under `evaluation_results/`.
- `runner.py`: Loads the baseline model plus a saved LoRA adapter and runs interactive/sample generation.
- `presentation_demo.py`: Gradio UI demo for running the model (loads the baseline model plus a LoRA adapter).

## Key files
- `final_dataset.jsonl`: the merged training dataset used by `trainer.py` and `evaluator.py`.
- `requirements.txt`: Python dependencies for all scripts in this repo.

## Datasets and `data_sanitizer.py`
The raw source datasets under `original_datasets/` are **too large to store directly in this repository**. The sanitization pipeline in `data_sanitizer.py` can process them, but the actual original dataset files are stored via **Git LFS** and **won’t be available / parseable unless you download/pull the LFS objects**.

Because of that, this repo includes a pre-created `final_dataset.jsonl` so you can train/evaluate immediately without first downloading the large raw datasets.

If you *do* have the raw datasets available locally, you can re-generate a merged, sanitized dataset by running:

```bash
py -3 data_sanitizer.py
```

This writes the merged output to `cleaned_datasets/final_dataset_v2.jsonl` (see the script constants/paths near the top of `data_sanitizer.py`).

## Folders
- `original_datasets/`: Raw source datasets.
  - `CodeTransOcean_dataset/`: JSONL dataset files (e.g. `niche_train.jsonl`).
  - `RosettaCodeData_dataset/`: Parquet dataset file(s).

## Setup
Create a virtual environment and install dependencies:

```bash
py -3 -m venv .venv
.venv\\Scripts\\activate
py -3 -m pip install -r requirements.txt
```
