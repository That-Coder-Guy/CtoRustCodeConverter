This repository trains and evaluates an LLM-based C to Rust translator.

## What to run
- `trainer.py`: Fine-tunes the baseline language model with LoRA on `final_dataset.jsonl`. Outputs go under `training_results/`.
- `evaluator.py`: Cross-validates (baseline vs proposed prompt/training) on `final_dataset.jsonl`. Outputs go under `evaluation_results/`.
- `runner.py`: Loads the baseline model plus a saved LoRA adapter and runs interactive/sample generation.
- `presentation_demo.py`: Gradio UI demo for running the model (loads the baseline model plus a LoRA adapter).

## Key files
- `final_dataset.jsonl`: the merged training dataset used by `trainer.py` and `evaluator.py`.
- `requirements.txt`: Python dependencies for all scripts in this repo.

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
