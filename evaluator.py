from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# --- 1. Experiment Configuration ---

BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_JSONL: str = "final_dataset.jsonl"
BASE_OUTPUT_DIR: str = "./evaluation_results"

# 5x2cv Requirements
SEEDS: List[int] = [42, 1337, 2024, 777, 999]
NUM_FOLDS: int = 2
TEST_MODELS: List[str] = ["Baseline", "Proposed"]

# Training hyperparameters
MAX_SEQ_LENGTH: int = 8192
PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
GRADIENT_ACCUMULATION_STEPS: int = 16
LEARNING_RATE: float = 5e-5
NUM_TRAIN_EPOCHS: float = 3.0
WARMUP_RATIO: float = 0.03
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 200
EVAL_STEPS: int = 200

# Runtime / debug settings
USE_4BIT: bool = False
USE_BF16: bool = True
USE_FP16: bool = False
GRADIENT_CHECKPOINTING: bool = True
DRY_RUN: bool = False
INFERENCE_BATCH_SIZE: int = 4


# --- 2. Data Processing Helpers ---

def _as_messages_baseline(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Formats the dataset example using the baseline prompt without complex system constraints.
    
    Args:
        example (Dict[str, Any]): A single dataset record containing 'c_code' and 'safe_rust_code'.
    
    Returns:
        List[Dict[str, str]]: A list of message dictionaries suitable for chat template application.
    """
    c_code: Optional[str] = example.get("c_code")
    safe: Optional[str] = example.get("safe_rust_code")
    
    if not (isinstance(c_code, str) and isinstance(safe, str)):
        return [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]

    return [
        {"role": "user", "content": f"Convert this C code to Rust:\n\n{c_code}"},
        {"role": "assistant", "content": safe}
    ]


def _as_messages_proposed(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Formats the dataset example using the proposed complex system prompt and rules.
    
    Args:
        example (Dict[str, Any]): A single dataset record containing 'c_code' and 'safe_rust_code'.
    
    Returns:
        List[Dict[str, str]]: A list of message dictionaries including the system prompt.
    """
    c_code: Optional[str] = example.get("c_code")
    safe: Optional[str] = example.get("safe_rust_code")
    
    if not (isinstance(c_code, str) and isinstance(safe, str)):
        return [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]

    system_prompt: str = """Convert the provided C program into safe, highly idiomatic stable Rust.

Rules for Translation:
1. Pure Safe Rust: Prefer 100% safe code. Use `unsafe` only if it is fundamentally impossible to achieve the behavior otherwise.
2. Native Types: Use standard Rust types (`i32`, `usize`, `str`, `String`, etc.). Strictly avoid FFI types (`c_int`, `c_char`, `c_void`, `*mut T`).
3. Idiomatic Memory & Ownership: Translate C-style manual memory management, pointer arithmetic, and raw arrays into Rust's ownership model (e.g., `Vec<T>`, `Box<T>`, slices, and iterators).
4. Error Handling: Refactor C-style error codes (e.g., returning -1 or NULL) into idiomatic `Result<T, E>` or `Option<T>` patterns.
5. Standard Library Only: Do not use any external crates unless they are commonly used. Rely almost entirely on `std`.
6. Zero Comments: Do not include any code comments in your output.
7. Inline Assembly: Replace __asm__ with safe equivalents or core::arch::asm!.
8. Identifier Preservation: Preserve original names but adjust to snake_case/PascalCase standards.

Output Format:
Respond STRICTLY with a single markdown code block tagged `rust`. Do not provide any conversational filler, explanations, or text before or after the code block. Your entire response must begin exactly with ```rust and end exactly with ```."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": c_code},
        {"role": "assistant", "content": safe}
    ]


def _tokenize_for_training(tokenizer: Any, messages: List[Dict[str, str]], max_seq_length: int) -> Dict[str, Any]:
    """
    Applies the chat template to messages and tokenizes them for training.
    
    Args:
        tokenizer (Any): The Hugging Face tokenizer instance.
        messages (List[Dict[str, str]]): The formatted conversation history.
        max_seq_length (int): The maximum allowed sequence length.
    
    Returns:
        Dict[str, Any]: The tokenized inputs including input_ids and attention_mask.
    """
    rendered: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc: Dict[str, Any] = tokenizer(
        rendered,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_attention_mask=True,
    )
    return enc


def _split_data(full_ds: Dataset, seed: int, fold: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Dynamically splits pre-loaded data into Train, Validation, and Test sets based on seed and fold.
    
    Args:
        full_ds (Dataset): The complete loaded dataset.
        seed (int): The random seed for the K-Fold split.
        fold (int): The specific fold index to extract.
    
    Returns:
        Tuple[Dataset, Dataset, Dataset]: The train, validation, and test datasets respectively.
    """
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    splits: List[Tuple[Any, Any]] = list(kf.split(range(len(full_ds))))
    train_val_idx, test_idx = splits[fold]

    # Take 10% of the train_val split for validation
    val_split_idx: int = int(len(train_val_idx) * 0.9)
    train_idx = train_val_idx[:val_split_idx]
    val_idx = train_val_idx[val_split_idx:]

    train_ds: Dataset = full_ds.select(train_idx)
    val_ds: Dataset = full_ds.select(val_idx)
    test_ds: Dataset = full_ds.select(test_idx)

    return train_ds, val_ds, test_ds


def cleanup_vram(*objects: Any) -> None:
    """
    Aggressively clears VRAM between distinct evaluation runs.
    
    Args:
        *objects: Any Python objects (typically models or tokenizers) to delete from memory.
    """
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- 3. Main Loop Engine ---

def run_experiment_fold(model_type: str, seed: int, fold: int, full_ds: Dataset) -> None:
    """
    Executes a single pipeline run (Zero-shot for Baseline, SFT for Proposed).
    
    Args:
        model_type (str): The type of model to run ("Baseline" or "Proposed").
        seed (int): The random seed for reproducibility and data splitting.
        fold (int): The current fold number in the cross-validation.
        full_ds (Dataset): The complete loaded Hugging Face dataset.
    """
    # Hierarchical Directory Setup
    run_dir: Path = Path(BASE_OUTPUT_DIR) / model_type / f"seed_{seed}" / f"fold_{fold}"
    checkpoint_dir: Path = run_dir / "checkpoints"
    adapter_dir: Path = run_dir / "final_adapter"
    predictions_file: Path = run_dir / "test_predictions.json"

    # Check if this run is already completed
    if predictions_file.exists():
        print(f"\n{'='*50}\nSkipping Run: {model_type} | Seed: {seed} | Fold: {fold} (Already exists)\n{'='*50}")
        return

    print(f"\n{'='*50}\nStarting Run: {model_type} | Seed: {seed} | Fold: {fold}\n{'='*50}")

    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Split Data
    train_ds, val_ds, test_ds = _split_data(full_ds, seed, fold)

    # 2. Setup Tokenizer & Model
    tokenizer: Any = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched generation

    quantization_config: Optional[BitsAndBytesConfig] = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if USE_4BIT else None
    )

    model_kw: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if USE_BF16 else ("float16" if USE_FP16 else "auto"),
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    if USE_4BIT:
        model_kw["quantization_config"] = quantization_config
        model_kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kw)

    # Variables for cleanup
    trainer: Optional[SFTTrainer] = None
    train_tok: Optional[Dataset] = None
    val_tok: Optional[Dataset] = None

    # 3. Fine-tuning
    if model_type == "Proposed":
        print(f"--- Training {model_type} Model ---")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
            return _tokenize_for_training(tokenizer, _as_messages_proposed(ex), MAX_SEQ_LENGTH)

        num_workers: int = os.cpu_count() or 1
        train_tok = train_ds.map(map_fn, remove_columns=list(train_ds.column_names), num_proc=num_workers)
        val_tok = val_ds.map(map_fn, remove_columns=list(val_ds.column_names), num_proc=num_workers)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        response_template: str = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

        train_args = SFTConfig(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            eval_strategy="steps",
            save_strategy="steps",
            seed=seed,
            max_seq_length=MAX_SEQ_LENGTH,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            bf16=USE_BF16,
            fp16=USE_FP16,
            dataloader_num_workers=min(4, num_workers),
        )

        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            peft_config=peft_config,
            data_collator=collator,
            processing_class=tokenizer
        )

        if not DRY_RUN:
            trainer.train()
            trainer.model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2)
    else:
        print(f"--- Zero-Shot {model_type} Mode ---")
        print("Bypassing training phase to test raw base model.")

    # 4. Generate Test Set Inferences
    predictions: List[Dict[str, Optional[str]]] = []

    model.config.use_cache = True
    model.eval()

    # Iterate through the test set in chunks
    for i in tqdm(range(0, len(test_ds), INFERENCE_BATCH_SIZE), desc=f"Testing {model_type} | Seed {seed} | Fold {fold}"):
        if DRY_RUN and i > 2: 
            break

        batch_end: int = min(i + INFERENCE_BATCH_SIZE, len(test_ds))
        batch_ex: List[Dict[str, Any]] = [test_ds[j] for j in range(i, batch_end)]

        rendered_prompts: List[str] = []
        for ex in batch_ex:
            messages: List[Dict[str, str]] = _as_messages_proposed(ex) if model_type == "Proposed" else _as_messages_baseline(ex)
            prompt_msgs: List[Dict[str, str]] = messages[:-1]
            rendered_prompt: str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            rendered_prompts.append(rendered_prompt)

        inputs: Dict[str, torch.Tensor] = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():  # Faster than torch.no_grad()
            outputs: torch.Tensor = model.generate(
                **inputs,
                max_new_tokens=1536,
                pad_token_id=tokenizer.pad_token_id,  # Avoids HF warnings
                do_sample=False  # Ensures deterministic greedy decoding for coding benchmarks
            )

        prompt_lengths: int = inputs["input_ids"].shape[-1]
        generated_tokens: torch.Tensor = outputs[:, prompt_lengths:]

        decoded_preds: List[str] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for ex, pred_text in zip(batch_ex, decoded_preds):
            predictions.append({
                "original_c": ex.get("c_code"),
                "true_rust": ex.get("safe_rust_code"),
                "predicted_rust": pred_text.strip()
            })

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # 5. Crucial cleanup before next loop iteration
    cleanup_vram(model, tokenizer, trainer, train_ds, val_ds, test_ds, train_tok, val_tok)


def main() -> int:
    """
    Main entry point for the script. Handles UTF-8 mode enforcement, dataset loading,
    and loops through models, seeds, and folds for cross-validation.

    Returns:
        int: Exit status code.
    """
    if getattr(sys.flags, "utf8_mode", 0) != 1:
        cmd: List[str] = [sys.executable, "-X", "utf8", *sys.argv]
        return subprocess.call(cmd)

    # Load dataset globally once
    print("Loading dataset globally...")
    full_ds: Dataset = load_dataset("json", data_files={"train": DATASET_JSONL})["train"]

    for model_type in TEST_MODELS:
        for seed in SEEDS:
            for fold in range(NUM_FOLDS):
                run_experiment_fold(model_type, seed, fold, full_ds)

    print("\nAll 5x2cv experiments complete. Ready for external statistical testing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())