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
BASE_OUTPUT_DIR: str = "./training_results"

# Single Run Requirements
SEED: int = 42

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

def _as_messages_proposed(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Formats the dataset example using the complex system prompt and rules.

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


def _split_data(full_ds: Dataset, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits the pre-loaded data into an 80/10/10 Train, Validation, and Test set.

    Args:
        full_ds (Dataset): The complete loaded dataset.
        seed (int): The random seed for deterministic splitting.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The train, validation, and test datasets respectively.
    """
    # 1. Split off 20% for validation and testing combined (80% train)
    train_test_split = full_ds.train_test_split(test_size=0.2, seed=seed)
    train_ds: Dataset = train_test_split["train"]
    temp_val_test_ds: Dataset = train_test_split["test"]

    # 2. Split the 20% perfectly in half (10% valid, 10% test)
    val_test_split = temp_val_test_ds.train_test_split(test_size=0.5, seed=seed)
    val_ds: Dataset = val_test_split["train"]
    test_ds: Dataset = val_test_split["test"]

    print(f"Data Split Output - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def cleanup_vram(*objects: Any) -> None:
    """
    Aggressively clears VRAM after the evaluation run to free resources.

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

def run_training_pipeline(seed: int, full_ds: Dataset) -> None:
    """
    Executes a single end-to-end SFT and inference pipeline.

    Args:
        seed (int): The random seed for reproducibility and data splitting.
        full_ds (Dataset): The complete loaded Hugging Face dataset.
    """
    # Hierarchical Directory Setup
    run_dir: Path = Path(BASE_OUTPUT_DIR) / f"seed_{seed}"
    checkpoint_dir: Path = run_dir / "checkpoints"
    adapter_dir: Path = run_dir / "final_adapter"
    predictions_file: Path = run_dir / "test_predictions.json"

    # Check if this run is already completed
    if predictions_file.exists():
        print(f"\n{'='*50}\nSkipping Run: Seed {seed} (Already exists)\n{'='*50}")
        return

    print(f"\n{'='*50}\nStarting Training Run | Seed: {seed}\n{'='*50}")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Split Data
    train_ds, val_ds, test_ds = _split_data(full_ds, seed)

    # 2. Setup Tokenizer & Model
    tokenizer: Any = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched generation

    quantization_config: Optional[BitsAndBytesConfig] = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if USE_4BIT else None
    )

    # device_map="auto" can leave meta/offloaded shards; SFTTrainer then calls model.to(device) and crashes.
    # Non-quantized: omit device_map so weights load normally and the Trainer moves the model.
    model_kw: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if USE_BF16 else ("float16" if USE_FP16 else "auto"),
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }
    if USE_4BIT:
        model_kw["quantization_config"] = quantization_config
        model_kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kw)

    # 3. Fine-tuning setup
    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        return _tokenize_for_training(tokenizer, _as_messages_proposed(ex), MAX_SEQ_LENGTH)

    num_workers: int = os.cpu_count() or 1
    train_tok: Dataset = train_ds.map(map_fn, remove_columns=list(train_ds.column_names), num_proc=num_workers)
    val_tok: Dataset = val_ds.map(map_fn, remove_columns=list(val_ds.column_names), num_proc=num_workers)

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

    # 4. Train the Model
    if not DRY_RUN:
        trainer.train()
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        with open(run_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2)

    # 5. Generate Test Set Inferences (Optimized Batched Generation)
    predictions: List[Dict[str, Optional[str]]] = []

    model.config.use_cache = True
    model.eval()

    # Iterate through the test set in chunks
    for i in tqdm(range(0, len(test_ds), INFERENCE_BATCH_SIZE), desc=f"Testing Inference | Seed {seed}"):
        if DRY_RUN and i > 2: 
            break

        batch_end: int = min(i + INFERENCE_BATCH_SIZE, len(test_ds))
        batch_ex: List[Dict[str, Any]] = [test_ds[j] for j in range(i, batch_end)]

        rendered_prompts: List[str] = []
        for ex in batch_ex:
            messages: List[Dict[str, str]] = _as_messages_proposed(ex)
            prompt_msgs: List[Dict[str, str]] = messages[:-1]
            rendered_prompt: str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            rendered_prompts.append(rendered_prompt)

        inputs: Dict[str, torch.Tensor] = tokenizer(rendered_prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            outputs: torch.Tensor = model.generate(
                **inputs,
                max_new_tokens=1536,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
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

    # 6. Crucial cleanup
    cleanup_vram(model, tokenizer, trainer, train_ds, val_ds, test_ds, train_tok, val_tok)


def main() -> int:
    """
    Main entry point for the script. Handles UTF-8 mode enforcement, dataset loading,
    and initiates the single training pipeline run.

    Returns:
        int: Exit status code.
    """
    if getattr(sys.flags, "utf8_mode", 0) != 1:
        cmd: List[str] = [sys.executable, "-X", "utf8", *sys.argv]
        return subprocess.call(cmd)

    print("Loading dataset globally...")
    full_ds: Dataset = load_dataset("json", data_files={"train": DATASET_JSONL})["train"]

    run_training_pipeline(SEED, full_ds)

    print("\nExperiment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())