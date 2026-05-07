from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---

BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Update this to match the specific seed folder you generated in the previous script
ADAPTER_DIR: str = "./training_results/seed_42/final_adapter"

# Generation settings
MAX_NEW_TOKENS: int = 1536
TEMPERATURE: float = 0.0


def setup_model_and_tokenizer(base_model_id: str, adapter_path: str) -> tuple[Any, Any]:
    """
    Loads the base causal LM, applies the trained LoRA adapter, and sets up the tokenizer.

    Args:
        base_model_id (str): The Hugging Face hub ID of the base model.
        adapter_path (str): The local directory containing the fine-tuned LoRA weights.

    Returns:
        tuple[Any, Any]: A tuple containing the loaded model and tokenizer.
    """
    print(f"Loading tokenizer: {base_model_id}")
    tokenizer: Any = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model: {base_model_id}")
    # Load base model in bfloat16 for optimal Ampere GPU performance
    base_model: Any = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Loading and applying LoRA adapter from: {adapter_path}")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Could not find adapter at {adapter_path}. Did the training script finish?")
        
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def format_prompt(c_code: str) -> List[Dict[str, str]]:
    """
    Wraps the input C code in the exact system and user prompt constraints used during training.

    Args:
        c_code (str): The raw C code snippet to translate.

    Returns:
        List[Dict[str, str]]: The chat conversation list ready for the chat template.
    """
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
        {"role": "user", "content": c_code}
    ]


def generate_translation(model: Any, tokenizer: Any, c_code: str) -> str:
    """
    Runs the forward pass to generate the Rust translation for a given C code snippet.

    Args:
        model (Any): The LoRA-adapted PeftModel.
        tokenizer (Any): The corresponding tokenizer.
        c_code (str): The C code to translate.

    Returns:
        str: The raw model output (which should be a Rust markdown block based on training).
    """
    messages: List[Dict[str, str]] = format_prompt(c_code)
    
    rendered_prompt: str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs: Dict[str, torch.Tensor] = tokenizer(
        rendered_prompt, 
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs: torch.Tensor = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
        )

    # Slice the output to exclude the original prompt tokens
    prompt_length: int = inputs["input_ids"].shape[-1]
    generated_tokens: torch.Tensor = outputs[0, prompt_length:]
    
    decoded_output: str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded_output.strip()


def main() -> int:
    """
    Main entry point for the runner script.
    """
    # Sample C Code to test the model
    sample_c_code: str = """
#include <stdio.h>
#include <stdlib.h>

int sum_array(int *arr, int size) {
    if (arr == NULL || size <= 0) {
        return -1;
    }
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}
    """

    print("=== C to Rust Translation Runner ===")
    
    try:
        model, tokenizer = setup_model_and_tokenizer(BASE_MODEL, ADAPTER_DIR)
    except Exception as e:
        print(f"\nError initializing model: {e}")
        return 1

    print("\n--- Input C Code ---")
    print(sample_c_code.strip())
    
    print("\nGenerating translation...\n")
    prediction: str = generate_translation(model, tokenizer, sample_c_code)

    print("--- Predicted Rust Code ---")
    print(prediction)
    print("---------------------------\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())