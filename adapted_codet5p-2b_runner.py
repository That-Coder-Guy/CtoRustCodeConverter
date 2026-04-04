"""
Run C -> Rust generation with a LoRA adapter trained via codet5p-2b_trainer.py.

Mirrors adapted_codet5p-770m_runner.py but uses Salesforce/codet5p-2b (custom HF code,
different attention module names in training) and training_results_codet5p-2b/final_model.
"""

import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "Salesforce/codet5p-2b"
LORA_ADAPTER = os.path.join(_TRAINING_DIR, "training_results_codet5p-2b", "final_model")
TRUST_REMOTE = True

# Same prefix as codet5p-2b_trainer.py preprocess_function
INPUT_PREFIX = "Translate this C code to Rust:\n"

device = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_seq2seq_config_ids(model, tokenizer) -> None:
    """codet5p-2b config may omit decoder_start_token_id; set for safe generation."""
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    cfg = model.config
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = pad
    if getattr(cfg, "decoder_start_token_id", None) is None:
        cfg.decoder_start_token_id = pad


print(f"Loading {BASE_MODEL} onto {device}...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=TRUST_REMOTE,
    torch_dtype=torch.float16,
    device_map="auto",
)
if hasattr(base_model, "tie_weights"):
    base_model.tie_weights()

if not os.path.isdir(LORA_ADAPTER):
    raise FileNotFoundError(
        f"LoRA adapter folder not found: {LORA_ADAPTER}\n"
        "Run codet5p-2b_trainer.py first or set LORA_ADAPTER."
    )

tokenizer = AutoTokenizer.from_pretrained(
    LORA_ADAPTER,
    clean_up_tokenization_spaces=False,
    trust_remote_code=TRUST_REMOTE,
)

print(f"Loading LoRA adapter from {LORA_ADAPTER}...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
_ensure_seq2seq_config_ids(model, tokenizer)

c_code_snippet = """
int multiply(int a, int b) {
    return a * b;
}
"""
prompt = INPUT_PREFIX + c_code_snippet.strip()

print("\nTokenizing and generating...")
inputs = tokenizer(prompt, return_tensors="pt")
if device == "cuda":
    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== AI OUTPUT ===")
print(generated)
print("=================\n")
