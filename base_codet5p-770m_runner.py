"""
Stock (non–LoRA) inference for Salesforce/codet5p-770m.

Uses Hub weights only. For a LoRA adapter saved under
training_results_codet5p-770m/final_model, use adapted_codet5p-770m_runner.py.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Salesforce/codet5p-770m"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} onto {device}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    clean_up_tokenization_spaces=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device)
if hasattr(model, "tie_weights"):
    model.tie_weights()


def _ensure_seq2seq_config_ids(model, tokenizer) -> None:
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    cfg = model.config
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = pad
    if getattr(cfg, "decoder_start_token_id", None) is None:
        cfg.decoder_start_token_id = pad


_ensure_seq2seq_config_ids(model, tokenizer)

BASELINE_INSTRUCTION = "Convert the following C code to Rust:\n"
c_code_snippet = """
int multiply(int a, int b) {
    return a * b;
}
"""
prompt = BASELINE_INSTRUCTION + c_code_snippet.strip()

print("Tokenizing input and generating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== AI OUTPUT ===")
print(generated_code)
print("=================\n")
