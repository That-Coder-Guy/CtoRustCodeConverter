"""
Stock (non–LoRA) inference for Salesforce/codet5p-2b.

Uses Hub weights only. For a trained adapter under training_results_codet5p-2b/,
use adapted_codet5p-2b_runner.py instead.
"""

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Salesforce/codet5p-2b"
TRUST_REMOTE = True

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} onto {device}...")

model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=TRUST_REMOTE)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    clean_up_tokenization_spaces=False,
    trust_remote_code=TRUST_REMOTE,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    config=model_config,
    torch_dtype=torch.float16,
    trust_remote_code=TRUST_REMOTE,
    device_map="auto",
)
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

# Baseline cue for zero-shot (wording independent of LoRA training)
BASELINE_INSTRUCTION = "Convert the following C code to Rust:\n"
c_code_snippet = """
int multiply(int a, int b) {
    return a * b;
}
"""
prompt = BASELINE_INSTRUCTION + c_code_snippet.strip()

print("Tokenizing input and generating response...")
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

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== AI OUTPUT ===")
print(generated_code)
print("=================\n")
