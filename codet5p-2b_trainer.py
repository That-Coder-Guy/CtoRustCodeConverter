import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Configuration ---
MODEL_NAME = "Salesforce/codet5p-2b" # "Salesforce/codet5p-770m"
TRAIN_FILE = os.path.join(_TRAINING_DIR, "training_dataset", "niche_train.jsonl")
VALID_FILE = os.path.join(_TRAINING_DIR, "training_dataset", "niche_valid.jsonl")
OUTPUT_DIR = os.path.join(_TRAINING_DIR, "training_results_codet5p-2b")


def _lora_target_modules(model_name: str) -> list[str]:
    if "codet5p-2b" in model_name:
        return ["qkv_proj", "out_proj", "q_attn"]
    return ["q", "v"]


def _ensure_seq2seq_config_ids(model, tokenizer) -> None:
    """codet5p-2b may omit decoder_start_token_id; DataCollatorForSeq2Seq requires it."""
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    cfg = model.config
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = pad
    if getattr(cfg, "decoder_start_token_id", None) is None:
        cfg.decoder_start_token_id = pad


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    clean_up_tokenization_spaces=False,
    trust_remote_code="codet5p-2b" in MODEL_NAME,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code="codet5p-2b" in MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
if hasattr(model, "tie_weights"):
    model.tie_weights()

# 2. Setup LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=_lora_target_modules(MODEL_NAME),
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
_ensure_seq2seq_config_ids(model, tokenizer)

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

# 3. Prepare the Datasets
print("\nLoading and formatting datasets...")
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
valid_dataset = load_dataset("json", data_files=VALID_FILE, split="train")

def preprocess_function(examples):
    inputs = ["Translate this C code to Rust:\n" + c for c in examples["C"]]
    targets = examples["Rust"]

    model_inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=256, padding="max_length", truncation=True)

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_valid = valid_dataset.map(preprocess_function, batched=True, remove_columns=valid_dataset.column_names)

# 4. Configure the Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=10,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # <-- 3. Add the Callback
)

# 5. Train and Save
print("\nStarting the training loop! Early stopping is active.")
trainer.train()

print("\nSaving the absolute best LoRA adapter...")
_final = os.path.join(OUTPUT_DIR, "final_model")
trainer.model.save_pretrained(_final)
tokenizer.save_pretrained(_final)
print("Training complete! Your optimal model is ready.")