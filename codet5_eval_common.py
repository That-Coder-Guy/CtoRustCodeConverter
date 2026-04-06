"""Shared test-set evaluation: base (pre–fine-tuning) vs LoRA (post–fine-tuning) on the same data."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _slug_from_base_model(base_model: str) -> str:
    return base_model.rsplit("/", 1)[-1].replace(".", "_")


def _default_results_path(base_model: str) -> str:
    out_dir = os.path.join(_REPO_ROOT, "evaluation_results")
    return os.path.join(out_dir, f"{_slug_from_base_model(base_model)}_test_eval.json")


def _json_safe_metrics(m: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in m.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


def _default_test_path() -> str:
    return os.path.join(_REPO_ROOT, "training_datasets", "test.jsonl")


def load_test_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "C" not in obj or "Rust" not in obj:
                continue
            rows.append(obj)
    return rows


def _normalize_for_exact_match(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Corpus BLEU + chrF (sacrebleu) and exact-match rate on normalized strings."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references length mismatch")
    n = len(predictions)
    if n == 0:
        return {
            "exact_match": 0.0,
            "bleu": 0.0,
            "chrf": 0.0,
            "num_examples": 0.0,
        }

    em = sum(
        1
        for p, r in zip(predictions, references, strict=True)
        if _normalize_for_exact_match(p) == _normalize_for_exact_match(r)
    )
    try:
        from sacrebleu import corpus_bleu, corpus_chrf

        bleu = corpus_bleu(predictions, [references])
        bleu_score = float(bleu.score)
        chrf = corpus_chrf(predictions, [references])
        chrf_score = float(chrf.score)
    except ImportError:
        bleu_score = float("nan")
        chrf_score = float("nan")

    return {
        "exact_match": 100.0 * em / n,
        "bleu": bleu_score,
        "chrf": chrf_score,
        "num_examples": float(n),
    }


def _ensure_seq2seq_config_ids(model: Any, tokenizer: Any) -> None:
    pad = tokenizer.pad_token_id
    if pad is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad = tokenizer.pad_token_id
    cfg = model.config
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = pad
    if getattr(cfg, "decoder_start_token_id", None) is None:
        cfg.decoder_start_token_id = pad


def _generate_all(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    input_prefix: str,
    device: str,
    embed_device: torch.device,
    max_new_tokens: int,
    desc: str,
) -> list[str]:
    preds: list[str] = []
    for row in tqdm(rows, desc=desc, unit="ex"):
        c_src = str(row["C"]).strip()
        prompt = input_prefix + c_src
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device == "cuda":
            inputs = {k: v.to(embed_device) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        preds.append(gen)
    return preds


def run_adapted_eval(
    *,
    base_model: str,
    lora_adapter: str,
    trust_remote_code: bool,
    input_prefix: str = "",
    test_jsonl: str | None = None,
    max_new_tokens: int = 512,
    output_path: str | None = None,
    eval_base_model: bool = True,
) -> int:
    """Load base, run test set, then wrap LoRA and run again; print/write both metrics.

    ``input_prefix`` is empty by default (raw ``C`` field only). Trainers may use a
    task string during training; pass it here only if you want eval to match that.
    """
    test_path = test_jsonl or _default_test_path()
    results_file = output_path or _default_results_path(base_model)
    if not os.path.isfile(test_path):
        print(f"Test file not found: {test_path}", file=sys.stderr)
        return 1
    if not os.path.isdir(lora_adapter):
        print(
            f"LoRA adapter folder not found: {lora_adapter}",
            file=sys.stderr,
        )
        return 1

    rows = load_test_jsonl(test_path)
    if not rows:
        print(f"No valid rows in {test_path}")
        return 0

    references = [str(row["Rust"]) for row in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {base_model} onto {device}...")
    model_kw: dict = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kw["device_map"] = "auto"
    if trust_remote_code:
        model_kw["trust_remote_code"] = True

    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, **model_kw)
    if hasattr(base, "tie_weights"):
        base.tie_weights()

    tok_kw: dict = {"clean_up_tokenization_spaces": False}
    if trust_remote_code:
        tok_kw["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter, **tok_kw)
    _ensure_seq2seq_config_ids(base, tokenizer)

    embed_device = base.get_input_embeddings().weight.device
    base.eval()

    preds_base: list[str] = []
    metrics_base: dict[str, float] | None = None
    if eval_base_model:
        print("\n--- Base model (pre fine-tuning, no LoRA) ---")
        preds_base = _generate_all(
            base,
            tokenizer,
            rows,
            input_prefix,
            device,
            embed_device,
            max_new_tokens,
            desc="Base model",
        )
        metrics_base = compute_metrics(preds_base, references)
        n0 = int(metrics_base["num_examples"])
        print(f"Examples: {n0}")
        print(f"Exact match (normalized): {metrics_base['exact_match']:.2f}%")
        if math.isnan(metrics_base["bleu"]):
            print("Corpus BLEU: n/a (pip install sacrebleu)")
            print("chrF++: n/a (pip install sacrebleu)")
        else:
            print(f"Corpus BLEU: {metrics_base['bleu']:.2f}")
            print(f"chrF++: {metrics_base['chrf']:.2f}")

    print(f"\nLoading LoRA from {lora_adapter}...")
    model = PeftModel.from_pretrained(base, lora_adapter)
    _ensure_seq2seq_config_ids(model, tokenizer)
    model.eval()
    embed_device = model.get_input_embeddings().weight.device

    print("\n--- Adapted model (LoRA, post fine-tuning) ---")
    preds_adapted = _generate_all(
        model,
        tokenizer,
        rows,
        input_prefix,
        device,
        embed_device,
        max_new_tokens,
        desc="LoRA model",
    )
    metrics_adapted = compute_metrics(preds_adapted, references)
    n = int(metrics_adapted["num_examples"])
    print(f"Examples: {n}")
    print(f"Exact match (normalized): {metrics_adapted['exact_match']:.2f}%")
    if math.isnan(metrics_adapted["bleu"]):
        print("Corpus BLEU: n/a (pip install sacrebleu)")
        print("chrF++: n/a (pip install sacrebleu)")
    else:
        print(f"Corpus BLEU: {metrics_adapted['bleu']:.2f}")
        print(f"chrF++: {metrics_adapted['chrf']:.2f}")

    meta_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        ref_rust = references[i]
        pb = preds_base[i] if preds_base else ""
        pa = preds_adapted[i]
        ex: dict[str, Any] = {
            "reference": ref_rust,
            "prediction_base": pb,
            "prediction_adapted": pa,
            "exact_match_base": _normalize_for_exact_match(pb)
            == _normalize_for_exact_match(ref_rust)
            if preds_base
            else None,
            "exact_match_adapted": _normalize_for_exact_match(pa)
            == _normalize_for_exact_match(ref_rust),
        }
        if "task_name" in row:
            ex["task_name"] = row["task_name"]
        meta_rows.append(ex)

    payload: dict[str, Any] = {
        "metrics": {
            "adapted_lora": _json_safe_metrics(metrics_adapted),
        },
        "config": {
            "base_model": base_model,
            "lora_adapter": lora_adapter,
            "test_jsonl": os.path.abspath(test_path),
            "max_new_tokens": max_new_tokens,
            "eval_base_model": eval_base_model,
            "input_prefix": input_prefix,
        },
        "examples": meta_rows,
    }
    if metrics_base is not None:
        payload["metrics"]["base_model"] = _json_safe_metrics(metrics_base)

    out_abs = os.path.abspath(results_file)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_abs, "w", encoding="utf-8", newline="\n") as rf:
        json.dump(payload, rf, ensure_ascii=False, indent=2)

    print(f"\nWrote results to: {out_abs}")

    return 0
