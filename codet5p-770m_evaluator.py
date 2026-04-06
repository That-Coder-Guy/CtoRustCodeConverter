"""Codet5p-770m test evaluation: base model vs LoRA (see codet5_eval_common)."""

from __future__ import annotations

import os

from codet5_eval_common import run_adapted_eval

__all__ = [
    "BASE_MODEL",
    "LORA_ADAPTER",
    "main",
    "run_adapted_eval",
]

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "Salesforce/codet5p-770m"
LORA_ADAPTER = os.path.join(
    _REPO_ROOT, "training_results", "codet5p-770m", "final_model"
)


def main() -> int:
    return run_adapted_eval(
        base_model=BASE_MODEL,
        lora_adapter=LORA_ADAPTER,
        trust_remote_code=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
