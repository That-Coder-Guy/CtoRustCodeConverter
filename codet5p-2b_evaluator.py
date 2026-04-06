"""Codet5p-2b test evaluation: base model vs LoRA (see codet5_eval_common)."""

from __future__ import annotations

import os

from codet5_eval_common import run_adapted_eval

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    raise SystemExit(
        run_adapted_eval(
            base_model="Salesforce/codet5p-2b",
            lora_adapter=os.path.join(
                _REPO_ROOT, "training_results", "codet5p-2b", "final_model"
            ),
            trust_remote_code=True,
        )
    )
