"""Run dataset cleaning, merge/split, CodeT5+ 2B training, then 2B evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_STEPS = (
    "dataset_cleaner.py",
    "dataset_merger.py",
    "codet5p-2b_trainer.py",
    "codet5p-2b_evaluator.py",
)


def main() -> int:
    py = sys.executable
    for i, name in enumerate(_STEPS, start=1):
        script = _REPO_ROOT / name
        if not script.is_file():
            print(f"Missing script: {script}", file=sys.stderr)
            return 1
        print(f"\n{'=' * 60}\n[{i}/{len(_STEPS)}] {name}\n{'=' * 60}\n")
        completed = subprocess.run([py, str(script)], cwd=_REPO_ROOT)
        if completed.returncode != 0:
            print(
                f"\nPipeline stopped: {name} exited with {completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode
    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
