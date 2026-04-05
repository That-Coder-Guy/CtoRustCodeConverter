from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Fixed I/O layout (edit here to change paths)
CLEANED_DATASETS_DIR: Path = _REPO_ROOT / "cleaned_datasets"
TRAINING_DATASETS_DIR: Path = _REPO_ROOT / "training_datasets"


def _dedup_key(obj: dict) -> tuple:
    return (obj.get("C"), obj.get("Rust"))


def _iter_jsonl_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.jsonl"))


def _load_merge_dedup(source_root: Path) -> tuple[list[dict], int]:
    """Load every object from all ``*.jsonl`` under ``source_root``; dedupe by (C, Rust).
    Returns (merged rows, number of non-empty lines read)."""
    seen: set[tuple] = set()
    merged: list[dict] = []
    lines_read = 0
    for path in _iter_jsonl_files(source_root):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                lines_read += 1
                obj = json.loads(line)
                if "C" not in obj or "Rust" not in obj:
                    continue
                key = _dedup_key(obj)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(obj)
    return merged, lines_read


def _split_sizes(n: int, train_r: float, valid_r: float, test_r: float) -> tuple[int, int, int]:
    if n == 0:
        return 0, 0, 0
    total = train_r + valid_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    n_train = int(n * train_r)
    n_valid = int(n * valid_r)
    n_test = n - n_train - n_valid
    if n_test < 0:
        n_valid += n_test
        n_test = 0
    return n_train, n_valid, n_test


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge all JSONL entries under CLEANED_DATASETS_DIR (recursively), deduplicate "
            "by (C, Rust), shuffle, and split into train / valid / test under TRAINING_DATASETS_DIR."
        ),
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of rows for training (after dedup and shuffle)",
    )
    p.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction for test (remaining mass goes here if rounding leaves slack)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling before split",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    source_root = CLEANED_DATASETS_DIR.resolve()
    dest_dir = TRAINING_DATASETS_DIR.resolve()

    if not source_root.is_dir():
        print(f"Source not found: {source_root}", file=sys.stderr)
        return 1

    rows, lines_read = _load_merge_dedup(source_root)
    print(f"Non-empty lines read (all files): {lines_read}")
    print(f"Unique (C, Rust) rows after dedup: {len(rows)}")

    if not rows:
        print("No valid rows; nothing written.")
        return 0

    random.seed(args.seed)
    random.shuffle(rows)

    n_train, n_valid, n_test = _split_sizes(
        len(rows), args.train_ratio, args.valid_ratio, args.test_ratio
    )
    i0 = 0
    i1 = i0 + n_train
    i2 = i1 + n_valid
    train_rows = rows[i0:i1]
    valid_rows = rows[i1:i2]
    test_rows = rows[i2 : i2 + n_test]

    out_train = dest_dir / "train.jsonl"
    out_valid = dest_dir / "valid.jsonl"
    out_test = dest_dir / "test.jsonl"
    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_valid, valid_rows)
    _write_jsonl(out_test, test_rows)

    print(
        f"Wrote train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)} "
        f"-> {dest_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
