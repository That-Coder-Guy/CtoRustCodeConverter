from __future__ import annotations

import json
import os

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SOURCE = os.path.join(_TRAINING_DIR, "CodeTransOcean_dataset")
_DEFAULT_DEST = os.path.join(_TRAINING_DIR, "training_dataset")


def _iter_jsonl_paths(source_dir: str) -> list[str]:
    names = sorted(
        n for n in os.listdir(source_dir)
        if n.endswith(".jsonl") and os.path.isfile(os.path.join(source_dir, n))
    )
    return [os.path.join(source_dir, n) for n in names]


def _dedup_key(obj: dict) -> tuple:
    """Identify a row by its C and Rust snippets (exact string equality)."""
    return (obj.get("C"), obj.get("Rust"))


def filter_file(src_path: str, dest_path: str) -> tuple[int, int, int]:
    """Returns (lines_read, rows_with_c_and_rust, unique_rows_written)."""
    lines_read = 0
    with_c_rust = 0
    seen: set[tuple] = set()
    written = 0
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(src_path, encoding="utf-8") as fin, open(
        dest_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            lines_read += 1
            obj = json.loads(line)
            if "C" not in obj or "Rust" not in obj:
                continue
            with_c_rust += 1
            key = _dedup_key(obj)
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return lines_read, with_c_rust, written


def main() -> None:
    source = _DEFAULT_SOURCE
    dest_root = _DEFAULT_DEST
    paths = _iter_jsonl_paths(source)
    if not paths:
        print(f"No .jsonl files found in {source}")
        return

    print(f"Source: {source}\nDestination: {dest_root}\n")
    for src in paths:
        name = os.path.basename(src)
        out = os.path.join(dest_root, name)
        lines_read, with_c_rust, written = filter_file(src, out)
        dups = with_c_rust - written
        print(
            f"{name}: {lines_read} lines read, {with_c_rust} with C+Rust, "
            f"{written} unique written"
            + (f", {dups} duplicates dropped" if dups else "")
            + f" -> {out}"
        )


if __name__ == "__main__":
    main()
