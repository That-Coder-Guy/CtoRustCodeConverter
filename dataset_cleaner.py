from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter_c as tsc
import tree_sitter_rust as tsr
from tree_sitter import Language, Node, Parser

_REPO_ROOT = Path(__file__).resolve().parent

# Fixed I/O layout (edit here to change paths)
ORIGINAL_DATASETS_DIR: Path = _REPO_ROOT / "original_datasets"
CLEANED_DATASETS_DIR: Path = _REPO_ROOT / "cleaned_datasets"

_CODETRANS_OCEAN_DIR = "CodeTransOcean_dataset"
_CODETRANS_DROP_KEYS = frozenset[str]({"id"})

_ALLOWED_LANGUAGE_NAMES = frozenset[str]({"C", "Rust"})
# NO-BREAK SPACE (common in web/wiki-sourced text); all occurrences are removed from code
_NBSP = "\u00a0"

_C_COMMENT_TYPES = frozenset({"comment"})
_RUST_COMMENT_TYPES = frozenset(
    {"line_comment", "block_comment", "doc_comment"}
)


@lru_cache(maxsize=1)
def _language_c() -> Language:
    return Language(tsc.language())


@lru_cache(maxsize=1)
def _language_rust() -> Language:
    return Language(tsr.language())


@lru_cache(maxsize=1)
def _parser_c() -> Parser:
    return Parser(_language_c())


@lru_cache(maxsize=1)
def _parser_rust() -> Parser:
    return Parser(_language_rust())


def _collect_comment_ranges(
    node: Node, types: frozenset[str], out: list[tuple[int, int]]
) -> None:
    if node.type in types:
        out.append((node.start_byte, node.end_byte))
        return
    for child in node.children:
        _collect_comment_ranges(child, types, out)


def _strip_comment_ranges(source: bytes, ranges: list[tuple[int, int]]) -> bytes:
    if not ranges:
        return source
    b = bytearray(source)
    for start, end in sorted(ranges, reverse=True):
        del b[start:end]
    return bytes(b)


def _strip_c_comments(code: str) -> str:
    if not code:
        return code
    data = code.encode("utf-8")
    tree = _parser_c().parse(data)
    ranges: list[tuple[int, int]] = []
    _collect_comment_ranges(tree.root_node, _C_COMMENT_TYPES, ranges)
    return _strip_comment_ranges(data, ranges).decode("utf-8")


def _strip_rust_comments(code: str) -> str:
    if not code:
        return code
    data = code.encode("utf-8")
    tree = _parser_rust().parse(data)
    ranges: list[tuple[int, int]] = []
    _collect_comment_ranges(tree.root_node, _RUST_COMMENT_TYPES, ranges)
    return _strip_comment_ranges(data, ranges).decode("utf-8")


def _normalize_code(code: str | None) -> str | None:
    if code is None:
        return None
    return code.replace(_NBSP, "").strip()


def _finalize_snippet(code: str | None, lang: str) -> str | None:
    """NBSP removal, comment stripping (tree-sitter), outer strip."""
    if code is None:
        return None
    text = _normalize_code(code)
    if not text:
        return None
    if lang == "C":
        text = _strip_c_comments(text)
    elif lang == "Rust":
        text = _strip_rust_comments(text)
    else:
        raise ValueError(f"unknown language: {lang!r}")
    return text.strip()


def _group_c_rust_by_task(
    inp: Path, batch_size: int
) -> dict[str, dict[str, str | None]]:
    """Map task_name -> {'C': code|None, 'Rust': code|None} (last wins per language)."""
    by_task: dict[str, dict[str, str | None]] = {}
    pf = pq.ParquetFile(inp)
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in pa.Table.from_batches([batch]).to_pylist():
            lang = row.get("language_name")
            if lang not in _ALLOWED_LANGUAGE_NAMES:
                continue
            task_name = row.get("task_name")
            if task_name is None:
                continue
            raw_code = row.get("code")
            code = (
                _finalize_snippet(raw_code, lang)
                if isinstance(raw_code, str)
                else None
            )
            if task_name not in by_task:
                by_task[task_name] = {"C": None, "Rust": None}
            by_task[task_name][lang] = code
    return by_task


def clean_parquet_rosetta(src: Path, dest: Path, batch_size: int) -> int:
    """Rosetta parquet: C/Rust by task_name, NBSP removed, strip; require both snippets.
    Writes JSONL to ``dest``. Returns number of lines written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    by_task = _group_c_rust_by_task(src, batch_size)
    n_written = 0
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for task_name in sorted(by_task.keys()):
            slots = by_task[task_name]
            c_code = slots.get("C")
            rust_code = slots.get("Rust")
            if not c_code or not rust_code:
                continue
            record = _normalize_record_descriptor(
                {"task_name": task_name, "C": c_code, "Rust": rust_code}
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


def _dedup_key(obj: dict) -> tuple:
    """Identify a row by its C and Rust snippets (exact string equality)."""
    return (obj.get("C"), obj.get("Rust"))


def _normalize_record_descriptor(obj: dict) -> dict:
    """Use ``task_name`` as the canonical label for the task; migrate from ``name`` if needed."""
    if "task_name" in obj:
        tn = obj["task_name"]
    elif "name" in obj:
        tn = obj["name"]
    else:
        tn = None
    rest = {k: v for k, v in obj.items() if k not in ("name", "task_name")}
    if tn is not None:
        return {"task_name": tn, **rest}
    return dict(obj)


def clean_jsonl_dedup(
    src: Path,
    dest: Path,
    *,
    drop_keys: frozenset[str] | None = None,
) -> tuple[int, int, int]:
    """Keep rows with both C and Rust; deduplicate by (C, Rust).
    Optional ``drop_keys`` are removed from each written object (after normalization).
    Returns (lines_read, rows_with_c_and_rust, unique_rows_written)."""
    lines_read = 0
    with_c_rust = 0
    seen: set[tuple] = set()
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dest.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            lines_read += 1
            obj = json.loads(line)
            if "C" not in obj or "Rust" not in obj:
                continue
            c_raw, r_raw = obj["C"], obj["Rust"]
            if not isinstance(c_raw, str) or not isinstance(r_raw, str):
                continue
            c_fin = _finalize_snippet(c_raw, "C")
            r_fin = _finalize_snippet(r_raw, "Rust")
            if not c_fin or not r_fin:
                continue
            obj = {**obj, "C": c_fin, "Rust": r_fin}
            with_c_rust += 1
            key = _dedup_key(obj)
            if key in seen:
                continue
            seen.add(key)
            normalized = _normalize_record_descriptor(obj)
            if drop_keys:
                for k in drop_keys:
                    normalized.pop(k, None)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1
    return lines_read, with_c_rust, written


def _iter_dataset_subdirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _iter_cleanable_files(dataset_dir: Path) -> list[Path]:
    """Supported inputs directly under a dataset folder (not nested subfolders)."""
    out: list[Path] = []
    for p in sorted(dataset_dir.iterdir()):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in (".parquet", ".jsonl"):
            out.append(p)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Clean datasets under ORIGINAL_DATASETS_DIR: each subfolder is processed; "
            ".parquet files use Rosetta C/Rust merge + normalization; .jsonl files use "
            "C/Rust deduplication. Task labels are normalized to the key task_name "
            "(e.g. name -> task_name). CodeTransOcean JSONL rows have id removed. "
            "C and Rust snippets have comments stripped (tree-sitter). "
            "Outputs mirror folder layout under CLEANED_DATASETS_DIR."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Parquet read batch size (lower uses less memory)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    source_root: Path = ORIGINAL_DATASETS_DIR.resolve()
    dest_root: Path = CLEANED_DATASETS_DIR.resolve()

    if not source_root.is_dir():
        print(f"Source root not found or not a directory: {source_root}", file=sys.stderr)
        return 1

    subdirs = _iter_dataset_subdirs(source_root)
    if not subdirs:
        print(f"No dataset subfolders in {source_root}")
        return 0

    print(f"Source: {source_root}\nDestination: {dest_root}\n")

    total_files = 0
    for dataset_dir in subdirs:
        rel_name = dataset_dir.name
        files = _iter_cleanable_files(dataset_dir)
        if not files:
            print(f"{rel_name}/: no .parquet or .jsonl files, skip")
            continue

        out_dir = dest_root / rel_name
        for src in files:
            total_files += 1
            suf = src.suffix.lower()
            dest_file = out_dir / src.name
            if suf == ".parquet":
                dest_jsonl = dest_file.with_suffix(".jsonl")
                n = clean_parquet_rosetta(src, dest_jsonl, args.batch_size)
                print(
                    f"{rel_name}/{src.name} [parquet -> jsonl]: "
                    f"{n} lines -> {dest_jsonl}"
                )
            elif suf == ".jsonl":
                drop_keys = (
                    _CODETRANS_DROP_KEYS
                    if rel_name == _CODETRANS_OCEAN_DIR
                    else None
                )
                lines_read, with_c_rust, written = clean_jsonl_dedup(
                    src, dest_file, drop_keys=drop_keys
                )
                dups = with_c_rust - written
                print(
                    f"{rel_name}/{src.name} [jsonl dedup]: "
                    f"{lines_read} lines read, {with_c_rust} with both C and Rust, "
                    f"{written} unique written"
                    + (f", {dups} duplicates dropped" if dups else "")
                    + f" -> {dest_file}"
                )

    if total_files == 0:
        print("No .parquet or .jsonl files found under any dataset subfolder.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
