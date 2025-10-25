"""
csv_path_rewriter
-----------------

Read a CSV with columns: "file_name","transcription"
- Replace file_name with just the basename (e.g., "6_1.wav")
- Add a new "relative_path" column

"relative_path" is computed:
- If --base-dir is provided: path relative to that directory
- Otherwise: path relative to the longest common directory prefix across all file paths

The output CSV will have columns in this order:
    file_name, relative_path, transcription

Usage:
    python csv_path_rewriter.py input.csv output.csv
    python csv_path_rewriter.py input.csv output.csv --base-dir /home/devops/broadcast-inference

Notes:
- UTF-8 I/O, robust quoting, and large field sizes are supported.
- No third-party dependencies.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Record:
    file_name: str
    transcription: str


class CSVPathRewriterError(Exception):
    """Base exception for csv_path_rewriter errors."""


class MissingColumnError(CSVPathRewriterError):
    """Raised when expected CSV columns are missing."""


def _read_records(csv_path: Path) -> List[Record]:
    """Read input CSV, validating required columns."""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"file_name", "transcription"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise MissingColumnError(
                    f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
                )
            records: List[Record] = []
            for row in reader:
                records.append(
                    Record(
                        file_name=row["file_name"],
                        transcription=row["transcription"],
                    )
                )
            return records
    except FileNotFoundError as e:
        raise CSVPathRewriterError(f"Input CSV not found: {csv_path}") from e


def _longest_common_dir_prefix(paths: Iterable[Path]) -> Path:
    """Compute the longest common directory prefix (as a Path)."""
    # Use only directory parts (exclude the filename), normalize with resolve=False
    dir_strings = [str(p.parent) for p in paths]
    if not dir_strings:
        return Path(".")
    common = os.path.commonpath(dir_strings)
    return Path(common)


def _compute_relpaths(
    full_paths: List[Path], base_dir: Path | None
) -> Tuple[List[str], Path]:
    """
    Compute relative paths for each full path based on base_dir.
    If base_dir is None, use the longest common directory prefix of all inputs.
    Returns (relative_paths, actual_base_dir_used)
    """
    if base_dir is None:
        base_dir = _longest_common_dir_prefix(full_paths)

    # Convert to absolute-ish semantics for relpath without requiring existence on disk.
    base_str = str(base_dir)
    rels: List[str] = []
    for p in full_paths:
        try:
            rels.append(str(Path(os.path.relpath(str(p), base_str))))
        except ValueError:
            # On different drives (Windows edge case), fallback to path minus drive & root
            rels.append(str(p.as_posix().lstrip("/").split(":/")[-1]))
    return rels, base_dir


def transform(records: Iterable[Record], base_dir: Path | None) -> List[dict]:
    """
    Transform records:
    - file_name -> basename
    - add relative_path
    - preserve transcription
    """
    full_paths = [Path(r.file_name) for r in records]
    rels, used_base = _compute_relpaths(full_paths, base_dir)

    out_rows: List[dict] = []
    for r, rel in zip(records, rels):
        p = Path(r.file_name)
        out_rows.append(
            {
                "file_name": p.name,  # basename only
                "relative_path": rel,  # relative to base
                "transcription": r.transcription,
            }
        )
    return out_rows


def _write_output(rows: List[dict], out_csv: Path) -> None:
    """Write transformed rows to CSV with a stable column order."""
    fieldnames = ["file_name", "relative_path", "transcription"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite CSV paths: basename file_name + relative_path column."
    )
    parser.add_argument("input_csv", type=Path, help="Path to input CSV.")
    parser.add_argument("output_csv", type=Path, help="Path to output CSV.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for computing relative_path. "
            "If omitted, the longest common directory prefix of all file paths is used."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    records = _read_records(args.input_csv)
    rows = transform(records, args.base_dir)
    _write_output(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
