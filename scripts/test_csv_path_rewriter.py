import csv
from pathlib import Path

import pytest

from csv_path_rewriter import (
    Record,
    transform,
    _compute_relpaths,
    _longest_common_dir_prefix,
    MissingColumnError,
    _read_records,
)


def test_longest_common_dir_prefix():
    paths = [
        Path("/a/b/c/d/file1.wav"),
        Path("/a/b/c/e/file2.wav"),
        Path("/a/b/c/file3.wav"),
    ]
    assert _longest_common_dir_prefix(paths) == Path("/a/b/c")


def test_compute_relpaths_with_base():
    full_paths = [
        Path("/root/x/y/1.wav"),
        Path("/root/x/y/z/2.wav"),
    ]
    rels, used = _compute_relpaths(full_paths, Path("/root/x"))
    assert rels == ["y/1.wav", "y/z/2.wav"]
    assert used == Path("/root/x")


def test_compute_relpaths_auto_base():
    full_paths = [
        Path("/root/x/y/1.wav"),
        Path("/root/x/y/z/2.wav"),
    ]
    rels, used = _compute_relpaths(full_paths, None)
    assert used == Path("/root/x/y")
    assert rels == ["1.wav", "z/2.wav"]


def test_transform_basename_and_relative():
    records = [
        Record("/data/a/b/clip1.wav", "hello"),
        Record("/data/a/b/c/clip2.wav", "world"),
    ]
    rows = transform(records, base_dir=Path("/data/a/b"))
    assert rows[0]["file_name"] == "clip1.wav"
    assert rows[0]["relative_path"] == "clip1.wav"
    assert rows[0]["transcription"] == "hello"
    assert rows[1]["file_name"] == "clip2.wav"
    assert rows[1]["relative_path"] == "c/clip2.wav"
    assert rows[1]["transcription"] == "world"


def test_read_records_missing_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wrong", "columns"])
        writer.writerow(["a", "b"])
    with pytest.raises(MissingColumnError):
        _ = _read_records(p)


def test_end_to_end(tmp_path: Path):
    input_csv = tmp_path / "in.csv"
    output_csv = tmp_path / "out.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_name", "transcription"])
        writer.writerow(["/home/devops/a/b/6.wav/6_1.wav", "Тест 1"])
        writer.writerow(["/home/devops/a/b/c/7.wav/7_2.wav", "Тест 2"])

    # Auto base-dir
    from csv_path_rewriter import main

    assert main([str(input_csv), str(output_csv)]) == 0

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == ["file_name", "relative_path", "transcription"]
    assert rows[0]["file_name"] == "6_1.wav"
    assert rows[0]["transcription"] == "Тест 1"
    # base is /home/devops/a/b in auto mode (common dir), so rels are:
    # '6.wav/6_1.wav' and 'c/7.wav/7_2.wav'
    assert rows[0]["relative_path"].endswith("6.wav/6_1.wav")
    assert rows[1]["file_name"] == "7_2.wav"
    assert rows[1]["relative_path"].endswith("c/7.wav/7_2.wav")
