# audios-to-dataset

`audios-to-dataset` is a Rust CLI that turns a folder full of audio files into chunked DuckDB or Parquet datasets that mirror the layout expected by the Hugging Face `datasets` library. It is designed for fast local preparation of corpora before pushing the data to object storage or the Hub.

## Highlights

- Recursively scans your input directory (with optional MIME-type filtering) and trims the traversal depth to a configurable level.
- Batches audio into evenly sized databases with optional DuckDB or Parquet outputs and Hugging Face metadata embedded in the Parquet files.
- Pulls per-file transcripts from a companion CSV so you can ship audio + text pairs without post-processing.
- Parallelizes decoding and packing across a configurable number of threads to keep local ingestion fast.
- Populates audio duration by reading WAV headers when available, keeping non-WAV formats in the dataset with a zero duration fallback.

## Installation

The binary is published as part of the repository; you can build it locally with Cargo:

```shell
cargo install --locked --path .
```

If you prefer to run it directly from source while developing:

```shell
cargo run --release -- --help
```

## Quick Start

1. Prepare an input directory that contains audio files (WAV, MP3, FLAC, OGG, AAC, …).  
2. (Optional) Create a CSV file with the columns `file_name` and `transcription` (plus an optional `relative_path`) if you want to attach transcripts.
3. Run the CLI and point it at the input directory and an empty output folder:

```shell
audios-to-dataset \
  --input ./recordings \
  --output ./recordings-packed \
  --files-per-db 1000 \
  --format parquet
```

During execution the tool:

- creates the output directory when it is missing,
- splits the corpus into batches of `files-per-db` files,
- writes files named `0.parquet`, `1.parquet`, … (or `.duckdb`) into the output directory, and
- prints progress for each chunk.

## Working With Transcriptions

Use `--metadata-file path/to/metadata.csv` to provide transcripts. The CSV is expected to have headers and contain:

| column           | description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `file_name`      | Base file name matching the audio files found during the scan.              |
| `relative_path`* | Path (relative to `--input`) for disambiguating duplicate file names.       |
| `transcription`  | Text transcription that will be associated with the audio sample.           |

`*` Optional. When you have multiple files with the same name under different subdirectories, include a `relative_path` column (use forward slashes) so each transcription maps to the correct file.

Rows without a matching audio file are skipped. When a match is missing the CLI stores `"-"` as the placeholder transcript.

## Command Reference

```
audios-to-dataset [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
      --input <INPUT>                 Directory to scan (recursively by default)
      --format <FORMAT>               Output format [default: parquet] [duck-db, parquet]
      --files-per-db <N>              Number of audio files per output shard [default: 500]
      --max-depth-size <N>            Maximum recursion depth when scanning [default: 50]
      --check-mime-type               Skip files whose MIME type is not audio/*
      --num-threads <N>               Worker threads used for processing [default: 5]
      --output <OUTPUT>               Destination folder for `.parquet` / `.duckdb` files
      --parquet-compression <TYPE>    Compression (Snappy, Zstd, Gzip, …) [default: snappy]
      --metadata-file <CSV>           CSV with `file_name` + `transcription` columns (and optional `relative_path`)
  -h, --help                          Print help
  -V, --version                       Print version
```

## Output Details

- **Parquet** files contain a struct column named `audio` with `bytes`, `sampling_rate`, and `path` (relative to your `--input`), plus `duration` and `transcription` columns. Hugging Face-specific metadata is embedded in the Parquet schema so that the Hub Data Viewer recognizes the dataset automatically.
- **DuckDB** files create a `files` table with the same schema. Existing files in the output directory are replaced shard by shard.
- When `--check-mime-type` is enabled the CLI keeps a curated allow list of audio MIME types; others are skipped with a log line.
- Durations are derived from WAV headers; non-WAV files remain with `duration = 0.0`.

## Examples

Produce DuckDB shards of 250 files and verify MIME types before packing:

```shell
audios-to-dataset \
  --format duckdb \
  --files-per-db 250 \
  --check-mime-type \
  --input ./speech-corpus \
  --output ./speech-corpus-duckdb
```

Create Parquet shards with Zstd compression and attach transcripts:

```shell
audios-to-dataset \
  --format parquet \
  --parquet-compression zstd \
  --metadata-file ./metadata-file.csv \
  --input ./speech-corpus \
  --output ./speech-corpus-parquet
```

## Building Release Artifacts

Requirements: `cargo`, `rustc`, [`cross`](https://github.com/cross-rs/cross), `podman`, and [`goreleaser`](https://goreleaser.com/).

1. Build the cross images and allocate more resources to Podman (once per machine):

   ```shell
   podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.aarch64-unknown-linux-gnu -t aarch64-unknown-linux-gnu:my-edge .
   podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.x86_64-unknown-linux-gnu -t x86_64-unknown-linux-gnu:my-edge .

   podman machine set --cpus 4 --memory 8192
   ```

2. Produce the binaries with GoReleaser:

   ```shell
   goreleaser build --clean --snapshot --id audios-to-dataset --timeout 60m
   ```

Helpful automation is captured in the `justfile`:

```shell
just fmt      # rustfmt
just clippy   # cargo clippy --all-targets
just release  # cargo build --release
```

## Hugging Face Dataset Card Snippet

To render an inline audio player on the Hugging Face Hub Data Viewer, prepend this front matter to the dataset README:

```
---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: duration
    dtype: float64
  - name: transcription
    dtype: string
task_categories:
- automatic-speech-recognition
tags:
  - audio
  - speech-processing
---
```

## License

Distributed under the terms of the [MIT License](LICENSE).
