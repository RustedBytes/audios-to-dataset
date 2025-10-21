# `audios-to-dataset`

Convert your audio files into DuckDB or Parquet files (the same thing as does Hugging Face `datasets` library).

## Usage

```
Usage: audios-to-dataset [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
      --input <INPUT>
          The path to the input folder (by default, the program will scan the entire folder recursively)
      --format <FORMAT>
          The format of the output database files [default: parquet] [possible values: duck-db, parquet]
      --files-per-db <FILES_PER_DB>
          How many files to put in each database [default: 500]
      --max-depth-size <MAX_DEPTH_SIZE>
          The maximum depth of the directory tree to scan [default: 50]
      --check-mime-type
          Check mime type of files
      --num-threads <NUM_THREADS>
          The number of threads used for processing [default: 5]
      --output <OUTPUT>
          The path to the output files
      --parquet-compression <PARQUET_COMPRESSION>
          The compression algorithm to use for Parquet files [default: snappy] [possible values: uncompressed, snappy, gzip, lzo, brotli, lz4, zstd, lz4-raw]
      --metadata-file <METADATA_FILE>
          CSV file where transcriptions reside
  -h, --help
          Print help
  -V, --version
          Print version
```

## Example

```shell
audios-to-dataset --format duckdb --input test-data --output test-data-packed

audios-to-dataset --format parquet --files-per-db 1000 --input test-data --output test-data-packed
```

## Build

You need: cargo, rustc, cross, podman, goreleaser.

0. build images and increase resources for podman:

```shell
podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.aarch64-unknown-linux-gnu -t aarch64-unknown-linux-gnu:my-edge .
podman build --platform=linux/amd64 -f dockerfiles/Dockerfile.x86_64-unknown-linux-gnu -t x86_64-unknown-linux-gnu:my-edge .

podman machine set --cpus 4 --memory 8192
```

1. make binaries:

```shell
goreleaser build --clean --snapshot --id audios-to-dataset --timeout 60m
```

## Data Viewer audio on HF

Insert the following header to `README.md` file on Hugging Face if you want to see audio HTML tag to listen to audios in the Data Viewer.

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
