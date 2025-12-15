use std::collections::{BTreeSet, HashMap};
use std::fs::File as StdFile;
use std::fs::create_dir_all;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::{
    io::{BufRead, BufReader, Read},
    num::NonZeroUsize,
};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use duckdb::types::Value as DuckValue;
use duckdb::{Connection, params_from_iter};
use hound::WavReader;
use polars::prelude::*;
use rayon::prelude::*;
use recv_dir::{Filter, MaxDepth, NoSymlink, RecursiveDirIterator};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Audio {
    path: String,
    sampling_rate: i32,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct File {
    duration: f64,
    audio: Audio,
    metadata: HashMap<String, Value>,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    DuckDB,
    Parquet,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum ParquetCompressionChoice {
    Uncompressed,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
    Lz4Raw,
}

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// The path to the input folder (by default, the program will scan the entire folder recursively)
    #[arg(long)]
    input: PathBuf,

    /// The format of the output database files
    #[arg(long)]
    #[clap(value_enum, default_value_t = Format::Parquet)]
    format: Format,

    /// How many files to put in each database
    #[arg(long, default_value_t = 500)]
    files_per_db: usize,

    /// The maximum depth of the directory tree to scan
    #[arg(long, default_value_t = 50)]
    max_depth_size: usize,

    /// Check mime type of files
    #[arg(long, default_value_t = false)]
    check_mime_type: bool,

    /// The number of threads used for processing
    #[arg(long, default_value_t = 5)]
    num_threads: usize,

    /// The path to the output files
    #[arg(long)]
    output: PathBuf,

    /// The compression algorithm to use for Parquet files
    #[arg(long)]
    #[clap(value_enum, default_value_t = ParquetCompressionChoice::Snappy)]
    parquet_compression: ParquetCompressionChoice,

    /// Metadata file (CSV or JSONL) describing per-file fields
    #[arg(long)]
    metadata_file: Option<PathBuf>,
}

const AUDIO_MIME_TYPES: [&str; 12] = [
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "audio/flac",
    "audio/vnd.wave",
    "audio/x-wav",
    "audio/x-flac",
    "audio/x-mpeg",
    "audio/x-aiff",
    "audio/aiff",
    "audio/x-aac",
    "audio/aac",
];

fn normalized_relative_path(path: &Path) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    normalized.trim_start_matches("./").to_string()
}

fn normalized_relative_path_str(value: &str) -> String {
    value
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetadataType {
    String,
    Bool,
    Float64,
}

impl MetadataType {
    fn merge(self, other: MetadataType) -> MetadataType {
        if self == other {
            self
        } else {
            MetadataType::String
        }
    }
}

#[derive(Default)]
struct MetadataStore {
    by_relative_path: HashMap<String, HashMap<String, Value>>,
    by_name: HashMap<String, HashMap<String, Value>>,
    keys: BTreeSet<String>,
    types: HashMap<String, MetadataType>,
}

impl MetadataStore {
    fn new() -> Self {
        let mut store = MetadataStore::default();
        store.ensure_transcription_key();
        store
    }

    fn ensure_transcription_key(&mut self) {
        self.keys.insert("transcription".to_string());
        self.types
            .entry("transcription".to_string())
            .or_insert(MetadataType::String);
    }

    fn update_types_from_record(&mut self, metadata: &HashMap<String, Value>) {
        for (key, value) in metadata {
            self.keys.insert(key.clone());
            if let Some(value_type) = infer_metadata_type(value) {
                self.types
                    .entry(key.clone())
                    .and_modify(|current| *current = current.merge(value_type))
                    .or_insert(value_type);
            }
        }
    }

    fn insert_record(
        &mut self,
        relative_path: Option<String>,
        file_name: Option<String>,
        metadata: HashMap<String, Value>,
    ) {
        if let Some(rel) = relative_path {
            self.by_relative_path
                .entry(rel)
                .or_insert_with(|| metadata.clone());
        }

        if let Some(name) = file_name {
            self.by_name.entry(name).or_insert(metadata);
        }
    }

    fn metadata_for_file(&self, relative_path: &str, file_name: &str) -> HashMap<String, Value> {
        let mut metadata = self
            .by_relative_path
            .get(relative_path)
            .cloned()
            .or_else(|| self.by_name.get(file_name).cloned())
            .unwrap_or_default();

        metadata
            .entry("transcription".to_string())
            .or_insert_with(|| Value::String("-".to_string()));

        metadata
    }
}

fn infer_metadata_type(value: &Value) -> Option<MetadataType> {
    match value {
        Value::Bool(_) => Some(MetadataType::Bool),
        Value::Number(_) => Some(MetadataType::Float64),
        Value::String(_) => Some(MetadataType::String),
        Value::Null => None,
        _ => Some(MetadataType::String),
    }
}

fn sanitize_column_name(name: &str) -> String {
    name.replace('"', "\"\"")
}

fn is_reserved_metadata_key(key: &str) -> bool {
    matches!(key, "duration" | "audio" | "id")
}

enum MetadataFormat {
    Csv,
    Jsonl,
}

fn metadata_format_from_path(path: &Path) -> MetadataFormat {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "jsonl" | "json" => MetadataFormat::Jsonl,
        _ => MetadataFormat::Csv,
    }
}

fn load_metadata_store(path: &Path) -> Result<MetadataStore> {
    match metadata_format_from_path(path) {
        MetadataFormat::Csv => load_csv_metadata(path),
        MetadataFormat::Jsonl => load_jsonl_metadata(path),
    }
}

fn load_csv_metadata(path: &Path) -> Result<MetadataStore> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut store = MetadataStore::new();

    for record in reader.records() {
        let record = record?;
        let mut file_name: Option<String> = None;
        let mut relative_path: Option<String> = None;
        let mut metadata = HashMap::new();

        for (header, value) in headers.iter().zip(record.iter()) {
            match header {
                "file_name" => {
                    if !value.is_empty() {
                        file_name = Some(value.to_string());
                    }
                }
                "relative_path" => {
                    if !value.is_empty() {
                        relative_path = Some(normalized_relative_path_str(value));
                    }
                }
                _ => {
                    if !value.is_empty() && !is_reserved_metadata_key(header) {
                        metadata.insert(header.to_string(), Value::String(value.to_string()));
                    }
                }
            }
        }

        metadata
            .entry("transcription".to_string())
            .or_insert_with(|| Value::String("-".to_string()));

        store.update_types_from_record(&metadata);

        if file_name.is_none() && relative_path.is_none() {
            continue;
        }

        store.insert_record(relative_path, file_name, metadata);
    }

    Ok(store)
}

fn load_jsonl_metadata(path: &Path) -> Result<MetadataStore> {
    let file = StdFile::open(path)?;
    let reader = BufReader::new(file);
    let mut store = MetadataStore::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(trimmed)?;
        let Some(object) = value.as_object() else {
            continue;
        };

        let mut file_name = object
            .get("file_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());
        let mut relative_path = object
            .get("relative_path")
            .and_then(|v| v.as_str())
            .map(normalized_relative_path_str)
            .filter(|s| !s.is_empty());

        let mut metadata: HashMap<String, Value> = object
            .iter()
            .filter_map(|(key, value)| {
                if key == "file_name" || key == "relative_path" || is_reserved_metadata_key(key) {
                    return None;
                }

                Some((key.clone(), value.clone()))
            })
            .collect();

        metadata
            .entry("transcription".to_string())
            .or_insert_with(|| Value::String("-".to_string()));

        store.update_types_from_record(&metadata);

        if file_name.is_none() && relative_path.is_none() {
            continue;
        }

        store.insert_record(relative_path.take(), file_name.take(), metadata);
    }

    Ok(store)
}

fn build_create_table_sql(
    metadata_keys: &BTreeSet<String>,
    metadata_types: &HashMap<String, MetadataType>,
) -> String {
    let mut columns = vec![
        "id INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq')".to_string(),
        "duration DOUBLE".to_string(),
        "audio STRUCT(path VARCHAR, sampling_rate INTEGER, bytes BLOB)".to_string(),
    ];

    for key in metadata_keys {
        let column_type = metadata_types
            .get(key)
            .copied()
            .unwrap_or(MetadataType::String);
        let sql_type = match column_type {
            MetadataType::Bool => "BOOLEAN",
            MetadataType::Float64 => "DOUBLE",
            MetadataType::String => "VARCHAR",
        };

        columns.push(format!("\"{}\" {}", sanitize_column_name(key), sql_type));
    }

    format!(
        "CREATE SEQUENCE seq; CREATE TABLE files ({columns});",
        columns = columns.join(", ")
    )
}

fn build_insert_sql(metadata_keys: &BTreeSet<String>) -> String {
    let mut column_names = vec![
        "id".to_string(),
        "duration".to_string(),
        "audio".to_string(),
    ];
    for key in metadata_keys {
        column_names.push(format!("\"{}\"", sanitize_column_name(key)));
    }

    let mut placeholders = vec!["?".to_string(), "?".to_string(), "row(?, ?, ?)".to_string()];
    placeholders.extend(std::iter::repeat_n("?".to_string(), metadata_keys.len()));

    format!(
        "INSERT INTO files ({columns}) VALUES ({placeholders})",
        columns = column_names.join(", "),
        placeholders = placeholders.join(", ")
    )
}

fn write_files_to_parquet<P: AsRef<Path>>(
    output_path: P,
    files: &[File],
    metadata_keys: &std::collections::BTreeSet<String>,
    metadata_types: &HashMap<String, MetadataType>,
    compression: ParquetCompressionChoice,
) -> Result<()> {
    let duration_data: Vec<Option<f64>> = files.iter().map(|file| Some(file.duration)).collect();

    let bytes_data: Vec<Option<Vec<u8>>> = files
        .iter()
        .map(|file| Some(file.audio.bytes.clone()))
        .collect();

    let sr_data: Vec<Option<i32>> = files
        .iter()
        .map(|file| Some(file.audio.sampling_rate))
        .collect();

    let path_data: Vec<Option<String>> = files
        .iter()
        .map(|file| Some(file.audio.path.clone()))
        .collect();

    let bytes_series = Series::new("bytes".into(), bytes_data);
    let sr_series = Series::new("sampling_rate".into(), sr_data);
    let path_series = Series::new("path".into(), path_data);
    let audio_struct_series: Series = StructChunked::from_series(
        "audio".into(),
        files.len(),
        [bytes_series, sr_series, path_series].iter(),
    )?
    .into_series();

    let duration_series = Series::new("duration".into(), duration_data);
    let mut columns = vec![
        audio_struct_series.into_column(),
        duration_series.into_column(),
    ];

    for key in metadata_keys {
        let column_type = metadata_types
            .get(key)
            .copied()
            .unwrap_or(MetadataType::String);

        match column_type {
            MetadataType::Bool => {
                let data: Vec<Option<bool>> = files
                    .iter()
                    .map(|file| file.metadata.get(key).and_then(|v| v.as_bool()))
                    .collect();
                columns.push(Series::new(key.as_str().into(), data).into_column());
            }
            MetadataType::Float64 => {
                let data: Vec<Option<f64>> = files
                    .iter()
                    .map(|file| file.metadata.get(key).and_then(|v| v.as_f64()))
                    .collect();
                columns.push(Series::new(key.as_str().into(), data).into_column());
            }
            MetadataType::String => {
                let data: Vec<Option<String>> = files
                    .iter()
                    .map(|file| {
                        file.metadata.get(key).map(|v| match v {
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        })
                    })
                    .collect();
                columns.push(Series::new(key.as_str().into(), data).into_column());
            }
        }
    }

    let mut df = DataFrame::new(columns)?;

    let pq_compression = match compression {
        ParquetCompressionChoice::Uncompressed => ParquetCompression::Uncompressed,
        ParquetCompressionChoice::Snappy => ParquetCompression::Snappy,
        ParquetCompressionChoice::Gzip => ParquetCompression::Gzip(None),
        ParquetCompressionChoice::Lzo => ParquetCompression::Lzo,
        ParquetCompressionChoice::Brotli => ParquetCompression::Brotli(None),
        ParquetCompressionChoice::Lz4 => ParquetCompression::Lzo,
        ParquetCompressionChoice::Zstd => ParquetCompression::Zstd(None),
        ParquetCompressionChoice::Lz4Raw => ParquetCompression::Lz4Raw,
    };

    let mut features = serde_json::Map::new();
    features.insert("audio".to_string(), serde_json::json!({"_type": "Audio"}));
    features.insert(
        "duration".to_string(),
        serde_json::json!({"dtype": "float64", "_type": "Value"}),
    );

    for key in metadata_keys {
        let dtype = match metadata_types
            .get(key)
            .copied()
            .unwrap_or(MetadataType::String)
        {
            MetadataType::Bool => "bool",
            MetadataType::Float64 => "float64",
            MetadataType::String => "string",
        };

        features.insert(
            key.clone(),
            serde_json::json!({"dtype": dtype, "_type": "Value"}),
        );
    }

    let hf_value = serde_json::json!({"info": {"features": features}});

    let custom_metadata =
        KeyValueMetadata::from_static(vec![("huggingface".to_string(), hf_value.to_string())]);

    let mut file = StdFile::create(output_path)?;
    ParquetWriter::new(&mut file)
        .with_key_value_metadata(Some(custom_metadata))
        .with_compression(pq_compression)
        .with_row_group_size(Some(256))
        .finish(&mut df)?;

    println!("Successfully wrote {} records to Parquet.", files.len());

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    let metadata_store = if let Some(metadata_path) = &args.metadata_file {
        load_metadata_store(metadata_path)?
    } else {
        MetadataStore::new()
    };

    let metadata_keys = metadata_store.keys.clone();
    let metadata_types = metadata_store.types.clone();

    let metadata_store = Arc::new(metadata_store);
    let metadata_keys = Arc::new(metadata_keys);
    let metadata_types = Arc::new(metadata_types);

    if !args.input.exists() {
        eprintln!("Input folder does not exist: {:?}", args.input);
        return Ok(());
    }
    if !args.input.is_dir() {
        eprintln!("Input path is not a directory: {:?}", args.input);
        return Ok(());
    }

    if !args.output.exists() {
        create_dir_all(&args.output)?;

        println!("Created output folder: {:?}", args.output);
    }

    let metadata_relative = args
        .metadata_file
        .as_ref()
        .and_then(|path| path.strip_prefix(&args.input).ok())
        .map(normalized_relative_path);

    let metadata_absolute = args
        .metadata_file
        .as_ref()
        .and_then(|path| std::fs::canonicalize(path).ok());

    // Scan the input folder for files
    let dir = RecursiveDirIterator::with_filter(
        &args.input,
        NoSymlink.and(MaxDepth::new(
            NonZeroUsize::new(args.max_depth_size).unwrap(),
        )),
    )?;

    let mut files = Vec::new();

    for entry in dir {
        if entry.is_dir() {
            println!("Skipping directory: {:?}", entry);
            continue;
        }

        if let Some(target_relative) = &metadata_relative
            && let Ok(entry_relative) = entry.strip_prefix(&args.input) {
                let normalized_entry = normalized_relative_path(entry_relative);
                if &normalized_entry == target_relative {
                    println!("Skipping metadata file: {:?}", entry);
                    continue;
                }
            }

        if let Some(target_abs) = &metadata_absolute
            && let Ok(entry_abs) = entry.canonicalize()
                && &entry_abs == target_abs {
                    println!("Skipping metadata file: {:?}", entry);
                    continue;
                }

        if args.check_mime_type {
            let mime_type = tree_magic_mini::from_filepath(&entry);
            if mime_type.is_none() {
                println!("No mime type found for {:?}", entry);
                continue;
            }

            let mime_type = mime_type.unwrap();
            if !AUDIO_MIME_TYPES.contains(&mime_type) {
                println!("Not an audio file: {:?}: {}", entry, mime_type);
                continue;
            }
        }

        files.push(entry);
    }

    println!("Found {} files", files.len());

    // Chunk the files into groups of `args.files_per_db`
    files
        .chunks(args.files_per_db)
        .enumerate()
        .par_bridge() // Convert to a parallel iterator
        .for_each(|(idx, chunk)| {
            let ext = match args.format {
                Format::DuckDB => "duckdb",
                Format::Parquet => "parquet",
            };
            let path = args.output.join(format!("{}.{}", idx, ext));

            println!(
                "Creating database {} and adding {} files to it",
                path.display(),
                args.files_per_db
            );

            if path.exists() {
                println!("Removing existing file: {:?}", path);
                std::fs::remove_file(&path).unwrap();
            }

            let mut files = Vec::new();
            for file_path in chunk {
                let mut file = std::fs::File::open(file_path.clone()).unwrap();
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).unwrap();

                let relative_path_str = {
                    let normalized_relative = file_path
                        .strip_prefix(&args.input)
                        .map(normalized_relative_path)
                        .unwrap_or_else(|_| normalized_relative_path(file_path));

                    if normalized_relative.is_empty() {
                        file_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| file_path.to_string_lossy().to_string())
                    } else {
                        normalized_relative
                    }
                };

                let (duration, sr) = match WavReader::new(&buffer[..]) {
                    Ok(reader) => {
                        let spec = reader.spec();
                        (
                            reader.duration() as f64 / spec.sample_rate as f64,
                            spec.sample_rate as i32,
                        )
                    }
                    Err(_) => (0.0, 0),
                };

                let file_name = match file_path.file_name().and_then(|s| s.to_str()) {
                    Some(name) => name.to_string(),
                    None => {
                        eprintln!(
                            "Could not get file name as a string for {:?}, skipping.",
                            file_path
                        );
                        continue;
                    }
                };

                let metadata = metadata_store.metadata_for_file(&relative_path_str, &file_name);

                let file = File {
                    duration,
                    audio: Audio {
                        path: relative_path_str,
                        sampling_rate: sr,
                        bytes: buffer,
                    },
                    metadata,
                };

                files.push(file);
            }

            if args.format == Format::DuckDB {
                let conn = Connection::open(&path).unwrap();
                let create_sql =
                    build_create_table_sql(metadata_keys.as_ref(), metadata_types.as_ref());
                conn.execute_batch(&create_sql).unwrap();

                let insert_sql = build_insert_sql(metadata_keys.as_ref());
                let mut insert_stmt = conn.prepare(&insert_sql).unwrap();

                conn.execute_batch("BEGIN TRANSACTION").unwrap();
                for (idx, file) in files.iter().enumerate() {
                    let mut params: Vec<DuckValue> = Vec::with_capacity(5 + metadata_keys.len());
                    params.push(DuckValue::from(idx));
                    params.push(DuckValue::from(file.duration));
                    params.push(DuckValue::from(file.audio.path.clone()));
                    params.push(DuckValue::from(file.audio.sampling_rate));
                    params.push(DuckValue::from(file.audio.bytes.clone()));

                    for key in metadata_keys.iter() {
                        let column_type = metadata_types
                            .get(key)
                            .copied()
                            .unwrap_or(MetadataType::String);
                        let value = file.metadata.get(key);

                        match column_type {
                            MetadataType::Bool => {
                                params.push(DuckValue::from(value.and_then(|v| v.as_bool())));
                            }
                            MetadataType::Float64 => {
                                params.push(DuckValue::from(value.and_then(|v| v.as_f64())));
                            }
                            MetadataType::String => {
                                params.push(DuckValue::from(value.map(|v| match v {
                                    Value::String(s) => s.clone(),
                                    _ => v.to_string(),
                                })));
                            }
                        }
                    }

                    let _ = insert_stmt.execute(params_from_iter(params));
                }
                conn.execute_batch("COMMIT TRANSACTION").unwrap();

                if let Err(e) = conn.close() {
                    eprintln!("Failed to close connection: {:?}", e);
                }
            } else if args.format == Format::Parquet {
                let _ = write_files_to_parquet(
                    path.clone(),
                    &files,
                    metadata_keys.as_ref(),
                    metadata_types.as_ref(),
                    args.parquet_compression,
                );
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{ParquetReader, SerReader};
    use std::collections::{BTreeSet, HashMap};
    use std::fs::File as StdFile;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn normalized_relative_path_cleans_input_paths() {
        let path = Path::new("./nested/./folder/file.wav");
        assert_eq!(normalized_relative_path(path), "nested/./folder/file.wav");

        let windows_path = Path::new(r".\audio\file.wav");
        assert_eq!(normalized_relative_path(windows_path), "audio/file.wav");
    }

    #[test]
    fn normalized_relative_path_str_normalizes_separators() {
        let input = r".\segment\clip.wav";
        assert_eq!(
            normalized_relative_path_str(input),
            "segment/clip.wav".to_string()
        );

        let already_clean = "dataset/audio.wav";
        assert_eq!(
            normalized_relative_path_str(already_clean),
            already_clean.to_string()
        );
    }

    #[test]
    fn write_files_to_parquet_persists_audio_records() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let output_path = temp_dir.path().join("sample.parquet");

        let mut metadata = HashMap::new();
        metadata.insert(
            "transcription".to_string(),
            Value::String("hello world".to_string()),
        );

        let mut metadata_types = HashMap::new();
        metadata_types.insert("transcription".to_string(), MetadataType::String);
        let metadata_keys = BTreeSet::from(["transcription".to_string()]);

        let bytes = vec![0_u8, 1, 2, 3, 4];
        let files = vec![File {
            duration: 1.25,
            audio: Audio {
                path: "clip.wav".to_string(),
                sampling_rate: 16_000,
                bytes: bytes.clone(),
            },
            metadata,
        }];

        write_files_to_parquet(
            &output_path,
            &files,
            &metadata_keys,
            &metadata_types,
            ParquetCompressionChoice::Snappy,
        )?;

        let mut file = StdFile::open(&output_path)?;
        let df = ParquetReader::new(&mut file).finish()?;

        assert_eq!(df.height(), 1);

        let duration = df.column("duration")?.f64()?.get(0).unwrap();
        assert!((duration - 1.25).abs() < f64::EPSILON);

        let transcription = df.column("transcription")?.str()?.get(0);
        assert_eq!(transcription, Some("hello world"));

        let audio_struct = df.column("audio")?.struct_()?;

        let path_value = audio_struct
            .field_by_name("path")
            .expect("path field to exist")
            .str()?
            .get(0)
            .map(|s| s.to_string());
        assert_eq!(path_value, Some("clip.wav".to_string()));

        let sr_value = audio_struct
            .field_by_name("sampling_rate")
            .expect("sampling_rate field to exist")
            .i32()?
            .get(0);
        assert_eq!(sr_value, Some(16_000));

        let bytes_value = audio_struct
            .field_by_name("bytes")
            .expect("bytes field to exist")
            .binary()?
            .get(0)
            .map(|b| b.to_vec());
        assert_eq!(bytes_value, Some(bytes));

        Ok(())
    }

    #[test]
    fn load_jsonl_metadata_uses_relative_path_matching() -> anyhow::Result<()> {
        let temp_dir = tempdir()?;
        let metadata_path = temp_dir.path().join("metadata.jsonl");
        std::fs::write(
            &metadata_path,
            r#"{"relative_path":"clip.wav","transcription":"jsonl text"}"#,
        )?;

        let store = load_jsonl_metadata(&metadata_path)?;
        let metadata = store.metadata_for_file("clip.wav", "clip.wav");

        assert_eq!(
            metadata.get("transcription").and_then(|v| v.as_str()),
            Some("jsonl text")
        );

        Ok(())
    }
}
