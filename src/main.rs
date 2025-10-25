use std::collections::HashMap;
use std::fs::File as StdFile;
use std::path::Path;
use std::path::PathBuf;
use std::{io::Read, num::NonZeroUsize};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use duckdb::{Connection, params};
use hound::WavReader;
use polars::prelude::*;
use rayon::prelude::*;
use recv_dir::{Filter, MaxDepth, NoSymlink, RecursiveDirIterator};
use serde::{Deserialize, Serialize};

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
    transcription: String,
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

    /// CSV file where transcriptions reside
    #[arg(long)]
    metadata_file: Option<PathBuf>,
}

const CREATE_TABLE: &str = r"
CREATE SEQUENCE seq;

CREATE TABLE files (
  id INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq'),
  duration DOUBLE,
  transcription VARCHAR,
  audio STRUCT(path VARCHAR, sampling_rate INTEGER, bytes BLOB)
);";

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

fn write_files_to_parquet<P: AsRef<Path>>(
    output_path: P,
    files: &[File],
    compression: ParquetCompressionChoice,
) -> Result<()> {
    let transcription_data: Vec<Option<String>> = files
        .iter()
        .map(|file| Some(file.transcription.clone()))
        .collect();

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
    let transcription_series = Series::new("transcription".into(), transcription_data);

    let mut df = DataFrame::new(vec![
        audio_struct_series.into_column(),
        duration_series.into_column(),
        transcription_series.into_column(),
    ])?;

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

    let hf_value = r#"{"info": {"features": {"audio": {"_type": "Audio"}, "duration": {"dtype": "float64", "_type": "Value"}, "transcription": {"dtype": "string", "_type": "Value"}}}}"#;

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

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{ParquetReader, SerReader};
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

        let bytes = vec![0_u8, 1, 2, 3, 4];
        let files = vec![File {
            duration: 1.25,
            audio: Audio {
                path: "clip.wav".to_string(),
                sampling_rate: 16_000,
                bytes: bytes.clone(),
            },
            transcription: "hello world".to_string(),
        }];

        write_files_to_parquet(&output_path, &files, ParquetCompressionChoice::Snappy)?;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    let (transcriptions_by_rel, transcriptions_by_name): (
        HashMap<String, String>,
        HashMap<String, String>,
    ) = if let Some(metadata_path) = &args.metadata_file {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(metadata_path.clone()))?
            .finish()?;

        let file_name_col = df.column("file_name")?.str()?;
        let transcription_col = df.column("transcription")?.str()?;

        let mut by_relative_path = HashMap::new();
        let mut by_name = HashMap::new();
        let row_count = df.height();
        let transcriptions: Vec<String> = (0..row_count)
            .map(|idx| transcription_col.get(idx).unwrap_or("-").to_string())
            .collect();

        if let Ok(relative_path_col) = df.column("relative_path") {
            let relative_path_col = relative_path_col.str()?;

            for idx in 0..row_count {
                if let Some(relative_path) = relative_path_col.get(idx) {
                    let key = normalized_relative_path_str(relative_path);
                    by_relative_path
                        .entry(key)
                        .or_insert_with(|| transcriptions[idx].clone());
                }
            }
        }

        for idx in 0..row_count {
            if let Some(name) = file_name_col.get(idx) {
                by_name
                    .entry(name.to_string())
                    .or_insert_with(|| transcriptions[idx].clone());
            }
        }

        (by_relative_path, by_name)
    } else {
        (HashMap::new(), HashMap::new())
    };

    if !args.input.exists() {
        eprintln!("Input folder does not exist: {:?}", args.input);
        return Ok(());
    }
    if !args.input.is_dir() {
        eprintln!("Input path is not a directory: {:?}", args.input);
        return Ok(());
    }

    let canonical_input = args
        .input
        .canonicalize()
        .unwrap_or_else(|_| args.input.clone());

    if !args.output.exists() {
        std::fs::create_dir_all(&args.output)?;

        println!("Created output folder: {:?}", args.output);
    }

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

                let relative_path = file_path
                    .strip_prefix(&args.input)
                    .ok()
                    .or_else(|| file_path.strip_prefix(&canonical_input).ok());
                let relative_path_str = relative_path
                    .map(|path| normalized_relative_path(path))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| {
                        file_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| file_path.to_string_lossy().to_string())
                    });

                let (duration, sr) = match WavReader::new(&buffer[..]) {
                    Ok(reader) => {
                        let spec = reader.spec();
                        (reader.duration() as f64 / spec.sample_rate as f64, spec.sample_rate as i32)
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

                let transcription = transcriptions_by_rel
                    .get(&relative_path_str)
                    .or_else(|| transcriptions_by_name.get(&file_name))
                    .cloned()
                    .unwrap_or_else(|| "-".to_string());

                let file = File {
                    duration,
                    audio: Audio {
                        path: relative_path_str,
                        sampling_rate: sr,
                        bytes: buffer,
                    },
                    transcription,
                };

                files.push(file);
            }

            if args.format == Format::DuckDB {
                let conn = Connection::open(&path).unwrap();
                conn.execute_batch(CREATE_TABLE).unwrap();

                let mut insert_stmt = conn
                    .prepare("INSERT INTO files (id, transcription, duration, audio) VALUES (?, ?, ?, row(?, ?, ?))")
                    .unwrap();

                conn.execute_batch("BEGIN TRANSACTION").unwrap();
                for (idx, file) in files.iter().enumerate() {
                    let _ = insert_stmt.execute(params![
                        idx,
                        file.transcription,
                        file.duration,
                        file.audio.path.clone(),
                        file.audio.sampling_rate.clone(),
                        file.audio.bytes.clone(),
                    ]);
                }
                conn.execute_batch("COMMIT TRANSACTION").unwrap();

                if let Err(e) = conn.close() {
                    eprintln!("Failed to close connection: {:?}", e);
                }
            } else if args.format == Format::Parquet {
                let _ = write_files_to_parquet(path.clone(), &files, args.parquet_compression);
            }
        });

    Ok(())
}
