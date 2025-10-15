use std::fs::File as StdFile;
use std::path::Path;
use std::path::PathBuf;
use std::{io::Read, num::NonZeroUsize};

use polars::prelude::*;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use duckdb::{Connection, params};
use hound::WavReader;
use rayon::prelude::*;
use recv_dir::{Filter, MaxDepth, NoSymlink, RecursiveDirIterator};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Audio {
    path: String,
    // sampling_rate: i32,
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
}

const CREATE_TABLE: &str = r"
CREATE SEQUENCE seq;

CREATE TABLE files (
  id INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq'),
  duration DOUBLE,
  transcription VARCHAR,
  audio STRUCT(path VARCHAR, bytes BLOB)
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

    let path_data: Vec<Option<String>> = files
        .iter()
        .map(|file| Some(file.audio.path.clone()))
        .collect();

    let bytes_series = Series::new("bytes".into(), bytes_data);
    let path_series = Series::new("path".into(), path_data);
    let audio_struct_series: Series = StructChunked::from_series(
        "audio".into(),
        files.len(),
        [bytes_series, path_series].iter(),
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;

    if !args.input.exists() {
        eprintln!("Input folder does not exist: {:?}", args.input);
        return Ok(());
    }
    if !args.input.is_dir() {
        eprintln!("Input path is not a directory: {:?}", args.input);
        return Ok(());
    }

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
            for file_name in chunk {
                let mut file = std::fs::File::open(file_name.clone()).unwrap();
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).unwrap();

                let duration = match WavReader::new(&buffer[..]) {
                    Ok(reader) => {
                        let spec = reader.spec();
                        reader.duration() as f64 / spec.sample_rate as f64
                    }
                    Err(_) => 0.0,
                };

                let file_name = match file_name.file_name().and_then(|s| s.to_str()) {
                    Some(name) => name.to_string(),
                    None => {
                        eprintln!(
                            "Could not get file name as a string for {:?}, skipping.",
                            file_name
                        );
                        continue;
                    }
                };

                let file = File {
                    duration,
                    audio: Audio {
                        path: file_name,
                        bytes: buffer,
                    },
                    transcription: "-".to_string(),
                };

                files.push(file);
            }

            if args.format == Format::DuckDB {
                let conn = Connection::open(&path).unwrap();
                conn.execute_batch(CREATE_TABLE).unwrap();

                let mut insert_stmt = conn
                    .prepare("INSERT INTO files (id, transcription, duration, audio) VALUES (?, ?, ?, row(?, ?))")
                    .unwrap();

                conn.execute_batch("BEGIN TRANSACTION").unwrap();
                for (idx, file) in files.iter().enumerate() {
                    let _ = insert_stmt.execute(params![
                        idx,
                        file.transcription,
                        file.duration,
                        file.audio.path.clone(),
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
