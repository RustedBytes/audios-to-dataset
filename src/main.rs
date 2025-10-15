use std::fs::File as StdFile;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::{io::Read, num::NonZeroUsize};

use anyhow::Result;
use arrow::array::{BinaryBuilder, Float64Builder, StringBuilder, StructBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueEnum};
use duckdb::{Connection, params};
use hound::WavReader;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::BrotliLevel;
use parquet::basic::Compression;
use parquet::basic::GzipLevel;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
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
enum ParquetCompression {
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
    #[clap(value_enum, default_value_t = ParquetCompression::Snappy)]
    parquet_compression: ParquetCompression,
}

const CREATE_TABLE: &str = r"
CREATE SEQUENCE seq;

CREATE TABLE files (
  id INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq'),
  duration INTEGER,
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

fn create_schema() -> SchemaRef {
    let audio_fields = vec![
        Field::new("bytes", DataType::Binary, false),
        Field::new("path", DataType::Utf8, false),
    ];

    let audio_struct = DataType::Struct(audio_fields.into());

    let schema = Schema::new(vec![
        Field::new("audio", audio_struct, false),
        Field::new("duration", DataType::Float64, false),
        Field::new("transcription", DataType::Utf8, false),
    ]);

    Arc::new(schema)
}

fn write_files_to_parquet<P: AsRef<Path>>(
    output_path: P,
    files: &[File],
    compression: ParquetCompression,
) -> Result<()> {
    let schema = create_schema();

    let mut duration_builder = Float64Builder::with_capacity(files.len());
    let mut transcription_builder = StringBuilder::with_capacity(files.len(), files.len() * 50);

    let audio_path_builder = StringBuilder::with_capacity(files.len(), files.len() * 50); // Estimate capacity
    let audio_bytes_builder = BinaryBuilder::with_capacity(files.len(), files.len() * 1024 * 10); // Estimate capacity

    let audio_fields_for_builder = vec![
        Field::new("bytes", DataType::Binary, false),
        Field::new("path", DataType::Utf8, false),
    ];
    let audio_field_builders: Vec<Box<dyn arrow::array::ArrayBuilder>> =
        vec![Box::new(audio_bytes_builder), Box::new(audio_path_builder)];
    let mut audio_struct_builder =
        StructBuilder::new(audio_fields_for_builder, audio_field_builders);

    for file in files {
        transcription_builder.append_value(file.transcription.clone());

        duration_builder.append_value(file.duration);

        audio_struct_builder
            .field_builder::<BinaryBuilder>(0)
            .unwrap()
            .append_value(&file.audio.bytes);
        audio_struct_builder
            .field_builder::<StringBuilder>(1)
            .unwrap()
            .append_value(&file.audio.path);

        audio_struct_builder.append(true);
    }

    let transcription_array = Arc::new(transcription_builder.finish());
    let duration_array = Arc::new(duration_builder.finish());
    let audio_array = Arc::new(audio_struct_builder.finish());

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![audio_array, duration_array, transcription_array],
    )?;

    let file = StdFile::create(output_path)?;
    let compression = match compression {
        ParquetCompression::Uncompressed => Compression::UNCOMPRESSED,
        ParquetCompression::Snappy => Compression::SNAPPY,
        ParquetCompression::Gzip => Compression::GZIP(GzipLevel::default()),
        ParquetCompression::Lzo => Compression::LZO,
        ParquetCompression::Brotli => Compression::BROTLI(BrotliLevel::default()),
        ParquetCompression::Lz4 => Compression::LZ4,
        ParquetCompression::Zstd => Compression::ZSTD(ZstdLevel::default()),
        ParquetCompression::Lz4Raw => Compression::LZ4_RAW,
    };
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    let _ = writer.write(&batch);
    writer.close()?;

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
