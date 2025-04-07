use std::fs::File as StdFile;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::{io::Read, num::NonZeroUsize};

use anyhow::Result;
use arrow::array::{BinaryBuilder, Int32Builder, StringBuilder, StructBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

use clap::{Parser, ValueEnum};
use duckdb::{Connection, params};
use rayon::prelude::*;
use recv_dir::{Filter, MaxDepth, NoSymlink, RecursiveDirIterator};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Audio {
    path: String,
    // sampling_rate: i32,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct File {
    id: i32,
    duration: i32,
    audio: Audio,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    DUCKDB,
    PARQUET,
}

#[derive(Parser, Debug)]
#[command(version, long_about = None)]
struct Args {
    /// The path to the input folder
    #[arg(long)]
    input: PathBuf,

    /// File format
    #[arg(long)]
    #[clap(value_enum, default_value_t = Format::PARQUET)]
    format: Format,

    /// How many files to put in each database
    #[arg(long, default_value_t = 500)]
    files_per_db: usize,

    /// The maximum depth of the directory tree to scan
    #[arg(long, default_value_t = 50)]
    max_depth_size: usize,

    /// The path to the output files
    #[arg(long)]
    output: PathBuf,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse the command line arguments
    let args = Args::parse();

    // Check if the input folder exists
    if !args.input.exists() {
        eprintln!("Input folder does not exist: {:?}", args.input);
        return Ok(());
    }

    // Check if the input folder is a directory
    if !args.input.is_dir() {
        eprintln!("Input path is not a directory: {:?}", args.input);
        return Ok(());
    }

    // Check if the output folder exists
    if !args.output.exists() {
        // Create the output folder if it doesn't exist
        std::fs::create_dir_all(&args.output)?;

        println!("Created output folder: {:?}", args.output);
    }

    // Scan the input folder for audio files
    let dir = RecursiveDirIterator::with_filter(
        &args.input,
        NoSymlink.and(MaxDepth::new(
            NonZeroUsize::new(args.max_depth_size).unwrap(),
        )),
    )
    .unwrap();

    let mut files = Vec::new();

    for entry in dir {
        // Skip folders
        if entry.is_dir() {
            println!("Skipping directory: {:?}", entry);
            continue;
        }

        // Determine the mime type to check if the file is an audio file
        let mime_type = tree_magic_mini::from_filepath(&entry);
        if mime_type.is_none() {
            println!("No mime type found for {:?}", entry);
            continue;
        }

        // Check if the mime type is in the list of audio mime types
        let mime_type = mime_type.unwrap();
        if !AUDIO_MIME_TYPES.contains(&mime_type) {
            println!("Not an audio file: {:?}: {}", entry, mime_type);
            continue;
        }

        files.push(entry);
    }

    println!("Found {} files", files.len());

    // Chunk the files into groups of 2000
    for (idx, chunk) in files.chunks(args.files_per_db).enumerate() {
        // Create the database file
        let ext = match args.format {
            Format::DUCKDB => "duckdb",
            Format::PARQUET => "parquet",
        };
        let path = args.output.join(format!("{}.{}", idx, ext));

        println!(
            "Creating database {} and adding {} files to it",
            path.display(),
            args.files_per_db
        );

        // Remove if the file already exists
        if std::path::Path::new(&path).exists() {
            std::fs::remove_file(&path)?;
        }

        let mut files = Vec::new();
        for (file_id, file_name) in chunk.iter().enumerate() {
            // Read the file into a vector of bytes
            let mut file = std::fs::File::open(file_name.clone())?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            // Extract the file name from the path
            let file_name = file_name.file_name().unwrap().to_string_lossy().to_string();

            let file = File {
                id: file_id as i32,
                duration: 0,
                audio: Audio {
                    path: file_name.clone(),
                    bytes: buffer,
                },
            };

            // Add the file to the list
            files.push(file);
        }

        if args.format == Format::DUCKDB {
            let conn = Connection::open(&path)?;
            conn.execute_batch(CREATE_TABLE)?;

            let mut insert_stmt =
                conn.prepare("INSERT INTO files (id, duration, audio) VALUES (?, ?, row(?, ?))")?;

            conn.execute_batch("BEGIN TRANSACTION")?;
            for file in files {
                insert_stmt.execute(params![
                    file.id,
                    file.duration,
                    file.audio.path.clone(),
                    file.audio.bytes.clone(),
                ])?;
            }
            conn.execute_batch("COMMIT TRANSACTION")?;

            if let Err(e) = conn.close() {
                eprintln!("Failed to close connection: {:?}", e);
            }
        } else if args.format == Format::PARQUET {
            write_files_to_parquet(path.clone(), &files)?;
        }
    }

    Ok(())
}

fn create_schema() -> SchemaRef {
    let audio_fields = vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("bytes", DataType::Binary, false),
    ];

    let audio_struct = DataType::Struct(audio_fields.into());

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("duration", DataType::Int32, false),
        Field::new("audio", audio_struct, false),
    ]);

    Arc::new(schema)
}

fn write_files_to_parquet<P: AsRef<Path>>(output_path: P, files: &[File]) -> Result<()> {
    let schema = create_schema();

    let mut id_builder = Int32Builder::with_capacity(files.len());
    let mut duration_builder = Int32Builder::with_capacity(files.len());

    let audio_path_builder = StringBuilder::with_capacity(files.len(), files.len() * 50); // Estimate capacity
    let audio_bytes_builder = BinaryBuilder::with_capacity(files.len(), files.len() * 1024 * 10); // Estimate capacity

    let audio_fields_for_builder = vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("bytes", DataType::Binary, false),
    ];
    let audio_field_builders: Vec<Box<dyn arrow::array::ArrayBuilder>> =
        vec![Box::new(audio_path_builder), Box::new(audio_bytes_builder)];
    let mut audio_struct_builder =
        StructBuilder::new(audio_fields_for_builder, audio_field_builders);

    for file in files {
        id_builder.append_value(file.id);
        duration_builder.append_value(file.duration);

        audio_struct_builder
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value(&file.audio.path);
        audio_struct_builder
            .field_builder::<BinaryBuilder>(1)
            .unwrap()
            .append_value(&file.audio.bytes);

        audio_struct_builder.append(true);
    }

    let id_array = Arc::new(id_builder.finish());
    let duration_array = Arc::new(duration_builder.finish());
    let audio_array = Arc::new(audio_struct_builder.finish());

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, duration_array, audio_array])?;

    let file = StdFile::create(output_path)?;
    let props = WriterProperties::builder().build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

    writer.write(&batch)?;
    writer.close()?;

    println!("Successfully wrote {} records to Parquet.", files.len());

    Ok(())
}
