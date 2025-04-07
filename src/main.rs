use duckdb::{Connection, params};
use recv_dir::{Filter, MaxDepth, NoSymlink, RecursiveDirIterator};
use std::path::PathBuf;
use std::{io::Read, num::NonZeroUsize};

#[derive(Debug)]
struct Audio {
    path: String,
    // sampling_rate: i32,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct File {
    id: i32,
    duration: i32,
    audio: Audio,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files_per_db = 2000;

    let root = PathBuf::from("test-data");
    let dir = RecursiveDirIterator::with_filter(
        &root,
        NoSymlink.and(MaxDepth::new(NonZeroUsize::new(2).unwrap())),
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

        // Add the file to the list
        files.push(entry);
    }

    println!("Found {} files", files.len());

    // Chunk the files into groups of 2000
    for (idx, chunk) in files.chunks(files_per_db).enumerate() {
        // Create the database file
        let path = format!("databases/{}.db3", idx);

        // Remove if the file already exists
        if std::path::Path::new(&path).exists() {
            std::fs::remove_file(&path)?;
        }

        println!("Creating database {} and adding {} files to it", path, files_per_db);

        // Open a connection to the database
        let conn = Connection::open(&path)?;

        conn.execute_batch(r"
        CREATE SEQUENCE seq;
        CREATE TABLE files (
                      id              INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq'),
                      duration           INTEGER,
                      audio           STRUCT(path VARCHAR, bytes BLOB)
            );
        ")?;

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
            conn.execute(
                "INSERT INTO files (id, duration, audio) VALUES (?, ?, row(?, ?))",
                params![file.id, file.duration, file.audio.path, file.audio.bytes],
            )?;
        }

        // if let Err(e) = conn.close() {
        //     eprintln!("Failed to close connection: {:?}", e);
        // }

        println!("Inserted {} files into {}", chunk.len(), path);
    }

    Ok(())
}
