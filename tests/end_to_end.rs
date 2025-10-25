use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use assert_cmd::Command;
use hound::{SampleFormat, WavSpec, WavWriter};
use polars::prelude::{ParquetReader, SerReader};
use tempfile::{tempdir, NamedTempFile};

#[test]
fn generates_parquet_dataset_with_metadata() -> Result<()> {
    let input_dir = tempdir()?;
    let output_dir = tempdir()?;

    let wav_path = input_dir.path().join("sample.wav");
    create_test_wav(&wav_path, 16_000, 16_000)?;

    let mut metadata = NamedTempFile::new()?;
    metadata.write_all(
        b"file_name,transcription,relative_path\nsample.wav,test transcription,sample.wav\n",
    )?;
    metadata.flush()?;

    Command::cargo_bin("audios-to-dataset")?
        .arg("--input")
        .arg(input_dir.path())
        .arg("--output")
        .arg(output_dir.path())
        .arg("--format")
        .arg("parquet")
        .arg("--files-per-db")
        .arg("1")
        .arg("--num-threads")
        .arg("1")
        .arg("--metadata-file")
        .arg(metadata.path())
        .assert()
        .success();

    let output_file = output_dir.path().join("0.parquet");
    assert!(
        output_file.exists(),
        "expected Parquet file at {:?}",
        output_file
    );

    let expected_bytes = std::fs::read(&wav_path)?;

    let mut file = File::open(&output_file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    assert_eq!(df.height(), 1);

    let duration = df.column("duration")?.f64()?.get(0).unwrap();
    assert!((duration - 1.0).abs() < 1e-6);

    let transcription = df.column("transcription")?.str()?.get(0);
    assert_eq!(transcription, Some("test transcription"));

    let audio_struct = df.column("audio")?.struct_()?;

    let path_value = audio_struct
        .field_by_name("path")
        .expect("path field to be present")
        .str()?
        .get(0)
        .map(|s| s.to_string());
    assert_eq!(path_value, Some("sample.wav".to_string()));

    let sr_value = audio_struct
        .field_by_name("sampling_rate")
        .expect("sampling_rate field to be present")
        .i32()?
        .get(0);
    assert_eq!(sr_value, Some(16_000));

    let bytes_value = audio_struct
        .field_by_name("bytes")
        .expect("bytes field to be present")
        .binary()?
        .get(0)
        .map(|b| b.to_vec());
    assert_eq!(bytes_value, Some(expected_bytes));

    Ok(())
}

fn create_test_wav(path: &Path, sample_rate: u32, samples: usize) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for idx in 0..samples {
        let time = idx as f32 / sample_rate as f32;
        let angle = time * 2.0 * std::f32::consts::PI * 440.0;
        let sample = (angle.sin() * i16::MAX as f32) as i16;
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}
