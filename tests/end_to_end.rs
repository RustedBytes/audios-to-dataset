use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use assert_cmd::cargo::cargo_bin_cmd;
use hound::{SampleFormat, WavSpec, WavWriter};
use polars::prelude::{AnyValue, ParquetReader, SerReader};
use tempfile::{NamedTempFile, tempdir};

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

    cargo_bin_cmd!("audios-to-dataset")
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

#[test]
fn generates_parquet_dataset_with_metadata_fallback_to_filename() -> Result<()> {
    let input_dir = tempdir()?;
    let output_dir = tempdir()?;

    let wav_path = input_dir.path().join("fallback.wav");
    create_test_wav(&wav_path, 22_050, 22_050)?;

    let mut metadata = NamedTempFile::new()?;
    metadata.write_all(b"file_name,transcription\nfallback.wav,using filename\n")?;
    metadata.flush()?;

    cargo_bin_cmd!("audios-to-dataset")
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

    let mut file = File::open(&output_file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    assert_eq!(df.height(), 1);

    let transcription = df.column("transcription")?.str()?.get(0);
    assert_eq!(transcription, Some("using filename"));

    Ok(())
}

#[test]
fn generates_parquet_dataset_from_jsonl_metadata_with_typed_fields() -> Result<()> {
    let input_dir = tempdir()?;
    let output_dir = tempdir()?;

    let wav_path = input_dir.path().join("jsonl.wav");
    create_test_wav(&wav_path, 8_000, 8_000)?;

    let metadata_path = input_dir.path().join("metadata.jsonl");
    let mut metadata = File::create(&metadata_path)?;
    metadata.write_all(
        br#"{"relative_path":"jsonl.wav","transcription":"jsonl text","speaker":"alice","verified":true,"snr":12.5}
"#,
    )?;
    metadata.flush()?;

    cargo_bin_cmd!("audios-to-dataset")
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
        .arg(&metadata_path)
        .assert()
        .success();

    let output_file = output_dir.path().join("0.parquet");
    assert!(
        output_file.exists(),
        "expected Parquet file at {:?}",
        output_file
    );

    let mut file = File::open(&output_file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    assert_eq!(df.height(), 1);

    let transcription = df.column("transcription")?.str()?.get(0);
    assert_eq!(transcription, Some("jsonl text"));

    let speaker = df.column("speaker")?.str()?.get(0);
    assert_eq!(speaker, Some("alice"));

    let verified = df.column("verified")?.bool()?.get(0);
    assert_eq!(verified, Some(true));

    let snr = df.column("snr")?.f64()?.get(0).unwrap();
    assert!((snr - 12.5).abs() < 1e-6);

    Ok(())
}

#[test]
fn generates_parquet_dataset_from_jsonl_metadata_with_arrays() -> Result<()> {
    let input_dir = tempdir()?;
    let output_dir = tempdir()?;

    let wav_path = input_dir.path().join("arrays.wav");
    create_test_wav(&wav_path, 16_000, 16_000)?;

    let metadata_path = input_dir.path().join("metadata.jsonl");
    let mut metadata = File::create(&metadata_path)?;
    metadata.write_all(
        br#"{"relative_path":"arrays.wav","transcription":"array text","tags":["music","test"],"scores":[0.1,0.2],"flags":[true,false]}
"#,
    )?;
    metadata.flush()?;

    cargo_bin_cmd!("audios-to-dataset")
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
        .arg(&metadata_path)
        .assert()
        .success();

    let output_file = output_dir.path().join("0.parquet");
    assert!(
        output_file.exists(),
        "expected Parquet file at {:?}",
        output_file
    );

    let mut file = File::open(&output_file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    assert_eq!(df.height(), 1);

    let tags = df.column("tags")?.get(0)?;
    let tags_values: Vec<String> = match tags {
        AnyValue::List(inner) => {
            let inner = inner.clone();
            inner
                .str()?
                .into_no_null_iter()
                .map(|s| s.to_string())
                .collect()
        }
        _ => panic!("expected list for tags"),
    };
    assert_eq!(tags_values, vec!["music".to_string(), "test".to_string()]);

    let scores = df.column("scores")?.get(0)?;
    let scores_values: Vec<f64> = match scores {
        AnyValue::List(inner) => {
            let inner = inner.clone();
            inner.f64()?.into_no_null_iter().collect()
        }
        _ => panic!("expected list for scores"),
    };
    assert_eq!(scores_values, vec![0.1, 0.2]);

    let flags = df.column("flags")?.get(0)?;
    let flags_values: Vec<bool> = match flags {
        AnyValue::List(inner) => {
            let inner = inner.clone();
            inner.bool()?.into_no_null_iter().collect()
        }
        _ => panic!("expected list for flags"),
    };
    assert_eq!(flags_values, vec![true, false]);

    Ok(())
}

#[test]
fn uses_jsonl_filename_with_path_for_metadata_lookup() -> Result<()> {
    let input_dir = tempdir()?;
    let output_dir = tempdir()?;

    let nested = input_dir.path().join("nested");
    std::fs::create_dir_all(&nested)?;

    let wav_path = nested.join("with_path.wav");
    create_test_wav(&wav_path, 44_100, 44_100)?;

    let metadata_path = input_dir.path().join("metadata.jsonl");
    let mut metadata = File::create(&metadata_path)?;
    metadata.write_all(
        br#"{"file_name":"nested/with_path.wav","transcription":"path lookup","speaker":"bob","verified":true,"snr":5.5}
"#,
    )?;
    metadata.flush()?;

    cargo_bin_cmd!("audios-to-dataset")
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
        .arg(&metadata_path)
        .assert()
        .success();

    let output_file = output_dir.path().join("0.parquet");
    assert!(
        output_file.exists(),
        "expected Parquet file at {:?}",
        output_file
    );

    let mut file = File::open(&output_file)?;
    let df = ParquetReader::new(&mut file).finish()?;
    assert_eq!(df.height(), 1);

    let transcription = df.column("transcription")?.str()?.get(0);
    assert_eq!(transcription, Some("path lookup"));

    let speaker = df.column("speaker")?.str()?.get(0);
    assert_eq!(speaker, Some("bob"));

    let verified = df.column("verified")?.bool()?.get(0);
    assert_eq!(verified, Some(true));

    let snr = df.column("snr")?.f64()?.get(0).unwrap();
    assert!((snr - 5.5).abs() < 1e-6);

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
