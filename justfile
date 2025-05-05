fmt:
    cargo fmt

release: fmt
    cargo build --release

clippy:
    cargo clippy --all-targets

archive:
    ouch compress dist/audios-to-dataset_aarch64-apple-darwin dist/audios-to-dataset_aarch64-apple-darwin.zip
    ouch compress dist/audios-to-dataset_aarch64-unknown-linux-gnu dist/audios-to-dataset_aarch64-unknown-linux-gnu.zip
    ouch compress dist/audios-to-dataset_x86_64-apple-darwin dist/audios-to-dataset_x86_64-apple-darwin.zip
    ouch compress dist/audios-to-dataset_x86_64-unknown-linux-gnu dist/audios-to-dataset_x86_64-unknown-linux-gnu.zip
    ouch compress dist/audios-to-dataset_x86_64-unknown-linux-musl dist/audios-to-dataset_x86_64-unknown-linux-musl.zip
