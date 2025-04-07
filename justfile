fmt:
    cargo fmt

release: fmt
    cargo build --release
