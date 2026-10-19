//! `CalibrationIndex::load_dir` contract: strict, deterministic, and loud.
//!
//! Embedders (the Python binding, test harnesses, anything that is not the
//! runtime's own loop) need to turn a directory of TOML into an index. A typo in
//! a shipped default must fail at startup with the file name, not degrade the
//! stream silently.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blazar_calibration::{CalibrationIndex, LoadDirError, Query};
use blazar_core::{ModelIdentity, Target};

const SVT_BALANCED: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "svtav1"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
keyint = 4
bframes = 0
pix_fmt = "yuv420p"
[video.codec_params]
svtav1 = "preset=6"
"#;

const SVT_QUALITY: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "svtav1"
target = "quality"
[video.common]
rate_control = "crf"
crf = 24
keyint = 4
bframes = 0
pix_fmt = "yuv420p"
[video.codec_params]
svtav1 = "preset=4"
"#;

/// Minimal RAII temp dir: unique per test, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "blazar-load-dir-{}-{}-{}",
            std::process::id(),
            label,
            n
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, name: &str, contents: &str) {
        fs::write(self.0.join(name), contents).expect("write fixture");
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn query(target: Target) -> Query {
    Query::new(
        &ModelIdentity {
            family: "wan".into(),
            version: "2.1".into(),
            quant: "bf16".into(),
            size: "1.3b".into(),
        },
        "*",
        "svtav1",
        target,
    )
}

#[test]
fn loads_every_toml_in_the_directory() {
    let dir = TempDir::new("valid");
    dir.write("svtav1_low_latency.toml", SVT_BALANCED);
    dir.write("svtav1_quality.toml", SVT_QUALITY);
    dir.write("README.md", "not a calibration");

    let index = CalibrationIndex::load_dir(dir.path()).expect("valid directory loads");
    assert_eq!(index.len(), 2, "non-TOML files must be ignored");
}

#[test]
fn loaded_index_resolves() {
    let dir = TempDir::new("resolve");
    dir.write("svtav1_balanced.toml", SVT_BALANCED);

    let index = CalibrationIndex::load_dir(dir.path()).expect("valid directory loads");
    let resolved = index
        .resolve(&query(Target::Balanced))
        .expect("terminal default resolves");
    assert_eq!(resolved.video.common.crf, Some(30));
}

#[test]
fn one_invalid_file_fails_the_whole_load_with_its_name() {
    let dir = TempDir::new("invalid");
    dir.write("svtav1_balanced.toml", SVT_BALANCED);
    dir.write("broken.toml", "schema_version = ");

    match CalibrationIndex::load_dir(dir.path()) {
        Err(LoadDirError::InvalidFile { path, .. }) => {
            assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some("broken.toml")
            );
        }
        other => panic!("expected InvalidFile naming broken.toml, got {other:?}"),
    }
}

#[test]
fn empty_directory_is_an_error() {
    let dir = TempDir::new("empty");
    dir.write("notes.txt", "no calibrations here");

    match CalibrationIndex::load_dir(dir.path()) {
        Err(LoadDirError::Empty { .. }) => {}
        other => panic!("expected Empty, got {other:?}"),
    }
}

#[test]
fn missing_directory_is_an_error_naming_the_path() {
    let missing = std::env::temp_dir().join("blazar-load-dir-does-not-exist");
    let _ = fs::remove_dir_all(&missing);

    match CalibrationIndex::load_dir(&missing) {
        Err(LoadDirError::Io { path, .. }) => assert_eq!(path, missing),
        other => panic!("expected Io, got {other:?}"),
    }
}
