// SPDX-License-Identifier: Apache-2.0
//! Validates the actual shipped default calibration files in `calibrations/`.
//!
//! This is the contract check between the data (TOML on disk) and the loader:
//! the files the runtime ships must load, validate, and resolve as the terminal
//! default for their (codec, target).

use std::fs;
use std::path::PathBuf;

use blazar_calibration::{load_str, CalibrationIndex, Query};
use blazar_core::{ModelIdentity, Target};

fn calibrations_dir() -> PathBuf {
    // The calibrations live in their own repository. Locally they sit beside the
    // project; in CI they are checked out into the workspace, so the environment
    // wins when it is set. The python tests use the same variable.
    if let Ok(path) = std::env::var("BLAZAR_CALIBRATIONS") {
        return PathBuf::from(path)
            .canonicalize()
            .expect("BLAZAR_CALIBRATIONS must point at a directory");
    }
    // crate dir is blazar/crates/blazar-calibration; repo calibrations/ is ../../../calibrations
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../calibrations/default")
        .canonicalize()
        .expect("calibrations/default must exist, or set BLAZAR_CALIBRATIONS")
}

fn load_all() -> CalibrationIndex {
    let mut idx = CalibrationIndex::new();
    for entry in fs::read_dir(calibrations_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            let text = fs::read_to_string(&path).unwrap();
            let cal = load_str(&text, path.display().to_string())
                .unwrap_or_else(|e| panic!("default file {} failed to load: {e}", path.display()));
            idx.insert(cal);
        }
    }
    idx
}

/// Codec keys that must ship a terminal default for every target. A deployment
/// whose backend has no default here fails resolution at stream open.
const CODEC_KEYS: [&str; 9] = [
    "svtav1",
    "nvenc_av1",
    "nvenc_h264",
    "nvenc_hevc",
    "videotoolbox_h264",
    "videotoolbox_hevc",
    "x264",
    "x265",
    "vp9",
];

/// `mock` ships one file: it emits no compressed data, so a quality ladder for
/// it would be meaningless, and the single `balanced` default is what tests and
/// pipelines resolve against.
const MOCK_KEY: &str = "mock";

#[test]
fn all_default_files_load() {
    let idx = load_all();
    assert_eq!(
        idx.len(),
        CODEC_KEYS.len() * 3 + 1,
        "expected three targets per real codec and one for the mock"
    );
}

#[test]
fn the_mock_has_a_balanced_default() {
    // Without this, every `backend="mock"` stream fails at open with
    // "no calibration resolved", which is how the rav1e leftover was found.
    let idx = load_all();
    let query = Query::new(
        &ModelIdentity {
            family: "wan".into(),
            version: "2.1".into(),
            quant: "bf16".into(),
            size: "1.3b".into(),
        },
        "*",
        MOCK_KEY,
        Target::Balanced,
    );
    assert!(idx.resolve(&query).is_some(), "mock default must resolve");
}

#[test]
fn each_codec_and_target_resolves_to_its_default() {
    let idx = load_all();
    let id = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
    for codec in CODEC_KEYS {
        for target in [Target::LowLatency, Target::Balanced, Target::Quality] {
            let q = Query::new(&id, "any_unseen_scene", codec, target);
            let cal = idx
                .resolve(&q)
                .unwrap_or_else(|| panic!("no default resolved for {codec}/{target:?}"));
            assert_eq!(cal.source, blazar_calibration::Source::Default);
            assert_eq!(cal.key.codec, codec);
            assert_eq!(cal.key.target, target.as_key());
        }
    }
}
