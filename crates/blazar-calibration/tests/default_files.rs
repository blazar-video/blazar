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
    // crate dir is blazar/crates/blazar-calibration; repo calibrations/ is ../../../calibrations
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../calibrations/default")
        .canonicalize()
        .expect("calibrations/default must exist")
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

#[test]
fn all_default_files_load() {
    let idx = load_all();
    // Three targets shipped for SVT-AV1.
    assert_eq!(idx.len(), 3, "expected 3 default SVT-AV1 calibrations");
}

#[test]
fn each_target_resolves_to_its_default() {
    let idx = load_all();
    let id = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
    for target in [Target::LowLatency, Target::Balanced, Target::Quality] {
        let q = Query::new(&id, "any_unseen_scene", "svtav1", target);
        let cal = idx
            .resolve(&q)
            .unwrap_or_else(|| panic!("no default resolved for {:?}", target));
        assert_eq!(cal.source, blazar_calibration::Source::Default);
        assert_eq!(cal.key.target, target.as_key());
    }
}
