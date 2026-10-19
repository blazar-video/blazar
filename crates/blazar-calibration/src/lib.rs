//! `blazar-calibration` — the calibration data contract: schema, loader, and
//! resolution index (calibration-format.md; architecture §5.1).
//!
//! The runtime's lookup code never changes as calibrations get richer — only
//! the data does. Specificity is the only ranking axis; `source` breaks ties
//! at equal specificity (User > Calibrator > Default) and the tie-break is
//! logged.

pub mod index;
pub mod loader;
pub mod schema;

pub use index::{CalibrationIndex, Query};
pub use loader::{load_str, Calibration, LoadError};
pub use schema::{
    CalibrationKey, MatchAxis, Meta, Source, VideoBody, VideoCommon, SUPPORTED_SCHEMA_VERSION,
};
