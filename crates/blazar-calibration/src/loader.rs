//! Loading + validation with per-file safe failure (calibration-format §4.1).
//!
//! A single malformed calibration is rejected individually with a warning; it
//! never aborts the whole load. The resolution chain still terminates at the
//! default, so a bad file degrades to less-specific tuning rather than breaking
//! the runtime.

use thiserror::Error;

use crate::schema::{
    CalibrationKey, Meta, RawCalibration, Source, VideoBody, SUPPORTED_SCHEMA_VERSION,
};

/// A validated calibration ready to enter the index.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub key: CalibrationKey,
    pub source: Source,
    pub video: VideoBody,
    pub meta: Meta,
    /// Where this calibration was loaded from, for conflict/resolution logging.
    pub origin: String,
}

/// Why a single calibration file was rejected. Rejection is per-file; the
/// caller logs and continues.
#[derive(Debug, Error, PartialEq)]
pub enum LoadError {
    #[error("TOML parse error: {0}")]
    Parse(String),

    #[error("unknown schema_version {found} (runtime supports {supported})")]
    UnknownSchemaVersion { found: u32, supported: u32 },

    #[error("unknown rate_control mode {0:?} (expected crf|vbr|cbr)")]
    UnknownRateControl(String),

    #[error(
        "codec_params names codec {found:?} but this calibration's key codec is {key_codec:?}"
    )]
    CodecParamsMismatch { found: String, key_codec: String },

    #[error("rate_control=crf requires a crf value")]
    MissingCrf,

    #[error("rate_control={mode} requires a bitrate_kbps value")]
    MissingBitrate { mode: String },
}

/// Parse and validate one calibration document from TOML text.
///
/// `origin` is a human label (file path) carried into the index for logging.
pub fn load_str(text: &str, origin: impl Into<String>) -> Result<Calibration, LoadError> {
    let origin = origin.into();
    let raw: RawCalibration = toml::from_str(text).map_err(|e| LoadError::Parse(e.to_string()))?;
    validate(raw, origin)
}

fn validate(raw: RawCalibration, origin: String) -> Result<Calibration, LoadError> {
    // schema_version first (calibration-format §2).
    if raw.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(LoadError::UnknownSchemaVersion {
            found: raw.schema_version,
            supported: SUPPORTED_SCHEMA_VERSION,
        });
    }

    let key = CalibrationKey::from_raw(&raw.key);

    // rate_control is a known mode.
    let rc = raw.video.common.rate_control.as_str();
    match rc {
        "crf" => {
            if raw.video.common.crf.is_none() {
                return Err(LoadError::MissingCrf);
            }
        }
        "vbr" | "cbr" => {
            if raw.video.common.bitrate_kbps.is_none() {
                return Err(LoadError::MissingBitrate {
                    mode: rc.to_string(),
                });
            }
        }
        other => return Err(LoadError::UnknownRateControl(other.to_string())),
    }

    // codec_params keys must name the active (key) codec, else load error
    // (the escape hatch must not silently target the wrong encoder).
    for cp_codec in raw.video.codec_params.keys() {
        if cp_codec != &key.codec {
            return Err(LoadError::CodecParamsMismatch {
                found: cp_codec.clone(),
                key_codec: key.codec.clone(),
            });
        }
    }

    Ok(Calibration {
        key,
        source: raw.source,
        video: raw.video,
        meta: raw.meta,
        origin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
schema_version = 1
source = "user"
[key]
model = { family = "wan", version = "2.1", quant = "*", size = "*" }
scene = "cinematic"
codec = "svtav1"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
keyint = 120
bframes = 3
pix_fmt = "yuv420p"
[video.codec_params]
svtav1 = "preset=6:tune=0"
[meta]
author = "jhurley"
note = "hand-tuned"
"#;

    #[test]
    fn loads_valid_calibration() {
        let c = load_str(GOOD, "users/wan21_cinematic.toml").unwrap();
        assert_eq!(c.source, Source::User);
        assert_eq!(c.key.codec, "svtav1");
        assert_eq!(c.key.specificity(), 3); // family+version+scene
        assert_eq!(c.meta.author.as_deref(), Some("jhurley"));
        assert_eq!(c.origin, "users/wan21_cinematic.toml");
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let txt = GOOD.replace("schema_version = 1", "schema_version = 99");
        let err = load_str(&txt, "x.toml").unwrap_err();
        assert_eq!(
            err,
            LoadError::UnknownSchemaVersion {
                found: 99,
                supported: 1
            }
        );
    }

    #[test]
    fn rejects_codec_params_for_wrong_codec() {
        // key codec is svtav1 but codec_params names x265.
        let txt = GOOD.replace(r#"svtav1 = "preset=6:tune=0""#, r#"x265 = "crf=20""#);
        let err = load_str(&txt, "x.toml").unwrap_err();
        assert_eq!(
            err,
            LoadError::CodecParamsMismatch {
                found: "x265".into(),
                key_codec: "svtav1".into()
            }
        );
    }

    #[test]
    fn rejects_unknown_rate_control() {
        let txt = GOOD.replace(r#"rate_control = "crf""#, r#"rate_control = "magic""#);
        let err = load_str(&txt, "x.toml").unwrap_err();
        assert_eq!(err, LoadError::UnknownRateControl("magic".into()));
    }

    #[test]
    fn crf_without_value_rejected() {
        let txt = GOOD.replace("crf = 30\n", "");
        let err = load_str(&txt, "x.toml").unwrap_err();
        assert_eq!(err, LoadError::MissingCrf);
    }

    #[test]
    fn malformed_toml_is_parse_error_not_panic() {
        let err = load_str("this is not toml = = =", "x.toml").unwrap_err();
        assert!(matches!(err, LoadError::Parse(_)));
    }
}
