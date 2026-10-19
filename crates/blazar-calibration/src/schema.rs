//! Calibration file schema (calibration-format §2–§6).
//!
//! A calibration is the contract between the Python calibrator and the Rust
//! runtime. It self-declares its key (with wildcards), its provenance, and the
//! encoder body (portable common params + opaque codec passthrough).

use serde::Deserialize;
use std::collections::BTreeMap;

/// The only schema version the runtime understands today. Unknown versions are
/// rejected rather than misread (calibration-format §2).
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Where a calibration came from. Also the tie-break order at equal
/// specificity: `User > Calibrator > Default` (calibration-format §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Default,
    Calibrator,
    User,
}

impl Source {
    /// Higher wins at equal specificity.
    pub fn rank(self) -> u8 {
        match self {
            Source::Default => 0,
            Source::Calibrator => 1,
            Source::User => 2,
        }
    }
}

/// One axis value in a calibration key: a concrete match or a wildcard.
///
/// `Wild` ("*") matches anything and contributes zero specificity; `Exact`
/// matches only its literal and contributes one point (calibration-format §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchAxis {
    Wild,
    Exact(String),
}

impl MatchAxis {
    pub fn matches(&self, concrete: &str) -> bool {
        match self {
            MatchAxis::Wild => true,
            MatchAxis::Exact(v) => v == concrete,
        }
    }

    pub fn specificity(&self) -> u32 {
        match self {
            MatchAxis::Wild => 0,
            MatchAxis::Exact(_) => 1,
        }
    }

    fn from_str(s: &str) -> Self {
        if s == "*" {
            MatchAxis::Wild
        } else {
            MatchAxis::Exact(s.to_string())
        }
    }
}

/// Raw TOML form of the model identity portion of the key.
#[derive(Debug, Clone, Deserialize)]
pub struct RawModelKey {
    pub family: String,
    pub version: String,
    pub quant: String,
    pub size: String,
}

/// Raw TOML form of `[key]`.
#[derive(Debug, Clone, Deserialize)]
pub struct RawKey {
    pub model: RawModelKey,
    pub scene: String,
    pub codec: String,
    pub target: String,
}

/// Raw TOML form of `[video.common]` (calibration-format §5). Small, fixed,
/// cross-codec semantics; the encoder backend translates.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VideoCommon {
    pub rate_control: String,
    #[serde(default)]
    pub crf: Option<u32>,
    #[serde(default)]
    pub bitrate_kbps: Option<u32>,
    pub keyint: u32,
    pub bframes: u32,
    pub pix_fmt: String,
}

/// Raw TOML form of `[video]`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VideoBody {
    pub common: VideoCommon,
    /// Opaque per-codec passthrough string, keyed by codec name. Blazar does
    /// not interpret the value; a key naming a non-active codec is a load error
    /// (checked in the loader, calibration-format §5).
    #[serde(default)]
    pub codec_params: BTreeMap<String, String>,
}

/// Optional provenance (calibration-format §4.1). Functionally ignored; logged.
#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
pub struct Meta {
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub created: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// The full TOML document, as deserialized before validation.
#[derive(Debug, Clone, Deserialize)]
pub struct RawCalibration {
    pub schema_version: u32,
    pub source: Source,
    pub key: RawKey,
    pub video: VideoBody,
    #[serde(default)]
    pub meta: Meta,
}

/// A validated, resolution-ready calibration key with wildcard-aware axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationKey {
    pub family: MatchAxis,
    pub version: MatchAxis,
    pub quant: MatchAxis,
    pub size: MatchAxis,
    pub scene: MatchAxis,
    /// Codec is NOT a wildcard axis: a calibration is always for one codec
    /// (the body's codec_params are codec-specific). Stored concrete.
    pub codec: String,
    /// Target is likewise concrete (low_latency|balanced|quality).
    pub target: String,
}

impl CalibrationKey {
    pub(crate) fn from_raw(raw: &RawKey) -> Self {
        Self {
            family: MatchAxis::from_str(&raw.model.family),
            version: MatchAxis::from_str(&raw.model.version),
            quant: MatchAxis::from_str(&raw.model.quant),
            size: MatchAxis::from_str(&raw.model.size),
            scene: MatchAxis::from_str(&raw.scene),
            codec: raw.codec.clone(),
            target: raw.target.clone(),
        }
    }

    /// Specificity is the count of non-wildcard wildcard-capable axes (codec and
    /// target are not wildcard axes and do not count — they must match exactly
    /// for a candidate to be considered at all).
    pub fn specificity(&self) -> u32 {
        self.family.specificity()
            + self.version.specificity()
            + self.quant.specificity()
            + self.size.specificity()
            + self.scene.specificity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_rank_orders_user_highest() {
        assert!(Source::User.rank() > Source::Calibrator.rank());
        assert!(Source::Calibrator.rank() > Source::Default.rank());
    }

    #[test]
    fn wildcard_axis_matches_anything_zero_specificity() {
        assert!(MatchAxis::Wild.matches("anything"));
        assert_eq!(MatchAxis::Wild.specificity(), 0);
    }

    #[test]
    fn exact_axis_matches_only_literal_one_specificity() {
        let a = MatchAxis::Exact("cinematic".into());
        assert!(a.matches("cinematic"));
        assert!(!a.matches("high_motion"));
        assert_eq!(a.specificity(), 1);
    }

    #[test]
    fn specificity_counts_non_wildcard_axes() {
        let raw = RawKey {
            model: RawModelKey {
                family: "wan".into(),
                version: "2.1".into(),
                quant: "*".into(),
                size: "*".into(),
            },
            scene: "cinematic".into(),
            codec: "svtav1".into(),
            target: "balanced".into(),
        };
        let key = CalibrationKey::from_raw(&raw);
        // family + version + scene = 3; quant + size wild = 0.
        assert_eq!(key.specificity(), 3);
    }

    #[test]
    fn full_wildcard_default_has_zero_specificity() {
        let raw = RawKey {
            model: RawModelKey {
                family: "*".into(),
                version: "*".into(),
                quant: "*".into(),
                size: "*".into(),
            },
            scene: "*".into(),
            codec: "svtav1".into(),
            target: "balanced".into(),
        };
        assert_eq!(CalibrationKey::from_raw(&raw).specificity(), 0);
    }
}
