//! Structured model identity — the calibration key axis for the model.
//!
//! Structured (`family/version/quant/size`) rather than a flat string so
//! calibrations can share and fall back across axes with wildcards
//! (architecture §3, calibration-format §3). This type is the *concrete*
//! identity of a running model; the calibration *key* uses a wildcard-capable
//! form that lives in `blazar-calibration`.

use serde::{Deserialize, Serialize};

/// Concrete identity of the model producing frames. Every field is known at
/// `create_encoder` time (it describes a specific deployment).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Model family, e.g. "wan", "ltx", "cosmos".
    pub family: String,
    /// Version string, e.g. "2.1".
    pub version: String,
    /// Quantization tag, e.g. "bf16", "fp8", "int8".
    pub quant: String,
    /// Size/variant tag, e.g. "1.3b", "14b".
    pub size: String,
}

impl ModelIdentity {
    pub fn new(
        family: impl Into<String>,
        version: impl Into<String>,
        quant: impl Into<String>,
        size: impl Into<String>,
    ) -> Self {
        Self {
            family: family.into(),
            version: version.into(),
            quant: quant.into(),
            size: size.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trips_and_compares() {
        let a = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
        let b = ModelIdentity::new("wan", "2.1", "bf16", "1.3b");
        assert_eq!(a, b);
        assert_eq!(a.family, "wan");
        assert_eq!(a.size, "1.3b");
    }
}
