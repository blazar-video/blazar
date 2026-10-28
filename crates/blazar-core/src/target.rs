// SPDX-License-Identifier: Apache-2.0
//! Operator target — states intent/constraint; resolves BOTH the calibration
//! and the scheduling/lookahead policy (architecture §3).
//!
//! v0 is the preset form. The explicit `{bitrate, max_latency_ms}` form is a
//! later addition; modeled as a separate variant so adding it does not change
//! the preset callers.

use serde::{Deserialize, Serialize};

/// What the operator is optimizing for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Target {
    /// Encode-as-available, zero lookahead. A streaming runtime can be asked
    /// for this; a batch runtime cannot stream regardless.
    LowLatency,
    /// Buffer a lookahead window; the default.
    Balanced,
    /// Larger lookahead window, quality-first.
    Quality,
}

impl Target {
    /// The lookahead window in frames implied by this target (architecture §4:
    /// scheduling falls out of the latency budget, it is not a separate knob).
    /// Concrete values are a starting point, tunable later.
    pub fn lookahead_frames(self) -> u32 {
        match self {
            Target::LowLatency => 0,
            Target::Balanced => 8,
            Target::Quality => 32,
        }
    }

    /// The canonical string used as the calibration `target` key axis
    /// (calibration-format §3).
    pub fn as_key(self) -> &'static str {
        match self {
            Target::LowLatency => "low_latency",
            Target::Balanced => "balanced",
            Target::Quality => "quality",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_latency_has_zero_lookahead() {
        assert_eq!(Target::LowLatency.lookahead_frames(), 0);
    }

    #[test]
    fn lookahead_increases_with_quality() {
        assert!(Target::Quality.lookahead_frames() > Target::Balanced.lookahead_frames());
        assert!(Target::Balanced.lookahead_frames() > Target::LowLatency.lookahead_frames());
    }

    #[test]
    fn key_strings_match_calibration_format() {
        assert_eq!(Target::LowLatency.as_key(), "low_latency");
        assert_eq!(Target::Balanced.as_key(), "balanced");
        assert_eq!(Target::Quality.as_key(), "quality");
    }
}
