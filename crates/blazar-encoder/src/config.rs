//! Translating a resolved calibration into encoder settings.
//!
//! `[video.common]` has fixed cross-codec semantics; each backend translates it
//! to its own knobs. `[video.codec_params]` is opaque passthrough appended
//! verbatim (and wins on overlap, like ffmpeg). This module produces the
//! backend-neutral [`EncoderConfig`]; each backend interprets it.

use blazar_calibration::{Calibration, VideoCommon};

/// Rate-control mode, backend-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    Crf(u32),
    Vbr(u32),
    Cbr(u32),
}

/// Backend-neutral encoder configuration derived from a calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub rate_control: RateControl,
    pub keyint: u32,
    pub bframes: u32,
    pub pix_fmt: String,
    /// Verbatim codec-specific passthrough for the active codec, or empty.
    pub codec_params: String,
}

impl EncoderConfig {
    /// Build from a resolved calibration. `active_codec` selects which
    /// codec_params entry applies (the loader already guaranteed any present
    /// entry names this codec).
    pub fn from_calibration(cal: &Calibration, active_codec: &str) -> Self {
        let c: &VideoCommon = &cal.video.common;
        let rate_control = match c.rate_control.as_str() {
            "crf" => RateControl::Crf(c.crf.unwrap_or(30)),
            "vbr" => RateControl::Vbr(c.bitrate_kbps.unwrap_or(3000)),
            "cbr" => RateControl::Cbr(c.bitrate_kbps.unwrap_or(3000)),
            // loader validated this; default defensively.
            _ => RateControl::Crf(30),
        };
        let codec_params = cal
            .video
            .codec_params
            .get(active_codec)
            .cloned()
            .unwrap_or_default();
        Self {
            rate_control,
            keyint: c.keyint,
            bframes: c.bframes,
            pix_fmt: c.pix_fmt.clone(),
            codec_params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_calibration::load_str;

    fn cal(body: &str) -> Calibration {
        let txt = format!(
            r#"
schema_version = 1
source = "calibrator"
[key]
model = {{ family = "wan", version = "2.1", quant = "*", size = "*" }}
scene = "cinematic"
codec = "svtav1"
target = "balanced"
{body}
"#
        );
        load_str(&txt, "t.toml").unwrap()
    }

    #[test]
    fn crf_translation() {
        let c = cal(r#"[video.common]
rate_control = "crf"
crf = 28
keyint = 120
bframes = 3
pix_fmt = "yuv420p"
[video.codec_params]
svtav1 = "preset=6:tune=0""#);
        let cfg = EncoderConfig::from_calibration(&c, "svtav1");
        assert_eq!(cfg.rate_control, RateControl::Crf(28));
        assert_eq!(cfg.keyint, 120);
        assert_eq!(cfg.codec_params, "preset=6:tune=0");
    }

    #[test]
    fn vbr_uses_bitrate() {
        let c = cal(r#"[video.common]
rate_control = "vbr"
bitrate_kbps = 5000
keyint = 60
bframes = 2
pix_fmt = "yuv420p""#);
        let cfg = EncoderConfig::from_calibration(&c, "svtav1");
        assert_eq!(cfg.rate_control, RateControl::Vbr(5000));
        assert_eq!(cfg.codec_params, ""); // none present
    }
}
