//! Translation from the backend-neutral calibration config to NVENC parameters.
//!
//! Kept free of SDK types on purpose: this is where the policy lives (what a
//! calibration's `rate_control`, `keyint`, `bframes`, `pix_fmt` mean to NVENC),
//! and it is unit-tested on machines without a GPU. The driver-facing code in
//! [`worker`](super::worker) only applies these values.
//!
//! Cross-backend equivalence is *not* assumed. A calibration is keyed by
//! backend (`svtav1`, `nvenc_av1`, ...), so `crf = 32` means "whatever 32 means
//! to this backend". Mapping CRF to NVENC's constant-QP mode is deliberate: it
//! is the closest single-knob analogue, and the calibrator is the place that
//! measures what value is right.

use thiserror::Error;

use crate::config::{EncoderConfig, RateControl};

/// NVENC rate-control settings, in the units the API expects (bits/second).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvencRateControl {
    /// Constant QP. NVENC's AV1 QP range is 0..=255 (not H.264's 0..=51).
    ConstQp { qp: u32 },
    Vbr {
        average_bitrate_bps: u32,
        max_bitrate_bps: u32,
        /// VBV size in bits (NVENC's unit), one second of target bitrate.
        vbv_bits: u32,
    },
    Cbr {
        bitrate_bps: u32,
        /// VBV size in bits (NVENC's unit), one second of target bitrate.
        vbv_bits: u32,
    },
}

/// Colour signalling written into the AV1 sequence header.
///
/// The CPU preprocess stage produces full-range YUV with the BT.709 matrix and
/// leaves colour untagged, so the encoder must tag it or players guess. This
/// does not change pixels; it stops the decoder from guessing wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvencColorConfig {
    pub full_range: bool,
    /// BT.709 primaries + transfer + matrix (the preprocess assumption).
    pub bt709: bool,
}

impl NvencColorConfig {
    pub const BT709_FULL_RANGE: Self = Self {
        full_range: true,
        bt709: true,
    };
}

/// Everything the worker needs to start a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvencParams {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub gop_length: u32,
    /// `NVENC` `frameIntervalP`: `1` = no B-frames, `1 + bframes` otherwise.
    pub frame_interval_p: i32,
    pub rate_control: NvencRateControl,
    pub color: NvencColorConfig,
}

#[derive(Debug, Error, PartialEq)]
pub enum NvencConfigError {
    #[error("NVENC takes NV12/yuv420p input, calibration asked for pix_fmt {0:?}")]
    UnsupportedPixelFormat(String),
    #[error("NVENC does not understand SVT-style codec_params {0:?}; use a nvenc_av1 calibration")]
    UnsupportedCodecParams(String),
    #[error("NVENC supports at most 4 B-frames (got bframes={0})")]
    BframesUnsupported(u32),
    #[error("keyint must be at least 1")]
    InvalidKeyint,
    #[error("4:2:0 needs even dimensions, got {width}x{height}")]
    OddDimensions { width: u32, height: u32 },
    #[error("invalid fps {0}")]
    InvalidFps(f64),
    #[error("rate-control value out of range: {0}")]
    InvalidRateControl(&'static str),
}

impl NvencParams {
    /// Translate a calibration-derived config for one stream.
    pub fn from_encoder_config(
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        fps: f64,
    ) -> Result<Self, NvencConfigError> {
        if cfg.pix_fmt != "yuv420p" {
            return Err(NvencConfigError::UnsupportedPixelFormat(
                cfg.pix_fmt.clone(),
            ));
        }
        if !cfg.codec_params.is_empty() {
            return Err(NvencConfigError::UnsupportedCodecParams(
                cfg.codec_params.clone(),
            ));
        }
        // NVENC's frameIntervalP is 1 + bframes and accepts up to 5.
        if cfg.bframes > 4 {
            return Err(NvencConfigError::BframesUnsupported(cfg.bframes));
        }
        if cfg.keyint == 0 {
            return Err(NvencConfigError::InvalidKeyint);
        }
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(NvencConfigError::OddDimensions { width, height });
        }
        let (fps_num, fps_den) = fps_rational(fps)?;

        Ok(Self {
            width,
            height,
            fps_num,
            fps_den,
            gop_length: cfg.keyint,
            frame_interval_p: cfg.bframes as i32 + 1,
            rate_control: rate_control(cfg.rate_control)?,
            color: NvencColorConfig::BT709_FULL_RANGE,
        })
    }
}

/// NVENC's framerate is a 32-bit rational. Integer rates are exact; fractional
/// rates use a milli-fps rational, which is close enough for the encoder's rate
/// hint. Real timing comes from the timebase and the supplied timestamps.
fn fps_rational(fps: f64) -> Result<(u32, u32), NvencConfigError> {
    if !fps.is_finite() || fps <= 0.0 {
        return Err(NvencConfigError::InvalidFps(fps));
    }
    let rounded = fps.round();
    let (num, den) = if (fps - rounded).abs() < 1e-6 {
        (rounded, 1.0)
    } else {
        ((fps * 1000.0).round(), 1000.0)
    };
    if num < 1.0 || num > f64::from(u32::MAX) {
        return Err(NvencConfigError::InvalidFps(fps));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok((num as u32, den as u32))
}

fn rate_control(rc: RateControl) -> Result<NvencRateControl, NvencConfigError> {
    match rc {
        RateControl::Crf(qp) => {
            if qp > 255 {
                return Err(NvencConfigError::InvalidRateControl(
                    "CRF/QP must be 0..=255 for NVENC AV1",
                ));
            }
            Ok(NvencRateControl::ConstQp { qp })
        }
        RateControl::Vbr(kbps) => {
            let average = bitrate_bps(kbps)?;
            Ok(NvencRateControl::Vbr {
                average_bitrate_bps: average,
                max_bitrate_bps: average
                    .checked_mul(2)
                    .ok_or(NvencConfigError::InvalidRateControl("bitrate too large"))?,
                // One second of buffering, in bits (NVENC's unit).
                vbv_bits: average,
            })
        }
        RateControl::Cbr(kbps) => {
            let bitrate = bitrate_bps(kbps)?;
            Ok(NvencRateControl::Cbr {
                bitrate_bps: bitrate,
                vbv_bits: bitrate,
            })
        }
    }
}

fn bitrate_bps(kbps: u32) -> Result<u32, NvencConfigError> {
    if kbps == 0 {
        return Err(NvencConfigError::InvalidRateControl(
            "bitrate must be greater than zero",
        ));
    }
    kbps.checked_mul(1000)
        .ok_or(NvencConfigError::InvalidRateControl("bitrate too large"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EncoderConfig;

    fn cfg(rc: RateControl, keyint: u32, bframes: u32) -> EncoderConfig {
        EncoderConfig {
            rate_control: rc,
            keyint,
            bframes,
            pix_fmt: "yuv420p".into(),
            codec_params: String::new(),
        }
    }

    #[test]
    fn crf_maps_to_constant_qp() {
        let p = NvencParams::from_encoder_config(&cfg(RateControl::Crf(32), 48, 0), 832, 480, 24.0)
            .unwrap();
        assert_eq!(p.rate_control, NvencRateControl::ConstQp { qp: 32 });
        assert_eq!(p.gop_length, 48);
        assert_eq!(p.frame_interval_p, 1);
        assert_eq!((p.fps_num, p.fps_den), (24, 1));
        assert_eq!(p.color, NvencColorConfig::BT709_FULL_RANGE);
    }

    #[test]
    fn vbr_derives_max_and_vbv_from_the_target() {
        let p = NvencParams::from_encoder_config(
            &cfg(RateControl::Vbr(5_000), 120, 0),
            1280,
            720,
            30.0,
        )
        .unwrap();
        assert_eq!(
            p.rate_control,
            NvencRateControl::Vbr {
                average_bitrate_bps: 5_000_000,
                max_bitrate_bps: 10_000_000,
                vbv_bits: 5_000_000,
            }
        );
    }

    #[test]
    fn cbr_uses_the_same_rate_for_average_and_max() {
        let p =
            NvencParams::from_encoder_config(&cfg(RateControl::Cbr(3_000), 60, 0), 640, 360, 30.0)
                .unwrap();
        assert_eq!(
            p.rate_control,
            NvencRateControl::Cbr {
                bitrate_bps: 3_000_000,
                vbv_bits: 3_000_000,
            }
        );
    }

    #[test]
    fn fractional_fps_uses_a_milli_rational() {
        let p =
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(30), 48, 0), 640, 360, 23.976)
                .unwrap();
        assert_eq!((p.fps_num, p.fps_den), (23_976, 1_000));
    }

    #[test]
    fn bframes_map_to_frame_interval_p() {
        let p =
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(30), 120, 3), 640, 360, 24.0)
                .unwrap();
        assert_eq!(p.frame_interval_p, 4, "frameIntervalP = 1 + bframes");
    }

    #[test]
    fn more_than_four_bframes_is_rejected() {
        assert_eq!(
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(30), 120, 5), 640, 360, 24.0)
                .unwrap_err(),
            NvencConfigError::BframesUnsupported(5)
        );
    }

    #[test]
    fn non_420_pixel_format_is_rejected() {
        let mut c = cfg(RateControl::Crf(30), 48, 0);
        c.pix_fmt = "yuv444p".into();
        assert!(matches!(
            NvencParams::from_encoder_config(&c, 640, 360, 24.0),
            Err(NvencConfigError::UnsupportedPixelFormat(_))
        ));
    }

    #[test]
    fn svt_codec_params_are_rejected() {
        let mut c = cfg(RateControl::Crf(30), 48, 0);
        c.codec_params = "preset=10".into();
        assert!(matches!(
            NvencParams::from_encoder_config(&c, 640, 360, 24.0),
            Err(NvencConfigError::UnsupportedCodecParams(_))
        ));
    }

    #[test]
    fn odd_dimensions_are_rejected() {
        assert_eq!(
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(30), 48, 0), 833, 480, 24.0)
                .unwrap_err(),
            NvencConfigError::OddDimensions {
                width: 833,
                height: 480
            }
        );
    }

    #[test]
    fn zero_keyint_is_rejected() {
        assert_eq!(
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(30), 0, 0), 640, 360, 24.0)
                .unwrap_err(),
            NvencConfigError::InvalidKeyint
        );
    }

    #[test]
    fn out_of_range_qp_is_rejected() {
        assert!(matches!(
            NvencParams::from_encoder_config(&cfg(RateControl::Crf(300), 48, 0), 640, 360, 24.0),
            Err(NvencConfigError::InvalidRateControl(_))
        ));
    }

    #[test]
    fn zero_bitrate_is_rejected() {
        assert!(matches!(
            NvencParams::from_encoder_config(&cfg(RateControl::Cbr(0), 48, 0), 640, 360, 24.0),
            Err(NvencConfigError::InvalidRateControl(_))
        ));
    }
}
