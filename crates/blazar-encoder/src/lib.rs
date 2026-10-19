//! `blazar-encoder` — the media-encoder abstraction (architecture §3.1, §6).
//!
//! The "encoder backend" generalizes to a **media encoder** with video
//! (SVT-AV1/NVENC) and audio (Opus) variants feeding one muxer. Phase 1 ships
//! the trait plus a [`MockVideoEncoder`] that implements the full contract
//! (keyframe cadence from `keyint`, ordered PTS, flush/finish) without a real
//! codec, so the pipeline is exercisable end-to-end before the SVT-AV1 FFI
//! bindings land behind the `svtav1` feature.

pub mod av1;
pub mod config;
pub mod nvenc;

use blazar_core::{EncodedUnit, Timebase, TrackId};
use blazar_preprocess::PreprocessedFrame;
use thiserror::Error;

pub use config::{EncoderConfig, RateControl};

#[derive(Debug, Error, PartialEq)]
pub enum EncodeError {
    #[error("encoder received a frame after finish()")]
    AfterFinish,
    #[error("backend error: {0}")]
    Backend(String),
}

/// A video encoder backend. Stateful: frame N depends on N-1's references and
/// rate-control history, which is why the runtime keeps one encoder per stream
/// and never reorders submits (architecture §4).
pub trait VideoEncoder {
    /// Feed one preprocessed frame. Returns zero or more encoded units (an
    /// encoder with lookahead/B-frames may emit nothing until enough frames
    /// have accumulated).
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError>;

    /// Drain buffered frames without ending the stream (segment boundary).
    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError>;

    /// End of stream: drain everything and refuse further frames.
    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError>;

    /// Force the next emitted frame to be a keyframe (WebRTC PLI / scene cut).
    fn request_keyframe(&mut self);

    /// Container-facing codec configuration (AV1 `av1C`, H.264 `avcC`, HEVC
    /// `hvcC` body), once the backend knows it.
    ///
    /// Real encoders only learn this from their first keyframe (the sequence
    /// header lives in the bitstream), so this returns `None` until then. The
    /// mock has no real codec and always returns `None`; the runtime then falls
    /// back to the explicit `Output::video_extradata` override. Returning
    /// `None` when no override exists is a hard error in the runtime: muxing an
    /// empty configuration would produce an unplayable stream.
    fn codec_config(&self) -> Option<Vec<u8>> {
        None
    }
}

/// A deterministic mock video encoder implementing the full contract.
///
/// It does not compress; each frame becomes one `EncodedUnit` whose bytes are a
/// tiny deterministic header (so output is inspectable and tests are exact). It
/// honors `keyint` for keyframe cadence and forced keyframes, assigns monotonic
/// PTS/DTS from frame PTS, and computes per-frame duration from fps. This is the
/// contract the real SVT-AV1 backend must also satisfy.
pub struct MockVideoEncoder {
    keyint: u32,
    fps: f64,
    timebase: Timebase,
    frames_since_key: u32,
    frame_count: u64,
    force_key: bool,
    finished: bool,
}

impl MockVideoEncoder {
    pub fn new(cfg: &EncoderConfig, fps: f64, timebase: Timebase) -> Self {
        Self {
            keyint: cfg.keyint.max(1),
            fps,
            timebase,
            frames_since_key: 0,
            frame_count: 0,
            force_key: false,
            finished: false,
        }
    }

    fn duration_ticks(&self) -> u64 {
        self.timebase.ticks_per_frame(self.fps)
    }
}

impl VideoEncoder for MockVideoEncoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        // Keyframe if it's the first frame, a forced key, or keyint elapsed.
        let keyframe =
            self.frame_count == 0 || self.force_key || self.frames_since_key >= self.keyint;
        if keyframe {
            self.frames_since_key = 0;
            self.force_key = false;
        }
        self.frames_since_key += 1;
        self.frame_count += 1;

        // Deterministic "bitstream": marker + dims + a checksum of luma. Enough
        // to prove the muxer wrote the right frames in order.
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(if keyframe { b"KEY0" } else { b"DLT0" });
        bytes.extend_from_slice(&frame.width.to_le_bytes());
        bytes.extend_from_slice(&frame.height.to_le_bytes());
        let luma_sum: u64 = frame.y.iter().map(|&b| b as u64).sum();
        bytes.extend_from_slice(&luma_sum.to_le_bytes());

        // No reordering in the mock: dts == pts. Real backends with B-frames
        // will produce dts < pts for reordered frames.
        let unit = EncodedUnit {
            track_id: TrackId::Video,
            bytes,
            pts: frame.pts,
            dts: frame.pts,
            keyframe,
            duration: self.duration_ticks(),
        };
        Ok(vec![unit])
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        // Mock has no internal buffer; nothing to drain.
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.finished = true;
        Ok(Vec::new())
    }

    fn request_keyframe(&mut self) {
        self.force_key = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::Pts;
    use blazar_preprocess::PreprocessedFrame;

    fn frame(pts: u64) -> PreprocessedFrame {
        PreprocessedFrame {
            width: 2,
            height: 2,
            y: vec![100, 100, 100, 100],
            u: vec![128],
            v: vec![128],
            pts: Pts(pts),
        }
    }

    fn cfg(keyint: u32) -> EncoderConfig {
        EncoderConfig {
            rate_control: RateControl::Crf(30),
            keyint,
            bframes: 0,
            pix_fmt: "yuv420p".into(),
            codec_params: String::new(),
        }
    }

    #[test]
    fn first_frame_is_keyframe() {
        let mut enc = MockVideoEncoder::new(&cfg(120), 24.0, Timebase::MPEG_90K);
        let out = enc.encode(&frame(0)).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].keyframe);
        assert_eq!(out[0].duration, 3750); // 90000/24
    }

    #[test]
    fn keyint_drives_cadence() {
        let mut enc = MockVideoEncoder::new(&cfg(3), 24.0, Timebase::MPEG_90K);
        let mut keyframes = 0;
        for i in 0..10 {
            let out = enc.encode(&frame(i * 3750)).unwrap();
            if out[0].keyframe {
                keyframes += 1;
            }
        }
        // keyint=3 over 10 frames: frames 0,3,6,9 → 4 keyframes.
        assert_eq!(keyframes, 4);
    }

    #[test]
    fn forced_keyframe_resets_cadence() {
        let mut enc = MockVideoEncoder::new(&cfg(120), 24.0, Timebase::MPEG_90K);
        enc.encode(&frame(0)).unwrap(); // key
        let d1 = enc.encode(&frame(3750)).unwrap();
        assert!(!d1[0].keyframe);
        enc.request_keyframe();
        let d2 = enc.encode(&frame(7500)).unwrap();
        assert!(d2[0].keyframe);
    }

    #[test]
    fn pts_is_preserved_and_ordered() {
        let mut enc = MockVideoEncoder::new(&cfg(120), 24.0, Timebase::MPEG_90K);
        let a = enc.encode(&frame(0)).unwrap();
        let b = enc.encode(&frame(3750)).unwrap();
        assert!(b[0].pts > a[0].pts);
    }

    #[test]
    fn encode_after_finish_errors() {
        let mut enc = MockVideoEncoder::new(&cfg(120), 24.0, Timebase::MPEG_90K);
        enc.finish().unwrap();
        assert_eq!(enc.encode(&frame(0)).unwrap_err(), EncodeError::AfterFinish);
    }
}
