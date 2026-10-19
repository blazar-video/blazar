//! Shared timebase and presentation timestamps.
//!
//! Both runtimes (xDiT, vLLM-Omni) expose no per-frame PTS — only an `fps`
//! (architecture §10). Blazar therefore *synthesizes* PTS from a frame index
//! and the frame rate. Video and audio share one timebase so the muxer can
//! interleave by PTS (architecture §3.1).
//!
//! A PTS is represented in integer *ticks* of a [`Timebase`] rather than a
//! float second, because container muxers (fMP4) and RTP want integer
//! timestamps and float accumulation drifts over long streams.

use serde::{Deserialize, Serialize};

/// A rational timebase: one tick is `1 / ticks_per_second` seconds.
///
/// e.g. `Timebase::new(90_000)` is the classic 90 kHz RTP/MPEG clock; a 24 fps
/// video has 3750 ticks per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timebase {
    ticks_per_second: u32,
}

impl Timebase {
    /// 90 kHz — the conventional MPEG/RTP media clock. A good default because
    /// it divides evenly for common frame rates and audio sample rates.
    pub const MPEG_90K: Timebase = Timebase {
        ticks_per_second: 90_000,
    };

    /// Create a timebase with the given tick resolution.
    ///
    /// # Panics
    /// Panics if `ticks_per_second` is zero (a zero-resolution clock is
    /// meaningless and would divide-by-zero downstream).
    pub fn new(ticks_per_second: u32) -> Self {
        assert!(
            ticks_per_second > 0,
            "timebase ticks_per_second must be > 0"
        );
        Self { ticks_per_second }
    }

    pub fn ticks_per_second(self) -> u32 {
        self.ticks_per_second
    }

    /// Ticks per video frame at `fps`. Rounded to nearest; non-integer frame
    /// rates (23.976) therefore accumulate a bounded rounding error per frame,
    /// which is why [`Pts::from_frame_index`] multiplies the index rather than
    /// summing per-frame deltas.
    pub fn ticks_per_frame(self, fps: f64) -> u64 {
        assert!(fps > 0.0, "fps must be > 0");
        (self.ticks_per_second as f64 / fps).round() as u64
    }
}

/// A presentation timestamp in [`Timebase`] ticks on the shared clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Pts(pub u64);

impl Pts {
    /// Synthesize a video PTS from a 0-based frame index and the frame rate.
    ///
    /// Multiplies `frame_index * ticks_per_second / fps` in a single rational
    /// step so error does not accumulate across frames (important for the
    /// vLLM-Omni streaming case where the index is cumulative across chunks,
    /// architecture §7 table).
    pub fn from_frame_index(frame_index: u64, fps: f64, tb: Timebase) -> Self {
        assert!(fps > 0.0, "fps must be > 0");
        let ticks = (frame_index as f64 * tb.ticks_per_second() as f64 / fps).round() as u64;
        Pts(ticks)
    }

    /// Synthesize an audio PTS from a 0-based sample index and the sample rate.
    pub fn from_sample_index(sample_index: u64, sample_rate: u32, tb: Timebase) -> Self {
        assert!(sample_rate > 0, "sample_rate must be > 0");
        let ticks =
            (sample_index as u128 * tb.ticks_per_second() as u128 / sample_rate as u128) as u64;
        Pts(ticks)
    }

    pub fn ticks(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_per_frame_24fps_on_90k() {
        // 90000 / 24 = 3750 exactly.
        assert_eq!(Timebase::MPEG_90K.ticks_per_frame(24.0), 3750);
    }

    #[test]
    fn video_pts_is_monotonic_and_non_accumulating() {
        let tb = Timebase::MPEG_90K;
        // Frame 0 is t=0; each subsequent frame advances by one frame of ticks.
        assert_eq!(Pts::from_frame_index(0, 24.0, tb), Pts(0));
        assert_eq!(Pts::from_frame_index(1, 24.0, tb), Pts(3750));
        assert_eq!(Pts::from_frame_index(100, 24.0, tb), Pts(375_000));
    }

    #[test]
    fn non_integer_fps_does_not_drift() {
        // 23.976 fps: a per-frame-delta sum would drift; the rational form keeps
        // frame N within half a tick of the true time.
        let tb = Timebase::MPEG_90K;
        let fps = 24000.0 / 1001.0; // 23.976...
        let p1000 = Pts::from_frame_index(1000, fps, tb).ticks();
        let true_ticks = (1000.0 * 90_000.0 / fps).round() as u64;
        assert_eq!(p1000, true_ticks);
    }

    #[test]
    fn audio_pts_matches_video_timebase() {
        let tb = Timebase::MPEG_90K;
        // 48 kHz audio, 48000 samples = exactly 1 second = 90000 ticks.
        assert_eq!(Pts::from_sample_index(48_000, 48_000, tb), Pts(90_000));
        // and that lines up with 24 video frames (1 second).
        assert_eq!(Pts::from_frame_index(24, 24.0, tb), Pts(90_000));
    }

    #[test]
    #[should_panic]
    fn zero_timebase_rejected() {
        Timebase::new(0);
    }
}
