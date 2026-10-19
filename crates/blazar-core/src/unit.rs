//! Encoded output units — what the encoder hands the sink (architecture §6).
//!
//! `write` takes structured units (bytes + pts/dts/keyframe/duration), not raw
//! bytes, so a file muxer can write the container AND an RTP packetizer can mark
//! keyframes / packetize on frame boundaries. `track_id` distinguishes video
//! from audio so the muxer interleaves by PTS (architecture §3.1, §6).

use serde::{Deserialize, Serialize};

use crate::timebase::Pts;

/// Which track an encoded unit belongs to. Video and audio are separate tracks
/// on a shared timebase; WebRTC carries them as separate RTP streams, so the id
/// rides on every unit (and on feedback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackId {
    Video,
    Audio,
}

/// One encoded access unit (a coded video frame or an audio packet).
#[derive(Debug, Clone)]
pub struct EncodedUnit {
    pub track_id: TrackId,
    pub bytes: Vec<u8>,
    /// Presentation timestamp on the shared timebase.
    pub pts: Pts,
    /// Decode timestamp. Equals `pts` when there are no reordered B-frames;
    /// distinct once the encoder emits B-frames.
    pub dts: Pts,
    pub keyframe: bool,
    /// Duration of this unit in timebase ticks.
    pub duration: u64,
}

impl EncodedUnit {
    pub fn video(bytes: Vec<u8>, pts: Pts, dts: Pts, keyframe: bool, duration: u64) -> Self {
        Self {
            track_id: TrackId::Video,
            bytes,
            pts,
            dts,
            keyframe,
            duration,
        }
    }

    pub fn audio(bytes: Vec<u8>, pts: Pts, duration: u64) -> Self {
        // Audio is never reordered: dts == pts, every packet independently
        // decodable (keyframe semantics do not apply, marked true).
        Self {
            track_id: TrackId::Audio,
            bytes,
            pts,
            dts: pts,
            keyframe: true,
            duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_unit_preserves_fields() {
        let u = EncodedUnit::video(vec![0, 1, 2], Pts(3750), Pts(0), true, 3750);
        assert_eq!(u.track_id, TrackId::Video);
        assert!(u.keyframe);
        assert_eq!(u.pts, Pts(3750));
        assert_eq!(u.dts, Pts(0)); // reordered: dts precedes pts
    }

    #[test]
    fn audio_unit_has_pts_equal_dts() {
        let u = EncodedUnit::audio(vec![9, 9], Pts(960), 480);
        assert_eq!(u.track_id, TrackId::Audio);
        assert_eq!(u.pts, u.dts);
    }
}
