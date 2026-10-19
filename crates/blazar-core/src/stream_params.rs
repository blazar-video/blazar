//! Stream parameters declared to a sink at `open` (architecture §6).
//!
//! Declares both tracks (video codec/res/fps, audio codec/rate/channels) and
//! the shared timebase up front, so a muxer can write container headers before
//! the first unit arrives.

use serde::{Deserialize, Serialize};

use crate::timebase::Timebase;

/// Video codec selected for output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoCodec {
    SvtAv1,
    /// Hardware AV1 (NVENC) — Phase 2.
    NvencAv1,
    Rav1e,
}

/// Audio codec. Opus pairs with AV1 for fMP4/WebM/WebRTC (architecture §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioCodec {
    Opus,
}

/// Output container. fMP4 is primary (architecture §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Container {
    /// Fragmented MP4 — valid file, matches vLLM-Omni's fragmented-MP4
    /// streaming, and is CMAF low-latency carriage.
    FragmentedMp4,
    WebM,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoTrackParams {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioTrackParams {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u8,
}

/// Declared at `Sink::open`. A stream always has video; audio is optional
/// (present only for omni models).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StreamParams {
    pub timebase: Timebase,
    pub video: VideoTrackParams,
    pub audio: Option<AudioTrackParams>,
    pub container: Container,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_only_stream_has_no_audio_track() {
        let p = StreamParams {
            timebase: Timebase::MPEG_90K,
            video: VideoTrackParams {
                codec: VideoCodec::SvtAv1,
                width: 832,
                height: 480,
                fps: 24.0,
            },
            audio: None,
            container: Container::FragmentedMp4,
        };
        assert!(p.audio.is_none());
        assert_eq!(p.video.codec, VideoCodec::SvtAv1);
        assert_eq!(p.container, Container::FragmentedMp4);
    }

    #[test]
    fn omni_stream_declares_both_tracks() {
        let p = StreamParams {
            timebase: Timebase::MPEG_90K,
            video: VideoTrackParams {
                codec: VideoCodec::SvtAv1,
                width: 1280,
                height: 720,
                fps: 24.0,
            },
            audio: Some(AudioTrackParams {
                codec: AudioCodec::Opus,
                sample_rate: 48_000,
                channels: 2,
            }),
            container: Container::FragmentedMp4,
        };
        assert_eq!(p.audio.unwrap().sample_rate, 48_000);
    }
}
