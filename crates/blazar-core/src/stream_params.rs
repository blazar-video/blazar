//! Stream parameters declared to a sink at `open` (architecture §6).
//!
//! Stream metadata describes the wire/container codec separately from the
//! encoder backend that produced it. Codec configuration bytes are mandatory
//! container-facing extradata: AV1 `av1C`, H.264 `avcC`, HEVC `hvcC`, AAC
//! AudioSpecificConfig, or Opus `OpusHead`/`dOps`.

use serde::{Deserialize, Serialize};

use crate::timebase::Timebase;

/// Video codec carried on the wire/container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoCodec {
    Av1,
    H264,
    Hevc,
}

/// Encoder backend selected by the deployment. This is separate from
/// [`VideoCodec`]: multiple backends can produce the same wire codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoEncoderBackend {
    SvtAv1,
    /// Hardware AV1 encoder.
    NvencAv1,
    /// Hardware H.264 encoder (same NVENC implementation as AV1).
    NvencH264,
    /// Hardware HEVC encoder (same NVENC implementation as AV1).
    NvencHevc,
    Rav1e,
    /// Software H.264 (x264). GPL; compiled in only with `enc-x264`.
    X264,
    /// Software HEVC (x265). GPL; compiled in only with `enc-x265`.
    X265,
    /// Software VP9 (libvpx); compiled in only with `enc-vpx`.
    Vp9,
}

/// Audio codec carried on the wire/container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioCodec {
    Opus,
    Aac,
}

/// Output container. fMP4 is primary (architecture §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Container {
    /// Fragmented MP4 — valid file, matches vLLM-Omni's fragmented-MP4
    /// streaming, and is CMAF low-latency carriage.
    FragmentedMp4,
    WebM,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTrackParams {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// Codec configuration record body, excluding the ISOBMFF box header.
    pub extradata: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioTrackParams {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channels: u8,
    /// Codec configuration bytes in the codec's native/container convention.
    pub extradata: Vec<u8>,
}

/// Declared at `Sink::open`. A stream always has video; audio is optional
/// (present only for omni models).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    fn video_codec_is_wire_identity_not_backend_identity() {
        assert_ne!(VideoCodec::Av1, VideoCodec::H264);
        assert_ne!(VideoEncoderBackend::SvtAv1, VideoEncoderBackend::Rav1e);
    }

    #[test]
    fn video_only_stream_has_no_audio_track() {
        let p = StreamParams {
            timebase: Timebase::MPEG_90K,
            video: VideoTrackParams {
                codec: VideoCodec::Av1,
                width: 832,
                height: 480,
                fps: 24.0,
                extradata: vec![0x81, 0x04, 0x0c, 0x00],
            },
            audio: None,
            container: Container::FragmentedMp4,
        };
        assert!(p.audio.is_none());
        assert_eq!(p.video.codec, VideoCodec::Av1);
        assert_eq!(p.container, Container::FragmentedMp4);
    }

    #[test]
    fn omni_stream_declares_both_tracks() {
        let p = StreamParams {
            timebase: Timebase::MPEG_90K,
            video: VideoTrackParams {
                codec: VideoCodec::Hevc,
                width: 1280,
                height: 720,
                fps: 24.0,
                extradata: vec![1, 1, 0, 0],
            },
            audio: Some(AudioTrackParams {
                codec: AudioCodec::Aac,
                sample_rate: 48_000,
                channels: 2,
                extradata: vec![0x12, 0x10],
            }),
            container: Container::FragmentedMp4,
        };
        assert_eq!(p.audio.as_ref().unwrap().sample_rate, 48_000);
        assert_eq!(p.audio.as_ref().unwrap().codec, AudioCodec::Aac);
    }
}
