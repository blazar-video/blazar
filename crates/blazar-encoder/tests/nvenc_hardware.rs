//! Hardware tests for the NVENC backend. Compiled only with `--features nvenc`.
//!
//! These are real encodes on whatever GPU is present, so they assert on
//! properties that must hold regardless of driver: frame count, timestamp
//! mapping, decode-order monotonicity, IDR cadence, and codec configuration.
//! They do not assert on bitrate or bytes; that is calibration territory.

#![cfg(feature = "nvenc")]

use blazar_core::Pts;
use blazar_core::{Timebase, VideoCodec};
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::nvenc::NvencAv1Encoder;
use blazar_encoder::VideoEncoder;
use blazar_preprocess::PreprocessedFrame;

const W: u32 = 320;
const H: u32 = 240;
const FPS: f64 = 24.0;
const FRAME_TICKS: u64 = 3750; // 90000 / 24

fn cfg(keyint: u32) -> EncoderConfig {
    EncoderConfig {
        rate_control: RateControl::Crf(40),
        keyint,
        bframes: 0,
        pix_fmt: "yuv420p".into(),
        codec_params: String::new(),
    }
}

/// Moving gradient: real inter-frame content, so the encoder emits P frames
/// instead of degenerating into intra-only.
fn frame(index: u64) -> PreprocessedFrame {
    let shift = (index * 4) as u32;
    let mut y = vec![0u8; (W * H) as usize];
    for row in 0..H {
        for col in 0..W {
            y[(row * W + col) as usize] = ((col + row + shift) & 0xff) as u8;
        }
    }
    let chroma = ((W / 2) * (H / 2)) as usize;
    PreprocessedFrame {
        width: W,
        height: H,
        y,
        u: vec![128; chroma],
        v: vec![128; chroma],
        pts: Pts(index * FRAME_TICKS),
    }
}

fn encoder(keyint: u32) -> NvencAv1Encoder {
    NvencAv1Encoder::new(&cfg(keyint), W, H, FPS, Timebase::MPEG_90K, VideoCodec::Av1)
        .expect("NVENC AV1 session opens")
}

#[test]
fn encodes_every_frame_with_ordered_timestamps_and_idr_cadence() {
    let mut enc = encoder(12);
    assert!(
        enc.codec_config().is_none(),
        "no configuration before the first keyframe is encoded"
    );

    let mut units = Vec::new();
    for i in 0..30u64 {
        units.extend(enc.encode(&frame(i)).expect("frame encodes"));
        if i == 0 {
            let config = enc.codec_config().expect("config after first keyframe");
            assert_eq!(config[0], 0x81, "av1C marker + version");
            assert_eq!(
                (config[4] >> 3) & 0x0f,
                1,
                "configOBUs must start with OBU_SEQUENCE_HEADER"
            );
        }
    }
    units.extend(enc.finish().expect("finish drains"));

    assert_eq!(units.len(), 30, "one unit per submitted frame");

    let mut pts: Vec<u64> = units.iter().map(|u| u.pts.ticks()).collect();
    pts.sort_unstable();
    let expected: Vec<u64> = (0..30).map(|i| i * FRAME_TICKS).collect();
    assert_eq!(pts, expected, "PTS are exactly the submitted timestamps");

    for pair in units.windows(2) {
        assert!(
            pair[1].dts > pair[0].dts,
            "decode timestamps must strictly increase"
        );
    }
    for (i, unit) in units.iter().enumerate() {
        assert_eq!(unit.duration, FRAME_TICKS);
        assert_eq!(
            unit.dts.ticks(),
            i as u64 * FRAME_TICKS,
            "decode order is submission order"
        );
        assert!(!unit.bytes.is_empty());
    }

    let keyframes: Vec<usize> = units
        .iter()
        .enumerate()
        .filter(|(_, u)| u.keyframe)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(keyframes, vec![0, 12, 24], "IDR every keyint frames");
}

#[test]
fn a_caller_requested_keyframe_is_an_idr_mid_gop() {
    let mut enc = encoder(100);
    let mut units = Vec::new();
    for i in 0..6u64 {
        if i == 3 {
            enc.request_keyframe();
        }
        units.extend(enc.encode(&frame(i)).expect("frame encodes"));
    }
    units.extend(enc.finish().expect("finish drains"));

    let keyframes: Vec<usize> = units
        .iter()
        .enumerate()
        .filter(|(_, u)| u.keyframe)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(keyframes, vec![0, 3]);
}

#[test]
fn finish_without_frames_is_not_an_error() {
    let mut enc = encoder(12);
    assert!(enc.finish().expect("empty finish is fine").is_empty());
}

#[test]
fn encoding_after_finish_is_rejected() {
    let mut enc = encoder(12);
    enc.encode(&frame(0)).unwrap();
    enc.finish().unwrap();
    assert_eq!(
        enc.encode(&frame(1)).unwrap_err(),
        blazar_encoder::EncodeError::AfterFinish
    );
}
