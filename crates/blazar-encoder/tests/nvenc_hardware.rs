// SPDX-License-Identifier: Apache-2.0
//! Hardware tests for the NVENC backend. Compiled only with `--features nvenc`.
//!
//! These are real encodes on whatever GPU is present, so they assert on
//! properties that must hold regardless of driver: frame count, timestamp
//! mapping, decode-order monotonicity, IDR cadence, and codec configuration.
//! They do not assert on bitrate or bytes; that is calibration territory.

#![cfg(feature = "nvenc")]

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_core::{Timebase, VideoCodec};
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::nvenc::NvencEncoder;
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

fn encoder(keyint: u32) -> NvencEncoder {
    NvencEncoder::new(&cfg(keyint), W, H, FPS, Timebase::MPEG_90K, VideoCodec::Av1)
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

fn encoder_codec(keyint: u32, bframes: u32, codec: VideoCodec) -> NvencEncoder {
    let mut c = cfg(keyint);
    c.bframes = bframes;
    NvencEncoder::new(&c, W, H, FPS, Timebase::MPEG_90K, codec)
        .unwrap_or_else(|e| panic!("NVENC {codec:?} session opens: {e}"))
}

/// Every unit must be a sequence of four-byte-length-prefixed NALs.
fn assert_length_prefixed(unit: &blazar_core::EncodedUnit) {
    let mut offset = 0usize;
    let bytes = &unit.bytes;
    while offset < bytes.len() {
        assert!(offset + 4 <= bytes.len(), "truncated length prefix");
        let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(len > 0, "zero-length NAL");
        offset += 4 + len;
    }
    assert_eq!(offset, bytes.len(), "framing must consume the sample");
}

/// H.264 and HEVC are the codecs that actually reorder on this driver, so they
/// are the regression test for DTS-based ordering and for the end-of-stream
/// drain that used to drop the reorder window.
#[test]
fn h264_and_hevc_emit_config_and_true_reordering() {
    for codec in [VideoCodec::H264, VideoCodec::Hevc] {
        let mut enc = encoder_codec(12, 2, codec);
        let mut units = Vec::new();
        for i in 0..24u64 {
            units.extend(enc.encode(&frame(i)).expect("frame encodes"));
        }
        units.extend(enc.finish().expect("finish drains the reorder window"));

        assert_eq!(units.len(), 24, "{codec:?} must emit every submitted frame");
        let config = enc.codec_config().expect("decoder config after keyframe");
        assert!(!config.is_empty());
        match codec {
            VideoCodec::H264 => assert_eq!(config[0], 1, "avcC configurationVersion"),
            VideoCodec::Hevc => assert_eq!(config[0], 1, "hvcC configurationVersion"),
            // Av1 and Vp9 are covered by their own backends; this test drives
            // the H.264/HEVC paths and must stay compilable without the CPU
            // codec features enabled.
            _ => unreachable!(),
        }

        let dts: Vec<u64> = units.iter().map(|u| u.dts.ticks()).collect();
        let pts: Vec<u64> = units.iter().map(|u| u.pts.ticks()).collect();
        assert!(
            dts.windows(2).all(|w| w[1] > w[0]),
            "{codec:?} DTS must be strictly increasing"
        );
        assert!(
            pts.windows(2).any(|w| w[1] < w[0]),
            "{codec:?} should produce a presentation-order descent with B-frames"
        );
        for unit in &units {
            assert_length_prefixed(unit);
        }
    }
}

fn rgb_frame(index: u64) -> FrameRef {
    let shift = (index * 4) as u32;
    let mut bytes = Vec::with_capacity((W * H * 3) as usize * 4);
    for row in 0..H {
        for col in 0..W {
            let v = ((col + row + shift) & 0xff) as f32 / 255.0;
            for channel in 0..3 {
                // Offsets per channel so a crossed frame or plane is visible.
                bytes.extend_from_slice(&(v * (1.0 - channel as f32 * 0.1)).to_le_bytes());
            }
        }
    }
    let desc = FrameDesc {
        width: W,
        height: H,
        dtype: Dtype::F32,
        range: ColorRange::Unit01,
        layout: Layout::Thwc,
        colorspace: ColorSpace::AssumedBt709,
        fps: FPS,
        pts: Pts(index * FRAME_TICKS),
    };
    FrameRef::new(FrameBuffer::host_owned(bytes), desc)
}

/// A batch must produce the same packets as the same frames encoded one at a
/// time. NVENC is deterministic for identical input and settings, so the
/// payloads are compared byte for byte, not just counted.
fn assert_batch_matches_singles(n: u64) {
    // Consumer GPUs cap concurrent NVENC sessions, and the rest of this suite
    // runs in parallel, so the two encoders are not held at the same time.
    let single_units = {
        let mut single = encoder(120);
        let mut units = Vec::new();
        for index in 0..n {
            units.extend(
                single
                    .encode_frame(rgb_frame(index))
                    .expect("single encode"),
            );
        }
        units
    };
    let frames: Vec<_> = (0..n).map(rgb_frame).collect();
    let batch_units = {
        let mut batched = encoder(120);
        batched.encode_batch(frames).expect("batch encode")
    };

    assert_eq!(
        batch_units.len(),
        single_units.len(),
        "batch emitted {} packets, singles emitted {}",
        batch_units.len(),
        single_units.len()
    );
    let batch_bytes: Vec<u8> = batch_units.iter().flat_map(|u| u.bytes.clone()).collect();
    let single_bytes: Vec<u8> = single_units.iter().flat_map(|u| u.bytes.clone()).collect();
    assert_eq!(
        batch_bytes, single_bytes,
        "batch and single payloads differ"
    );

    let batch_dts: Vec<u64> = batch_units.iter().map(|u| u.dts.0).collect();
    let single_dts: Vec<u64> = single_units.iter().map(|u| u.dts.0).collect();
    assert_eq!(
        batch_dts, single_dts,
        "batch and single decode order differ"
    );
}

#[test]
fn a_batch_matches_encoding_the_same_frames_one_at_a_time() {
    // 30 is what a second of 480p vLLM-Omni rollout produces.
    assert_batch_matches_singles(30);
}

#[test]
fn a_batch_larger_than_the_pool_is_chunked_not_serialised() {
    // Above POOL_CAPACITY the worker splits into launches; the result must be
    // the same as if the whole batch had been submitted at once.
    assert_batch_matches_singles(40);
}
