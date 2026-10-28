// SPDX-License-Identifier: Apache-2.0
//! VideoToolbox encoder tests.
//!
//! These run only on macOS with `enc-videotoolbox` enabled: the backend links
//! system frameworks that exist nowhere else. The final test decodes the
//! encoded stream with `ffprobe` when it is installed, so a stream that is
//! structurally plausible but undecodable fails.

#![cfg(all(feature = "enc-videotoolbox", target_os = "macos"))]

use std::process::Command;

use blazar_core::{Pts, Timebase};
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::videotoolbox::{Codec, VideoToolboxEncoder};
use blazar_encoder::VideoEncoder;
use blazar_preprocess::PreprocessedFrame;

const W: u32 = 320;
const H: u32 = 240;

/// A frame whose luma moves so the encoder has something to code.
fn frame(n: u32) -> PreprocessedFrame {
    let w = W as usize;
    let h = H as usize;
    let mut y = vec![16u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let band = ((col as u32 + n * 8) / 16) % 2;
            y[row * w + col] = if band == 0 { 40 } else { 200 };
        }
    }
    PreprocessedFrame {
        width: W,
        height: H,
        y,
        u: vec![128u8; (w / 2) * (h / 2)],
        v: vec![128u8; (w / 2) * (h / 2)],
        pts: Pts(Timebase::MPEG_90K.ticks_per_frame(24.0) * n as u64),
    }
}

fn encoder(codec: Codec) -> VideoToolboxEncoder {
    // Built directly rather than from a calibration file: this test covers the
    // encoder, and a missing calibration in the sibling repo would make it fail
    // for an unrelated reason.
    let cfg = EncoderConfig {
        rate_control: RateControl::Crf(23),
        keyint: 48,
        bframes: 0,
        pix_fmt: "nv12".into(),
        codec_params: String::new(),
    };
    VideoToolboxEncoder::new(W, H, codec, &cfg).expect("session opens")
}

/// Encode `count` frames, returning the units in order.
fn encode_units(enc: &mut VideoToolboxEncoder, count: u32) -> Vec<blazar_core::EncodedUnit> {
    let mut units = Vec::new();
    for n in 0..count {
        units.extend(enc.encode(&frame(n)).expect("frame encodes"));
    }
    units.extend(enc.flush().expect("flush succeeds"));
    units
}

#[test]
fn h264_produces_units_and_a_codec_config() {
    let mut enc = encoder(Codec::H264);
    let units = encode_units(&mut enc, 30);

    assert!(!units.is_empty(), "no access units came back");
    assert!(units.iter().all(|u| !u.bytes.is_empty()), "empty unit");
    assert!(units[0].keyframe, "the first unit must be a keyframe");
    let config = enc.codec_config().expect("avcC record");
    assert_eq!(config[0], 1, "avcC starts with configurationVersion 1");
}

#[test]
fn hevc_produces_units_and_a_codec_config() {
    let mut enc = encoder(Codec::Hevc);
    let units = encode_units(&mut enc, 30);

    assert!(!units.is_empty(), "no access units came back");
    assert!(units[0].keyframe, "the first unit must be a keyframe");
    let config = enc.codec_config().expect("hvcC record");
    assert_eq!(config[0], 1, "hvcC starts with configurationVersion 1");
}

#[test]
fn timestamps_are_carried_through() {
    let mut enc = encoder(Codec::H264);
    let units = encode_units(&mut enc, 8);
    let expected: Vec<u64> = (0..8)
        .map(|n| Timebase::MPEG_90K.ticks_per_frame(24.0) * n)
        .collect();
    let got: Vec<u64> = units.iter().map(|u| u.pts.0).collect();
    assert_eq!(got, expected, "pts must come back unchanged");
}

#[test]
fn a_mismatched_frame_size_is_rejected() {
    let mut enc = encoder(Codec::H264);
    let mut wrong = frame(0);
    wrong.width = W + 2;
    let err = enc.encode(&wrong).expect_err("size mismatch must fail");
    assert!(
        format!("{err}").contains("frame is"),
        "unexpected error: {err}"
    );
}

/// The stream has to decode, not merely look like H.264.
#[test]
fn the_encoded_stream_decodes_with_ffprobe() {
    let mut enc = encoder(Codec::H264);
    let units = encode_units(&mut enc, 30);

    // Rebuild an Annex B elementary stream: start codes instead of the
    // length prefixes the sink consumes, with the parameter sets in front.
    let mut stream = Vec::new();
    let config = enc.codec_config().expect("avcC record");
    for nal in parameter_sets(&config) {
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&nal);
    }
    for unit in &units {
        for nal in length_prefixed(&unit.bytes) {
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(nal);
        }
    }

    let path = std::env::temp_dir().join("blazar_videotoolbox.h264");
    std::fs::write(&path, &stream).expect("write the elementary stream");

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(&path)
        .output();
    let Ok(output) = output else {
        eprintln!("ffprobe not installed; skipping the decode check");
        return;
    };
    let reported = String::from_utf8_lossy(&output.stdout);
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "ffprobe rejected the stream: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        reported.trim(),
        "320,240,30",
        "ffprobe decoded a different stream than we encoded"
    );
}

/// Split an `avcC` record into its SPS and PPS payloads.
fn parameter_sets(config: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if config.len() < 6 {
        return out;
    }
    let mut i = 5;
    let num_sps = (config[i] & 0x1f) as usize;
    i += 1;
    for _ in 0..num_sps {
        let len = u16::from_be_bytes([config[i], config[i + 1]]) as usize;
        out.push(config[i + 2..i + 2 + len].to_vec());
        i += 2 + len;
    }
    let num_pps = config[i] as usize;
    i += 1;
    for _ in 0..num_pps {
        let len = u16::from_be_bytes([config[i], config[i + 1]]) as usize;
        out.push(config[i + 2..i + 2 + len].to_vec());
        i += 2 + len;
    }
    out
}

/// Split a length-prefixed sample into its NAL units.
fn length_prefixed(sample: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= sample.len() {
        let len =
            u32::from_be_bytes([sample[i], sample[i + 1], sample[i + 2], sample[i + 3]]) as usize;
        i += 4;
        if i + len > sample.len() {
            break;
        }
        out.push(&sample[i..i + len]);
        i += len;
    }
    out
}

/// Times the encoder on its own, at 1080p, with no muxer or runtime in the way.
///
/// `cargo test -p blazar-encoder --features enc-videotoolbox --test videotoolbox
///  -- --ignored --nocapture encoder_timings`
#[test]
#[ignore]
fn encoder_timings() {
    use std::time::Instant;
    for (w, h) in [(320u32, 240u32), (1280, 720), (1920, 1080)] {
        let cfg = EncoderConfig {
            rate_control: RateControl::Crf(23),
            keyint: 48,
            bframes: 0,
            pix_fmt: "nv12".into(),
            codec_params: String::new(),
        };
        let mut enc = VideoToolboxEncoder::new(w, h, Codec::H264, &cfg).expect("session opens");
        let (fw, fh) = (w as usize, h as usize);
        let make = |n: u32| PreprocessedFrame {
            width: w,
            height: h,
            y: vec![(40 + (n % 3) * 60) as u8; fw * fh],
            u: vec![128u8; (fw / 2) * (fh / 2)],
            v: vec![128u8; (fw / 2) * (fh / 2)],
            pts: Pts(Timebase::MPEG_90K.ticks_per_frame(24.0) * n as u64),
        };
        let frames: Vec<PreprocessedFrame> = (0..20u32).map(make).collect();
        let started = Instant::now();
        for f in &frames {
            let _ = enc.encode(f).expect("encode");
        }
        let per = started.elapsed().as_secs_f64() * 1000.0 / frames.len() as f64;
        let _ = enc.flush();
        println!("{w}x{h}: {per:8.2} ms/frame (encoder only)");
    }
}
