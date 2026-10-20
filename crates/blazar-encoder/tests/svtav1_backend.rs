//! SVT-AV1 backend: real encode on the CPU.

#![cfg(feature = "enc-svtav1")]

use blazar_core::Pts;
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::svtav1::SvtAv1Encoder;
use blazar_encoder::VideoEncoder;
use blazar_preprocess::PreprocessedFrame;

// SVT-AV1 2.x rejects frames shorter than 64 rows.
const W: u32 = 96;
const H: u32 = 64;

fn cfg(keyint: u32, bframes: u32) -> EncoderConfig {
    EncoderConfig {
        rate_control: RateControl::Crf(32),
        keyint,
        bframes,
        pix_fmt: "yuv420p".into(),
        codec_params: "preset=10".into(),
    }
}

fn frame(index: u64) -> PreprocessedFrame {
    let shift = (index * 3) as u32;
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
        pts: Pts(index * 3750),
    }
}

#[test]
fn encodes_frames_and_reports_av1c() {
    let mut enc = SvtAv1Encoder::new(&cfg(5, 0), W, H, 24.0, 3750).expect("open svt-av1");
    let mut units = Vec::new();
    for index in 0..10 {
        units.extend(enc.encode(&frame(index)).expect("encode"));
    }
    units.extend(enc.finish().expect("finish"));

    assert_eq!(units.len(), 10, "svt-av1 dropped frames");
    assert!(units[0].keyframe, "first frame must be a keyframe");
    assert!(
        units.iter().skip(1).any(|u| u.keyframe),
        "keyint=5 should produce a second keyframe"
    );
    let config = enc.codec_config().expect("av1C from the first keyframe");
    assert!(!config.is_empty(), "av1C must not be empty");
    let dts: Vec<u64> = units.iter().map(|u| u.dts.0).collect();
    assert!(dts.windows(2).all(|w| w[0] < w[1]), "DTS must increase");
    assert!(units.iter().all(|u| !u.bytes.is_empty()));
}

#[test]
fn bframes_reorder_presentation_timestamps() {
    let mut enc = SvtAv1Encoder::new(&cfg(60, 2), W, H, 24.0, 3750).expect("open svt-av1");
    let mut units = Vec::new();
    for index in 0..16 {
        units.extend(enc.encode(&frame(index)).expect("encode"));
    }
    units.extend(enc.finish().expect("finish"));
    assert_eq!(units.len(), 16);
    let dts: Vec<u64> = units.iter().map(|u| u.dts.0).collect();
    assert!(dts.windows(2).all(|w| w[0] < w[1]), "DTS must increase");
}
