//! libvpx VP9 backend: real encode on the CPU.

#![cfg(feature = "enc-vp9")]

use blazar_core::Pts;
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::vp9::Vp9Encoder;
use blazar_encoder::VideoEncoder;
use blazar_preprocess::PreprocessedFrame;

const W: u32 = 64;
const H: u32 = 48;

fn cfg(keyint: u32) -> EncoderConfig {
    EncoderConfig {
        rate_control: RateControl::Crf(32),
        keyint,
        bframes: 0,
        pix_fmt: "yuv420p".into(),
        codec_params: String::new(),
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
fn encodes_frames_and_reports_vpcc() {
    let mut enc = Vp9Encoder::new(&cfg(5), W, H, 24.0, 3750).expect("open vp9");
    let mut units = Vec::new();
    for index in 0..10 {
        units.extend(enc.encode(&frame(index)).expect("encode"));
    }
    units.extend(enc.finish().expect("finish"));

    assert_eq!(units.len(), 10, "vp9 dropped frames");
    assert!(units[0].keyframe, "first frame must be a keyframe");
    assert!(
        units.iter().skip(1).any(|u| u.keyframe),
        "keyint=5 should produce a second keyframe"
    );
    let config = enc.codec_config().expect("vpcC is fixed for profile 0");
    assert_eq!(config.len(), 12, "vpcC body length");
    assert_eq!(config[0], 1, "vpcC version");
    assert_eq!(config[3] & 0x0f, 0, "profile 0");
    let dts: Vec<u64> = units.iter().map(|u| u.dts.0).collect();
    assert!(dts.windows(2).all(|w| w[0] < w[1]), "DTS must increase");
    assert!(units.iter().all(|u| !u.bytes.is_empty()));
}
