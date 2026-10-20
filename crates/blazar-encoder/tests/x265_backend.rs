//! libx265 backend: real HEVC encode on the CPU, no GPU required.

#![cfg(feature = "enc-x265")]

use blazar_core::Pts;
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::x265::X265Encoder;
use blazar_encoder::VideoEncoder;
use blazar_preprocess::PreprocessedFrame;

// x265 rejects anything smaller than one CTU, and the default CTU is 64x64:
// `x265_encoder_open` fails with "Picture size must be at least one CTU".
const W: u32 = 96;
const H: u32 = 64;

fn cfg(keyint: u32, bframes: u32) -> EncoderConfig {
    EncoderConfig {
        rate_control: RateControl::Crf(30),
        keyint,
        bframes,
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
fn encodes_frames_and_reports_hvcc() {
    let mut enc = X265Encoder::new(&cfg(5, 0), W, H, 24.0, 3750).expect("open x265");
    let mut units = Vec::new();
    for index in 0..10 {
        units.extend(enc.encode(&frame(index)).expect("encode"));
    }
    units.extend(enc.finish().expect("finish"));

    assert_eq!(units.len(), 10, "x265 dropped frames");
    assert!(units[0].keyframe, "first frame must be an IDR");
    assert!(
        units.iter().skip(1).any(|u| u.keyframe),
        "keyint=5 should produce a second keyframe"
    );
    let config = enc.codec_config().expect("hvcC after the first frame");
    assert_eq!(*config.first().unwrap(), 1, "hvcC configurationVersion");
    for unit in &units {
        // A sample is one or more length-prefixed NALs; walk them all.
        let mut offset = 0usize;
        let mut nals = 0usize;
        while offset < unit.bytes.len() {
            assert!(unit.bytes.len() - offset >= 4, "truncated length prefix");
            let declared = u32::from_be_bytes([
                unit.bytes[offset],
                unit.bytes[offset + 1],
                unit.bytes[offset + 2],
                unit.bytes[offset + 3],
            ]) as usize;
            offset += 4 + declared;
            nals += 1;
        }
        assert_eq!(offset, unit.bytes.len(), "length prefixes must cover the sample");
        assert!(nals >= 1);
        assert_eq!(unit.duration, 3750);
    }
    // The set of presentation timestamps must be exactly the frames we fed,
    // whatever order the encoder emits them in.
    let pts: Vec<u64> = units.iter().map(|u| u.pts.0).collect();
    let mut sorted = pts.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..units.len() as u64).map(|i| i * 3750).collect::<Vec<_>>(),
        "PTS set does not match the submitted frames"
    );
    assert_eq!(units[0].pts.0, 0, "first emitted unit must be frame 0");
}

#[test]
fn bframes_reorder_decode_timestamps() {
    let mut enc = X265Encoder::new(&cfg(60, 2), W, H, 24.0, 3750).expect("open x265");
    let mut units = Vec::new();
    for index in 0..12 {
        units.extend(enc.encode(&frame(index)).expect("encode"));
    }
    units.extend(enc.finish().expect("finish"));
    assert_eq!(units.len(), 12);
    // With B-frames the decode order differs from display order at least once.
    assert!(
        units.windows(2).any(|w| w[0].pts.0 > w[1].pts.0),
        "expected reordered PTS in decode order"
    );
    let dts: Vec<u64> = units.iter().map(|u| u.dts.0).collect();
    assert!(dts.windows(2).all(|w| w[0] < w[1]), "DTS must increase");
}
