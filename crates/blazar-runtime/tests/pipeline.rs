//! End-to-end Phase-1 pipeline test: host frame → preprocess → encode → sink.
//!
//! Exercises the full operator API against the DebugFileSink and asserts the
//! stream's structural invariants (keyframe cadence, ordering, unit count).

use blazar_calibration::{load_str, CalibrationIndex};
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec,
};
use blazar_preprocess::rgb_f32_len;
use blazar_runtime::{Boundary, Encoder, Output};
use blazar_sink::fmp4::DebugFileSink;

const DEFAULT_CAL: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "svtav1"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
keyint = 4
bframes = 0
pix_fmt = "yuv420p"
[video.codec_params]
svtav1 = "preset=6"
"#;

fn index() -> CalibrationIndex {
    let mut idx = CalibrationIndex::new();
    idx.insert(load_str(DEFAULT_CAL, "default/svtav1_balanced.toml").unwrap());
    idx
}

fn host_frame(w: u32, h: u32, luma: f32, frame_index: u64) -> FrameRef {
    let mut bytes = Vec::with_capacity(rgb_f32_len(w, h));
    for _ in 0..(w * h) {
        for _ in 0..3 {
            bytes.extend_from_slice(&luma.to_le_bytes());
        }
    }
    let desc = FrameDesc {
        width: w,
        height: h,
        dtype: Dtype::F32,
        range: ColorRange::Unit01,
        layout: Layout::Thwc,
        colorspace: ColorSpace::AssumedBt709,
        fps: 24.0,
        pts: Pts::from_frame_index(frame_index, 24.0, Timebase::MPEG_90K),
    };
    FrameRef::new(FrameBuffer::host_owned(bytes), desc)
}

fn parse_units(buf: &[u8]) -> Vec<(u8, bool, u64)> {
    // Mirror DebugFileSink framing: skip 8-byte magic, then records of
    // [track:u8][key:u8][pts:u64][dts:u64][dur:u64][len:u32][bytes].
    assert_eq!(&buf[0..8], b"BLZRDBG1");
    let mut out = Vec::new();
    let mut i = 8;
    while i < buf.len() {
        let track = buf[i];
        let key = buf[i + 1] != 0;
        let pts = u64::from_le_bytes(buf[i + 2..i + 10].try_into().unwrap());
        let len = u32::from_le_bytes(buf[i + 26..i + 30].try_into().unwrap()) as usize;
        out.push((track, key, pts));
        i += 30 + len;
    }
    out
}

fn run(n: u32) -> Vec<u8> {
    let sink = DebugFileSink::new(Vec::new());
    let mut enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::Balanced,
        Output {
            codec: VideoCodec::SvtAv1,
            container: Container::FragmentedMp4,
        },
        sink,
        index(),
    );
    {
        let mut stream = enc.open_stream("req-1");
        assert_eq!(stream.request_id(), "req-1");
        for i in 0..n {
            let seq = stream
                .submit_video(host_frame(8, 8, 0.5, i as u64), None, None)
                .unwrap();
            assert_eq!(seq, i as u64);
        }
        stream.close().unwrap();
    }
    // Recover the sink's buffer.
    enc_into_buffer(enc)
}

// Helper: the Encoder owns the sink; expose it for the test by reconstructing.
// Since Encoder has no accessor, we re-run through a local that returns bytes.
fn enc_into_buffer(enc: Encoder<DebugFileSink<Vec<u8>>>) -> Vec<u8> {
    // Encoder does not expose the sink; instead we rely on close() having
    // flushed, and reach in via a dedicated test-only destructuring.
    enc.into_sink_for_test().into_inner()
}

#[test]
fn produces_one_unit_per_frame() {
    let buf = run(10);
    let units = parse_units(&buf);
    assert_eq!(units.len(), 10);
}

#[test]
fn keyframe_cadence_matches_keyint() {
    let buf = run(10);
    let units = parse_units(&buf);
    // keyint=4: keyframes at frame 0,4,8 → 3 keyframes.
    let keyframes = units.iter().filter(|(_, k, _)| *k).count();
    assert_eq!(keyframes, 3);
    assert!(units[0].1, "first frame must be a keyframe");
}

#[test]
fn pts_is_monotonic_across_stream() {
    let buf = run(10);
    let units = parse_units(&buf);
    let mut last = 0u64;
    for (i, (_, _, pts)) in units.iter().enumerate() {
        if i > 0 {
            assert!(*pts > last, "pts must increase");
        }
        last = *pts;
    }
    // Frame 9 at 24fps/90k = 9*3750 = 33750.
    assert_eq!(units[9].2, 33_750);
}

#[test]
fn scene_cut_boundary_forces_keyframe() {
    let sink = DebugFileSink::new(Vec::new());
    let mut enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::Balanced,
        Output {
            codec: VideoCodec::SvtAv1,
            container: Container::FragmentedMp4,
        },
        sink,
        index(),
    );
    {
        let mut stream = enc.open_stream("req-cut");
        stream
            .submit_video(host_frame(8, 8, 0.5, 0), None, None)
            .unwrap(); // key
        stream
            .submit_video(host_frame(8, 8, 0.5, 1), None, None)
            .unwrap(); // delta
                       // Force a scene cut on frame 2 (would otherwise be a delta under keyint=4).
        stream
            .submit_video(host_frame(8, 8, 0.5, 2), None, Some(Boundary::SceneCut))
            .unwrap();
        stream.close().unwrap();
    }
    let buf = enc.into_sink_for_test().into_inner();
    let units = parse_units(&buf);
    assert!(units[0].1, "frame 0 keyframe");
    assert!(!units[1].1, "frame 1 delta");
    assert!(units[2].1, "frame 2 forced keyframe by scene cut");
}
