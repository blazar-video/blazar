//! VideoToolbox -> fMP4, through the same runtime and sink the Python binding
//! uses.
//!
//! The encoder's own tests cover the elementary stream. This one covers the
//! whole path: preprocess, encode, mux, and a decode check with `ffprobe` when
//! it is installed. It is the test that would have caught a backend which
//! produces valid NAL units but a container that no player accepts.

#![cfg(all(feature = "enc-videotoolbox", target_os = "macos"))]

use std::fs;
use std::process::Command;

use blazar_calibration::{load_str, CalibrationIndex};
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_preprocess::rgb_f32_len;
use blazar_runtime::{Encoder, Output};
use blazar_sink::fmp4::Fmp4FileSink;

const VT_CAL: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "videotoolbox_h264"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 23
bitrate_kbps = 3000
keyint = 24
bframes = 0
pix_fmt = "nv12"
"#;

const W: u32 = 640;
const H: u32 = 360;
const FRAMES: u64 = 48;

fn index() -> CalibrationIndex {
    let mut idx = CalibrationIndex::new();
    idx.insert(load_str(VT_CAL, "default/videotoolbox_h264_balanced.toml").expect("calibration loads"));
    idx
}

/// A moving gradient, so the encoder sees real inter-frame content rather than
/// a still image it could code in one keyframe.
fn host_frame(index: u64) -> FrameRef {
    let shift = (index * 4) as f32 / 255.0;
    let mut bytes = Vec::with_capacity(rgb_f32_len(W, H));
    for row in 0..H {
        for col in 0..W {
            let r = ((col as f32 / W as f32) + shift) % 1.0;
            let g = (row as f32 / H as f32 + shift) % 1.0;
            let b = ((col + row) as f32 / (W + H) as f32 + shift) % 1.0;
            for component in [r, g, b] {
                bytes.extend_from_slice(&component.to_le_bytes());
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
        fps: 24.0,
        pts: Pts::from_frame_index(index, 24.0, Timebase::MPEG_90K),
    };
    FrameRef::new(FrameBuffer::host_owned(bytes), desc)
}

fn count_box(bytes: &[u8], fourcc: &[u8; 4]) -> usize {
    bytes.windows(4).filter(|window| *window == fourcc).count()
}

fn encode_to(path: &std::path::Path) {
    let _ = fs::remove_file(path);
    let sink = Fmp4FileSink::new(std::io::BufWriter::new(
        fs::File::create(path).expect("create output file"),
    ));
    let enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::Balanced,
        Output {
            codec: VideoCodec::H264,
            backend: VideoEncoderBackend::VideoToolboxH264,
            container: Container::FragmentedMp4,
            video_extradata: None,
            audio_extradata: None,
        },
        sink,
        index(),
    );
    let mut stream = enc.open_stream("videotoolbox-e2e");
    for i in 0..FRAMES {
        stream
            .submit_video(host_frame(i), None, None)
            .expect("submit");
    }
    stream.close().expect("close");
}

#[test]
fn videotoolbox_writes_a_playable_fragmented_mp4() {
    let path = std::env::temp_dir().join("blazar-videotoolbox.mp4");
    encode_to(&path);

    // Read while the stream is out of scope but the file still exists: nothing
    // may depend on a destructor to make the container complete.
    let bytes = fs::read(&path).expect("output file exists");
    assert!(
        bytes.len() > 10_000,
        "expected a real bitstream, got {} bytes",
        bytes.len()
    );
    assert_eq!(count_box(&bytes, b"ftyp"), 1, "one file type box");
    assert_eq!(count_box(&bytes, b"moov"), 1, "one movie box");
    assert!(count_box(&bytes, b"moof") >= 1, "fragments present");
    assert!(count_box(&bytes, b"mdat") >= 1, "media data present");
    assert_eq!(count_box(&bytes, b"avcC"), 1, "one decoder config record");
}

/// The container has to decode, not merely look like MP4.
#[test]
fn the_muxed_file_decodes_with_ffprobe() {
    let path = std::env::temp_dir().join("blazar-videotoolbox-probe.mp4");
    encode_to(&path);

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames,avg_frame_rate,color_range",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output();
    let Ok(output) = output else {
        eprintln!("ffprobe not installed; skipping the decode check");
        return;
    };
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert!(
        output.status.success(),
        "ffprobe rejected the file: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Key/value rather than positional: ffprobe does not promise an order.
    let field = |key: &str| -> String {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(field("codec_name"), "h264");
    assert_eq!(field("width"), W.to_string());
    assert_eq!(field("height"), H.to_string());
    assert_eq!(field("nb_read_frames"), FRAMES.to_string(), "decoded frames");
    // Timestamps, not just pixels: each unit has to carry its duration or the
    // track looks like a 90000 fps stream one frame long.
    assert_eq!(field("avg_frame_rate"), "24/1", "frame rate from durations");
    // The preprocessor emits full-range planes, so the stream must say `pc`.
    // Declaring them video-range shifts every colour.
    assert_eq!(field("color_range"), "pc", "full-range signalling");
}
