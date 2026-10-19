//! End-to-end hardware smoke test: host frame → preprocess → NVENC → fMP4 file.
//!
//! Compiled only with `--features nvenc` and only meaningful on a machine with
//! an NVENC-capable GPU. Unlike the synthetic-payload sink tests, every byte
//! here is a real AV1 bitstream, so the output can be handed to a decoder.
//!
//! The file is left in the temp directory and its path printed, so an external
//! probe (ffprobe/ffmpeg/dav1d) can validate it: `ffmpeg -xerror -i <file> -f null -`
//! must decode cleanly and `ffprobe -count_frames` must report every frame.

#![cfg(feature = "nvenc")]

use std::fs;

use blazar_calibration::{load_str, CalibrationIndex};
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_preprocess::rgb_f32_len;
use blazar_runtime::{Encoder, Output};
use blazar_sink::fmp4::Fmp4FileSink;

const NVENC_CAL: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "nvenc_av1"
target = "low_latency"
[video.common]
rate_control = "crf"
crf = 40
keyint = 12
bframes = 0
pix_fmt = "yuv420p"
"#;

const W: u32 = 320;
const H: u32 = 240;

fn index() -> CalibrationIndex {
    let mut idx = CalibrationIndex::new();
    idx.insert(
        load_str(NVENC_CAL, "default/nvenc_av1_low_latency.toml").expect("calibration loads"),
    );
    idx
}

/// Moving pattern so the encoder has real inter-frame content.
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

#[test]
fn nvenc_to_fmp4_produces_a_real_av1_container() {
    let path = std::env::temp_dir().join("blazar-nvenc-smoke.mp4");
    let _ = fs::remove_file(&path);

    let sink = Fmp4FileSink::new(fs::File::create(&path).expect("create smoke file"));
    let mut enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::LowLatency,
        Output {
            codec: VideoCodec::Av1,
            backend: VideoEncoderBackend::NvencAv1,
            container: Container::FragmentedMp4,
            // Real encoder: it derives av1C from the first keyframe.
            video_extradata: None,
            audio_extradata: None,
        },
        sink,
        index(),
    );
    {
        let mut stream = enc.open_stream("nvenc-smoke");
        for i in 0..24u64 {
            stream
                .submit_video(host_frame(i), None, None)
                .expect("submit");
        }
        stream.close().expect("close");
    }
    drop(enc);

    let bytes = fs::read(&path).expect("smoke file exists");
    assert!(
        bytes.len() > 10_000,
        "expected a real bitstream, got {} bytes",
        bytes.len()
    );
    assert_eq!(count_box(&bytes, b"ftyp"), 1);
    assert_eq!(count_box(&bytes, b"moov"), 1);
    assert!(count_box(&bytes, b"moof") >= 1);
    assert!(count_box(&bytes, b"mdat") >= 1);
    assert_eq!(
        count_box(&bytes, b"av01"),
        1,
        "AV1 sample entry must be present"
    );
    assert_eq!(
        count_box(&bytes, b"av1C"),
        1,
        "codec configuration must be present"
    );

    println!("SMOKE_MP4={}", path.display());
}

const ROW_W: u32 = 834;
const ROW_H: u32 = 480;

/// Every row is one constant colour, alternating black and white. Adjacent rows
/// differ maximally, so a stride mistake (which mixes neighbouring rows within a
/// decoded row) is unmistakable, and it cannot be confused with quantization
/// noise the way a low-contrast ramp can.
fn row_frame(index: u64) -> FrameRef {
    let mut bytes = Vec::with_capacity(rgb_f32_len(ROW_W, ROW_H));
    for row in 0..ROW_H {
        let v: f32 = if row % 2 == 0 { 0.0 } else { 1.0 };
        for _ in 0..ROW_W {
            for _ in 0..3 {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let desc = FrameDesc {
        width: ROW_W,
        height: ROW_H,
        dtype: Dtype::F32,
        range: ColorRange::Unit01,
        layout: Layout::Thwc,
        colorspace: ColorSpace::AssumedBt709,
        fps: 24.0,
        pts: Pts::from_frame_index(index, 24.0, Timebase::MPEG_90K),
    };
    FrameRef::new(FrameBuffer::host_owned(bytes), desc)
}

/// 834 is not a stride the driver allocates (it uses 896), so this is the case
/// that breaks if NV12 rows are padded to the width instead of the lock pitch.
#[test]
fn unaligned_width_produces_an_av1_container() {
    let path = std::env::temp_dir().join("blazar-nvenc-unaligned.mp4");
    let _ = fs::remove_file(&path);

    let sink = Fmp4FileSink::new(fs::File::create(&path).expect("create unaligned file"));
    let mut enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::LowLatency,
        Output {
            codec: VideoCodec::Av1,
            backend: VideoEncoderBackend::NvencAv1,
            container: Container::FragmentedMp4,
            video_extradata: None,
            audio_extradata: None,
        },
        sink,
        index(),
    );
    {
        let mut stream = enc.open_stream("nvenc-unaligned");
        for i in 0..2u64 {
            stream
                .submit_video(row_frame(i), None, None)
                .expect("submit");
        }
        stream.close().expect("close");
    }
    drop(enc);

    let bytes = fs::read(&path).expect("unaligned file exists");
    assert_eq!(count_box(&bytes, b"av01"), 1);
    assert!(count_box(&bytes, b"moof") >= 1);

    println!("SMOKE_UNALIGNED_MP4={}", path.display());
}
