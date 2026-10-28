// SPDX-License-Identifier: Apache-2.0
//! Host frame -> preprocess -> x264 -> fMP4, through the buffered writer the
//! Python binding uses.
//!
//! The regression this exists for: `close` wrote the trailer into a
//! `BufWriter` and returned, so everything stayed buffered until the `Stream`
//! was dropped, after the caller had already been told the stream was closed.
//! An encoder that holds every frame until finish then looked like it had
//! produced a 0-byte file.

#![cfg(feature = "enc-x264")]

use std::fs;

use blazar_calibration::{load_str, CalibrationIndex};
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_preprocess::rgb_f32_len;
use blazar_runtime::{Encoder, Output};
use blazar_sink::fmp4::Fmp4FileSink;

// keyint is deeper than the stream on purpose: x264 then holds every frame in
// its lookahead and emits the whole payload on finish.
const X264_CAL: &str = r#"
schema_version = 1
source = "default"
[key]
model = { family = "*", version = "*", quant = "*", size = "*" }
scene = "*"
codec = "x264"
target = "balanced"
[video.common]
rate_control = "crf"
crf = 30
bitrate_kbps = 3000
keyint = 120
bframes = 3
pix_fmt = "yuv420p"
"#;

const W: u32 = 320;
const H: u32 = 240;
const FRAMES: u64 = 24;

fn index() -> CalibrationIndex {
    let mut idx = CalibrationIndex::new();
    idx.insert(load_str(X264_CAL, "default/x264_balanced.toml").expect("calibration loads"));
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
fn x264_close_flushes_the_buffered_writer() {
    let path = std::env::temp_dir().join("blazar-x264-close.mp4");
    let _ = fs::remove_file(&path);

    let sink = Fmp4FileSink::new(std::io::BufWriter::new(
        fs::File::create(&path).expect("create smoke file"),
    ));
    let enc = Encoder::new(
        ModelIdentity::new("wan", "2.1", "bf16", "1.3b"),
        Target::Balanced,
        Output {
            codec: VideoCodec::H264,
            backend: VideoEncoderBackend::X264,
            container: Container::FragmentedMp4,
            video_extradata: None,
            audio_extradata: None,
        },
        sink,
        index(),
    );
    let mut stream = enc.open_stream("x264-close");
    for i in 0..FRAMES {
        stream
            .submit_video(host_frame(i), None, None)
            .expect("submit");
    }
    stream.close().expect("close");

    // Deliberately read while the stream is still in scope: nothing may depend
    // on its destructor to make the file complete.
    let bytes = fs::read(&path).expect("smoke file exists");
    assert!(
        bytes.len() > 1000,
        "expected a real bitstream, got {} bytes",
        bytes.len()
    );
    assert_eq!(count_box(&bytes, b"ftyp"), 1);
    assert_eq!(count_box(&bytes, b"moov"), 1);
    assert!(count_box(&bytes, b"moof") >= 1);
    assert!(count_box(&bytes, b"mdat") >= 1);
    assert_eq!(count_box(&bytes, b"avcC"), 1);
}
