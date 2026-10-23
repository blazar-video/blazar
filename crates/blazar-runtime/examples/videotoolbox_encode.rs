//! Encode a synthetic clip with Apple VideoToolbox and write an fMP4 file.
//!
//! ```sh
//! cargo run -p blazar-runtime --features enc-videotoolbox \
//!     --example videotoolbox_encode -- /tmp/out.mp4 1920 1080 120
//! ```
//!
//! The pattern moves so the encoder sees real inter-frame content: a still
//! image would be coded in one keyframe and prove nothing about the rate
//! control.

use std::fs;

use blazar_calibration::{load_str, CalibrationIndex};
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_preprocess::rgb_f32_len;
use blazar_runtime::{Encoder, Output};
use blazar_sink::fmp4::Fmp4FileSink;

const CAL: &str = r#"
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
bitrate_kbps = 8000
keyint = 48
bframes = 0
pix_fmt = "nv12"
"#;

fn frame(index: u64, w: u32, h: u32) -> FrameRef {
    let shift = (index * 4) as f32 / 255.0;
    let mut bytes = Vec::with_capacity(rgb_f32_len(w, h));
    for row in 0..h {
        for col in 0..w {
            // Three moving gradients plus a checkerboard band, so there are
            // hard edges as well as smooth ones.
            let checker = if ((col / 64) + (row / 64) + index as u32 / 8) % 2 == 0 { 0.15 } else { 0.0 };
            let r = ((col as f32 / w as f32) + shift) % 1.0 * (1.0 - checker) + checker;
            let g = ((row as f32 / h as f32) + shift) % 1.0 * (1.0 - checker) + checker;
            let b = (((col + row) as f32 / (w + h) as f32) + shift) % 1.0 * (1.0 - checker) + checker;
            for component in [r, g, b] {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
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
        pts: Pts::from_frame_index(index, 24.0, Timebase::MPEG_90K),
    };
    FrameRef::new(FrameBuffer::host_owned(bytes), desc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "/tmp/videotoolbox.mp4".into());
    let w: u32 = args.next().unwrap_or_else(|| "1920".into()).parse()?;
    let h: u32 = args.next().unwrap_or_else(|| "1080".into()).parse()?;
    let frames: u64 = args.next().unwrap_or_else(|| "120".into()).parse()?;

    let mut index = CalibrationIndex::new();
    index.insert(load_str(CAL, "default/videotoolbox_h264_balanced.toml")?);

    let _ = fs::remove_file(&path);
    let sink = Fmp4FileSink::new(std::io::BufWriter::new(fs::File::create(&path)?));
    let encoder = Encoder::new(
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
        index,
    );

    let started = std::time::Instant::now();
    let mut stream = encoder.open_stream("videotoolbox-example");
    for i in 0..frames {
        stream.submit_video(frame(i, w, h), None, None)?;
    }
    stream.close()?;
    let elapsed = started.elapsed().as_secs_f64();

    let size = fs::metadata(&path)?.len();
    println!(
        "{path}: {frames} frames {w}x{h} in {elapsed:.2}s ({:.2} ms/frame), {size} bytes",
        elapsed * 1000.0 / frames as f64
    );
    Ok(())
}
