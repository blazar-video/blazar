// SPDX-License-Identifier: Apache-2.0
//! Where should a host u8 frame be converted to NV12 before NVENC?
//!
//! The public API is the same either way — `submit(u8_frame)`. The choice is
//! internal, so the only way to pick is to measure:
//!
//! * **GPU round trip** (current): upload RGB, convert on the GPU, download
//!   NV12, hand NVENC host memory. 6.2 + 4.7 + 4.7 MB per frame.
//! * **CPU convert**: convert with the SIMD kernels already used by the x264
//!   path, upload NV12 only. 4.7 MB per frame, plus ~0.5 ms of CPU time.
//!
//! Both rows hand the same `PreprocessedFrame` to the same `NvencEncoder`, so
//! the only variable is who converted and what crossed the bus. Frames are
//! generated before the clock; the encoder is constructed before the clock.
//!
//! Run with:
//! `cargo test -p blazar-encoder --release --features nvenc -- --ignored --nocapture host_convert_timings`

#![cfg(feature = "nvenc")]

use std::time::Instant;

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_core::{Timebase, VideoCodec};
use blazar_encoder::config::{EncoderConfig, RateControl};
use blazar_encoder::gpu_preprocess::GpuPreprocess;
use blazar_encoder::nvenc::NvencEncoder;
use blazar_encoder::VideoEncoder;
use blazar_preprocess::{CpuPreprocess, PreprocessStage};

fn w() -> u32 {
    std::env::var("W")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1920)
}
fn h() -> u32 {
    std::env::var("H")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1088)
}
const FPS: f64 = 24.0;
const FRAMES: usize = 48;
const ROUNDS: usize = 3;

fn desc(pts: u64) -> FrameDesc {
    FrameDesc {
        width: w(),
        height: h(),
        dtype: Dtype::U8,
        range: ColorRange::FullU8,
        layout: Layout::Thwc,
        colorspace: ColorSpace::AssumedBt709,
        fps: FPS,
        pts: Pts(pts),
    }
}

fn cfg() -> EncoderConfig {
    let crf = std::env::var("CRF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30u32);
    let bframes = std::env::var("BF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2u32);
    EncoderConfig {
        rate_control: RateControl::Crf(crf),
        keyint: 120,
        bframes,
        pix_fmt: "yuv420p".into(),
        codec_params: String::new(),
    }
}

/// A moving pattern, so the encoder does real work rather than intra-only.
fn frames() -> Vec<Vec<u8>> {
    let (W, H) = (w() as usize, h() as usize);
    let mut out = Vec::with_capacity(FRAMES);
    for index in 0..FRAMES {
        let shift = (index * 7) as u8;
        let mut bytes = vec![0u8; (W * H * 3) as usize];
        for y in 0..H {
            let row = y * W * 3;
            for x in 0..W {
                let p = row + x * 3;
                bytes[p] = (x as u8).wrapping_add(shift);
                bytes[p + 1] = (y as u8).wrapping_add(shift / 2);
                bytes[p + 2] = 128u8.wrapping_add(shift);
            }
        }
        out.push(bytes);
    }
    out
}

fn encoder() -> NvencEncoder {
    NvencEncoder::new(&cfg(), w(), h(), FPS, Timebase::MPEG_90K, VideoCodec::H264)
        .expect("NVENC H.264 session opens")
}

/// GPU convert, then hand the downloaded NV12 to NVENC.
fn gpu_round_trip(src: &[Vec<u8>], gpu: &mut GpuPreprocess, enc: &mut NvencEncoder) -> usize {
    let mut bytes = 0;
    for (index, frame) in src.iter().enumerate() {
        let nv12 = gpu
            .convert(frame, &desc(index as u64 * 3750))
            .expect("gpu convert");
        if index == 0 {
            let cw = (w() as usize).div_ceil(2) * (h() as usize).div_ceil(2);
            eprintln!(
                "  planes: y={} u={} v={}  expected y={} u={} v={}",
                nv12.y.len(),
                nv12.u.len(),
                nv12.v.len(),
                (w() * h()) as usize,
                cw,
                cw
            );
        }
        for unit in enc.encode(&nv12).expect("encode") {
            bytes += unit.bytes.len();
        }
    }
    for unit in enc.finish().expect("finish") {
        bytes += unit.bytes.len();
    }
    bytes
}

/// CPU convert, then hand NV12 straight to NVENC. No device round trip.
fn cpu_convert(src: &[Vec<u8>], enc: &mut NvencEncoder) -> usize {
    let cpu = CpuPreprocess::new();
    let mut bytes = 0;
    for (index, frame) in src.iter().enumerate() {
        let host = FrameRef::new(
            FrameBuffer::host_owned(frame.clone()),
            desc(index as u64 * 3750),
        );
        let nv12 = cpu.process(&host).expect("cpu preprocess");
        if index == 0 {
            let cw = (w() as usize).div_ceil(2) * (h() as usize).div_ceil(2);
            eprintln!(
                "  planes: y={} u={} v={}  expected y={} u={} v={}",
                nv12.y.len(),
                nv12.u.len(),
                nv12.v.len(),
                (w() * h()) as usize,
                cw,
                cw
            );
        }
        for unit in enc.encode(&nv12).expect("encode") {
            bytes += unit.bytes.len();
        }
    }
    for unit in enc.finish().expect("finish") {
        bytes += unit.bytes.len();
    }
    bytes
}

/// Conversion alone, so the two costs can be attributed.
fn conversion_only(src: &[Vec<u8>], gpu: &mut GpuPreprocess) -> (f64, f64) {
    let cpu = CpuPreprocess::new();
    let t = Instant::now();
    for (index, frame) in src.iter().enumerate() {
        let host = FrameRef::new(
            FrameBuffer::host_owned(frame.clone()),
            desc(index as u64 * 3750),
        );
        std::hint::black_box(cpu.process(&host).expect("cpu preprocess"));
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / src.len() as f64;

    let t = Instant::now();
    for (index, frame) in src.iter().enumerate() {
        std::hint::black_box(
            gpu.convert(frame, &desc(index as u64 * 3750))
                .expect("gpu convert"),
        );
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / src.len() as f64;
    (cpu_ms, gpu_ms)
}

#[test]
#[ignore = "timing run: needs NVENC hardware, invoked explicitly"]
fn host_convert_timings() {
    let src = frames();
    let mut gpu = GpuPreprocess::new().expect("gpu preprocess");
    let (cpu_conv, gpu_conv) = conversion_only(&src, &mut gpu);
    println!("conversion only, {}x{} u8:", w(), h());
    println!("  cpu simd   {cpu_conv:6.2} ms/frame");
    println!("  gpu kernel {gpu_conv:6.2} ms/frame");

    // Warm-up: session open, kernel load and NVENC's first-frame cost are setup,
    // not steady-state throughput.
    if std::env::var("ROW").unwrap_or_default() != "cpu" {
        let mut enc = encoder();
        let mut warm = GpuPreprocess::new().expect("gpu preprocess");
        let _ = gpu_round_trip(&src[..4], &mut warm, &mut enc);
    }

    let mut gpu_best = f64::MAX;
    let mut cpu_best = f64::MAX;
    let mut gpu_bytes = 0;
    let mut cpu_bytes = 0;
    let only = std::env::var("ROW").unwrap_or_default();
    for _ in 0..ROUNDS {
        if only != "cpu" {
            let mut enc = encoder();
            let t = Instant::now();
            let bytes = gpu_round_trip(&src, &mut gpu, &mut enc);
            let ms = t.elapsed().as_secs_f64() * 1e3 / src.len() as f64;
            if ms < gpu_best {
                gpu_best = ms;
                gpu_bytes = bytes;
            }
        }

        if only != "gpu" {
            let mut enc = encoder();
            let t = Instant::now();
            let bytes = cpu_convert(&src, &mut enc);
            let ms = t.elapsed().as_secs_f64() * 1e3 / src.len() as f64;
            if ms < cpu_best {
                cpu_best = ms;
                cpu_bytes = bytes;
            }
        }
    }

    println!();
    println!("| path | ms/frame | fps | output |");
    println!("|---|---|---|---|");
    // A row filtered out by ROW never runs, so it has no timing: report only
    // the rows that did.
    if gpu_best < f64::MAX {
        println!(
            "| gpu convert, NVENC reads the device surface | {gpu_best:.2} | {:.0} | {:.2} MB |",
            FRAMES as f64 * 1e3 / gpu_best,
            gpu_bytes as f64 / 1e6
        );
    }
    if cpu_best < f64::MAX {
        println!(
            "| cpu convert, upload nv12 | {cpu_best:.2} | {:.0} | {:.2} MB |",
            FRAMES as f64 * 1e3 / cpu_best,
            cpu_bytes as f64 / 1e6
        );
    }
    assert!(
        gpu_bytes > 0 || cpu_bytes > 0,
        "at least one path must produce output"
    );
}
