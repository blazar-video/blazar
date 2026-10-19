//! GPU preprocessing must agree with the CPU path byte-for-byte modulo float
//! ordering. This is the test that decides whether the kernel's colour math and
//! NV12 layout are right; encode/decode can hide either mistake.

#![cfg(feature = "nvenc")]

use std::sync::Arc;

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_encoder::nvenc::cuda::Nv12Converter;
use blazar_encoder::nvenc::nv12::to_nv12;
use blazar_preprocess::{CpuPreprocess, PreprocessStage};
use cudarc::driver::CudaContext;

fn frame_desc(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) -> FrameDesc {
    FrameDesc {
        width: w,
        height: h,
        dtype,
        range,
        layout,
        colorspace: ColorSpace::AssumedBt709,
        fps: 24.0,
        pts: Pts(0),
    }
}

fn host_rgb(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let components = (w * h * 3) as usize;
    let mut bytes = Vec::with_capacity(components * if dtype == Dtype::Bf16 { 2 } else { 4 });
    for _ in 0..components {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 40) as f32 / 16_777_215.0;
        let value = match range {
            ColorRange::SignedUnit => unit * 2.0 - 1.0,
            ColorRange::FullU8 => (state >> 40) as f32 % 256.0,
            ColorRange::Unit01 => unit,
        };
        match dtype {
            Dtype::Bf16 => bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes()),
            _ => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    bytes
}

fn check(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) {
    let pitch = w.div_ceil(256) * 256; // driver-style aligned stride
    let desc = frame_desc(w, h, layout, dtype, range);
    let bytes = host_rgb(w, h, layout, dtype, range);
    let frame = FrameRef::new(FrameBuffer::host_owned(bytes.clone()), desc);

    let cpu = to_nv12(
        &CpuPreprocess::new()
            .process(&frame)
            .expect("cpu preprocess"),
        pitch,
    )
    .expect("cpu nv12");

    let ctx: Arc<CudaContext> = CudaContext::new(0).expect("cuda");
    let stream = ctx.default_stream();
    let src = stream.clone_htod(&bytes).expect("upload rgb");
    let chroma_rows = h.div_ceil(2);
    let mut dst = stream
        .alloc_zeros::<u8>((pitch * (h + chroma_rows)) as usize)
        .expect("alloc nv12");
    let converter = Nv12Converter::new(&ctx).expect("compile kernel");
    converter
        .convert(&stream, &src, &mut dst, &desc, pitch)
        .expect("gpu convert");
    let gpu = stream.clone_dtoh(&dst).expect("download nv12");
    stream.synchronize().expect("sync");

    assert_eq!(gpu.len(), cpu.len(), "NV12 buffer size");
    let worst = gpu
        .iter()
        .zip(&cpu)
        .map(|(g, c)| (*g as i32 - *c as i32).abs())
        .max()
        .unwrap_or(0);
    assert!(
        worst <= 1,
        "{layout:?}/{dtype:?}/{range:?} {w}x{h}: max |gpu-cpu| = {worst} (CPU/GPU disagree)"
    );
}

#[test]
fn gpu_matches_cpu_for_thwc_f32() {
    check(320, 240, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
    check(317, 241, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
}

#[test]
fn gpu_matches_cpu_for_bcthw_f32_signed() {
    check(320, 240, Layout::Bcthw, Dtype::F32, ColorRange::SignedUnit);
    check(31, 17, Layout::Bcthw, Dtype::F32, ColorRange::SignedUnit);
}

#[test]
fn gpu_matches_cpu_for_bf16() {
    check(160, 120, Layout::Thwc, Dtype::Bf16, ColorRange::Unit01);
    check(33, 17, Layout::Bcthw, Dtype::Bf16, ColorRange::SignedUnit);
}
