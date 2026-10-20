//! GPU preprocessing must agree with the CPU path byte-for-byte modulo float
//! ordering. This is the test that decides whether the kernel's colour math and
//! NV12 layout are right; encode/decode can hide either mistake.

#![cfg(feature = "nvenc")]

use std::sync::Arc;

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_encoder::nvenc::cuda::Nv12Converter;
use blazar_encoder::nvenc::nv12::to_nv12;
use blazar_preprocess::{CpuPreprocess, PreprocessStage};
use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

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

fn host_rgb(
    w: u32,
    h: u32,
    layout: Layout,
    dtype: Dtype,
    range: ColorRange,
    seed: u64,
) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ seed;
    let components = (w * h * 3) as usize;
    // Values are generated once in sample order (row-major, RGB interleaved) and
    // then written in the requested memory layout. Generating them in layout
    // order matters: BCTHW and THWC otherwise feed identical bytes to both
    // implementations and the bcthw kernel path goes untested.
    let mut values = Vec::with_capacity(components);
    for _ in 0..components {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 40) as f32 / 16_777_215.0;
        values.push(match range {
            ColorRange::SignedUnit => unit * 2.0 - 1.0,
            ColorRange::FullU8 => (state >> 40) as f32 % 256.0,
            ColorRange::Unit01 => unit,
        });
    }
    let order: Vec<usize> = match layout {
        Layout::Thwc => (0..components).collect(),
        // BCTHW with a single frame: channel, then row, then column.
        Layout::Bcthw => (0..3)
            .flat_map(|c| (0..(w * h) as usize).map(move |px| px * 3 + c))
            .collect(),
    };
    let mut bytes = Vec::with_capacity(components * if dtype == Dtype::Bf16 { 2 } else { 4 });
    for index in order {
        match dtype {
            Dtype::Bf16 => {
                bytes.extend_from_slice(&((values[index].to_bits() >> 16) as u16).to_le_bytes());
            }
            _ => bytes.extend_from_slice(&values[index].to_le_bytes()),
        }
    }
    bytes
}

fn check(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) {
    let pitch = w.div_ceil(256) * 256; // driver-style aligned stride
    let desc = frame_desc(w, h, layout, dtype, range);
    let bytes = host_rgb(w, h, layout, dtype, range, 0);
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
    let (worst_idx, worst_diff) = gpu
        .iter()
        .zip(&cpu)
        .enumerate()
        .map(|(i, (g, c))| (i, (*g as i32 - *c as i32).abs()))
        .max_by_key(|(_, d)| *d)
        .unwrap_or((0, 0));
    eprintln!(
        "diag {layout:?}/{dtype:?}/{range:?} {w}x{h}: worst {worst_diff} at {worst_idx} \
         (gpu={} cpu={}) luma_len={} pitch={pitch}",
        gpu[worst_idx],
        cpu[worst_idx],
        pitch * h
    );
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
    check(318, 240, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
}

#[test]
fn gpu_matches_cpu_for_bcthw_f32_signed() {
    check(320, 240, Layout::Bcthw, Dtype::F32, ColorRange::SignedUnit);
    check(32, 18, Layout::Bcthw, Dtype::F32, ColorRange::SignedUnit);
}

#[test]
fn gpu_matches_cpu_for_bf16() {
    check(160, 120, Layout::Thwc, Dtype::Bf16, ColorRange::Unit01);
    check(34, 18, Layout::Bcthw, Dtype::Bf16, ColorRange::SignedUnit);
}

/// Frames in a batch must be read from their own slice of the staging buffer
/// and written to their own destination. A seed per frame makes a wrong frame
/// index visible: identical frames would pass even if every thread read frame 0.
#[test]
fn gpu_matches_cpu_for_a_batch_of_frames() {
    let (w, h, n) = (64u32, 48u32, 3u32);
    let pitch = w.div_ceil(256) * 256;
    let desc = frame_desc(w, h, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
    let chroma_rows = h.div_ceil(2);

    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for frame_index in 0..n {
        let data = host_rgb(w, h, Layout::Thwc, Dtype::F32, ColorRange::Unit01, frame_index as u64 + 1);
        let frame = FrameRef::new(FrameBuffer::host_owned(data.clone()), desc);
        expected.push(
            to_nv12(&CpuPreprocess::new().process(&frame).expect("cpu preprocess"), pitch)
                .expect("cpu nv12"),
        );
        bytes.extend_from_slice(&data);
    }

    let ctx: Arc<CudaContext> = CudaContext::new(0).expect("cuda");
    let stream = ctx.default_stream();
    let src = stream.clone_htod(&bytes).expect("upload rgb");
    let dsts: Vec<CudaSlice<u8>> = (0..n)
        .map(|_| {
            stream
                .alloc_zeros::<u8>((pitch * (h + chroma_rows)) as usize)
                .expect("alloc nv12")
        })
        .collect();
    let ptrs: Vec<u64> = dsts
        .iter()
        .map(|dst| {
            let (ptr, guard) = dst.device_ptr(&stream);
            drop(guard);
            ptr
        })
        .collect();
    let ptr_buf = stream.clone_htod(&ptrs).expect("upload dst pointers");

    let converter = Nv12Converter::new(&ctx).expect("compile kernel");
    converter
        .convert_batch(&stream, &src, &ptr_buf, &desc, n, pitch)
        .expect("gpu convert batch");
    stream.synchronize().expect("sync");

    for (frame_index, dst) in dsts.iter().enumerate() {
        let gpu = stream.clone_dtoh(dst).expect("download nv12");
        let worst = gpu
            .iter()
            .zip(&expected[frame_index])
            .map(|(g, c)| (*g as i32 - *c as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 1,
            "batch frame {frame_index}: max |gpu-cpu| = {worst} (frames crossed?)"
        );
    }
}

/// BCTHW with T > 1 has frame stride H*W and channel stride T*H*W. Two frames
/// of distinct data catch a kernel that assumes per-frame channel planes.
#[test]
fn gpu_matches_cpu_for_a_bcthw_batch() {
    let (w, h, n) = (48u32, 32u32, 3u32);
    let pitch = w.div_ceil(256) * 256;
    let desc = frame_desc(w, h, Layout::Bcthw, Dtype::F32, ColorRange::Unit01);
    let chroma_rows = h.div_ceil(2);
    let pixels = (w * h) as usize;

    // (C, T, H, W) host buffer, which is what (1, C, T, H, W) puts on the wire.
    let mut bytes = vec![0u8; 3 * n as usize * pixels * 4];
    let mut expected = Vec::new();
    for frame_index in 0..n {
        let data = host_rgb(w, h, Layout::Thwc, Dtype::F32, ColorRange::Unit01, frame_index as u64 + 7);
        // The CPU reference must see the same samples in the layout its desc
        // claims: one frame of BCTHW is (C, 1, H, W).
        let cpu_data = host_rgb(w, h, Layout::Bcthw, Dtype::F32, ColorRange::Unit01, frame_index as u64 + 7);
        let frame = FrameRef::new(FrameBuffer::host_owned(cpu_data), desc);
        expected.push(
            to_nv12(&CpuPreprocess::new().process(&frame).expect("cpu preprocess"), pitch)
                .expect("cpu nv12"),
        );
        // Copy the THWC samples into BCTHW planes: channel, frame, row, col.
        for c in 0..3usize {
            for p in 0..pixels {
                let src = (p * 3 + c) * 4;
                let dst = ((c * n as usize + frame_index as usize) * pixels + p) * 4;
                bytes[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
            }
        }
    }

    let ctx: Arc<CudaContext> = CudaContext::new(0).expect("cuda");
    let stream = ctx.default_stream();
    let src = stream.clone_htod(&bytes).expect("upload rgb");
    let dsts: Vec<CudaSlice<u8>> = (0..n)
        .map(|_| {
            stream
                .alloc_zeros::<u8>((pitch * (h + chroma_rows)) as usize)
                .expect("alloc nv12")
        })
        .collect();
    let ptrs: Vec<u64> = dsts
        .iter()
        .map(|dst| {
            let (ptr, guard) = dst.device_ptr(&stream);
            drop(guard);
            ptr
        })
        .collect();
    let ptr_buf = stream.clone_htod(&ptrs).expect("upload dst pointers");
    let converter = Nv12Converter::new(&ctx).expect("compile kernel");
    converter
        .convert_batch(&stream, &src, &ptr_buf, &desc, n, pitch)
        .expect("gpu convert");
    stream.synchronize().expect("sync");

    for (frame_index, dst) in dsts.iter().enumerate() {
        let gpu = stream.clone_dtoh(dst).expect("download nv12");
        let worst = gpu
            .iter()
            .zip(&expected[frame_index])
            .map(|(g, c)| (*g as i32 - *c as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 1,
            "bcthw batch frame {frame_index}: max |gpu-cpu| = {worst} (channel/frame stride wrong?)"
        );
    }
}
