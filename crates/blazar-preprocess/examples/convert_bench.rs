// SPDX-License-Identifier: Apache-2.0
//! Conversion cost: the float SIMD path against the fixed-point reference.
//!
//! Not a benchmark harness — no warm-up statistics, no confidence intervals. It
//! answers one question: is the scalar fixed-point version in the same range as
//! the existing SIMD float path, i.e. is a fixed-point *kernel* worth writing.

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_preprocess::{integer, CpuPreprocess, PreprocessStage};
use std::time::Instant;

fn desc(w: u32, h: u32) -> FrameDesc {
    FrameDesc {
        width: w,
        height: h,
        dtype: Dtype::U8,
        layout: Layout::Thwc,
        range: ColorRange::FullU8,
        colorspace: ColorSpace::AssumedBt709,
        fps: 24.0,
        pts: Pts(0),
    }
}

fn pixels(w: u32, h: u32) -> Vec<u8> {
    let mut state = 0x1234_5678u32;
    (0..w * h * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 7) as u8
        })
        .collect()
}

fn main() {
    let stage = CpuPreprocess::new();
    for (w, h) in [(1280u32, 720u32), (1920, 1088)] {
        let rgb = pixels(w, h);
        let d = desc(w, h);
        let reps = 50;

        // Float SIMD path, through the same entry point the encoder uses.
        let frame = FrameRef::new(FrameBuffer::host_owned(rgb.clone()), d);
        for _ in 0..3 {
            std::hint::black_box(stage.process(&frame).unwrap());
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(stage.process(&frame).unwrap());
        }
        let float_simd = t0.elapsed().as_secs_f64() / f64::from(reps);

        // Scalar fixed-point reference.
        for _ in 0..3 {
            std::hint::black_box(integer::convert(&rgb, &d, w as usize, h as usize));
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(integer::convert(&rgb, &d, w as usize, h as usize));
        }
        let integer_scalar = t0.elapsed().as_secs_f64() / f64::from(reps);

        println!(
            "{w}x{h}  float SIMD {:6.2} ms   integer scalar {:6.2} ms   scalar is {:5.2}x",
            float_simd * 1e3,
            integer_scalar * 1e3,
            integer_scalar / float_simd
        );
    }
}
