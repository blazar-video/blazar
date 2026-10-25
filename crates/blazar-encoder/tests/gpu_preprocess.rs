//! The GPU conversion stage must produce the same NV12 the CPU stage does: it
//! is the same pixels, just converted on the other processor.

#![cfg(feature = "nvenc")]

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_encoder::gpu_preprocess::GpuPreprocess;
use blazar_preprocess::{CpuPreprocess, PreprocessStage};

fn desc(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) -> FrameDesc {
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
        Layout::Bcthw => (0..3)
            .flat_map(|c| (0..(w * h) as usize).map(move |px| px * 3 + c))
            .collect(),
    };
    let mut bytes = Vec::with_capacity(components * if dtype == Dtype::Bf16 { 2 } else { 4 });
    for index in order {
        match dtype {
            Dtype::Bf16 => {
                bytes.extend_from_slice(&((values[index].to_bits() >> 16) as u16).to_le_bytes())
            }
            _ => bytes.extend_from_slice(&values[index].to_le_bytes()),
        }
    }
    bytes
}

fn check(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) {
    let desc = desc(w, h, layout, dtype, range);
    let bytes = host_rgb(w, h, layout, dtype, range);

    let frame = FrameRef::new(FrameBuffer::host_owned(bytes.clone()), desc);
    let cpu = CpuPreprocess::new().process(&frame).expect("cpu preprocess");

    let mut gpu = GpuPreprocess::new().expect("gpu preprocess");
    let out = gpu.convert(&bytes, &desc).expect("gpu convert");

    assert_eq!(out.y.len(), cpu.y.len(), "luma size");
    assert_eq!(out.u.len(), cpu.u.len(), "cb size");
    assert_eq!(out.v.len(), cpu.v.len(), "cr size");

    for (plane, (g, c)) in [("y", (&out.y, &cpu.y)), ("u", (&out.u, &cpu.u)), ("v", (&out.v, &cpu.v))] {
        let worst = g
            .iter()
            .zip(c)
            .map(|(g, c)| (*g as i32 - *c as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 1,
            "{plane} plane {layout:?}/{dtype:?}/{range:?} {w}x{h}: max |gpu-cpu| = {worst}"
        );
    }
}

#[test]
fn gpu_host_output_matches_cpu_preprocess() {
    check(64, 48, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
    check(66, 50, Layout::Bcthw, Dtype::F32, ColorRange::SignedUnit);
    check(34, 18, Layout::Bcthw, Dtype::Bf16, ColorRange::Unit01);
    check(48, 32, Layout::Thwc, Dtype::F32, ColorRange::FullU8);
}

#[test]
fn geometry_changes_reallocate() {
    let mut gpu = GpuPreprocess::new().expect("gpu preprocess");
    for (w, h) in [(32, 24), (64, 48), (32, 24)] {
        let desc = desc(w, h, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
        let bytes = host_rgb(w, h, Layout::Thwc, Dtype::F32, ColorRange::Unit01);
        let out = gpu.convert(&bytes, &desc).expect("gpu convert");
        assert_eq!(out.y.len(), (w * h) as usize);
        assert!(gpu.pitch() >= w);
    }
}

fn host_rgb_u8(w: u32, h: u32, layout: Layout) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let components = (w * h * 3) as usize;
    let mut values = Vec::with_capacity(components);
    for _ in 0..components {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push((state >> 40) as u8);
    }
    match layout {
        Layout::Thwc => values,
        Layout::Bcthw => {
            let mut out = Vec::with_capacity(components);
            for c in 0..3 {
                for px in 0..(w * h) as usize {
                    out.push(values[px * 3 + c]);
                }
            }
            out
        }
    }
}

/// u8 RGB is full-range by definition: the kernel normalises it and ignores the
/// range argument, so its output must match the CPU stage's u8 path.
#[test]
fn u8_host_output_matches_cpu_preprocess() {
    for layout in [Layout::Thwc, Layout::Bcthw] {
        // Odd dimensions exercise the chroma tail; 1 frame cannot expose a
        // frame-stride bug, so the layout check runs over the same buffer shape
        // the caller stacks.
        let (w, h) = (131u32, 67u32);
        let desc = desc(w, h, layout, Dtype::U8, ColorRange::FullU8);
        let bytes = host_rgb_u8(w, h, layout);

        let frame = FrameRef::new(FrameBuffer::host_owned(bytes.clone()), desc);
        let cpu = CpuPreprocess::new().process(&frame).expect("cpu preprocess");
        let mut gpu = GpuPreprocess::new().expect("gpu preprocess");
        let out = gpu.convert(&bytes, &desc).expect("gpu convert");

        assert_eq!(out.y.len(), cpu.y.len(), "luma size");
        assert_eq!(out.u.len(), cpu.u.len(), "cb size");
        assert_eq!(out.v.len(), cpu.v.len(), "cr size");
        for (plane, (g, c)) in
            [("y", (&out.y, &cpu.y)), ("u", (&out.u, &cpu.u)), ("v", (&out.v, &cpu.v))]
        {
            let worst = g
                .iter()
                .zip(c)
                .map(|(g, c)| (*g as i32 - *c as i32).abs())
                .max()
                .unwrap_or(0);
            assert!(
                worst <= 1,
                "{plane} plane {layout:?}/U8 {w}x{h}: max |gpu-cpu| = {worst}"
            );
        }
    }
}
