// SPDX-License-Identifier: Apache-2.0
//! The fixed-point path must agree with the float path to within one code value.
//!
//! One is enough: the float path rounds once at the end, so the integer path can
//! differ by at most a rounding step. Anything larger is a real error — a
//! coefficient that does not sum to unity, chroma computed from rounded luma, or an
//! edge cell divided by the wrong count. The odd dimensions are deliberate: the
//! last row and column produce 1- and 2-pixel chroma cells.

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout, Pts};
use blazar_preprocess::{integer, CpuPreprocess, PreprocessStage};

fn desc(w: u32, h: u32, colorspace: ColorSpace) -> FrameDesc {
    FrameDesc {
        width: w,
        height: h,
        dtype: Dtype::U8,
        layout: Layout::Thwc,
        range: ColorRange::FullU8,
        colorspace,
        fps: 24.0,
        pts: Pts(0),
    }
}

fn pixels(w: u32, h: u32, seed: u32) -> Vec<u8> {
    // Noise rather than a gradient: gradients miss the clamp and near-saturation
    // cases at both ends of every channel.
    let mut state = seed | 1;
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..w * h * 3 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push((state >> 7) as u8);
    }
    out
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "plane lengths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn thwc_u8_matches_the_float_path_within_one_code_value() {
    for (w, h) in [
        (131, 67),
        (64, 64),
        (127, 65),
        (2, 2),
        (3, 3),
        (1, 1),
        (130, 66),
    ] {
        for colorspace in [ColorSpace::AssumedBt709, ColorSpace::AssumedBt601] {
            let rgb = pixels(w, h, (w * h) ^ 0x9e37);
            let d = desc(w, h, colorspace);

            let frame = FrameRef::new(FrameBuffer::host_owned(rgb.clone()), d.clone());
            let float = CpuPreprocess::new().process(&frame).expect("float path");
            let (y, u, v) = integer::convert_thwc_u8(&rgb, &d, w as usize, h as usize);

            for (name, int, flt) in [
                ("Y", &y, &float.y),
                ("U", &u, &float.u),
                ("V", &v, &float.v),
            ] {
                let diff = max_diff(int, flt);
                assert!(
                    diff <= 1,
                    "{name} differs by {diff} at {w}x{h} {colorspace:?}"
                );
            }
        }
    }
}
