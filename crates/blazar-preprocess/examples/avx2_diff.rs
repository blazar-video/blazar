// SPDX-License-Identifier: Apache-2.0
//! Differentially compare the AVX2 conversion against the scalar reference.
//!
//! Prints the first few differing samples per plane with their indices, which is
//! the only information that reliably locates a fault in interleaved-byte SIMD.

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameDesc, Layout, Pts};
use blazar_preprocess::integer;

fn main() {
    for (w, h) in [(16usize, 8usize), (16, 2), (18, 2), (131, 67), (1920, 1088)] {
        let d = FrameDesc {
            width: w as u32,
            height: h as u32,
            dtype: Dtype::U8,
            layout: Layout::Thwc,
            range: ColorRange::FullU8,
            colorspace: ColorSpace::AssumedBt709,
            fps: 24.0,
            pts: Pts(0),
        };
        let mut state = 0x9e37_79b9u32;
        let bytes: Vec<u8> = (0..w * h * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 7) as u8
            })
            .collect();

        let want = integer::convert_thwc_u8(&bytes, &d, w, h);
        let Some(got) = integer::convert_avx2(&bytes, &d, w, h) else {
            println!("no avx2");
            return;
        };

        if w == 16 && h == 8 {
            println!("16x8 per-row Y diff counts:");
            for r in 0..h {
                let n = (0..w)
                    .filter(|&x| want.0[r * w + x] != got.0[r * w + x])
                    .count();
                println!(
                    "  row {r}: {n} differ   want {:?}   got {:?}",
                    &want.0[r * w..r * w + 8],
                    &got.0[r * w..r * w + 8]
                );
            }
        }

        for (name, a, b, stride) in [
            ("Y", &want.0, &got.0, w),
            ("U", &want.1, &got.1, w.div_ceil(2)),
            ("V", &want.2, &got.2, w.div_ceil(2)),
        ] {
            let diffs: Vec<(usize, u8, u8)> = a
                .iter()
                .zip(b)
                .enumerate()
                .filter(|(_, (x, y))| x != y)
                .map(|(i, (x, y))| (i, *x, *y))
                .collect();
            match diffs.first() {
                None => println!("{w}x{h} {name}: identical"),
                Some(&(i, x, y)) => println!(
                    "{w}x{h} {name}: {} differ, first at ({}, {}) want {x} got {y}",
                    diffs.len(),
                    i / stride,
                    i % stride
                ),
            }
        }
    }
}
