//! aarch64 NEON kernel for interleaved 8-bit RGB.
//!
//! `vld3q_u8` deinterleaves 16 RGB triplets in one instruction, which is why
//! the interleaved layout costs nothing here: no shuffles to unpack channels.

use std::arch::aarch64::*;

use rayon::prelude::*;

use super::{pixel, Coeffs};

/// Pixels handled by one vector iteration.
const LANES: usize = 16;

pub(crate) fn convert(
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
) {
    let c = Coeffs::new(desc);
    let cw = w.div_ceil(2);
    let pixels = w * h;

    let row_pair = |cy: usize, y_rows: &mut [u8], u_row: &mut [u8], v_row: &mut [u8]| {
        // Scratch holds per-pixel chroma for the two luma rows of this pair.
        // Two rows of f32 at 1080p is 15 KB: it stays in L1/L2, unlike a whole
        // chroma plane, which is what made the two-pass version slower.
        let mut cb0 = vec![0.0f32; w];
        let mut cr0 = vec![0.0f32; w];
        let mut cb1 = vec![0.0f32; w];
        let mut cr1 = vec![0.0f32; w];
        let rows = y_rows.len() / w;
        for (dy, y_row) in y_rows.chunks_exact_mut(w).enumerate() {
            let sy = cy * 2 + dy;
            debug_assert!(sy < h);
            let src = &bytes[sy * w * 3..(sy + 1) * w * 3];
            // SAFETY: the pointer arithmetic stays inside `src`; the scratch
            // pointers cover `w` floats each.
            unsafe {
                let (cb, cr) = if dy == 0 {
                    (&mut cb0, &mut cr0)
                } else {
                    (&mut cb1, &mut cr1)
                };
                row(
                    &c,
                    src.as_ptr(),
                    w,
                    y_row.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            }
        }
        // Odd heights: the last chroma row averages one luma row over itself.
        if rows == 1 {
            cb1.copy_from_slice(&cb0);
            cr1.copy_from_slice(&cr0);
        }
        combine(&cb0, &cr0, &cb1, &cr1, cw, u_row, v_row);
    };

    if pixels < crate::PARALLEL_MIN_PIXELS {
        for (cy, ((y_rows, u_row), v_row)) in y_plane
            .chunks_mut(2 * w)
            .zip(u_plane.chunks_mut(cw))
            .zip(v_plane.chunks_mut(cw))
            .enumerate()
        {
            row_pair(cy, y_rows, u_row, v_row);
        }
    } else {
        y_plane
            .par_chunks_mut(2 * w)
            .zip(u_plane.par_chunks_mut(cw))
            .zip(v_plane.par_chunks_mut(cw))
            .enumerate()
            .for_each(|(cy, ((y_rows, u_row), v_row))| row_pair(cy, y_rows, u_row, v_row));
    }
}

/// Fold the 2x2 (or 1x2 at an odd height) chroma block and quantize.
fn combine(cb0: &[f32], cr0: &[f32], cb1: &[f32], cr1: &[f32], cw: usize, u: &mut [u8], v: &mut [u8]) {
    // Row 1 always holds data, because an odd height duplicates row 0 into it;
    // every 2x2 block is therefore four samples.
    let inv = 0.25f32;
    let mut cx = 0;
    while cx + 4 <= cw && 2 * cx + 8 <= cb0.len() {
        // SAFETY: four chroma columns are in range of every scratch row.
        unsafe {
            let sum = |a: &[f32], b: &[f32]| -> float32x4_t {
                let lo = vpaddq_f32(vld1q_f32(a.as_ptr().add(2 * cx)), vld1q_f32(a.as_ptr().add(2 * cx + 4)));
                let hi = vpaddq_f32(vld1q_f32(b.as_ptr().add(2 * cx)), vld1q_f32(b.as_ptr().add(2 * cx + 4)));
                vmulq_n_f32(vaddq_f32(lo, hi), inv)
            };
            let uq = round_u8(sum(cb0, cb1));
            let vq = round_u8(sum(cr0, cr1));
            vst1_lane_u32::<0>(u.as_mut_ptr().add(cx) as *mut u32, vreinterpret_u32_u8(uq));
            vst1_lane_u32::<0>(v.as_mut_ptr().add(cx) as *mut u32, vreinterpret_u32_u8(vq));
        }
        cx += 4;
    }
    while cx < cw {
        // Odd widths: the last chroma column owns a single luma column, so the
        // average is over two samples rather than four — the same rule the
        // scalar pass applies when `sx >= w`.
        let paired = 2 * cx + 1 < cb0.len();
        let inv = if paired { inv } else { 0.5f32 };
        let s = |r: &[f32]| if paired { r[2 * cx] + r[2 * cx + 1] } else { r[2 * cx] };
        let uu = (s(cb0) + s(cb1)) * inv;
        let vv = (s(cr0) + s(cr1)) * inv;
        u[cx] = q8(uu);
        v[cx] = q8(vv);
        cx += 1;
    }
}

#[inline]
fn q8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Round four f32 lanes half away from zero, clamp and narrow to bytes.
///
/// # Safety
/// Requires NEON, which aarch64 guarantees.
#[inline]
unsafe fn round_u8(v: float32x4_t) -> uint8x8_t {
    // Round half away from zero, matching `f32::round`: bias by ±0.5 then
    // truncate. `vcvtn` would be shorter but rounds ties to even.
    let zero = vdupq_n_f32(0.0);
    let bias = vbslq_f32(vcltq_f32(v, zero), vdupq_n_f32(-0.5), vdupq_n_f32(0.5));
    let biased = vaddq_f32(v, bias);
    let clamped = vminq_f32(vmaxq_f32(biased, zero), vdupq_n_f32(255.0));
    let ints: int32x4_t = vcvtq_s32_f32(clamped);
    vmovn_u16(vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(ints)), vdup_n_u16(0)))
}

/// Convert one row: luma bytes out, per-pixel chroma into scratch.
///
/// # Safety
/// `src` must have `w * 3` readable bytes; the output pointers must have `w`
/// writable elements.
unsafe fn row(
    c: &Coeffs,
    src: *const u8,
    w: usize,
    y_out: *mut u8,
    cb_out: *mut f32,
    cr_out: *mut f32,
) {
    let scale = vdupq_n_f32(c.scale);
    let offset = vdupq_n_f32(c.offset);
    let zero = vdupq_n_f32(0.0);
    let max = vdupq_n_f32(255.0);
    let kr = vdupq_n_f32(c.kr);
    let kg = vdupq_n_f32(c.kg);
    let kb = vdupq_n_f32(c.kb);
    let cb_scale = vdupq_n_f32(c.cb_scale);
    let cr_scale = vdupq_n_f32(c.cr_scale);
    let half = vdupq_n_f32(127.5);

    let mut x = 0;
    while x + LANES <= w {
        let v = vld3q_u8(src.add(x * 3));
        // Process the low and high halves as separate f32x4 pairs.
        let mut ys = [0u8; LANES];
        // 16 pixels arrive as three u8x16 once `vld3q_u8` splits the channels.
        // Each channel is widened in quarters so four f32x4 lanes cover them.
        let quarters = |c8: uint8x16_t, q: usize| -> float32x4_t {
            let lo = vmovl_u8(vget_low_u8(c8));
            let hi = vmovl_u8(vget_high_u8(c8));
            let u32x4 = match q {
                0 => vmovl_u16(vget_low_u16(lo)),
                1 => vmovl_u16(vget_high_u16(lo)),
                2 => vmovl_u16(vget_low_u16(hi)),
                _ => vmovl_u16(vget_high_u16(hi)),
            };
            let f = vcvtq_f32_u32(u32x4);
            vminq_f32(vmaxq_f32(vmlaq_f32(offset, f, scale), zero), max)
        };
        for q in 0..4usize {
            let r = quarters(v.0, q);
            let g = quarters(v.1, q);
            let b = quarters(v.2, q);
            let y = vmlaq_f32(vmlaq_f32(vmulq_f32(kr, r), kg, g), kb, b);
            let cbv = vmlaq_f32(half, vsubq_f32(b, y), cb_scale);
            let crv = vmlaq_f32(half, vsubq_f32(r, y), cr_scale);
            let yq = round_u8(y);
            vst1_lane_u32::<0>(
                ys.as_mut_ptr().add(q * 4) as *mut u32,
                vreinterpret_u32_u8(yq),
            );
            vst1q_f32(cb_out.add(x + q * 4), cbv);
            vst1q_f32(cr_out.add(x + q * 4), crv);
        }
        std::ptr::copy_nonoverlapping(ys.as_ptr(), y_out.add(x), LANES);
        x += LANES;
    }
    while x < w {
        let base = src.add(x * 3);
        let (y8, cb, cr) = pixel(c, *base, *base.add(1), *base.add(2));
        *y_out.add(x) = y8;
        *cb_out.add(x) = cb;
        *cr_out.add(x) = cr;
        x += 1;
    }
}
