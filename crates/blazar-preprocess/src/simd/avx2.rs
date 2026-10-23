//! x86_64 AVX2/SSSE3 kernel for interleaved 8-bit RGB.
//!
//! Four pixels per iteration: one 16-byte load covers the 12 bytes of four RGB
//! triplets, and three `pshufb` masks deinterleave them. The float maths then
//! runs 256 bits wide. Sixteen pixels per iteration was tried first and needed
//! three overlapping loads plus blended shuffle masks for the same throughput,
//! because the arithmetic was already the wider part.

use std::arch::x86_64::*;

use rayon::prelude::*;

use super::{pixel, Coeffs};

/// Pixels per vector iteration.
const LANES: usize = 4;

pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("fma")
}

/// # Safety
/// The caller must have checked [`available`].
pub(crate) unsafe fn convert(
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
        let mut cb0 = vec![0.0f32; w];
        let mut cr0 = vec![0.0f32; w];
        let mut cb1 = vec![0.0f32; w];
        let mut cr1 = vec![0.0f32; w];
        let rows = y_rows.len() / w;
        for (dy, y_row) in y_rows.chunks_exact_mut(w).enumerate() {
            let sy = cy * 2 + dy;
            debug_assert!(sy < h);
            let src = &bytes[sy * w * 3..(sy + 1) * w * 3];
            let (cb, cr) = if dy == 0 {
                (&mut cb0, &mut cr0)
            } else {
                (&mut cb1, &mut cr1)
            };
            // SAFETY: `src` holds `w * 3` bytes and the outputs `w` elements.
            unsafe {
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

fn combine(cb0: &[f32], cr0: &[f32], cb1: &[f32], cr1: &[f32], cw: usize, u: &mut [u8], v: &mut [u8]) {
    // Row 1 always holds data because an odd height duplicates row 0 into it,
    // so every 2x2 block is four samples and the divisor is always four.
    let mut cx = 0;
    // SAFETY: the loop bounds keep every load inside the scratch rows.
    unsafe {
        let quarter = _mm_set1_ps(0.25);
        // Adjacent pairs: lanes 0 and 1 hold x0+x1 and x2+x3.
        let pair = |a: __m128| -> __m128 {
            let even = _mm_shuffle_ps(a, a, 0b10_00_10_00);
            let odd = _mm_shuffle_ps(a, a, 0b11_01_11_01);
            _mm_add_ps(even, odd)
        };
        // Eight f32 from each of the two rows -> four 2x2 averages.
        let quad = |r0: &[f32], r1: &[f32], cx: usize| -> __m128 {
            let lo0 = pair(_mm_loadu_ps(r0.as_ptr().add(2 * cx)));
            let hi0 = pair(_mm_loadu_ps(r0.as_ptr().add(2 * cx + 4)));
            let lo1 = pair(_mm_loadu_ps(r1.as_ptr().add(2 * cx)));
            let hi1 = pair(_mm_loadu_ps(r1.as_ptr().add(2 * cx + 4)));
            let top = _mm_movelh_ps(lo0, hi0);
            let bottom = _mm_movelh_ps(lo1, hi1);
            _mm_mul_ps(_mm_add_ps(top, bottom), quarter)
        };
        while cx + 4 <= cw && 2 * cx + 8 <= cb0.len() {
            // round_u8 yields four i32 lanes: narrow before storing or the
            // 32-bit store would write one value as four bytes.
            store4(u.as_mut_ptr().add(cx), round_u8(quad(cb0, cb1, cx)));
            store4(v.as_mut_ptr().add(cx), round_u8(quad(cr0, cr1, cx)));
            cx += 4;
        }
    }
    while cx < cw {
        let paired = 2 * cx + 1 < cb0.len();
        let inv = if paired { 0.25f32 } else { 0.5f32 };
        let s = |r: &[f32]| if paired { r[2 * cx] + r[2 * cx + 1] } else { r[2 * cx] };
        u[cx] = q8((s(cb0) + s(cb1)) * inv);
        v[cx] = q8((s(cr0) + s(cr1)) * inv);
        cx += 1;
    }
}

/// Narrow four i32 lanes to four bytes and store them.
///
/// # Safety
/// `out` must have four writable bytes.
#[inline]
unsafe fn store4(out: *mut u8, v: __m128i) {
    let packed = _mm_packus_epi32(v, v);
    let packed = _mm_packus_epi16(packed, packed);
    _mm_storeu_si32(out.cast(), packed);
}

#[inline]
fn q8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// # Safety
/// Requires AVX2/SSE4.1.
#[inline]
unsafe fn round_u8(v: __m128) -> __m128i {
    // Round half away from zero like `f32::round`: bias then truncate.
    let zero = _mm_setzero_ps();
    let bias = _mm_blendv_ps(_mm_set1_ps(0.5), _mm_set1_ps(-0.5), _mm_cmplt_ps(v, zero));
    let biased = _mm_add_ps(v, bias);
    let clamped = _mm_min_ps(_mm_max_ps(biased, zero), _mm_set1_ps(255.0));
    _mm_cvttps_epi32(clamped)
}

/// # Safety
/// `src` must hold `w * 3` bytes; outputs must hold `w` elements each.
unsafe fn row(c: &Coeffs, src: *const u8, w: usize, y_out: *mut u8, cb_out: *mut f32, cr_out: *mut f32) {
    // Byte offsets of the first four pixels' R, G and B inside one 16-byte load.
    // A load spans five and a third pixels; the masks drop the surplus bytes.
    let mask_r = _mm_setr_epi8(0, 3, 6, 9, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128);
    let mask_g = _mm_setr_epi8(1, 4, 7, 10, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128);
    let mask_b = _mm_setr_epi8(2, 5, 8, 11, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128);

    let scale = _mm_set1_ps(c.scale);
    let offset = _mm_set1_ps(c.offset);
    let lo = _mm_setzero_ps();
    let hi = _mm_set1_ps(255.0);
    let kr = _mm_set1_ps(c.kr);
    let kg = _mm_set1_ps(c.kg);
    let kb = _mm_set1_ps(c.kb);
    let cb_scale = _mm_set1_ps(c.cb_scale);
    let cr_scale = _mm_set1_ps(c.cr_scale);
    let half = _mm_set1_ps(127.5);

    let mut x = 0;
    while x + LANES <= w {
        let v = _mm_loadu_si128(src.add(x * 3).cast());
        let widen = |m: __m128i| {
            let ints = _mm_cvtepu8_epi32(_mm_shuffle_epi8(v, m));
            _mm_min_ps(_mm_max_ps(_mm_fmadd_ps(_mm_cvtepi32_ps(ints), scale, offset), lo), hi)
        };
        let (r, g, b) = (widen(mask_r), widen(mask_g), widen(mask_b));
        let y = _mm_fmadd_ps(kr, r, _mm_fmadd_ps(kg, g, _mm_mul_ps(kb, b)));
        let cbv = _mm_fmadd_ps(_mm_sub_ps(b, y), cb_scale, half);
        let crv = _mm_fmadd_ps(_mm_sub_ps(r, y), cr_scale, half);
        let packed = _mm_packus_epi32(round_u8(y), round_u8(y));
        let packed = _mm_packus_epi16(packed, packed);
        let mut bytes4 = [0u8; 8];
        _mm_storel_epi64(bytes4.as_mut_ptr().cast(), packed);
        std::ptr::copy_nonoverlapping(bytes4.as_ptr(), y_out.add(x), LANES);
        _mm_storeu_ps(cb_out.add(x), cbv);
        _mm_storeu_ps(cr_out.add(x), crv);
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
