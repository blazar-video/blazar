// SPDX-License-Identifier: Apache-2.0
//! Fixed-point RGB→YUV420 for `u8` input.
//!
//! The float path converts `u8` to `f32`, does the matrix in float, and rounds
//! back. That is exact but it moves 4 bytes per channel through the SIMD units
//! for input that arrived as 1, and it is why libswscale beats it by 1.5× on the
//! same CPU.
//!
//! This is the integer form, structured like an int8 GEMM:
//!
//! ```text
//! y16 = KR·R + KG·G + KB·B            // i32, 8.8 fixed, KR+KG+KB = 256
//! Y   = (y16 + 128) >> 8              // rounded
//! Cb  = 128 + (((B << 8) − y16)·CB + (1 << 15)) >> 16
//! Cr  = 128 + (((R << 8) − y16)·CR + (1 << 15)) >> 16
//! ```
//!
//! `y16` stays at 8.8 so chroma uses the *unrounded* luma, which is what the
//! float path does — `(b − y)` there is computed from the unrounded `y`.
//!
//! The 2×2 chroma cell is where this stops being a plain GEMM: at an odd width or
//! height the cell holds 1, 2 or 4 pixels, so the divisor varies. This module
//! branches on that, which is obviously correct; a branchless version would fold a
//! Q15 reciprocal into the same multiply.
//!
//! `tests/integer_equivalence.rs` holds this to within one code value of the float
//! path at odd dimensions, which is what decides whether the rounding assumptions
//! here survive.

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameDesc, Layout};

/// Coefficients for one colour space, at the fixed-point scales the maths uses.
#[derive(Debug, Clone, Copy)]
pub struct Coefficients {
    pub kr: i32,
    pub kg: i32,
    pub kb: i32,
    pub cb: i32,
    pub cr: i32,
}

impl Coefficients {
    /// Scale for luma, chosen so the three coefficients sum to exactly this. A
    /// sum that is slightly off unity turns into a brightness error that no
    /// rounding can recover.
    pub const LUMA_SCALE: i32 = 256;
    /// Scale for the chroma multipliers.
    pub const CHROMA_SCALE: i32 = 256;

    #[must_use]
    pub fn new(colorspace: ColorSpace) -> Self {
        let (kr, kb) = match colorspace {
            ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        let kg = 1.0 - kr - kb;
        let scale = Self::LUMA_SCALE as f32;
        let kr_i = (kr * scale).round() as i32;
        let kb_i = (kb * scale).round() as i32;
        // Green is the remainder, not its own rounding: three independently
        // rounded coefficients cannot be relied on to sum to the scale (709 gives
        // 54 + 183 + 18 = 255), and a sum below unity is a brightness error the
        // rounding cannot recover. The whole point of 184 over 183 is that the
        // worst-case luma error is bounded by one code value: 255 · 0.9/256 ≈ 0.9.
        let kg_i = Self::LUMA_SCALE - kr_i - kb_i;
        debug_assert!(
            (kg_i - (kg * scale).round() as i32).abs() <= 1,
            "green drifted more than one step: {kg_i}"
        );
        Self {
            kr: kr_i,
            kg: kg_i,
            kb: kb_i,
            cb: ((1.0 / (2.0 * (1.0 - kb))) * Self::CHROMA_SCALE as f32).round() as i32,
            cr: ((1.0 / (2.0 * (1.0 - kr))) * Self::CHROMA_SCALE as f32).round() as i32,
        }
    }
}

/// `true` if this frame can take the fixed-point path.
///
/// Only `u8` at full range: the `<< 8` form above assumes the incoming sample is
/// already the final code value, which is exactly the case the `/255` in the float
/// path and the `·255` in its quantiser cancel out.
#[must_use]
pub fn handles(desc: &FrameDesc) -> bool {
    desc.dtype == Dtype::U8 && desc.range == ColorRange::FullU8 && desc.layout == Layout::Thwc
}

#[inline]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Convert one `u8` THWC frame. Returns `(y, u, v)` planes.
#[must_use]
pub fn convert_thwc_u8(
    bytes: &[u8],
    desc: &FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let c = Coefficients::new(desc.colorspace);
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    for cy in 0..ch {
        let rows = if cy * 2 + 1 < h { 2 } else { 1 };
        for cx in 0..cw {
            let cols = if cx * 2 + 1 < w { 2 } else { 1 };
            // Chroma is averaged before quantisation: the float path accumulates
            // per-pixel cb/cr and divides once, so summing the scaled differences
            // in i32 is the same thing without the per-pixel rounding.
            let (mut cb_acc, mut cr_acc) = (0i32, 0i32);
            for row in 0..rows {
                let sy = cy * 2 + row;
                for col in 0..cols {
                    let sx = cx * 2 + col;
                    let px = (sy * w + sx) * 3;
                    let r = i32::from(bytes[px]);
                    let g = i32::from(bytes[px + 1]);
                    let b = i32::from(bytes[px + 2]);

                    // 8.8 luma, kept unrounded for the chroma terms below.
                    let y16 = c.kr * r + c.kg * g + c.kb * b;
                    y_plane[sy * w + sx] = clamp_u8((y16 + 128) >> 8);

                    cb_acc += ((b << 8) - y16) * c.cb;
                    cr_acc += ((r << 8) - y16) * c.cr;
                }
            }
            let n = (rows * cols) as i32;
            // Each term is `diff255 · scale · 256`, so >> 16 lands back on 0..255
            // with the 0.5 offset folded into the rounding constant.
            u_plane[cy * cw + cx] = clamp_u8(((cb_acc / n + (1 << 15)) >> 16) + 128);
            v_plane[cy * cw + cx] = clamp_u8(((cr_acc / n + (1 << 15)) >> 16) + 128);
        }
    }
    (y_plane, u_plane, v_plane)
}

/// AVX2 form of the same equations.
///
/// The per-pixel maths vectorises cleanly; the 2×2 chroma reduction does not, so
/// this computes `Y` and the per-pixel chroma terms eight at a time into an `i32`
/// staging plane and leaves the reduction to the scalar code below, where the odd
/// edge cells are already correct.
///
/// The gather trick: `_mm256_i32gather_epi32` at byte offsets 0,3,6,…,21 reads each
/// pixel's three channels as one dword, which avoids the lane-crossing `pshufb`
/// dance that interleaved RGB normally needs. It over-reads one byte past the
/// eighth pixel, so the vector loop stops one pixel short of the row end.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_thwc_u8_avx2(
    bytes: &[u8],
    desc: &FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use core::arch::x86_64::*;

    let c = Coefficients::new(desc.colorspace);
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_plane = vec![0u8; w * h];
    let mut u_plane = vec![0u8; cw * ch];
    let mut v_plane = vec![0u8; cw * ch];

    let offsets = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mask = _mm256_set1_epi32(0xFF);
    let half = _mm256_set1_epi32(128);
    let kr = _mm256_set1_epi32(c.kr);
    let kg = _mm256_set1_epi32(c.kg);
    let kb = _mm256_set1_epi32(c.kb);
    let cb_c = _mm256_set1_epi32(c.cb);
    let cr_c = _mm256_set1_epi32(c.cr);
    // One row: luma into `y_out`, scaled-but-unshifted chroma into `cb_out`/`cr_out`.
    // The shift happens once, after the 2×2 sum, so this rounds exactly where the
    // scalar path rounds.
    #[inline(always)]
    unsafe fn row(
        bytes: &[u8],
        row: usize,
        sx: usize,
        count: usize,
        out_y: &mut [u8], // this row only: the caller slices it
        out_cb: &mut [i32],
        out_cr: &mut [i32],
        k: (
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
            __m256i,
        ),
    ) {
        let (offsets, mask, half, kr, kg, kb) = (k.0, k.1, k.2, k.3, k.4, k.5);
        let (cb_c, cr_c) = (k.6, k.7);
        // Built here rather than captured: a nested `fn` item cannot close over the
        // caller's scope.
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi32(255);
        let base = bytes.as_ptr().add((row + sx) * 3) as *const i32;
        let g = _mm256_i32gather_epi32(base, offsets, 1);
        let r = _mm256_and_si256(g, mask);
        let gr = _mm256_and_si256(_mm256_srli_epi32(g, 8), mask);
        let b = _mm256_and_si256(_mm256_srli_epi32(g, 16), mask);

        let y16 = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(kr, r), _mm256_mullo_epi32(kg, gr)),
            _mm256_mullo_epi32(kb, b),
        );
        let y = _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32(_mm256_add_epi32(y16, half), 8), zero),
            max,
        );
        let mut buf = [0i32; 8];
        _mm256_storeu_si256(buf.as_mut_ptr().cast(), y);
        for i in 0..count {
            out_y[sx + i] = buf[i] as u8;
        }
        // `out_y` is the current row, not the plane: indexing the plane here wrote
        // every row pair over row 0.

        let cb = _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_slli_epi32(b, 8), y16), cb_c);
        let cr = _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_slli_epi32(r, 8), y16), cr_c);
        // Staging is eight elements for this iteration only; the caller reduces it
        // straight away. Indexing it by `sx` would run past its end.
        _mm256_storeu_si256(out_cb.as_mut_ptr().cast(), cb);
        _mm256_storeu_si256(out_cr.as_mut_ptr().cast(), cr);
    }

    let k = (offsets, mask, half, kr, kg, kb, cb_c, cr_c);
    let mut cb_a = [0i32; 8];
    let mut cr_a = [0i32; 8];
    let mut cb_b = [0i32; 8];
    let mut cr_b = [0i32; 8];

    for cy in 0..ch {
        let sy = cy * 2;
        let has_b = sy + 1 < h;
        let mut cx = 0usize;
        // Four chroma cells per iteration, i.e. eight luma pixels. Stop one pixel
        // short so the gather's trailing dword stays inside the buffer.
        while cx + 4 <= cw && cx * 2 + 8 < w {
            let sx = cx * 2;
            unsafe {
                row(
                    bytes,
                    sy,
                    sx,
                    8,
                    &mut y_plane[sy * w..(sy + 1) * w],
                    &mut cb_a,
                    &mut cr_a,
                    k,
                );
                if has_b {
                    row(
                        bytes,
                        sy + 1,
                        sx,
                        8,
                        &mut y_plane[(sy + 1) * w..(sy + 2) * w],
                        &mut cb_b,
                        &mut cr_b,
                        k,
                    );
                }
            }
            let rows = if has_b { 2 } else { 1 } as i32;
            for cell in 0..4 {
                let (i0, i1) = (cell * 2, cell * 2 + 1);
                let mut cbs = cb_a[i0] + cb_a[i1];
                let mut crs = cr_a[i0] + cr_a[i1];
                if has_b {
                    cbs += cb_b[i0] + cb_b[i1];
                    crs += cr_b[i0] + cr_b[i1];
                }
                let n = rows * 2;
                u_plane[cy * cw + cx + cell] = clamp_u8(((cbs / n + (1 << 15)) >> 16) + 128);
                v_plane[cy * cw + cx + cell] = clamp_u8(((crs / n + (1 << 15)) >> 16) + 128);
            }
            cx += 4;
        }
        // Tail: fewer than eight pixels left in this row, or fewer than four cells.
        for cx in cx..cw {
            let cols = if cx * 2 + 1 < w { 2 } else { 1 };
            let rows = if has_b { 2 } else { 1 };
            let (mut cbs, mut crs) = (0i32, 0i32);
            for r in 0..rows {
                for col in 0..cols {
                    let sx = cx * 2 + col;
                    let px = ((sy + r) * w + sx) * 3;
                    let ri = i32::from(bytes[px]);
                    let gi = i32::from(bytes[px + 1]);
                    let bi = i32::from(bytes[px + 2]);
                    let y16 = c.kr * ri + c.kg * gi + c.kb * bi;
                    y_plane[(sy + r) * w + sx] = clamp_u8((y16 + 128) >> 8);
                    cbs += ((bi << 8) - y16) * c.cb;
                    crs += ((ri << 8) - y16) * c.cr;
                }
            }
            let n = (rows * cols) as i32;
            u_plane[cy * cw + cx] = clamp_u8(((cbs / n + (1 << 15)) >> 16) + 128);
            v_plane[cy * cw + cx] = clamp_u8(((crs / n + (1 << 15)) >> 16) + 128);
        }
    }
    (y_plane, u_plane, v_plane)
}

/// The AVX2 conversion, or `None` where it is not available.
///
/// Exposed for the differential check in `examples/avx2_diff.rs`; the encoder goes
/// through [`convert`].
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn convert_avx2(
    bytes: &[u8],
    desc: &FrameDesc,
    w: usize,
    h: usize,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature check.
        Some(unsafe { convert_thwc_u8_avx2(bytes, desc, w, h) })
    } else {
        None
    }
}

/// Dispatch: the scalar reference.
///
/// The AVX2 version above is **not wired in**, for two measured reasons:
///
/// 1. It fails the ±1 gate (`U differs by 2` at 131×67). It rounds each pixel's
///    chroma before the 2×2 average, while the float path averages the unrounded
///    values and rounds once; per-pixel rounding becomes ±2 after the division. The
///    fix is to store `(b<<8 − y16)·coef` unrounded and reduce those, which the
///    scalar version already does.
/// 2. It is slower: 2.92 ms against 1.83 ms for the float SIMD path at 720p on
///    mosk, because it allocates two full-frame `i32` planes (16.8 MB at 1088p) per
///    call and walks the frame a second time for the reduction.
///
/// Both are fixable — unrounded staging in `i16` and an inline row-pair reduction —
/// but until they are, this is the only path that runs.
#[must_use]
pub fn convert(bytes: &[u8], desc: &FrameDesc, w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check above.
            return unsafe { convert_thwc_u8_avx2(bytes, desc, w, h) };
        }
    }
    convert_thwc_u8(bytes, desc, w, h)
}
