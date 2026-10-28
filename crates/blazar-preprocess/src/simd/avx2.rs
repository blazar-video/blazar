// SPDX-License-Identifier: Apache-2.0
//! x86_64 AVX2 kernels for RGB -> YUV420.
//!
//! Six source shapes, three loading strategies:
//!
//! * interleaved u8 — `pshufb` masks pull four pixels out of one 16-byte load,
//! * planar u8 — eight bytes straight into eight f32 lanes,
//! * planar f32 — two 256-bit loads per plane,
//! * interleaved f32 — `vgatherdps`, because a 12-float stride has no shuffle
//!   pattern that beats a gather on this microarchitecture,
//! * bf16 (either layout) — widened into an f32 scratch row, then the f32 kernel.
//!
//! The arithmetic core is 256-bit; the four-pixel u8 path uses a 128-bit
//! instance of it.

use std::arch::x86_64::*;

use rayon::prelude::*;

use super::{pixel_unit, q8, Coeffs, Kind};

pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("sse4.1")
}

/// Everything a row driver needs that is constant across rows.
struct Ctx<'a> {
    c: &'a Coeffs,
    k: K,
    /// Bytes per plane: `w * h * component_bytes`.
    plane_bytes: usize,
    w: usize,
}

struct K {
    scale: __m256,
    offset: __m256,
    zero: __m256,
    max: __m256,
    kr: __m256,
    kg: __m256,
    kb: __m256,
    cb_scale: __m256,
    cr_scale: __m256,
    half: __m256,
    /// 128-bit copies for the four-pixel path.
    scale4: __m128,
    offset4: __m128,
    zero4: __m128,
    max4: __m128,
    kr4: __m128,
    kg4: __m128,
    kb4: __m128,
    cb_scale4: __m128,
    cr_scale4: __m128,
    half4: __m128,
}

impl K {
    /// # Safety
    /// Requires AVX.
    unsafe fn new(c: &Coeffs) -> Self {
        Self {
            scale: _mm256_set1_ps(c.scale),
            offset: _mm256_set1_ps(c.offset),
            zero: _mm256_setzero_ps(),
            max: _mm256_set1_ps(255.0),
            kr: _mm256_set1_ps(c.kr),
            kg: _mm256_set1_ps(c.kg),
            kb: _mm256_set1_ps(c.kb),
            cb_scale: _mm256_set1_ps(c.cb_scale),
            cr_scale: _mm256_set1_ps(c.cr_scale),
            half: _mm256_set1_ps(127.5),
            scale4: _mm_set1_ps(c.scale),
            offset4: _mm_set1_ps(c.offset),
            zero4: _mm_setzero_ps(),
            max4: _mm_set1_ps(255.0),
            kr4: _mm_set1_ps(c.kr),
            kg4: _mm_set1_ps(c.kg),
            kb4: _mm_set1_ps(c.kb),
            cb_scale4: _mm_set1_ps(c.cb_scale),
            cr_scale4: _mm_set1_ps(c.cr_scale),
            half4: _mm_set1_ps(127.5),
        }
    }
}

/// # Safety
/// The caller must have checked [`available`].
/// Row-pair scratch, kept per thread so a frame costs no allocations.
///
/// Rayon runs row pairs on every worker thread, and four fresh vectors per pair
/// meant 2,160 allocations per 1080p frame; on Linux the allocator contention
/// across 28 threads cost more than the kernels saved.
#[derive(Default)]
struct Scratch {
    cb0: Vec<f32>,
    cr0: Vec<f32>,
    cb1: Vec<f32>,
    cr1: Vec<f32>,
    /// Widened bf16 rows; empty for the other layouts.
    wide: Vec<f32>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

impl Scratch {
    fn ensure(&mut self, w: usize, wide: usize) {
        for v in [&mut self.cb0, &mut self.cr0, &mut self.cb1, &mut self.cr1] {
            if v.len() < w {
                v.resize(w, 0.0);
            }
            v.truncate(w);
        }
        if self.wide.len() < wide {
            self.wide.resize(wide, 0.0);
        }
        self.wide.truncate(wide);
    }
}

/// The attribute is what lets LLVM compile and schedule this function for AVX2.
/// Without it the intrinsics still emit their instructions, but everything
/// around them is generated for baseline x86-64, which measured slower than the
/// scalar pass.
///
/// # Safety
/// The caller must have checked [`available`].
#[allow(clippy::too_many_arguments)] // the kernel ABI, not a design choice
#[target_feature(enable = "avx2,fma,ssse3,sse4.1")]
pub(crate) unsafe fn convert(
    kind: Kind,
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
) -> bool {
    let component_bytes = match kind {
        Kind::ThwcU8 | Kind::BcthwU8 => 1,
        Kind::ThwcBf16 | Kind::BcthwBf16 => 2,
        Kind::ThwcF32 | Kind::BcthwF32 => 4,
    };
    let c = match kind {
        Kind::ThwcU8 | Kind::BcthwU8 => Coeffs::for_u8(desc),
        _ => Coeffs::for_unit(desc),
    };
    let cw = w.div_ceil(2);
    let pixels = w * h;
    // Carried as an address so the parallel closure stays `Send`.
    let base = bytes.as_ptr() as usize;
    let ctx = Ctx {
        c: &c,
        // SAFETY: broadcast constants only.
        k: K::new(&c),
        plane_bytes: pixels * component_bytes,
        w,
    };
    let scratch_len = match kind {
        Kind::ThwcBf16 | Kind::BcthwBf16 => 3 * w,
        _ => 0,
    };

    let row_pair = |cy: usize, y_rows: &mut [u8], u_row: &mut [u8], v_row: &mut [u8]| {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure(w, scratch_len);
            let Scratch {
                cb0,
                cr0,
                cb1,
                cr1,
                wide,
            } = &mut *scratch;
            let rows = y_rows.len() / w;
            for (dy, y_row) in y_rows.chunks_exact_mut(w).enumerate() {
                let sy = cy * 2 + dy;
                debug_assert!(sy < h);
                let (cb, cr) = if dy == 0 {
                    (&mut *cb0, &mut *cr0)
                } else {
                    (&mut *cb1, &mut *cr1)
                };
                // SAFETY: the driver stays inside the row it is given; the scratch
                // rows hold `w` floats each.
                row(
                    kind,
                    &ctx,
                    base,
                    sy,
                    wide,
                    y_row.as_mut_ptr(),
                    cb.as_mut_ptr(),
                    cr.as_mut_ptr(),
                );
            }
            if rows == 1 {
                cb1.copy_from_slice(cb0);
                cr1.copy_from_slice(cr0);
            }
            combine(cb0, cr0, cb1, cr1, cw, u_row, v_row);
        });
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
    true
}

/// Dispatch one source row to the loader for its shape.
///
/// # Safety
/// `base` must cover the whole frame; the outputs hold `w` elements.
#[allow(clippy::too_many_arguments)] // the row ABI, not a design choice
unsafe fn row(
    kind: Kind,
    ctx: &Ctx<'_>,
    base_addr: usize,
    sy: usize,
    scratch: &mut [f32],
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let base = base_addr as *const u8;
    match kind {
        Kind::ThwcU8 => row_thwc_u8(ctx, base, sy, y_out, cb, cr),
        Kind::BcthwU8 => row_bcthw_u8(ctx, base, sy, y_out, cb, cr),
        Kind::ThwcF32 => row_thwc_f32(ctx, base, sy, y_out, cb, cr),
        Kind::BcthwF32 => {
            let stride = ctx.plane_bytes / 4;
            row_bcthw_f32(ctx, base, sy * ctx.w, stride, y_out, cb, cr)
        }
        Kind::ThwcBf16 => {
            widen_bf16(base, sy, ctx.w, true, ctx.plane_bytes, scratch);
            row_thwc_f32(ctx, scratch.as_ptr() as *const u8, 0, y_out, cb, cr)
        }
        Kind::BcthwBf16 => {
            widen_bf16(base, sy, ctx.w, false, ctx.plane_bytes, scratch);
            row_bcthw_f32(ctx, scratch.as_ptr() as *const u8, 0, ctx.w, y_out, cb, cr)
        }
    }
}

/// Widen one row of bf16 samples into an f32 scratch buffer.
///
/// An interleaved row is one run of `3 * w` samples; a planar row is three runs
/// of `w`, written back to back, so the scratch layout is the same either way.
///
/// # Safety
/// `base` must cover the frame; `scratch` must hold `3 * w` floats.
unsafe fn widen_bf16(
    base: *const u8,
    sy: usize,
    w: usize,
    interleaved: bool,
    plane_bytes: usize,
    scratch: &mut [f32],
) {
    let src = base as *const u16;
    let runs: [(usize, usize); 3] = if interleaved {
        [(sy * w * 3, 3 * w), (0, 0), (0, 0)]
    } else {
        [
            (sy * w, w),
            (plane_bytes / 2 + sy * w, w),
            (plane_bytes + sy * w, w),
        ]
    };
    let mut written = 0usize;
    for &(start, count) in runs.iter() {
        if count == 0 {
            continue;
        }
        let mut i = 0usize;
        while i + 8 <= count {
            // Eight u16 -> eight f32: widen to i32 then reinterpret as the high
            // half of each float.
            let bits = _mm_loadu_si128(src.add(start + i).cast());
            let wide = _mm256_cvtepu16_epi32(bits);
            let float_bits = _mm256_slli_epi32(wide, 16);
            _mm256_storeu_ps(
                scratch.as_mut_ptr().add(written + i),
                _mm256_castsi256_ps(float_bits),
            );
            i += 8;
        }
        while i < count {
            let bits = *src.add(start + i);
            scratch[written + i] = f32::from_bits((bits as u32) << 16);
            i += 1;
        }
        written += count;
    }
    debug_assert_eq!(written, 3 * w);
}

/// Shared maths, eight pixels at a time.
///
/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn core8(k: &K, r: __m256, g: __m256, b: __m256) -> (__m256i, __m256, __m256) {
    let r = _mm256_min_ps(
        _mm256_max_ps(_mm256_fmadd_ps(r, k.scale, k.offset), k.zero),
        k.max,
    );
    let g = _mm256_min_ps(
        _mm256_max_ps(_mm256_fmadd_ps(g, k.scale, k.offset), k.zero),
        k.max,
    );
    let b = _mm256_min_ps(
        _mm256_max_ps(_mm256_fmadd_ps(b, k.scale, k.offset), k.zero),
        k.max,
    );
    let y = _mm256_fmadd_ps(k.kr, r, _mm256_fmadd_ps(k.kg, g, _mm256_mul_ps(k.kb, b)));
    let cb = _mm256_fmadd_ps(_mm256_sub_ps(b, y), k.cb_scale, k.half);
    let cr = _mm256_fmadd_ps(_mm256_sub_ps(r, y), k.cr_scale, k.half);
    (round_u8_256(y), cb, cr)
}

/// Shared maths, four pixels at a time.
///
/// # Safety
/// Requires SSE4.1.
#[inline(always)]
unsafe fn core4(k: &K, r: __m128, g: __m128, b: __m128) -> (__m128i, __m128, __m128) {
    let r = _mm_min_ps(
        _mm_max_ps(_mm_fmadd_ps(r, k.scale4, k.offset4), k.zero4),
        k.max4,
    );
    let g = _mm_min_ps(
        _mm_max_ps(_mm_fmadd_ps(g, k.scale4, k.offset4), k.zero4),
        k.max4,
    );
    let b = _mm_min_ps(
        _mm_max_ps(_mm_fmadd_ps(b, k.scale4, k.offset4), k.zero4),
        k.max4,
    );
    let y = _mm_fmadd_ps(k.kr4, r, _mm_fmadd_ps(k.kg4, g, _mm_mul_ps(k.kb4, b)));
    let cb = _mm_fmadd_ps(_mm_sub_ps(b, y), k.cb_scale4, k.half4);
    let cr = _mm_fmadd_ps(_mm_sub_ps(r, y), k.cr_scale4, k.half4);
    (round_u8_128(y), cb, cr)
}

/// Narrow four i32 lanes to four bytes and store them.
///
/// # Safety
/// `out` must have four writable bytes.
#[inline(always)]
unsafe fn store4(out: *mut u8, v: __m128i) {
    let packed = _mm_packus_epi32(v, v);
    let packed = _mm_packus_epi16(packed, packed);
    _mm_storeu_si32(out.cast(), packed);
}

/// Narrow eight i32 lanes to eight bytes and store them.
///
/// # Safety
/// `out` must have eight writable bytes.
#[inline(always)]
unsafe fn store8(out: *mut u8, v: __m256i) {
    // Packing inside each 128-bit half then joining keeps the lane order, which
    // a single `packus` across the whole register would not.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let packed = _mm_packus_epi32(lo, hi);
    let packed = _mm_packus_epi16(packed, packed);
    _mm_storel_epi64(out.cast(), packed);
}

/// Interleaved u8: four pixels per iteration.
///
/// # Safety
/// `base` must cover the frame; the outputs hold `w` elements.
#[inline(always)]
unsafe fn row_thwc_u8(
    ctx: &Ctx<'_>,
    base: *const u8,
    sy: usize,
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let w = ctx.w;
    // Byte offsets of four pixels' R, G and B inside one 16-byte load.
    let mask_r = _mm_setr_epi8(
        0, 3, 6, 9, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let mask_g = _mm_setr_epi8(
        1, 4, 7, 10, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let mask_b = _mm_setr_epi8(
        2, 5, 8, 11, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let p = base.add(sy * w * 3);
    let mut x = 0;
    while x + 4 <= w {
        let v = _mm_loadu_si128(p.add(x * 3).cast());
        let widen = |m: __m128i| _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi8(v, m)));
        let (yq, cbv, crv) = core4(&ctx.k, widen(mask_r), widen(mask_g), widen(mask_b));
        store4(y_out.add(x), yq);
        _mm_storeu_ps(cb.add(x), cbv);
        _mm_storeu_ps(cr.add(x), crv);
        x += 4;
    }
    while x < w {
        let i = x * 3;
        let (y8, cbv, crv) = pixel_unit(
            ctx.c,
            *p.add(i) as f32,
            *p.add(i + 1) as f32,
            *p.add(i + 2) as f32,
        );
        *y_out.add(x) = y8;
        *cb.add(x) = cbv;
        *cr.add(x) = crv;
        x += 1;
    }
}

/// Planar u8: eight pixels per iteration.
///
/// # Safety
/// Each plane must cover the frame; the outputs hold `w` elements.
#[inline(always)]
unsafe fn row_bcthw_u8(
    ctx: &Ctx<'_>,
    base: *const u8,
    sy: usize,
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let w = ctx.w;
    let offset = sy * w;
    let (rp, gp, bp) = (
        base.add(offset),
        base.add(offset + ctx.plane_bytes),
        base.add(offset + 2 * ctx.plane_bytes),
    );
    let mut x = 0;
    while x + 8 <= w {
        let widen = |p: *const u8| {
            let bytes = _mm_loadl_epi64(p.add(x).cast());
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes))
        };
        let (yq, cbv, crv) = core8(&ctx.k, widen(rp), widen(gp), widen(bp));
        store8(y_out.add(x), yq);
        _mm256_storeu_ps(cb.add(x), cbv);
        _mm256_storeu_ps(cr.add(x), crv);
        x += 8;
    }
    while x < w {
        let (y8, cbv, crv) = pixel_unit(
            ctx.c,
            *rp.add(x) as f32,
            *gp.add(x) as f32,
            *bp.add(x) as f32,
        );
        *y_out.add(x) = y8;
        *cb.add(x) = cbv;
        *cr.add(x) = crv;
        x += 1;
    }
}

/// Interleaved f32: eight pixels per iteration via `vgatherdps`.
///
/// A 12-float pixel stride has no shuffle pattern that beats a gather on this
/// microarchitecture: deinterleaving would cost three loads plus six permutes
/// per group, against three gathers.
///
/// # Safety
/// `base` must cover the frame; the outputs hold `w` elements.
#[inline(always)]
unsafe fn row_thwc_f32(
    ctx: &Ctx<'_>,
    base: *const u8,
    sy: usize,
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let w = ctx.w;
    let p = (base as *const f32).add(sy * w * 3);
    let index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    let mut x = 0;
    while x + 8 <= w {
        let at = p.add(x * 3);
        let r = _mm256_i32gather_ps(at, index, 4);
        let g = _mm256_i32gather_ps(at.add(1), index, 4);
        let b = _mm256_i32gather_ps(at.add(2), index, 4);
        let (yq, cbv, crv) = core8(&ctx.k, r, g, b);
        store8(y_out.add(x), yq);
        _mm256_storeu_ps(cb.add(x), cbv);
        _mm256_storeu_ps(cr.add(x), crv);
        x += 8;
    }
    while x < w {
        let i = x * 3;
        let (y8, cbv, crv) = pixel_unit(ctx.c, *p.add(i), *p.add(i + 1), *p.add(i + 2));
        *y_out.add(x) = y8;
        *cb.add(x) = cbv;
        *cr.add(x) = crv;
        x += 1;
    }
}

/// Planar f32: eight pixels per iteration.
///
/// The same driver serves bf16 frames, whose scratch row has this layout.
///
/// # Safety
/// Each plane must cover the frame; the outputs hold `w` elements.
#[inline(always)]
unsafe fn row_bcthw_f32(
    ctx: &Ctx<'_>,
    base: *const u8,
    offset: usize,
    stride: usize,
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let w = ctx.w;
    let (rp, gp, bp) = (
        (base as *const f32).add(offset),
        (base as *const f32).add(offset + stride),
        (base as *const f32).add(offset + 2 * stride),
    );
    let mut x = 0;
    while x + 8 <= w {
        let (r, g, b) = (
            _mm256_loadu_ps(rp.add(x)),
            _mm256_loadu_ps(gp.add(x)),
            _mm256_loadu_ps(bp.add(x)),
        );
        let (yq, cbv, crv) = core8(&ctx.k, r, g, b);
        store8(y_out.add(x), yq);
        _mm256_storeu_ps(cb.add(x), cbv);
        _mm256_storeu_ps(cr.add(x), crv);
        x += 8;
    }
    while x < w {
        let (y8, cbv, crv) = pixel_unit(ctx.c, *rp.add(x), *gp.add(x), *bp.add(x));
        *y_out.add(x) = y8;
        *cb.add(x) = cbv;
        *cr.add(x) = crv;
        x += 1;
    }
}

fn combine(
    cb0: &[f32],
    cr0: &[f32],
    cb1: &[f32],
    cr1: &[f32],
    cw: usize,
    u: &mut [u8],
    v: &mut [u8],
) {
    // Row 1 always holds data, because an odd height duplicates row 0 into it,
    // so every block is four samples and the divisor is always four.
    let mut cx = 0;
    // SAFETY: the bounds keep every load inside the scratch rows.
    unsafe {
        let quarter = _mm256_set1_ps(0.25);
        // Deinterleave even/odd lanes so a 256-bit vector carries four sums.
        let pairs = |a: __m256| -> __m256 {
            let even = _mm256_shuffle_ps::<0b10_00_10_00>(a, a);
            let odd = _mm256_shuffle_ps::<0b11_01_11_01>(a, a);
            _mm256_add_ps(even, odd)
        };
        while cx + 4 <= cw && 2 * cx + 8 <= cb0.len() {
            // Each 256-bit load covers exactly the 8 samples of four blocks.
            let quad = |r0: &[f32], r1: &[f32]| -> __m128 {
                let top = pairs(_mm256_loadu_ps(r0.as_ptr().add(2 * cx)));
                let bottom = pairs(_mm256_loadu_ps(r1.as_ptr().add(2 * cx)));
                let summed = _mm256_mul_ps(_mm256_add_ps(top, bottom), quarter);
                // Lanes 0..3 of each half hold the four sums.
                _mm_movelh_ps(
                    _mm256_castps256_ps128(summed),
                    _mm256_extractf128_ps::<1>(summed),
                )
            };
            store4(u.as_mut_ptr().add(cx), round_u8_128(quad(cb0, cb1)));
            store4(v.as_mut_ptr().add(cx), round_u8_128(quad(cr0, cr1)));
            cx += 4;
        }
    }
    while cx < cw {
        // Odd widths: the last chroma column owns a single luma column.
        let paired = 2 * cx + 1 < cb0.len();
        let inv = if paired { 0.25f32 } else { 0.5f32 };
        let s = |r: &[f32]| {
            if paired {
                r[2 * cx] + r[2 * cx + 1]
            } else {
                r[2 * cx]
            }
        };
        u[cx] = q8((s(cb0) + s(cb1)) * inv);
        v[cx] = q8((s(cr0) + s(cr1)) * inv);
        cx += 1;
    }
}

/// # Safety
/// Requires AVX2.
#[inline(always)]
unsafe fn round_u8_256(v: __m256) -> __m256i {
    // Round half away from zero like `f32::round`: bias then truncate.
    let zero = _mm256_setzero_ps();
    let bias = _mm256_blendv_ps(
        _mm256_set1_ps(0.5),
        _mm256_set1_ps(-0.5),
        _mm256_cmp_ps::<{ _CMP_LT_OQ }>(v, zero),
    );
    let biased = _mm256_add_ps(v, bias);
    let clamped = _mm256_min_ps(_mm256_max_ps(biased, zero), _mm256_set1_ps(255.0));
    _mm256_cvttps_epi32(clamped)
}

/// # Safety
/// Requires SSE4.1.
#[inline(always)]
unsafe fn round_u8_128(v: __m128) -> __m128i {
    let zero = _mm_setzero_ps();
    let bias = _mm_blendv_ps(_mm_set1_ps(0.5), _mm_set1_ps(-0.5), _mm_cmplt_ps(v, zero));
    let biased = _mm_add_ps(v, bias);
    let clamped = _mm_min_ps(_mm_max_ps(biased, zero), _mm_set1_ps(255.0));
    _mm_cvttps_epi32(clamped)
}
