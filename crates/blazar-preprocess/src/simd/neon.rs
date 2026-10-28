// SPDX-License-Identifier: Apache-2.0
//! aarch64 NEON kernels for RGB -> YUV420.
//!
//! Six source shapes share one arithmetic core. NEON's structure loads do the
//! deinterleaving for free (`vld3q_u8` for 16 pixels, `vld3_u16`/`vld3q_f32` for
//! 4), so the interleaved layouts cost nothing extra here.
//!
//! bf16 rows are widened into an f32 scratch row and then run through the f32
//! kernel: widening is one shift per sample, and keeping a single f32 loader per
//! layout stops the bf16 and f32 paths from drifting apart.

use std::arch::aarch64::*;

use rayon::prelude::*;

use super::{pixel_unit, q8, Coeffs, Kind};

/// Everything a row driver needs that is constant across rows.
struct Ctx<'a> {
    c: &'a Coeffs,
    k: K,
    /// Bytes per plane: `w * h * component_bytes`.
    plane_bytes: usize,
    w: usize,
}

/// Broadcast coefficients, set up once per row pair rather than per group.
struct K {
    scale: float32x4_t,
    offset: float32x4_t,
    zero: float32x4_t,
    max: float32x4_t,
    kr: float32x4_t,
    kg: float32x4_t,
    kb: float32x4_t,
    cb_scale: float32x4_t,
    cr_scale: float32x4_t,
    half: float32x4_t,
}

impl K {
    /// # Safety
    /// Requires NEON, which aarch64 guarantees.
    unsafe fn new(c: &Coeffs) -> Self {
        Self {
            scale: vdupq_n_f32(c.scale),
            offset: vdupq_n_f32(c.offset),
            zero: vdupq_n_f32(0.0),
            max: vdupq_n_f32(255.0),
            kr: vdupq_n_f32(c.kr),
            kg: vdupq_n_f32(c.kg),
            kb: vdupq_n_f32(c.kb),
            cb_scale: vdupq_n_f32(c.cb_scale),
            cr_scale: vdupq_n_f32(c.cr_scale),
            half: vdupq_n_f32(127.5),
        }
    }
}

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

pub(crate) fn convert(
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
        k: unsafe { K::new(&c) },
        plane_bytes: pixels * component_bytes,
        w,
    };
    // bf16 rows are widened here. Two rows share the buffer: they are processed
    // one after the other, so there is no aliasing.
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
                // SAFETY: the driver stays inside the row it is given, and the two
                // scratch rows hold `w` floats each.
                unsafe {
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
            }
            // Odd heights: the last chroma row averages one luma row over itself.
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
            // Native planes are a whole frame apart.
            let stride = ctx.plane_bytes / 4;
            row_bcthw_f32(ctx, base, sy * ctx.w, stride, y_out, cb, cr)
        }
        Kind::ThwcBf16 => {
            widen_bf16(base, sy, ctx.w, true, ctx.plane_bytes, scratch);
            row_thwc_f32(ctx, scratch.as_ptr() as *const u8, 0, y_out, cb, cr)
        }
        Kind::BcthwBf16 => {
            widen_bf16(base, sy, ctx.w, false, ctx.plane_bytes, scratch);
            // The scratch is three w-float planes with no frame padding.
            row_bcthw_f32(ctx, scratch.as_ptr() as *const u8, 0, ctx.w, y_out, cb, cr)
        }
    }
}

/// Widen one row of bf16 samples into an f32 scratch buffer.
///
/// An interleaved row is one run of `3 * w` samples. A planar row is three runs
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
    let runs: &[(usize, usize)] = if interleaved {
        &[(sy * w * 3, 3 * w)]
    } else {
        // SAFETY: unused; the slice below is built from plain integers.
        &[
            (sy * w, w),
            (plane_bytes / 2 + sy * w, w),
            (2 * plane_bytes / 2 + sy * w, w),
        ]
    };
    let mut written = 0usize;
    for &(start, count) in runs {
        let mut i = 0usize;
        while i + 8 <= count {
            let bits = vld1q_u16(src.add(start + i));
            let lo = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(bits)), 16));
            let hi = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(bits)), 16));
            vst1q_f32(scratch.as_mut_ptr().add(written + i), lo);
            vst1q_f32(scratch.as_mut_ptr().add(written + i + 4), hi);
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

/// Four pixels through the shared maths: luma bytes plus chroma values.
///
/// # Safety
/// Requires NEON.
#[inline]
unsafe fn core4(
    k: &K,
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
) -> (uint8x8_t, float32x4_t, float32x4_t) {
    let r = vminq_f32(vmaxq_f32(vmlaq_f32(k.offset, r, k.scale), k.zero), k.max);
    let g = vminq_f32(vmaxq_f32(vmlaq_f32(k.offset, g, k.scale), k.zero), k.max);
    let b = vminq_f32(vmaxq_f32(vmlaq_f32(k.offset, b, k.scale), k.zero), k.max);
    let y = vmlaq_f32(vmlaq_f32(vmulq_f32(k.kr, r), k.kg, g), k.kb, b);
    let cb = vmlaq_f32(k.half, vsubq_f32(b, y), k.cb_scale);
    let cr = vmlaq_f32(k.half, vsubq_f32(r, y), k.cr_scale);
    (round_u8(y), cb, cr)
}

/// Store four luma bytes and the chroma lanes of one group of four pixels.
///
/// # Safety
/// The pointers must have four elements of room.
#[inline]
unsafe fn store4(
    group: usize,
    y: uint8x8_t,
    cb: float32x4_t,
    cr: float32x4_t,
    y_out: *mut u8,
    cb_out: *mut f32,
    cr_out: *mut f32,
) {
    vst1_lane_u32::<0>(y_out.add(group * 4) as *mut u32, vreinterpret_u32_u8(y));
    vst1q_f32(cb_out.add(group * 4), cb);
    vst1q_f32(cr_out.add(group * 4), cr);
}

/// Interleaved u8: 16 pixels per iteration.
///
/// # Safety
/// `base` must cover the frame; the outputs hold `w` elements.
unsafe fn row_thwc_u8(
    ctx: &Ctx<'_>,
    base: *const u8,
    sy: usize,
    y_out: *mut u8,
    cb: *mut f32,
    cr: *mut f32,
) {
    let w = ctx.w;
    let p = base.add(sy * w * 3);
    let mut x = 0;
    while x + 16 <= w {
        let v = vld3q_u8(p.add(x * 3));
        let mut ys = [0u8; 16];
        for q in 0..4usize {
            let pick = |c8: uint8x16_t| -> float32x4_t {
                let half = if q < 2 {
                    vget_low_u8(c8)
                } else {
                    vget_high_u8(c8)
                };
                let wide = vmovl_u8(half);
                let quarters = if q % 2 == 0 {
                    vget_low_u16(wide)
                } else {
                    vget_high_u16(wide)
                };
                vcvtq_f32_u32(vmovl_u16(quarters))
            };
            let (yq, cbv, crv) = core4(&ctx.k, pick(v.0), pick(v.1), pick(v.2));
            vst1_lane_u32::<0>(
                ys.as_mut_ptr().add(q * 4) as *mut u32,
                vreinterpret_u32_u8(yq),
            );
            vst1q_f32(cb.add(x + q * 4), cbv);
            vst1q_f32(cr.add(x + q * 4), crv);
        }
        std::ptr::copy_nonoverlapping(ys.as_ptr(), y_out.add(x), 16);
        x += 16;
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

/// Planar u8: 8 pixels per iteration.
///
/// # Safety
/// Each plane must cover the frame; the outputs hold `w` elements.
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
        let (rv, gv, bv) = (vld1_u8(rp.add(x)), vld1_u8(gp.add(x)), vld1_u8(bp.add(x)));
        for (group, high) in [(0usize, false), (1, true)] {
            let pick = |v: uint8x8_t| -> float32x4_t {
                let wide = vmovl_u8(v);
                let quarters = if high {
                    vget_high_u16(wide)
                } else {
                    vget_low_u16(wide)
                };
                vcvtq_f32_u32(vmovl_u16(quarters))
            };
            let (yq, cbv, crv) = core4(&ctx.k, pick(rv), pick(gv), pick(bv));
            store4(group, yq, cbv, crv, y_out.add(x), cb.add(x), cr.add(x));
        }
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

/// Interleaved f32: 4 pixels per iteration via `vld3q_f32`.
///
/// # Safety
/// `base` must cover the frame; the outputs hold `w` elements.
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
    let mut x = 0;
    while x + 4 <= w {
        let v = vld3q_f32(p.add(x * 3));
        let (yq, cbv, crv) = core4(&ctx.k, v.0, v.1, v.2);
        store4(0, yq, cbv, crv, y_out.add(x), cb.add(x), cr.add(x));
        x += 4;
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

/// Planar f32: 8 pixels per iteration.
///
/// The same driver serves bf16 frames, whose scratch row has the same layout.
///
/// # Safety
/// Each plane must cover the frame; the outputs hold `w` elements.
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
        let (r0, g0, b0) = (
            vld1q_f32(rp.add(x)),
            vld1q_f32(gp.add(x)),
            vld1q_f32(bp.add(x)),
        );
        let (r1, g1, b1) = (
            vld1q_f32(rp.add(x + 4)),
            vld1q_f32(gp.add(x + 4)),
            vld1q_f32(bp.add(x + 4)),
        );
        let (yq, cbv, crv) = core4(&ctx.k, r0, g0, b0);
        store4(0, yq, cbv, crv, y_out.add(x), cb.add(x), cr.add(x));
        let (yq, cbv, crv) = core4(&ctx.k, r1, g1, b1);
        store4(1, yq, cbv, crv, y_out.add(x), cb.add(x), cr.add(x));
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

/// Fold the 2x2 chroma blocks and quantize.
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
    let inv = 0.25f32;
    let mut cx = 0;
    // SAFETY: the bounds keep every load inside the scratch rows.
    unsafe {
        while cx + 4 <= cw && 2 * cx + 8 <= cb0.len() {
            let sum = |a: &[f32], b: &[f32]| -> float32x4_t {
                let lo = vpaddq_f32(
                    vld1q_f32(a.as_ptr().add(2 * cx)),
                    vld1q_f32(a.as_ptr().add(2 * cx + 4)),
                );
                let hi = vpaddq_f32(
                    vld1q_f32(b.as_ptr().add(2 * cx)),
                    vld1q_f32(b.as_ptr().add(2 * cx + 4)),
                );
                vmulq_n_f32(vaddq_f32(lo, hi), inv)
            };
            let uq = round_u8(sum(cb0, cb1));
            let vq = round_u8(sum(cr0, cr1));
            vst1_lane_u32::<0>(u.as_mut_ptr().add(cx) as *mut u32, vreinterpret_u32_u8(uq));
            vst1_lane_u32::<0>(v.as_mut_ptr().add(cx) as *mut u32, vreinterpret_u32_u8(vq));
            cx += 4;
        }
    }
    while cx < cw {
        // Odd widths: the last chroma column owns a single luma column.
        let paired = 2 * cx + 1 < cb0.len();
        let inv = if paired { inv } else { 0.5f32 };
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
/// Requires NEON.
#[inline]
unsafe fn round_u8(v: float32x4_t) -> uint8x8_t {
    // Round half away from zero like `f32::round`: bias then truncate.
    let zero = vdupq_n_f32(0.0);
    let bias = vbslq_f32(vcltq_f32(v, zero), vdupq_n_f32(-0.5), vdupq_n_f32(0.5));
    let clamped = vminq_f32(vmaxq_f32(vaddq_f32(v, bias), zero), vdupq_n_f32(255.0));
    let ints: int32x4_t = vcvtq_s32_f32(clamped);
    vmovn_u16(vcombine_u16(
        vmovn_u32(vreinterpretq_u32_s32(ints)),
        vdup_n_u16(0),
    ))
}
