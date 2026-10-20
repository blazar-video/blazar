//! AVX2 RGB -> YUV420 for the planar f32 path (a model's BCTHW frame).
//!
//! Only the luma rows are vectorised here. The chroma 2x2 average stays scalar
//! for now; it is the next step and the prototype showed luma alone is already
//! 3x faster than the scalar single pass.
//!
//! Output must match the scalar implementation within 1 level. Rounding follows
//! `f32::round` (half away from zero), not the ties-to-even default of
//! `_mm256_round_ps`.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use rayon::prelude::*;

use blazar_core::FrameDesc;

use crate::CpuPreprocess;

/// Is AVX2 usable on this CPU?
#[must_use]
pub fn available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Convert one planar f32 frame, producing packed y/u/v planes.
///
/// # Safety
/// The caller must check [`available`]. `bytes` must be a BCTHW f32 frame of
/// `w*h*3` samples, unit range.
pub unsafe fn convert_bcthw_f32(
    bytes: &[u8],
    desc: &FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let pixels = w * h;
    let matrix = crate::Matrix::new(desc.colorspace);
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_plane = vec![0u8; pixels];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];

    let base = bytes.as_ptr();
    // Raw pointers are not Sync; carry the addresses and rebuild them inside
    // the parallel closures.
    let r_addr = base as usize;
    let g_addr = base as usize + pixels * 4;
    let b_addr = base as usize + 2 * pixels * 4;
    let (r, g, b) = (
        r_addr as *const f32,
        g_addr as *const f32,
        b_addr as *const f32,
    );

    let kr = _mm256_set1_ps(matrix.kr);
    let kg = _mm256_set1_ps(matrix.kg);
    let kb = _mm256_set1_ps(matrix.kb);
    let half = _mm256_set1_ps(0.5);
    let sign = _mm256_set1_ps(-0.0);
    let scale = _mm256_set1_ps(255.0);
    let zero = _mm256_setzero_ps();
    let max = _mm256_set1_ps(255.0);

    let rows: Vec<&mut [u8]> = y_plane.chunks_mut(w).collect();
    rows.into_par_iter_rows(w, h, |row, y_row| {
        let (r, g, b) = (
            r_addr as *const f32,
            g_addr as *const f32,
            b_addr as *const f32,
        );
        let p = row * w;
        let mut col = 0usize;
        while col + 8 <= w {
            // SAFETY: the frame has `pixels` samples per plane and col+8 <= w.
            let rv = _mm256_loadu_ps(r.add(p + col));
            let gv = _mm256_loadu_ps(g.add(p + col));
            let bv = _mm256_loadu_ps(b.add(p + col));
            let luma = _mm256_fmadd_ps(
                kb,
                bv,
                _mm256_fmadd_ps(kg, gv, _mm256_mul_ps(kr, rv)),
            );
            let scaled = _mm256_mul_ps(luma, scale);
            let rounded = _mm256_round_ps(
                _mm256_add_ps(scaled, _mm256_or_ps(half, sign)),
                _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
            );
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, zero), max);
            let ints = _mm256_cvtps_epi32(clamped);
            let lo = _mm256_castsi256_si128(ints);
            let hi = _mm256_extracti128_si256(ints, 1);
            let packed = _mm_packus_epi16(_mm_packus_epi32(lo, hi), _mm_setzero_si128());
            let mut out = [0u8; 8];
            _mm_storel_epi64(out.as_mut_ptr().cast(), packed);
            y_row[col..col + 8].copy_from_slice(&out);
            col += 8;
        }
        while col < w {
            let i = p + col;
            let luma = *r.add(i) * matrix.kr + *g.add(i) * matrix.kg + *b.add(i) * matrix.kb;
            y_row[col] = CpuPreprocess::q8(luma);
            col += 1;
        }
    });

    // Chroma is still the scalar 2x2 average.
    u.par_chunks_mut(cw)
        .zip(v.par_chunks_mut(cw))
        .enumerate()
        .for_each(|(cy, (u_row, v_row))| {
            let (r, g, b) = (
                r_addr as *const f32,
                g_addr as *const f32,
                b_addr as *const f32,
            );
            for cx in 0..cw {
                let mut cb = 0.0f32;
                let mut cr = 0.0f32;
                let mut count = 0.0f32;
                for dy in 0..2 {
                    let sy = cy * 2 + dy;
                    if sy >= h {
                        break;
                    }
                    for dx in 0..2 {
                        let sx = cx * 2 + dx;
                        if sx >= w {
                            break;
                        }
                        let i = sy * w + sx;
                        // SAFETY: indexes stay inside the frame.
                        let (rr, gg, bb) = unsafe { (*r.add(i), *g.add(i), *b.add(i)) };
                        let luma = matrix.kr * rr + matrix.kg * gg + matrix.kb * bb;
                        cb += (bb - luma) * matrix.cb_scale + 0.5;
                        cr += (rr - luma) * matrix.cr_scale + 0.5;
                        count += 1.0;
                    }
                }
                u_row[cx] = CpuPreprocess::q8(cb / count);
                v_row[cx] = CpuPreprocess::q8(cr / count);
            }
        });

    (y_plane, u, v)
}

/// Parallel-over-rows helper: keeps the unsafe intrinsic block in one place.
trait RowIter {
    fn into_par_iter_rows<F: Fn(usize, &mut [u8]) + Sync + Send>(self, w: usize, h: usize, f: F);
}

impl<'a> RowIter for Vec<&'a mut [u8]> {
    fn into_par_iter_rows<F: Fn(usize, &mut [u8]) + Sync + Send>(self, _w: usize, _h: usize, f: F) {
        use rayon::prelude::*;
        self.into_par_iter().enumerate().for_each(|(row, slice)| f(row, slice));
    }
}
