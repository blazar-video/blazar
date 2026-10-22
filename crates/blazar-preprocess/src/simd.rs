//! Fused single-pass RGB -> YUV420 kernels for 8-bit interleaved frames.
//!
//! One pass over the source row pair produces both the luma rows and the
//! per-pixel chroma values; the 2x2 chroma average is then folded from a small
//! cache-resident scratch row. Splitting luma and chroma into separate passes
//! over the frame was measured slower (the second pass costs more than the
//! vector maths saves), which is why everything happens here.
//!
//! The maths mirrors `CpuPreprocess` exactly — same coefficients, same order —
//! so the vector result differs from the scalar one by at most one level, and
//! only where a value lands within rounding distance of a boundary.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod avx2;

/// Convert an interleaved `RGBRGB…` 8-bit frame, producing packed planes.
///
/// Returns `None` when this CPU cannot run the vector path, so the caller falls
/// back to the scalar implementation.
#[must_use]
pub fn convert_thwc_u8(
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64: no runtime check, no fallback.
        neon::convert(bytes, desc, w, h, y_plane, u_plane, v_plane);
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !avx2::available() {
            return false;
        }
        // SAFETY: the CPU supports the instructions the kernel uses.
        unsafe { avx2::convert(bytes, desc, w, h, y_plane, u_plane, v_plane) };
        true
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (bytes, desc, w, h, y_plane, u_plane, v_plane);
        false
    }
}

/// Maths shared by both architectures, so the two kernels cannot drift apart.
pub(crate) struct Coeffs {
    pub(crate) kr: f32,
    pub(crate) kg: f32,
    pub(crate) kb: f32,
    pub(crate) cb_scale: f32,
    pub(crate) cr_scale: f32,
    /// Multiplier and offset applied to a source byte before the matrix.
    /// Both are the identity here; kept so the kernels stay range-agnostic.
    pub(crate) scale: f32,
    pub(crate) offset: f32,
}

impl Coeffs {
    pub(crate) fn new(desc: &blazar_core::FrameDesc) -> Self {
        // Mirrors `Matrix::new` and `U8Sample::to_unit`. Chroma is produced on
        // the 0..255 scale (`0.5` shifted chroma becomes `127.5`) so the average
        // over a 2x2 block is a plain sum.
        let m = Coeffs::matrix(desc.colorspace);
        // A u8 frame is already full-range code values: `U8Sample::to_unit`
        // ignores the caller's `range` (dividing by 255 twice would crush the
        // frame), so this kernel does the same and keeps the scale at one.
        let (scale, offset) = (1.0f32, 0.0f32);
        // Luma weights apply to the 0..255 component, so the unit-range
        // coefficients are used unchanged: 255 * (kr*r + kg*g + kb*b) with
        // r, g, b already on the 0..255 scale.
        Coeffs {
            kr: m.0,
            kg: m.1,
            kb: m.2,
            cb_scale: m.3,
            cr_scale: m.4,
            scale,
            offset,
        }
    }

    fn matrix(cs: blazar_core::ColorSpace) -> (f32, f32, f32, f32, f32) {
        let (kr, kb) = match cs {
            blazar_core::ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            blazar_core::ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        (
            kr,
            1.0 - kr - kb,
            kb,
            1.0 / (2.0 * (1.0 - kb)),
            1.0 / (2.0 * (1.0 - kr)),
        )
    }
}

/// Scalar tail and the odd-height row case, kept identical to the vector body.
#[inline]
pub(crate) fn pixel(
    c: &Coeffs,
    c0: u8,
    c1: u8,
    c2: u8,
) -> (u8, f32, f32) {
    let r = (c0 as f32 * c.scale + c.offset).clamp(0.0, 255.0);
    let g = (c1 as f32 * c.scale + c.offset).clamp(0.0, 255.0);
    let b = (c2 as f32 * c.scale + c.offset).clamp(0.0, 255.0);
    let y = c.kr * r + c.kg * g + c.kb * b;
    let cb = (b - y) * c.cb_scale + 127.5;
    let cr = (r - y) * c.cr_scale + 127.5;
    (y.round().clamp(0.0, 255.0) as u8, cb, cr)
}
