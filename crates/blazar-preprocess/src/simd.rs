//! Vector RGB -> YUV420 kernels, dispatched per architecture.
//!
//! Every source the pipeline can produce has a kernel: interleaved or planar,
//! 8-bit, f32 or bf16. Each one keeps luma and chroma in a single pass over the
//! source and folds the 2x2 chroma block from a two-row scratch that stays in
//! cache; a separate pass over the planes was measured slower than the maths it
//! saved.
//!
//! The kernels mirror `CpuPreprocess` exactly — same coefficients, same rounding
//! rule — so a vector result differs from the scalar one by at most one level,
//! and only where a value lands within rounding distance of a boundary.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod avx2;

/// Which frame shape a kernel is reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Interleaved `RGBRGB…`, 8-bit.
    ThwcU8,
    /// Three planes, 8-bit.
    BcthwU8,
    /// Interleaved `RGBRGB…`, f32.
    ThwcF32,
    /// Three planes, f32.
    BcthwF32,
    /// Interleaved `RGBRGB…`, bf16.
    ThwcBf16,
    /// Three planes, bf16.
    BcthwBf16,
}

/// Is a vector kernel available on this CPU?
pub(crate) fn available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        avx2::available()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// Run the vector path for `kind`, or return `false` when this CPU has none.
#[must_use]
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
    #[cfg(target_arch = "aarch64")]
    {
        neon::convert(kind, bytes, desc, w, h, y_plane, u_plane, v_plane)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !avx2::available() {
            return false;
        }
        // SAFETY: the CPU supports the instructions the kernels use.
        unsafe { avx2::convert(kind, bytes, desc, w, h, y_plane, u_plane, v_plane) }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = (kind, bytes, desc, w, h, y_plane, u_plane, v_plane);
        false
    }
}

/// Maths shared by every kernel and by the scalar tails.
pub(crate) struct Coeffs {
    pub(crate) kr: f32,
    pub(crate) kg: f32,
    pub(crate) kb: f32,
    pub(crate) cb_scale: f32,
    pub(crate) cr_scale: f32,
    /// Bring a raw sample onto the 0..255 scale the matrix works in.
    pub(crate) scale: f32,
    pub(crate) offset: f32,
}

impl Coeffs {
    /// Coefficients for an 8-bit frame.
    ///
    /// A u8 frame is already full-range code values, and `U8Sample::to_unit`
    /// ignores the caller's `range` (dividing by 255 twice would crush the
    /// frame), so these kernels do the same and keep the scale at one.
    pub(crate) fn for_u8(desc: &blazar_core::FrameDesc) -> Self {
        Self::with_scale(desc, 1.0, 0.0)
    }

    /// Coefficients for a float frame (f32 or bf16), whose range mapping does
    /// apply.
    ///
    /// The scalar path normalises to `[0,1]` and multiplies by 255 at the end.
    /// Both steps are linear, so folding them into one scale and clamping once
    /// gives the same result.
    pub(crate) fn for_unit(desc: &blazar_core::FrameDesc) -> Self {
        let (scale, offset) = match desc.range {
            blazar_core::ColorRange::Unit01 => (255.0f32, 0.0f32),
            blazar_core::ColorRange::SignedUnit => (127.5, 127.5),
            blazar_core::ColorRange::FullU8 => (1.0, 0.0),
        };
        Self::with_scale(desc, scale, offset)
    }

    fn with_scale(desc: &blazar_core::FrameDesc, scale: f32, offset: f32) -> Self {
        let (kr, kb) = match desc.colorspace {
            blazar_core::ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            blazar_core::ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        Self {
            kr,
            kg: 1.0 - kr - kb,
            kb,
            cb_scale: 1.0 / (2.0 * (1.0 - kb)),
            cr_scale: 1.0 / (2.0 * (1.0 - kr)),
            scale,
            offset,
        }
    }
}

/// Convert one pixel on the scalar path and in the vector tails.
///
/// Returns `(Y, Cb, Cr)` with chroma on the 0..255 scale so a 2x2 block can be
/// averaged before quantizing.
#[inline]
pub(crate) fn pixel_unit(c: &Coeffs, r: f32, g: f32, b: f32) -> (u8, f32, f32) {
    let r = (r * c.scale + c.offset).clamp(0.0, 255.0);
    let g = (g * c.scale + c.offset).clamp(0.0, 255.0);
    let b = (b * c.scale + c.offset).clamp(0.0, 255.0);
    let y = c.kr * r + c.kg * g + c.kb * b;
    let cb = (b - y) * c.cb_scale + 127.5;
    let cr = (r - y) * c.cr_scale + 127.5;
    (q8(y), cb, cr)
}

#[inline]
pub(crate) fn q8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
