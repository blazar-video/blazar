//! `blazar-preprocess` — the preprocess stage (architecture §7).
//!
//! Between VAE output and encoder input there is mandatory, per-pixel work:
//! color-space convert (RGB→YUV), range convert (normalized float → video
//! range), chroma subsample (→ 4:2:0), and quantize (float → 8-bit). On the
//! device path this is a set of CUDA kernels; Phase 1 is the CPU fallback for
//! the **host** ingest path (the xDiT seam), which is what this crate
//! implements.
//!
//! The stage is parameterized by [`blazar_core::FrameDesc`] because the two
//! runtimes hand over different conventions (xDiT `[0,1]` THWC; vLLM-Omni
//! `[-1,1]` BCTHW) — the conversion must read range + layout, not assume them.

use blazar_core::{ColorRange, ColorSpace, Dtype, FrameRef, Layout, Pts};
use rayon::prelude::*;

/// Fused luma+chroma vector kernels (NEON, AVX2) for 8-bit interleaved frames.
mod simd;
use thiserror::Error;

/// An 8-bit planar YUV 4:2:0 frame — the small representation encoders consume
/// (architecture §7: the device→host copy, when needed, happens on *this*, not
/// the fat RGB float).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessedFrame {
    pub width: u32,
    pub height: u32,
    /// Full-resolution luma plane, row-major, `width * height` bytes.
    pub y: Vec<u8>,
    /// Subsampled Cb plane, `((width+1)/2) * ((height+1)/2)` bytes.
    pub u: Vec<u8>,
    /// Subsampled Cr plane, same size as `u`.
    pub v: Vec<u8>,
    pub pts: Pts,
}

impl PreprocessedFrame {
    pub fn chroma_width(&self) -> u32 {
        self.width.div_ceil(2)
    }
    pub fn chroma_height(&self) -> u32 {
        self.height.div_ceil(2)
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum PreprocessError {
    #[error("device ingest is not handled by the CPU preprocess stage (Phase 2)")]
    DeviceNotSupported,
    #[error("unsupported dtype for CPU preprocess: {0:?} (Phase 1 handles F32)")]
    UnsupportedDtype(Dtype),
    #[error("frame byte length {got} != expected {expected} for {w}x{h} RGB")]
    SizeMismatch {
        got: usize,
        expected: usize,
        w: u32,
        h: u32,
    },
}

/// The preprocess stage trait. The CPU and (future) CUDA stages both implement
/// it so the pipeline is residency-agnostic above this seam.
pub trait PreprocessStage {
    fn process(&self, frame: &FrameRef) -> Result<PreprocessedFrame, PreprocessError>;
}

/// CPU preprocess for the host ingest path (Phase 1).
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuPreprocess;

impl CpuPreprocess {
    pub fn new() -> Self {
        CpuPreprocess
    }

    /// Normalize one float RGB component to `[0,1]` given the source range.
    #[inline]
    fn to_unit(v: f32, range: ColorRange) -> f32 {
        match range {
            ColorRange::Unit01 => v,
            // [-1,1] → [0,1]
            ColorRange::SignedUnit => (v + 1.0) * 0.5,
            ColorRange::FullU8 => v / 255.0,
        }
        .clamp(0.0, 1.0)
    }

    #[inline]
    fn q8(v: f32) -> u8 {
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

/// Component reader, generic so the per-pixel loop has no dtype branch.
trait Sample {
    fn component_bytes() -> usize;
    fn read(bytes: &[u8], component: usize) -> f32;
    /// Normalize one read component to `[0,1]`.
    ///
    /// On the trait rather than at the call site so the per-pixel loop stays
    /// monomorphised per dtype, with no branch on the dtype inside it.
    fn to_unit(value: f32, range: ColorRange) -> f32;
}

struct F32Sample;
struct Bf16Sample;
/// Full-range 8-bit code values: 0..=255, no range scaling to apply.
struct U8Sample;

impl Sample for F32Sample {
    #[inline(always)]
    fn component_bytes() -> usize {
        4
    }
    #[inline(always)]
    fn read(bytes: &[u8], component: usize) -> f32 {
        let off = component * 4;
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    #[inline(always)]
    fn to_unit(value: f32, range: ColorRange) -> f32 {
        CpuPreprocess::to_unit(value, range)
    }
}

impl Sample for Bf16Sample {
    #[inline(always)]
    fn component_bytes() -> usize {
        2
    }
    #[inline(always)]
    fn read(bytes: &[u8], component: usize) -> f32 {
        let off = component * 2;
        // Truncation, not rounding: bf16 is the top half of f32.
        f32::from_bits((u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32) << 16)
    }

    #[inline(always)]
    fn to_unit(value: f32, range: ColorRange) -> f32 {
        CpuPreprocess::to_unit(value, range)
    }
}

impl Sample for U8Sample {
    #[inline(always)]
    fn component_bytes() -> usize {
        1
    }
    #[inline(always)]
    fn read(bytes: &[u8], component: usize) -> f32 {
        bytes[component] as f32
    }

    /// A u8 array is already full-range code values, so the caller's `range`
    /// does not apply: dividing by 255 twice would crush the frame.
    #[inline(always)]
    fn to_unit(value: f32, _range: ColorRange) -> f32 {
        (value / 255.0).clamp(0.0, 1.0)
    }
}

/// BT.709/BT.601 matrix coefficients, computed once per frame.
struct Matrix {
    kr: f32,
    kg: f32,
    kb: f32,
    cb_scale: f32,
    cr_scale: f32,
}

impl Matrix {
    fn new(colorspace: ColorSpace) -> Self {
        let (kr, kb) = match colorspace {
            ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        Self {
            kr,
            kg: 1.0 - kr - kb,
            kb,
            cb_scale: 1.0 / (2.0 * (1.0 - kb)),
            cr_scale: 1.0 / (2.0 * (1.0 - kr)),
        }
    }
}

impl PreprocessStage for CpuPreprocess {
    fn process(&self, frame: &FrameRef) -> Result<PreprocessedFrame, PreprocessError> {
        let desc = &frame.desc;

        if frame.buffer.is_device() {
            return Err(PreprocessError::DeviceNotSupported);
        }
        let component_bytes = match desc.dtype {
            Dtype::F32 => F32Sample::component_bytes(),
            Dtype::Bf16 => Bf16Sample::component_bytes(),
            Dtype::U8 => U8Sample::component_bytes(),
            other => return Err(PreprocessError::UnsupportedDtype(other)),
        };

        let bytes = frame
            .buffer
            .host_bytes()
            .ok_or(PreprocessError::DeviceNotSupported)?;

        let w = desc.width as usize;
        let h = desc.height as usize;
        let pixels = w * h;
        let expected = pixels * 3 * component_bytes;
        if bytes.len() != expected {
            return Err(PreprocessError::SizeMismatch {
                got: bytes.len(),
                expected,
                w: desc.width,
                h: desc.height,
            });
        }

        // Dispatch dtype+layout once; the pixel loops then have no branches on
        // either. This is the difference between ~19 ms and a few ms at 1080p.
        let (y, u, v) = match (desc.layout, desc.dtype) {
            // Every supported pair goes through the vector kernel, which
            // falls back to the scalar pass when the CPU has none. The scalar
            // implementations stay because they are the reference the kernels
            // are tested against.
            (Layout::Thwc, Dtype::U8) => convert_simd(simd::Kind::ThwcU8, bytes, desc, w, h),
            (Layout::Bcthw, Dtype::U8) => convert_simd(simd::Kind::BcthwU8, bytes, desc, w, h),
            (Layout::Thwc, Dtype::F32) => convert_simd(simd::Kind::ThwcF32, bytes, desc, w, h),
            (Layout::Bcthw, Dtype::F32) => convert_simd(simd::Kind::BcthwF32, bytes, desc, w, h),
            (Layout::Thwc, Dtype::Bf16) => convert_simd(simd::Kind::ThwcBf16, bytes, desc, w, h),
            (Layout::Bcthw, Dtype::Bf16) => convert_simd(simd::Kind::BcthwBf16, bytes, desc, w, h),
            (_, other) => return Err(PreprocessError::UnsupportedDtype(other)),
        };

        Ok(PreprocessedFrame {
            width: desc.width,
            height: desc.height,
            y,
            u,
            v,
            pts: desc.pts,
        })
    }
}

/// Convert through the vector kernel, or fall back to the scalar pass.
///
/// The vector path is an optimisation, not a capability requirement, so a CPU
/// without one still produces a correct frame.
fn convert_simd(
    kind: simd::Kind,
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    if simd::convert(kind, bytes, desc, w, h, &mut y, &mut u, &mut v) {
        return (y, u, v);
    }
    match (desc.layout, desc.dtype) {
        (Layout::Thwc, Dtype::U8) => convert_thwc::<U8Sample>(bytes, desc, w, h),
        (Layout::Bcthw, Dtype::U8) => convert_bcthw::<U8Sample>(bytes, desc, w, h),
        (Layout::Thwc, Dtype::F32) => convert_thwc::<F32Sample>(bytes, desc, w, h),
        (Layout::Bcthw, Dtype::F32) => convert_bcthw::<F32Sample>(bytes, desc, w, h),
        (Layout::Thwc, Dtype::Bf16) => convert_thwc::<Bf16Sample>(bytes, desc, w, h),
        (Layout::Bcthw, Dtype::Bf16) => convert_bcthw::<Bf16Sample>(bytes, desc, w, h),
        (_, other) => (Vec::new(), Vec::new(), Vec::new()),
    }
}

/// Below this, rayon's per-task overhead costs more than the parallel gain.
/// Measured on the 28-thread mosk box: 480x320 is ~9% slower parallel, 720p is
/// 4x faster.
pub(crate) const PARALLEL_MIN_PIXELS: usize = 640 * 480;

/// Everything a row-pair task needs that is constant across rows.
struct RowCtx<'a> {
    bytes: &'a [u8],
    w: usize,
    pixels: usize,
    desc: &'a blazar_core::FrameDesc,
    matrix: &'a Matrix,
}

#[inline]
fn row_pair_thwc<S: Sample>(
    ctx: &RowCtx<'_>,
    cy: usize,
    y_rows: &mut [u8],
    u_row: &mut [u8],
    v_row: &mut [u8],
) {
    let RowCtx {
        bytes,
        w,
        desc,
        matrix,
        ..
    } = *ctx;
    for cx in 0..u_row.len() {
        let (mut cb, mut cr, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for (dy, y_row) in y_rows.chunks_exact_mut(w).enumerate() {
            for dx in 0..2 {
                let sx = cx * 2 + dx;
                if sx >= w {
                    break;
                }
                let component = ((cy * 2 + dy) * w + sx) * 3;
                let (yy, ccb, ccr) = pixel_into::<S>(
                    S::read(bytes, component),
                    S::read(bytes, component + 1),
                    S::read(bytes, component + 2),
                    desc,
                    matrix,
                );
                y_row[sx] = yy;
                cb += ccb;
                cr += ccr;
                n += 1.0;
            }
        }
        u_row[cx] = CpuPreprocess::q8(cb / n);
        v_row[cx] = CpuPreprocess::q8(cr / n);
    }
}

/// THWC (xDiT): each pixel is three consecutive components.
fn convert_thwc<S: Sample>(
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let matrix = Matrix::new(desc.colorspace);
    let ctx = RowCtx {
        bytes,
        w,
        pixels: w * h,
        desc,
        matrix: &matrix,
    };
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_plane = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];

    // One task per chroma row: it writes that row of U/V and the one or two
    // luma rows it owns, so the borrowed chunks are disjoint by construction.
    if w * h < PARALLEL_MIN_PIXELS {
        for (cy, ((y_rows, u_row), v_row)) in y_plane
            .chunks_mut(2 * w)
            .zip(u.chunks_mut(cw))
            .zip(v.chunks_mut(cw))
            .enumerate()
        {
            row_pair_thwc::<S>(&ctx, cy, y_rows, u_row, v_row);
        }
    } else {
        y_plane
            .par_chunks_mut(2 * w)
            .zip(u.par_chunks_mut(cw))
            .zip(v.par_chunks_mut(cw))
            .enumerate()
            .for_each(|(cy, ((y_rows, u_row), v_row))| {
                row_pair_thwc::<S>(&ctx, cy, y_rows, u_row, v_row);
            });
    }
    (y_plane, u, v)
}

#[inline]
fn row_pair_bcthw<S: Sample>(
    ctx: &RowCtx<'_>,
    cy: usize,
    y_rows: &mut [u8],
    u_row: &mut [u8],
    v_row: &mut [u8],
) {
    let RowCtx {
        bytes,
        w,
        pixels,
        desc,
        matrix,
    } = *ctx;
    for cx in 0..u_row.len() {
        let (mut cb, mut cr, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for (dy, y_row) in y_rows.chunks_exact_mut(w).enumerate() {
            for dx in 0..2 {
                let sx = cx * 2 + dx;
                if sx >= w {
                    break;
                }
                let px = (cy * 2 + dy) * w + sx;
                let (yy, ccb, ccr) = pixel_into::<S>(
                    S::read(bytes, px),
                    S::read(bytes, pixels + px),
                    S::read(bytes, 2 * pixels + px),
                    desc,
                    matrix,
                );
                y_row[sx] = yy;
                cb += ccb;
                cr += ccr;
                n += 1.0;
            }
        }
        u_row[cx] = CpuPreprocess::q8(cb / n);
        v_row[cx] = CpuPreprocess::q8(cr / n);
    }
}

/// BCTHW (vLLM-Omni host fallback): three full-resolution planes.
fn convert_bcthw<S: Sample>(
    bytes: &[u8],
    desc: &blazar_core::FrameDesc,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let matrix = Matrix::new(desc.colorspace);
    let pixels = w * h;
    let ctx = RowCtx {
        bytes,
        w,
        pixels,
        desc,
        matrix: &matrix,
    };
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_plane = vec![0u8; pixels];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];

    if pixels < PARALLEL_MIN_PIXELS {
        for (cy, ((y_rows, u_row), v_row)) in y_plane
            .chunks_mut(2 * w)
            .zip(u.chunks_mut(cw))
            .zip(v.chunks_mut(cw))
            .enumerate()
        {
            row_pair_bcthw::<S>(&ctx, cy, y_rows, u_row, v_row);
        }
    } else {
        y_plane
            .par_chunks_mut(2 * w)
            .zip(u.par_chunks_mut(cw))
            .zip(v.par_chunks_mut(cw))
            .enumerate()
            .for_each(|(cy, ((y_rows, u_row), v_row))| {
                row_pair_bcthw::<S>(&ctx, cy, y_rows, u_row, v_row);
            });
    }
    (y_plane, u, v)
}

/// Normalize, convert and quantize one pixel. Returns `(Y, Cb, Cr)` with chroma
/// as the shifted `[0,1]` value, so blocks can be averaged before quantizing.
#[inline(always)]
fn pixel_into<S: Sample>(
    r: f32,
    g: f32,
    b: f32,
    desc: &blazar_core::FrameDesc,
    matrix: &Matrix,
) -> (u8, f32, f32) {
    let r = S::to_unit(r, desc.range);
    let g = S::to_unit(g, desc.range);
    let b = S::to_unit(b, desc.range);
    let y = matrix.kr * r + matrix.kg * g + matrix.kb * b;
    let cb = (b - y) * matrix.cb_scale + 0.5;
    let cr = (r - y) * matrix.cr_scale + 0.5;
    (CpuPreprocess::q8(y), cb, cr)
}

/// Byte length of an RGB `f32` frame. Callers (and tests) allocate buffers with
/// this so they cannot disagree with the stage about the expected size.
pub fn rgb_f32_len(width: u32, height: u32) -> usize {
    width as usize * height as usize * 3 * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::{BufferOwner, FrameBuffer, FrameDesc};

    fn desc(w: u32, h: u32, range: ColorRange, layout: Layout) -> FrameDesc {
        FrameDesc {
            width: w,
            height: h,
            dtype: Dtype::F32,
            range,
            layout,
            colorspace: ColorSpace::AssumedBt709,
            fps: 24.0,
            pts: Pts(0),
        }
    }

    fn thwc_solid(w: u32, h: u32, r: f32, g: f32, b: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity(rgb_f32_len(w, h));
        for _ in 0..(w * h) {
            v.extend_from_slice(&r.to_le_bytes());
            v.extend_from_slice(&g.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
        }
        v
    }

    #[test]
    fn solid_white_maps_to_full_luma_neutral_chroma() {
        let (w, h) = (4, 4);
        let fr = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid(w, h, 1.0, 1.0, 1.0)),
            desc(w, h, ColorRange::Unit01, Layout::Thwc),
        );
        let out = CpuPreprocess::new().process(&fr).unwrap();
        // White → Y≈255, chroma centered at 128.
        assert!(out.y.iter().all(|&y| y >= 254));
        assert!(out.u.iter().all(|&u| (u as i32 - 128).abs() <= 1));
        assert!(out.v.iter().all(|&v| (v as i32 - 128).abs() <= 1));
    }

    #[test]
    fn solid_black_maps_to_zero_luma() {
        let (w, h) = (4, 4);
        let fr = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid(w, h, 0.0, 0.0, 0.0)),
            desc(w, h, ColorRange::Unit01, Layout::Thwc),
        );
        let out = CpuPreprocess::new().process(&fr).unwrap();
        assert!(out.y.iter().all(|&y| y == 0));
    }

    #[test]
    fn plane_sizes_are_420() {
        let (w, h) = (8, 6);
        let fr = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid(w, h, 0.5, 0.5, 0.5)),
            desc(w, h, ColorRange::Unit01, Layout::Thwc),
        );
        let out = CpuPreprocess::new().process(&fr).unwrap();
        assert_eq!(out.y.len(), (w * h) as usize);
        assert_eq!(out.u.len(), (4 * 3) as usize); // 4x3 chroma
        assert_eq!(out.v.len(), out.u.len());
    }

    #[test]
    fn signed_unit_range_is_normalized() {
        // vLLM-Omni convention: [-1,1]. A value of -1 (black) must map to Y=0,
        // not to a clamped mid-gray.
        let (w, h) = (2, 2);
        let fr = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid(w, h, -1.0, -1.0, -1.0)),
            desc(w, h, ColorRange::SignedUnit, Layout::Thwc),
        );
        let out = CpuPreprocess::new().process(&fr).unwrap();
        assert!(out.y.iter().all(|&y| y == 0));
    }

    #[test]
    fn device_buffer_rejected_in_phase1() {
        let fr = FrameRef::new(
            FrameBuffer::Device {
                dlpack: core::ptr::null_mut(),
                stream: blazar_core::CudaStream(0),
                ready: blazar_core::CudaEvent(0),
                ipc: None,
            },
            desc(2, 2, ColorRange::Unit01, Layout::Bcthw),
        );
        assert_eq!(
            CpuPreprocess::new().process(&fr).unwrap_err(),
            PreprocessError::DeviceNotSupported
        );
    }

    #[test]
    fn wrong_size_rejected() {
        let fr = FrameRef::new(
            FrameBuffer::Host {
                ptr: [0u8; 8].as_ptr(),
                len: 8,
                owner: BufferOwner::Owned(vec![0u8; 8]),
            },
            desc(4, 4, ColorRange::Unit01, Layout::Thwc),
        );
        // Note: host_bytes reads the owned Vec (len 8), expected is 4*4*3*4=192.
        let err = CpuPreprocess::new().process(&fr).unwrap_err();
        assert!(matches!(err, PreprocessError::SizeMismatch { .. }));
    }

    // --- bf16 (vLLM-Omni VAE dtype) -----------------------------------------

    /// bf16 is the top 16 bits of f32. Encoding by truncation (not rounding)
    /// gives an exact reference: the same value widened back to f32.
    fn to_bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    fn as_truncated_f32(v: f32) -> f32 {
        f32::from_bits((to_bf16_bits(v) as u32) << 16)
    }

    fn thwc_solid_bf16(w: u32, h: u32, r: f32, g: f32, b: f32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3 * 2) as usize);
        for _ in 0..(w * h) {
            for c in [r, g, b] {
                v.extend_from_slice(&to_bf16_bits(c).to_le_bytes());
            }
        }
        v
    }

    fn desc_bf16(w: u32, h: u32, range: ColorRange, layout: Layout) -> FrameDesc {
        FrameDesc {
            dtype: Dtype::Bf16,
            ..desc(w, h, range, layout)
        }
    }

    /// The original two-pass algorithm, kept as the regression oracle.
    fn reference(frame: &FrameRef) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let desc = &frame.desc;
        let bytes = frame.buffer.host_bytes().unwrap();
        let w = desc.width as usize;
        let h = desc.height as usize;
        let pixels = w * h;
        let comp = if desc.dtype == Dtype::Bf16 { 2 } else { 4 };
        let read = |component: usize| -> f32 {
            let off = component * comp;
            match desc.dtype {
                Dtype::Bf16 => {
                    f32::from_bits((u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32) << 16)
                }
                _ => {
                    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
                }
            }
        };
        let read_rgb = |px: usize| match desc.layout {
            Layout::Thwc => (read(px * 3), read(px * 3 + 1), read(px * 3 + 2)),
            Layout::Bcthw => (read(px), read(pixels + px), read(2 * pixels + px)),
        };
        let to_unit = |v: f32| {
            match desc.range {
                ColorRange::Unit01 => v,
                ColorRange::SignedUnit => (v + 1.0) * 0.5,
                ColorRange::FullU8 => v / 255.0,
            }
            .clamp(0.0, 1.0)
        };
        let q8 = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
        let (kr, kb) = match desc.colorspace {
            ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        let kg = 1.0 - kr - kb;
        let mut y_plane = vec![0u8; pixels];
        let mut cb_full = vec![0f32; pixels];
        let mut cr_full = vec![0f32; pixels];
        for px in 0..pixels {
            let (r, g, b) = read_rgb(px);
            let (r, g, b) = (to_unit(r), to_unit(g), to_unit(b));
            let y = kr * r + kg * g + kb * b;
            let cb = (b - y) / (2.0 * (1.0 - kb));
            let cr = (r - y) / (2.0 * (1.0 - kr));
            y_plane[px] = q8(y);
            cb_full[px] = cb + 0.5;
            cr_full[px] = cr + 0.5;
        }
        let cw = desc.width.div_ceil(2) as usize;
        let ch = desc.height.div_ceil(2) as usize;
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut cb, mut cr, mut n) = (0f32, 0f32, 0f32);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (sx, sy) = (cx * 2 + dx, cy * 2 + dy);
                        if sx < w && sy < h {
                            cb += cb_full[sy * w + sx];
                            cr += cr_full[sy * w + sx];
                            n += 1.0;
                        }
                    }
                }
                u[cy * cw + cx] = q8(cb / n);
                v[cy * cw + cx] = q8(cr / n);
            }
        }
        (y_plane, u, v)
    }

    fn filled_frame(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange) -> FrameRef {
        // Deterministic pseudo-random pixels, odd dimensions included.
        let mut state = 0x1234_5678_9abc_def0u64;
        let components = (w * h * 3) as usize;
        let mut bytes = Vec::new();
        for _ in 0..components {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = match range {
                ColorRange::SignedUnit => ((state >> 40) as f32 / 16_777_215.0) * 2.0 - 1.0,
                ColorRange::FullU8 => ((state >> 40) % 256) as f32,
                ColorRange::Unit01 => (state >> 40) as f32 / 16_777_215.0,
            };
            match dtype {
                Dtype::Bf16 => {
                    let bits = (v.to_bits() >> 16) as u16;
                    bytes.extend_from_slice(&bits.to_le_bytes());
                }
                _ => bytes.extend_from_slice(&v.to_le_bytes()),
            }
        }
        let mut d = desc(w, h, range, layout);
        d.dtype = dtype;
        FrameRef::new(FrameBuffer::host_owned(bytes), d)
    }

    #[test]
    fn matches_the_two_pass_reference_across_conventions() {
        for layout in [Layout::Thwc, Layout::Bcthw] {
            for dtype in [Dtype::F32, Dtype::Bf16] {
                for range in [
                    ColorRange::Unit01,
                    ColorRange::SignedUnit,
                    ColorRange::FullU8,
                ] {
                    for (w, h) in [(7, 5), (16, 16), (33, 17)] {
                        let frame = filled_frame(w, h, layout, dtype, range);
                        let new = CpuPreprocess::new().process(&frame).unwrap();
                        let (ry, ru, rv) = reference(&frame);
                        assert_eq!(new.y, ry, "luma {layout:?}/{dtype:?}/{range:?} {w}x{h}");
                        // Chroma divides by a precomputed reciprocal, so a
                        // last-ulp difference can round one level apart.
                        let close = |a: &[u8], b: &[u8]| {
                            a.iter()
                                .zip(b)
                                .all(|(x, y)| (*x as i32 - *y as i32).abs() <= 1)
                        };
                        assert!(
                            close(&new.u, &ru),
                            "Cb {layout:?}/{dtype:?}/{range:?} {w}x{h}"
                        );
                        assert!(
                            close(&new.v, &rv),
                            "Cr {layout:?}/{dtype:?}/{range:?} {w}x{h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn u8_matches_the_same_values_as_f32() {
        // A u8 frame is the same picture as an f32 frame holding v/255, so the
        // two paths must agree. Luma is exact; chroma may round one level.
        let (w, h) = (9, 7);
        let px = (w * h) as usize;
        for layout in [Layout::Thwc, Layout::Bcthw] {
            let rgb: Vec<[u8; 3]> = (0..px)
                .map(|i| {
                    [
                        ((i * 37) % 256) as u8,
                        ((i * 91 + 7) % 256) as u8,
                        ((i * 53 + 199) % 256) as u8,
                    ]
                })
                .collect();

            let mut u8_bytes = Vec::new();
            let mut f32_bytes = Vec::new();
            match layout {
                Layout::Thwc => {
                    for c in &rgb {
                        u8_bytes.extend_from_slice(c);
                        for v in c {
                            f32_bytes.extend_from_slice(&(*v as f32 / 255.0).to_le_bytes());
                        }
                    }
                }
                Layout::Bcthw => {
                    for channel in 0..3 {
                        for c in &rgb {
                            u8_bytes.push(c[channel]);
                        }
                    }
                    for channel in 0..3 {
                        for c in &rgb {
                            f32_bytes.extend_from_slice(&(c[channel] as f32 / 255.0).to_le_bytes());
                        }
                    }
                }
                other => panic!("unexpected layout {other:?}"),
            }

            let u8_frame = FrameRef::new(
                FrameBuffer::host_owned(u8_bytes),
                FrameDesc {
                    dtype: Dtype::U8,
                    ..desc(w, h, ColorRange::Unit01, layout)
                },
            );
            let f32_frame = FrameRef::new(
                FrameBuffer::host_owned(f32_bytes),
                desc(w, h, ColorRange::Unit01, layout),
            );

            let from_u8 = CpuPreprocess::new().process(&u8_frame).unwrap();
            let from_f32 = CpuPreprocess::new().process(&f32_frame).unwrap();
            assert_eq!(from_u8.y, from_f32.y, "luma {layout:?}");
            let close = |a: &[u8], b: &[u8]| {
                a.iter()
                    .zip(b)
                    .all(|(x, y)| (*x as i32 - *y as i32).abs() <= 1)
            };
            assert!(close(&from_u8.u, &from_f32.u), "Cb {layout:?}");
            assert!(close(&from_u8.v, &from_f32.v), "Cr {layout:?}");
        }
    }

    #[test]
    fn bf16_matches_the_same_values_as_f32() {
        let (w, h) = (4, 4);
        let (r, g, b) = (0.25f32, 0.5f32, 0.75f32);

        let f32_frame = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid(w, h, r, g, b)),
            desc(w, h, ColorRange::Unit01, Layout::Thwc),
        );
        let bf16_frame = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid_bf16(w, h, r, g, b)),
            desc_bf16(w, h, ColorRange::Unit01, Layout::Thwc),
        );

        let from_f32 = CpuPreprocess::new()
            .process(&FrameRef::new(
                FrameBuffer::host_owned(thwc_solid(
                    w,
                    h,
                    as_truncated_f32(r),
                    as_truncated_f32(g),
                    as_truncated_f32(b),
                )),
                desc(w, h, ColorRange::Unit01, Layout::Thwc),
            ))
            .unwrap();
        let from_bf16 = CpuPreprocess::new().process(&bf16_frame).unwrap();
        let direct_f32 = CpuPreprocess::new().process(&f32_frame).unwrap();

        assert_eq!(
            from_bf16.y, from_f32.y,
            "luma must match the truncated f32 reference"
        );
        assert_eq!(from_bf16.u, from_f32.u);
        assert_eq!(from_bf16.v, from_f32.v);
        // Sanity: truncation is lossy, so the reference differs from full f32 by
        // at most one luma level for these values.
        assert!(from_bf16
            .y
            .iter()
            .zip(&direct_f32.y)
            .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1));
    }

    #[test]
    fn bf16_signed_unit_range_is_normalized() {
        // vLLM-Omni streams [-1,1] bf16 tensors: -1 must be black, not mid-gray.
        let (w, h) = (2, 2);
        let fr = FrameRef::new(
            FrameBuffer::host_owned(thwc_solid_bf16(w, h, -1.0, -1.0, -1.0)),
            desc_bf16(w, h, ColorRange::SignedUnit, Layout::Thwc),
        );
        let out = CpuPreprocess::new().process(&fr).unwrap();
        assert!(out.y.iter().all(|&y| y == 0));
    }

    #[test]
    fn bf16_size_mismatch_reports_two_byte_components() {
        let fr = FrameRef::new(
            FrameBuffer::host_owned(vec![0u8; 8]),
            desc_bf16(4, 4, ColorRange::Unit01, Layout::Thwc),
        );
        match CpuPreprocess::new().process(&fr).unwrap_err() {
            PreprocessError::SizeMismatch { expected, .. } => {
                assert_eq!(expected, 4 * 4 * 3 * 2)
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn f16_is_still_rejected() {
        let fr = FrameRef::new(
            FrameBuffer::host_owned(vec![0u8; 2 * 4 * 4 * 3]),
            FrameDesc {
                dtype: Dtype::F16,
                ..desc(4, 4, ColorRange::Unit01, Layout::Thwc)
            },
        );
        assert_eq!(
            CpuPreprocess::new().process(&fr).unwrap_err(),
            PreprocessError::UnsupportedDtype(Dtype::F16)
        );
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use blazar_core::{ColorRange, ColorSpace, FrameBuffer, FrameDesc, FrameRef};

    fn desc(w: u32, h: u32, cs: ColorSpace) -> FrameDesc {
        FrameDesc {
            width: w,
            height: h,
            dtype: Dtype::U8,
            range: ColorRange::FullU8,
            layout: Layout::Thwc,
            colorspace: cs,
            fps: 24.0,
            pts: Pts(0),
        }
    }

    /// Deterministic pseudo-random RGB with structure, so gradients and edges
    /// both appear: a flat frame would let a broken kernel pass.
    fn frame(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 7 + y * 13) % 256) as u8;
                let g = ((x * 3) ^ (y * 5)) as u8;
                let b = ((x + y * 11) % 251) as u8;
                v.extend_from_slice(&[r, g, b]);
            }
        }
        v
    }

    fn max_diff(a: &[u8], b: &[u8]) -> u8 {
        a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
    }

    #[test]
    fn vector_path_matches_scalar_within_one_level() {
        let (w, h) = (257usize, 129usize); // odd dimensions exercise the tails
        for cs in [ColorSpace::AssumedBt709, ColorSpace::AssumedBt601] {
            let d = desc(w as u32, h as u32, cs);
            let bytes = frame(w, h);

            let (sy, su, sv) = convert_thwc::<U8Sample>(&bytes, &d, w, h);
            let (vy, vu, vv) = convert_simd(simd::Kind::ThwcU8, &bytes, &d, w, h);

            assert!(max_diff(&sy, &vy) <= 1, "luma differs by more than 1 level");
            assert!(max_diff(&su, &vu) <= 1, "Cb differs by more than 1 level");
            assert!(max_diff(&sv, &vv) <= 1, "Cr differs by more than 1 level");
        }
    }

    #[test]
    fn vector_path_is_used_and_agrees_on_a_flat_frame() {
        let (w, h) = (64usize, 64usize);
        let d = desc(w as u32, h as u32, ColorSpace::AssumedBt709);
        let bytes = vec![255u8; w * h * 3];

        let (vy, vu, vv) = convert_simd(simd::Kind::ThwcU8, &bytes, &d, w, h);
        assert!(vy.iter().all(|&v| v == 255), "white must be full luma");
        assert!(vu.iter().all(|&v| v == 128), "white chroma must be neutral");
        assert!(vv.iter().all(|&v| v == 128));
    }

    /// The SIMD path is reachable through the stage, not only through the
    /// helper: this is the regression guard for the dispatch.
    #[test]
    fn stage_uses_the_vector_path_for_interleaved_u8() {
        let (w, h) = (300usize, 200usize);
        let d = desc(w as u32, h as u32, ColorSpace::AssumedBt709);
        let bytes = frame(w, h);
        let stage = CpuPreprocess::new();
        let frame = FrameRef {
            desc: d.clone(),
            buffer: FrameBuffer::host_owned(bytes.clone()),
        };
        let out = stage.process(&frame).expect("stage runs");
        let (sy, su, sv) = convert_thwc::<U8Sample>(&bytes, &d, w, h);
        assert!(max_diff(&out.y, &sy) <= 1);
        assert!(max_diff(&out.u, &su) <= 1);
        assert!(max_diff(&out.v, &sv) <= 1);
    }
}

#[cfg(test)]
mod vector_kernels {
    use super::*;
    use blazar_core::{ColorRange, ColorSpace, FrameBuffer, FrameDesc, FrameRef};

    fn desc(w: u32, h: u32, layout: Layout, dtype: Dtype, range: ColorRange, cs: ColorSpace) -> FrameDesc {
        FrameDesc {
            width: w,
            height: h,
            dtype,
            range,
            layout,
            colorspace: cs,
            fps: 24.0,
            pts: Pts(0),
        }
    }

    /// One sample's worth of deterministic, structured content.
    fn units(px: usize, component: usize) -> f32 {
        let i = px * 3 + component;
        (((i * 37 + i / 13) % 200) as f32) / 199.0
    }

    fn build(w: usize, h: usize, layout: Layout, dtype: Dtype, range: ColorRange) -> Vec<u8> {
        let pixels = w * h;
        let order: Vec<usize> = match layout {
            Layout::Thwc => (0..pixels * 3).collect(),
            // Plane-major: the index of the sample at plane-major position i.
            Layout::Bcthw => (0..pixels * 3)
                .map(|i| {
                    let plane = i / pixels;
                    let px = i % pixels;
                    px * 3 + plane
                })
                .collect(),
        };
        let mut out = Vec::with_capacity(pixels * 3 * dtype_bytes(dtype));
        for i in order {
            let value = units(i / 3, i % 3);
            match dtype {
                Dtype::U8 => out.push((value * 255.0).round() as u8),
                Dtype::Bf16 => {
                    let bits = (value.to_bits() >> 16) as u16;
                    out.extend_from_slice(&bits.to_le_bytes());
                }
                Dtype::F32 => {
                    let raw = match range {
                        ColorRange::SignedUnit => value * 2.0 - 1.0,
                        ColorRange::FullU8 => value * 255.0,
                        ColorRange::Unit01 => value,
                    };
                    out.extend_from_slice(&raw.to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        out
    }

    fn dtype_bytes(dtype: Dtype) -> usize {
        match dtype {
            Dtype::U8 => 1,
            Dtype::Bf16 => 2,
            _ => 4,
        }
    }

    fn scalar(desc: &FrameDesc, bytes: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        match (desc.layout, desc.dtype) {
            (Layout::Thwc, Dtype::U8) => convert_thwc::<U8Sample>(bytes, desc, w, h),
            (Layout::Bcthw, Dtype::U8) => convert_bcthw::<U8Sample>(bytes, desc, w, h),
            (Layout::Thwc, Dtype::F32) => convert_thwc::<F32Sample>(bytes, desc, w, h),
            (Layout::Bcthw, Dtype::F32) => convert_bcthw::<F32Sample>(bytes, desc, w, h),
            (Layout::Thwc, Dtype::Bf16) => convert_thwc::<Bf16Sample>(bytes, desc, w, h),
            (Layout::Bcthw, Dtype::Bf16) => convert_bcthw::<Bf16Sample>(bytes, desc, w, h),
            _ => unreachable!(),
        }
    }

    fn max_diff(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len(), "plane length differs");
        a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
    }

    /// Every layout, dtype and range agrees with the scalar reference.
    #[test]
    fn all_kernels_match_the_scalar_reference() {
        // Odd dimensions exercise both the vector tails and the single-column
        // chroma edge.
        let (w, h) = (131usize, 67usize);
        let stage = CpuPreprocess::new();
        for layout in [Layout::Thwc, Layout::Bcthw] {
            for dtype in [Dtype::U8, Dtype::Bf16, Dtype::F32] {
                for range in [ColorRange::Unit01, ColorRange::SignedUnit] {
                    if dtype == Dtype::U8 && range != ColorRange::Unit01 {
                        // A u8 frame ignores the range entirely; covered below.
                        continue;
                    }
                    for cs in [ColorSpace::AssumedBt709, ColorSpace::AssumedBt601] {
                        let d = desc(w as u32, h as u32, layout, dtype, range, cs);
                        let bytes = build(w, h, layout, dtype, range);
                        let (sy, su, sv) = scalar(&d, &bytes, w, h);
                        let frame = FrameRef {
                            desc: d.clone(),
                            buffer: FrameBuffer::host_owned(bytes.clone()),
                        };
                        let out = stage.process(&frame).expect("stage runs");
                        let label = format!("{layout:?}/{dtype:?}/{range:?}/{cs:?}");
                        assert!(max_diff(&out.y, &sy) <= 1, "luma {label}");
                        assert!(max_diff(&out.u, &su) <= 1, "Cb {label}");
                        assert!(max_diff(&out.v, &sv) <= 1, "Cr {label}");
                    }
                }
            }
        }
    }

    /// A u8 frame is code values regardless of the declared range.
    #[test]
    fn u8_ignores_the_declared_range() {
        let (w, h) = (32usize, 16usize);
        let bytes = build(w, h, Layout::Thwc, Dtype::U8, ColorRange::Unit01);
        let a = desc(w as u32, h as u32, Layout::Thwc, Dtype::U8, ColorRange::Unit01, ColorSpace::AssumedBt709);
        let b = FrameDesc { range: ColorRange::FullU8, ..a.clone() };
        let sa = scalar(&a, &bytes, w, h);
        let sb = scalar(&b, &bytes, w, h);
        assert_eq!(sa.0, sb.0);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use blazar_core::{ColorRange, ColorSpace, FrameBuffer, FrameDesc, FrameRef};
    use std::time::Instant;

    /// Times every kernel at 1080p through the stage, which allocates the output
    /// planes on each call. Ignored by default: it is a measurement, not a
    /// correctness check. Run with
    /// `cargo test -p blazar-preprocess --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_timings() {
        let (w, h) = (1920usize, 1080usize);
        let pixels = w * h;
        let stage = CpuPreprocess::new();
        println!("vector kernels available: {}", simd::available());
        let cases = [
            ("thwc_u8", Layout::Thwc, Dtype::U8, 1usize),
            ("bcthw_u8", Layout::Bcthw, Dtype::U8, 1),
            ("thwc_f32", Layout::Thwc, Dtype::F32, 4),
            ("bcthw_f32", Layout::Bcthw, Dtype::F32, 4),
            ("thwc_bf16", Layout::Thwc, Dtype::Bf16, 2),
            ("bcthw_bf16", Layout::Bcthw, Dtype::Bf16, 2),
        ];
        for (name, layout, dtype, size) in cases {
            let mut data = vec![0u8; pixels * 3 * size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i * 37 + i / 13) % 251) as u8;
            }
            if dtype != Dtype::U8 {
                // Build in-range values: for bf16 the top half of an f32 is the
                // encoding, and f32 needs the float itself.
                let samples = data.len() / size;
                for i in 0..samples {
                    let v = ((i * 37 + i / 13) % 1001) as f32 / 1000.0;
                    if dtype == Dtype::Bf16 {
                        let bits = (v.to_bits() >> 16) as u16;
                        data[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
                    } else {
                        data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let d = FrameDesc {
                width: w as u32,
                height: h as u32,
                dtype,
                range: if dtype == Dtype::U8 {
                    ColorRange::FullU8
                } else {
                    ColorRange::Unit01
                },
                layout,
                colorspace: ColorSpace::AssumedBt709,
                fps: 24.0,
                pts: Pts(0),
            };
            let frame = FrameRef {
                desc: d,
                buffer: FrameBuffer::host_owned(data),
            };
            for _ in 0..2 {
                let _ = stage.process(&frame).unwrap();
            }
            let iters = 10;
            let started = Instant::now();
            for _ in 0..iters {
                let out = stage.process(&frame).unwrap();
                std::hint::black_box((out.y.len(), out.u.len(), out.v.len()));
            }
            let per = started.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("{name:>10}: {per:8.3} ms/frame");
        }
    }
}

#[cfg(test)]
mod bench_direct {
    use super::*;
    use blazar_core::{ColorRange, ColorSpace, FrameBuffer, FrameDesc, FrameRef};
    use std::time::Instant;

    const W: usize = 1920;
    const H: usize = 1080;

    struct Case {
        name: &'static str,
        layout: Layout,
        dtype: Dtype,
        size: usize,
        kind: simd::Kind,
    }

    fn cases() -> Vec<Case> {
        vec![
            Case { name: "thwc_u8", layout: Layout::Thwc, dtype: Dtype::U8, size: 1, kind: simd::Kind::ThwcU8 },
            Case { name: "bcthw_u8", layout: Layout::Bcthw, dtype: Dtype::U8, size: 1, kind: simd::Kind::BcthwU8 },
            Case { name: "thwc_f32", layout: Layout::Thwc, dtype: Dtype::F32, size: 4, kind: simd::Kind::ThwcF32 },
            Case { name: "bcthw_f32", layout: Layout::Bcthw, dtype: Dtype::F32, size: 4, kind: simd::Kind::BcthwF32 },
            Case { name: "thwc_bf16", layout: Layout::Thwc, dtype: Dtype::Bf16, size: 2, kind: simd::Kind::ThwcBf16 },
            Case { name: "bcthw_bf16", layout: Layout::Bcthw, dtype: Dtype::Bf16, size: 2, kind: simd::Kind::BcthwBf16 },
        ]
    }

    /// Deterministic content whose values sit inside the declared range.
    fn build(size: usize, dtype: Dtype) -> Vec<u8> {
        let samples = W * H * 3;
        let mut data = vec![0u8; samples * size];
        for i in 0..samples {
            let v = ((i * 37 + i / 13) % 1001) as f32 / 1000.0;
            match dtype {
                Dtype::U8 => data[i] = (v * 255.0) as u8,
                Dtype::Bf16 => {
                    let bits = (v.to_bits() >> 16) as u16;
                    data[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
                }
                _ => data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes()),
            }
        }
        data
    }

    fn desc(case: &Case) -> FrameDesc {
        FrameDesc {
            width: W as u32,
            height: H as u32,
            dtype: case.dtype,
            range: if case.dtype == Dtype::U8 {
                ColorRange::FullU8
            } else {
                ColorRange::Unit01
            },
            layout: case.layout,
            colorspace: ColorSpace::AssumedBt709,
            fps: 24.0,
            pts: Pts(0),
        }
    }

    /// Scalar reference, timed with the same preallocated outputs.
    fn scalar_into(case: &Case, d: &FrameDesc, bytes: &[u8], y: &mut [u8], u: &mut [u8], v: &mut [u8]) {
        let (sy, su, sv) = match (case.layout, case.dtype) {
            (Layout::Thwc, Dtype::U8) => convert_thwc::<U8Sample>(bytes, d, W, H),
            (Layout::Bcthw, Dtype::U8) => convert_bcthw::<U8Sample>(bytes, d, W, H),
            (Layout::Thwc, Dtype::F32) => convert_thwc::<F32Sample>(bytes, d, W, H),
            (Layout::Bcthw, Dtype::F32) => convert_bcthw::<F32Sample>(bytes, d, W, H),
            (Layout::Thwc, Dtype::Bf16) => convert_thwc::<Bf16Sample>(bytes, d, W, H),
            (Layout::Bcthw, Dtype::Bf16) => convert_bcthw::<Bf16Sample>(bytes, d, W, H),
            _ => unreachable!(),
        };
        y.copy_from_slice(&sy);
        u.copy_from_slice(&su);
        v.copy_from_slice(&sv);
    }

    #[test]
    #[ignore]
    fn direct_kernel_timings() {
        let cw = W.div_ceil(2);
        let ch = H.div_ceil(2);
        // Allocated once, outside every clock.
        let mut y = vec![0u8; W * H];
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        let stage = CpuPreprocess::new();
        println!(
            "vector kernels available: {}  ({}x{})",
            simd::available(),
            W,
            H
        );
        println!("{:<11} {:>9} {:>9} {:>9}", "kind", "scalar", "vector", "stage");
        for case in cases() {
            let bytes = build(case.size, case.dtype);
            let d = desc(&case);

            // Warm up every path once so page tables and caches are settled.
            scalar_into(&case, &d, &bytes, &mut y, &mut u, &mut v);
            let used = simd::convert(case.kind, &bytes, &d, W, H, &mut y, &mut u, &mut v);
            let frame = FrameRef {
                desc: d.clone(),
                buffer: FrameBuffer::host_owned(bytes.clone()),
            };
            let _ = stage.process(&frame).unwrap();

            // Three rounds, best of each: a shared machine drifts, and the
            // minimum is the closest thing to the kernel's own cost.
            let iters = 20;
            let rounds = 3;
            let mut best = [f64::MAX; 3];
            for _ in 0..rounds {
                let t = Instant::now();
                for _ in 0..iters {
                    scalar_into(&case, &d, &bytes, &mut y, &mut u, &mut v);
                    std::hint::black_box(&y);
                }
                best[0] = best[0].min(t.elapsed().as_secs_f64() * 1000.0 / iters as f64);

                let t = Instant::now();
                for _ in 0..iters {
                    simd::convert(case.kind, &bytes, &d, W, H, &mut y, &mut u, &mut v);
                    std::hint::black_box(&y);
                }
                best[1] = best[1].min(t.elapsed().as_secs_f64() * 1000.0 / iters as f64);

                // The stage allocates its planes, which is what the pipeline pays.
                let t = Instant::now();
                for _ in 0..iters {
                    let out = stage.process(&frame).unwrap();
                    std::hint::black_box(&out.y);
                }
                best[2] = best[2].min(t.elapsed().as_secs_f64() * 1000.0 / iters as f64);
            }
            let (scalar, vector, stage_ms) = (best[0], best[1], best[2]);

            println!(
                "{:<11} {:>8.3} {:>9.3} {:>9.3}{}",
                case.name,
                scalar,
                vector,
                stage_ms,
                if used { "" } else { "   (no vector path)" }
            );
        }
    }
}
