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

    /// RGB (each in `[0,1]`) → full-range 8-bit YUV using the BT.709 or BT.601
    /// matrix. Full-range (0..255) keeps Phase 1 simple; studio-range swing is a
    /// later refinement tied to the colorspace signalling the encoder emits.
    #[inline]
    fn rgb_to_yuv(r: f32, g: f32, b: f32, cs: ColorSpace) -> (f32, f32, f32) {
        let (kr, kb) = match cs {
            ColorSpace::AssumedBt709 => (0.2126_f32, 0.0722_f32),
            ColorSpace::AssumedBt601 => (0.299_f32, 0.114_f32),
        };
        let kg = 1.0 - kr - kb;
        let y = kr * r + kg * g + kb * b;
        // Cb/Cr in [-0.5, 0.5], then shifted to [0,1] around 0.5.
        let cb = (b - y) / (2.0 * (1.0 - kb));
        let cr = (r - y) / (2.0 * (1.0 - kr));
        (y, cb + 0.5, cr + 0.5)
    }

    #[inline]
    fn q8(v: f32) -> u8 {
        (v * 255.0).round().clamp(0.0, 255.0) as u8
    }
}

impl PreprocessStage for CpuPreprocess {
    fn process(&self, frame: &FrameRef) -> Result<PreprocessedFrame, PreprocessError> {
        let desc = &frame.desc;

        // Phase 1 only handles host F32 RGB. Device buffers are Phase 2.
        if frame.buffer.is_device() {
            return Err(PreprocessError::DeviceNotSupported);
        }
        // Phase 1 handles host f32 (both seams' numpy) and bf16 (vLLM-Omni's
        // native VAE dtype). f16 is uncommon at these seams; reject it loudly
        // rather than guess. The decode is exact: bf16 is the top half of f32.
        let component_bytes = match desc.dtype {
            Dtype::F32 => 4,
            Dtype::Bf16 => 2,
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

        // Interpret the bytes as RGB components of `desc.dtype`. Phase 1 assumes
        // a single frame with the channel order implied by `layout`: THWC packs
        // each pixel as [r,g,b]; BCTHW is channel-planar. Only THWC (xDiT) is
        // exercised in Phase 1, but both are decoded so the vLLM-Omni
        // host-fallback works.
        //
        // `component` is an index into the component array, not a byte offset,
        // so the same closure serves both dtypes.
        let read = |component: usize| -> f32 {
            let off = component * component_bytes;
            match desc.dtype {
                Dtype::F32 => {
                    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
                }
                Dtype::Bf16 => {
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    // Truncation, not rounding: the payload has no low bits.
                    f32::from_bits((bits as u32) << 16)
                }
                Dtype::F16 | Dtype::U8 => unreachable!("dtype checked above"),
            }
        };
        let read_rgb = |px: usize| -> (f32, f32, f32) {
            match desc.layout {
                Layout::Thwc => {
                    let base = px * 3;
                    (read(base), read(base + 1), read(base + 2))
                }
                Layout::Bcthw => {
                    // planar: R plane, then G, then B.
                    (read(px), read(pixels + px), read(2 * pixels + px))
                }
            }
        };

        let mut y_plane = vec![0u8; pixels];
        // Store full-res chroma first, then box-average down to 4:2:0.
        let mut cb_full = vec![0f32; pixels];
        let mut cr_full = vec![0f32; pixels];

        for px in 0..pixels {
            let (r, g, b) = read_rgb(px);
            let r = Self::to_unit(r, desc.range);
            let g = Self::to_unit(g, desc.range);
            let b = Self::to_unit(b, desc.range);
            let (y, cb, cr) = Self::rgb_to_yuv(r, g, b, desc.colorspace);
            y_plane[px] = Self::q8(y);
            cb_full[px] = cb;
            cr_full[px] = cr;
        }

        // 4:2:0 subsample: average each 2x2 block (edge blocks clamp).
        let cw = (desc.width).div_ceil(2) as usize;
        let ch = (desc.height).div_ceil(2) as usize;
        let mut u_plane = vec![0u8; cw * ch];
        let mut v_plane = vec![0u8; cw * ch];

        for cy in 0..ch {
            for cx in 0..cw {
                let mut cb_sum = 0f32;
                let mut cr_sum = 0f32;
                let mut n = 0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = cx * 2 + dx;
                        let sy = cy * 2 + dy;
                        if sx < w && sy < h {
                            let idx = sy * w + sx;
                            cb_sum += cb_full[idx];
                            cr_sum += cr_full[idx];
                            n += 1.0;
                        }
                    }
                }
                u_plane[cy * cw + cx] = Self::q8(cb_sum / n);
                v_plane[cy * cw + cx] = Self::q8(cr_sum / n);
            }
        }

        Ok(PreprocessedFrame {
            width: desc.width,
            height: desc.height,
            y: y_plane,
            u: u_plane,
            v: v_plane,
            pts: desc.pts,
        })
    }
}

/// Expected RGB-f32 byte length for a frame of `w*h`. Exposed for adapters that
/// build host frames.
pub fn rgb_f32_len(w: u32, h: u32) -> usize {
    (w as usize) * (h as usize) * 3 * core::mem::size_of::<f32>()
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
