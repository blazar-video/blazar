//! Planar YUV 4:2:0 → NV12, with explicit row pitch.
//!
//! The CPU preprocess stage produces three separate planes (`y`, `u`, `v`),
//! which is the natural output of per-pixel conversion. NVENC wants NV12: a
//! luma plane followed by interleaved Cb/Cr, both addressed with a row stride
//! (`pitch`) that is not necessarily the frame width.
//!
//! `BufferLock::write` copies contiguously and ignores the pitch, so this is
//! where the padding happens. The caller passes the pitch reported by the
//! driver and gets a buffer whose size exactly matches `pitch`-strided NV12.

use blazar_preprocess::PreprocessedFrame;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Nv12Error {
    #[error("4:2:0 needs even dimensions, got {width}x{height}")]
    OddDimensions { width: u32, height: u32 },
    #[error("pitch {pitch} is smaller than the frame width {width}")]
    PitchTooSmall { pitch: u32, width: u32 },
    #[error("pitch {0} must be even to address interleaved chroma pairs")]
    OddPitch(u32),
    #[error("luma plane has {got} bytes, expected {expected}")]
    LumaSize { got: usize, expected: usize },
    #[error("chroma plane has {got} bytes, expected {expected}")]
    ChromaSize { got: usize, expected: usize },
}

/// Interleave a planar 4:2:0 frame into NV12, padding every row to `pitch`.
///
/// Padding bytes are zero; NVENC only reads `width` (luma) or `width` (chroma
/// pairs) bytes per row, so their content does not matter.
pub fn to_nv12(frame: &PreprocessedFrame, pitch: u32) -> Result<Vec<u8>, Nv12Error> {
    let width = frame.width;
    let height = frame.height;

    if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(Nv12Error::OddDimensions { width, height });
    }
    if pitch < width {
        return Err(Nv12Error::PitchTooSmall { pitch, width });
    }
    if !pitch.is_multiple_of(2) {
        return Err(Nv12Error::OddPitch(pitch));
    }

    let luma_len = (width as usize) * (height as usize);
    if frame.y.len() != luma_len {
        return Err(Nv12Error::LumaSize {
            got: frame.y.len(),
            expected: luma_len,
        });
    }
    let chroma_width = width as usize / 2;
    let chroma_height = height as usize / 2;
    let chroma_len = chroma_width * chroma_height;
    if frame.u.len() != chroma_len {
        return Err(Nv12Error::ChromaSize {
            got: frame.u.len(),
            expected: chroma_len,
        });
    }
    if frame.v.len() != chroma_len {
        return Err(Nv12Error::ChromaSize {
            got: frame.v.len(),
            expected: chroma_len,
        });
    }

    let pitch = pitch as usize;
    let rows = height as usize;
    let mut out = vec![0u8; pitch * rows + pitch * (rows / 2)];

    // Luma: one source row into one pitched destination row.
    for row in 0..rows {
        let src = row * width as usize;
        let dst = row * pitch;
        out[dst..dst + width as usize].copy_from_slice(&frame.y[src..src + width as usize]);
    }

    // Chroma: interleave U and V per sample pair.
    let uv_base = pitch * rows;
    for row in 0..chroma_height {
        let src = row * chroma_width;
        let dst = uv_base + row * pitch;
        for col in 0..chroma_width {
            out[dst + col * 2] = frame.u[src + col];
            out[dst + col * 2 + 1] = frame.v[src + col];
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::Pts;

    fn frame(width: u32, height: u32) -> PreprocessedFrame {
        let y: Vec<u8> = (0..width * height).map(|v| v as u8).collect();
        let chroma = (width / 2) * (height / 2);
        let u: Vec<u8> = (0..chroma).map(|v| 100 + v as u8).collect();
        let v: Vec<u8> = (0..chroma).map(|v| 200 - v as u8).collect();
        PreprocessedFrame {
            width,
            height,
            y,
            u,
            v,
            pts: Pts(0),
        }
    }

    #[test]
    fn interleaves_chroma_and_keeps_luma_in_place() {
        let f = frame(4, 2);
        let nv12 = to_nv12(&f, 4).unwrap();
        assert_eq!(nv12.len(), 4 * 2 + 4);
        assert_eq!(&nv12[0..4], &[0, 1, 2, 3]);
        assert_eq!(&nv12[4..8], &[4, 5, 6, 7]);
        // One chroma row: (U0,V0, U1,V1)
        assert_eq!(&nv12[8..12], &[100, 200, 101, 199]);
    }

    #[test]
    fn pads_each_row_to_the_driver_pitch() {
        let f = frame(4, 2);
        let nv12 = to_nv12(&f, 8).unwrap();
        assert_eq!(nv12.len(), 8 * 2 + 8);
        assert_eq!(&nv12[0..8], &[0, 1, 2, 3, 0, 0, 0, 0]);
        assert_eq!(&nv12[8..16], &[4, 5, 6, 7, 0, 0, 0, 0]);
        assert_eq!(&nv12[16..24], &[100, 200, 101, 199, 0, 0, 0, 0]);
    }

    #[test]
    fn rejects_odd_dimensions() {
        assert_eq!(
            to_nv12(&frame(5, 2), 8).unwrap_err(),
            Nv12Error::OddDimensions {
                width: 5,
                height: 2
            }
        );
    }

    #[test]
    fn rejects_a_pitch_smaller_than_the_width() {
        assert_eq!(
            to_nv12(&frame(4, 2), 2).unwrap_err(),
            Nv12Error::PitchTooSmall { pitch: 2, width: 4 }
        );
    }

    #[test]
    fn rejects_an_odd_pitch() {
        assert_eq!(
            to_nv12(&frame(4, 2), 5).unwrap_err(),
            Nv12Error::OddPitch(5)
        );
    }

    #[test]
    fn rejects_plane_sizes_that_do_not_match_the_dimensions() {
        let mut f = frame(4, 2);
        f.y.push(0);
        assert!(matches!(to_nv12(&f, 4), Err(Nv12Error::LumaSize { .. })));

        let mut f = frame(4, 2);
        f.v.pop();
        assert!(matches!(to_nv12(&f, 4), Err(Nv12Error::ChromaSize { .. })));
    }
}
