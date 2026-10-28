// SPDX-License-Identifier: Apache-2.0
//! Frame handoff contract — the shape crossing the ingest boundary.
//!
//! A [`FrameRef`] is a tagged union over residency (architecture §7.1): a
//! device-resident GPU buffer (the primary path, the one vLLM-Omni exposes) or
//! a host-resident CPU buffer (xDiT's public seam, and the Phase-1 path). The
//! accompanying [`FrameDesc`] declares everything the preprocess stage needs to
//! interpret the bytes — crucially *color range* and *layout*, which differ
//! between the two runtimes:
//!
//! | Runtime    | Residency | Range    | Layout | (verified on mosk) |
//! |------------|-----------|----------|--------|--------------------|
//! | xDiT       | Host      | `[0,1]`  | THWC   | numpy float32      |
//! | vLLM-Omni  | Device    | `[-1,1]` | BCTHW  | torch float        |
//!
//! Colorspace is never tagged by either runtime, so it is an *assumed*
//! descriptor Blazar fills in (lean BT.709) and documents.

use serde::{Deserialize, Serialize};

use crate::timebase::Pts;

/// Pixel component element type as handed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    F32,
    F16,
    /// bf16 — vLLM-Omni VAE output is commonly bf16.
    Bf16,
    U8,
}

/// Numeric range the float pixels occupy. The preprocess stage normalizes to
/// video range from whichever of these the runtime produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorRange {
    /// `[0, 1]` normalized float — xDiT / diffusers `output_type="np"`.
    Unit01,
    /// `[-1, 1]` normalized float — raw VAE output, vLLM-Omni streaming decode.
    SignedUnit,
    /// Full-range 8-bit `[0, 255]`.
    FullU8,
}

/// Memory layout of the frame tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    /// Time, Height, Width, Channels (channels-last) — xDiT numpy.
    Thwc,
    /// Batch, Channels, Time, Height, Width (channels-first) — vLLM-Omni torch.
    Bcthw,
}

/// Color primaries / matrix. Never provided by the runtimes; Blazar assumes
/// and records which assumption it made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    /// Assumed BT.709 (HD) — Blazar's default assumption for generative HD.
    AssumedBt709,
    /// Assumed BT.601 (SD).
    AssumedBt601,
}

/// Descriptor mandatory on every frame, both residencies (architecture §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameDesc {
    pub width: u32,
    pub height: u32,
    pub dtype: Dtype,
    pub range: ColorRange,
    pub layout: Layout,
    pub colorspace: ColorSpace,
    /// Frames per second (the only timing signal either runtime provides).
    pub fps: f64,
    /// Synthesized presentation timestamp on the shared timebase.
    pub pts: Pts,
}

/// Opaque handle to a CUDA stream the producer used. Phase 1 never populates a
/// real one; it is here so the `Device` variant shape is final. Replaced by a
/// real `cust`/`cudarc` handle when the device path lands (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream(pub u64);

/// Opaque handle to a CUDA event the consumer waits on before reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaEvent(pub u64);

/// A CUDA IPC handle, present only in the out-of-process topology.
#[derive(Debug, Clone)]
pub struct IpcHandle(pub Vec<u8>);

/// Who owns the host bytes, so Blazar knows when it may free / must copy.
#[derive(Debug)]
pub enum BufferOwner {
    /// Blazar owns this allocation and will drop it.
    Owned(Vec<u8>),
    /// Borrowed from the caller for the duration of the submit call only.
    Borrowed,
}

/// The residency-tagged buffer. GPU is the primary variant; Host is a
/// first-class fallback (and the Phase-1 path), not a second code path.
pub enum FrameBuffer {
    /// PRIMARY: zero-copy GPU. `dlpack` carries device ptr/dtype/shape/strides.
    /// Phase 1 does not construct this variant; it exists so the contract is
    /// final.
    Device {
        /// Raw DLPack capsule pointer (opaque until the device path lands).
        dlpack: *mut core::ffi::c_void,
        stream: CudaStream,
        ready: CudaEvent,
        /// `Some` only in the out-of-process topology.
        ipc: Option<IpcHandle>,
    },
    /// DEGRADED / Phase-1: runtime already copied to host.
    Host {
        ptr: *const u8,
        len: usize,
        owner: BufferOwner,
    },
}

impl FrameBuffer {
    /// Construct a host buffer that owns its bytes (the common Phase-1 case:
    /// an adapter copies numpy into a Rust-owned Vec).
    pub fn host_owned(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        // Store the ptr for the zero-interpretation view; `owner` keeps it alive.
        // We read the ptr back from the owner to avoid a dangling raw pointer.
        let owner = BufferOwner::Owned(bytes);
        let ptr = match &owner {
            BufferOwner::Owned(v) => v.as_ptr(),
            BufferOwner::Borrowed => core::ptr::null(),
        };
        FrameBuffer::Host { ptr, len, owner }
    }

    pub fn is_device(&self) -> bool {
        matches!(self, FrameBuffer::Device { .. })
    }

    pub fn is_host(&self) -> bool {
        matches!(self, FrameBuffer::Host { .. })
    }

    /// Host bytes as a slice, or `None` if this is a device buffer.
    pub fn host_bytes(&self) -> Option<&[u8]> {
        match self {
            FrameBuffer::Host {
                owner: BufferOwner::Owned(v),
                ..
            } => Some(v.as_slice()),
            // Borrowed host bytes: reconstruct from ptr/len. Safe only while the
            // caller's buffer is alive, which the submit contract guarantees.
            FrameBuffer::Host {
                ptr,
                len,
                owner: BufferOwner::Borrowed,
            } => {
                if ptr.is_null() {
                    None
                } else {
                    // SAFETY: ptr/len describe a valid host region for the
                    // lifetime of this FrameBuffer per the submit contract.
                    Some(unsafe { core::slice::from_raw_parts(*ptr, *len) })
                }
            }
            FrameBuffer::Device { .. } => None,
        }
    }
}

/// A frame reference crossing the ingest boundary: buffer + descriptor.
pub struct FrameRef {
    pub buffer: FrameBuffer,
    pub desc: FrameDesc,
}

impl FrameRef {
    pub fn new(buffer: FrameBuffer, desc: FrameDesc) -> Self {
        Self { buffer, desc }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timebase::{Pts, Timebase};

    fn host_desc() -> FrameDesc {
        FrameDesc {
            width: 832,
            height: 480,
            dtype: Dtype::F32,
            range: ColorRange::Unit01,
            layout: Layout::Thwc,
            colorspace: ColorSpace::AssumedBt709,
            fps: 24.0,
            pts: Pts::from_frame_index(0, 24.0, Timebase::MPEG_90K),
        }
    }

    #[test]
    fn host_owned_roundtrips_bytes() {
        let bytes = vec![1u8, 2, 3, 4];
        let fb = FrameBuffer::host_owned(bytes.clone());
        assert!(fb.is_host());
        assert!(!fb.is_device());
        assert_eq!(fb.host_bytes(), Some(bytes.as_slice()));
    }

    #[test]
    fn frame_ref_carries_desc() {
        let fb = FrameBuffer::host_owned(vec![0u8; 16]);
        let fr = FrameRef::new(fb, host_desc());
        assert_eq!(fr.desc.width, 832);
        assert_eq!(fr.desc.range, ColorRange::Unit01);
        assert_eq!(fr.desc.layout, Layout::Thwc);
    }

    #[test]
    fn xdit_and_omni_descs_differ_in_range_and_layout() {
        // Encodes the investigation finding: the two runtimes hand over
        // different normalized conventions. The descriptor must distinguish them.
        let xdit = host_desc();
        let omni = FrameDesc {
            range: ColorRange::SignedUnit,
            layout: Layout::Bcthw,
            dtype: Dtype::Bf16,
            ..host_desc()
        };
        assert_ne!(xdit.range, omni.range);
        assert_ne!(xdit.layout, omni.layout);
    }
}
