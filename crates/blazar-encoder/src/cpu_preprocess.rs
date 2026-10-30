// SPDX-License-Identifier: Apache-2.0
//! CPU-side conversion stage, usable on any backend.
//!
//! This is the explicit `preprocess="host"` placement for a backend that would
//! otherwise convert on the device. Placement is the caller's decision: they
//! may want the conversion on the CPU because the device is busy, because
//! their build has no GPU preprocessing, or because the frames are already in
//! host memory.

use blazar_core::{EncodedUnit, FrameRef};
use blazar_preprocess::{CpuPreprocess, PreprocessStage, PreprocessedFrame};

use crate::{EncodeError, VideoEncoder};

/// Converts host frames on the CPU before handing them to the inner encoder.
///
/// This is the explicit `preprocess="host"` placement for a backend that would
/// otherwise convert on the device. Placement is a caller's decision, not ours:
/// a caller may want the conversion on the CPU because the device is busy with
/// something else, because a GPU kernel is unavailable in their build, or
/// because the frames are already staged in host memory and a device round trip
/// is not wanted.
///
/// Device tensors are not handled here — the caller that submits one has to
/// download it first, which is what `GpuPreprocessEncoder` exists to avoid.
pub struct CpuPreprocessEncoder {
    inner: Box<dyn VideoEncoder>,
    cpu: CpuPreprocess,
}

impl CpuPreprocessEncoder {
    /// Wrap an encoder so host frames convert on the CPU.
    #[must_use]
    pub fn new(inner: Box<dyn VideoEncoder>) -> Self {
        Self {
            inner,
            cpu: CpuPreprocess::new(),
        }
    }
}

impl VideoEncoder for CpuPreprocessEncoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.inner.encode(frame)
    }

    fn encode_frame(&mut self, frame: FrameRef) -> Result<Vec<EncodedUnit>, EncodeError> {
        let _convert = blazar_trace::span(blazar_trace::Stage::Convert);
        let converted = self
            .cpu
            .process(&frame)
            .map_err(|e| EncodeError::Backend(format!("cpu preprocess: {e}")))?;
        self.inner.encode(&converted)
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.inner.flush()
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.inner.finish()
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.inner.codec_config()
    }

    fn request_keyframe(&mut self) {
        self.inner.request_keyframe();
    }
}
