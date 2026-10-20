//! GPU RGB -> NV12 conversion as a reusable stage.
//!
//! This is the first piece of the pipeline split: the conversion no longer
//! belongs to a backend. A CPU encoder consumes the [`PreprocessedFrame`] this
//! produces, so the pixel work stays on the GPU and only NV12 (1.5 B/px) crosses
//! the bus, instead of the CPU redoing the conversion from RGB.
//!
//! The device half (`Nv12Frame::Device`) is what NVENC already reads; this
//! module implements the host half, which needs a download.

use std::sync::Arc;

use blazar_core::{Dtype, FrameDesc, Pts};
use blazar_preprocess::PreprocessedFrame;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use crate::nvenc::cuda::Nv12Converter;
use crate::EncodeError;

/// Reusable GPU conversion stage.
///
/// Buffers grow on demand and are reused across frames, so a stream pays the
/// allocation once. The CUDA context is the process-wide one shared with NVENC.
pub struct GpuPreprocess {
    stream: Arc<CudaStream>,
    converter: Nv12Converter,
    staging: CudaSlice<u8>,
    device: CudaSlice<u8>,
    dst_ptrs: CudaSlice<u64>,
    pitch: u32,
    width: u32,
    height: u32,
}

impl GpuPreprocess {
    /// Create the stage on device 0. Fails if CUDA or NVRTC is unavailable.
    pub fn new() -> Result<Self, EncodeError> {
        let ctx = CudaContext::new(0)
            .map_err(|e| EncodeError::Backend(format!("CUDA context on device 0: {e:?}")))?;
        let stream = ctx.default_stream();
        let converter = Nv12Converter::new(&ctx)?;
        let staging = stream
            .alloc_zeros(1)
            .map_err(|e| EncodeError::Backend(format!("allocate staging: {e}")))?;
        let device = stream
            .alloc_zeros(1)
            .map_err(|e| EncodeError::Backend(format!("allocate nv12: {e}")))?;
        let dst_ptrs = stream
            .alloc_zeros(1)
            .map_err(|e| EncodeError::Backend(format!("allocate pointer array: {e}")))?;
        Ok(Self {
            stream,
            converter,
            staging,
            device,
            dst_ptrs,
            pitch: 0,
            width: 0,
            height: 0,
        })
    }

    /// Bytes per pixel row of the internal NV12 buffer (>= width).
    #[must_use]
    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Convert one frame on the GPU and return packed NV12 planes on the host.
    ///
    /// The caller supplies the sample layout exactly as `CpuPreprocess` expects
    /// it; the two produce the same bytes (within 1 level) and a test asserts it.
    pub fn convert(
        &mut self,
        bytes: &[u8],
        desc: &FrameDesc,
    ) -> Result<PreprocessedFrame, EncodeError> {
        if !matches!(desc.dtype, Dtype::F32 | Dtype::Bf16) {
            return Err(EncodeError::Backend(format!(
                "GPU preprocess does not handle {:?}",
                desc.dtype
            )));
        }
        self.resize(desc)?;

        if bytes.len() > self.staging.len() {
            return Err(EncodeError::Backend(format!(
                "frame is {} bytes, staging is {}",
                bytes.len(),
                self.staging.len()
            )));
        }
        self.stream
            .memcpy_htod(bytes, &mut self.staging.slice_mut(0..bytes.len()))
            .map_err(|e| EncodeError::Backend(format!("upload frame: {e}")))?;

        let (ptr, guard) = self.device.device_ptr(&self.stream);
        drop(guard);
        self.stream
            .memcpy_htod(&[ptr], &mut self.dst_ptrs.slice_mut(0..1))
            .map_err(|e| EncodeError::Backend(format!("upload dst pointer: {e}")))?;
        self.converter
            .convert_batch(&self.stream, &self.staging, &self.dst_ptrs, desc, 1, self.pitch)?;
        let nv12 = self
            .stream
            .clone_dtoh(&self.device)
            .map_err(|e| EncodeError::Backend(format!("download nv12: {e}")))?;

        Ok(depitch(&nv12, desc.width, desc.height, self.pitch, desc.pts))
    }

    /// (Re)allocate buffers when the frame geometry changes.
    fn resize(&mut self, desc: &FrameDesc) -> Result<(), EncodeError> {
        if self.width == desc.width && self.height == desc.height && self.pitch != 0 {
            return Ok(());
        }
        // A pitched NV12 buffer: luma rows plus half as many chroma rows.
        let pitch = desc.width.div_ceil(64) * 64;
        let rows = desc.height + desc.height.div_ceil(2);
        let nv12_len = pitch as usize * rows as usize;
        let rgb_len = (desc.width as usize) * (desc.height as usize) * 3
            * if matches!(desc.dtype, Dtype::F32) { 4 } else { 2 };
        if nv12_len > self.device.len() {
            self.device = self
                .stream
                .alloc_zeros(nv12_len)
                .map_err(|e| EncodeError::Backend(format!("allocate nv12: {e}")))?;
        }
        if rgb_len > self.staging.len() {
            self.staging = self
                .stream
                .alloc_zeros(rgb_len)
                .map_err(|e| EncodeError::Backend(format!("allocate staging: {e}")))?;
        }
        self.pitch = pitch;
        self.width = desc.width;
        self.height = desc.height;
        Ok(())
    }
}

/// Strip the pitch from a downloaded NV12 buffer into packed y/u/v planes.
fn depitch(nv12: &[u8], width: u32, height: u32, pitch: u32, pts: Pts) -> PreprocessedFrame {
    let (w, h, pitch) = (width as usize, height as usize, pitch as usize);
    let chroma_rows = h.div_ceil(2);
    let chroma_cols = w.div_ceil(2);
    let mut y = Vec::with_capacity(w * h);
    for row in 0..h {
        let start = row * pitch;
        y.extend_from_slice(&nv12[start..start + w]);
    }
    let chroma_base = pitch * h;
    let mut u = Vec::with_capacity(chroma_cols * chroma_rows);
    let mut v = Vec::with_capacity(chroma_cols * chroma_rows);
    for row in 0..chroma_rows {
        let start = chroma_base + row * pitch;
        let line = &nv12[start..start + chroma_cols * 2];
        for pair in line.chunks_exact(2) {
            u.push(pair[0]);
            v.push(pair[1]);
        }
    }
    PreprocessedFrame {
        width,
        height,
        y,
        u,
        v,
        pts,
    }
}
