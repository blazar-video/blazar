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

    /// Convert `frames` frames that already live on the GPU, returning packed
    /// NV12 for each. Nothing is uploaded: the kernel reads the caller's
    /// memory, which is what a model's VAE output looks like.
    pub fn convert_device_frames(
        &mut self,
        ptr: u64,
        desc: &FrameDesc,
        frames: u32,
    ) -> Result<Vec<PreprocessedFrame>, EncodeError> {
        if frames == 0 {
            return Ok(Vec::new());
        }
        if !matches!(desc.dtype, Dtype::F32 | Dtype::Bf16) {
            return Err(EncodeError::Backend(format!(
                "GPU preprocess does not handle {:?}",
                desc.dtype
            )));
        }
        self.resize(desc)?;
        let elem = if matches!(desc.dtype, Dtype::F32) { 4 } else { 2 };
        let pixels = desc.width as u64 * desc.height as u64;
        // Device layout: (C,T,H,W) advances one plane per frame; THWC advances
        // a whole frame.
        let step = match desc.layout {
            blazar_core::Layout::Thwc => pixels * 3 * elem,
            blazar_core::Layout::Bcthw => pixels * elem,
        };

        let (device_ptr, guard) = self.device.device_ptr(&self.stream);
        drop(guard);
        let mut out = Vec::with_capacity(frames as usize);
        for index in 0..frames {
            self.stream
                .memcpy_htod(&[device_ptr], &mut self.dst_ptrs.slice_mut(0..1))
                .map_err(|e| EncodeError::Backend(format!("upload dst pointer: {e}")))?;
            self.converter.convert_batch_ptr(
                &self.stream,
                ptr + index as u64 * step,
                &self.dst_ptrs,
                desc,
                1,
                frames,
                false,
                self.pitch,
            )?;
            let nv12 = self
                .stream
                .clone_dtoh(&self.device)
                .map_err(|e| EncodeError::Backend(format!("download nv12: {e}")))?;
            out.push(depitch(&nv12, desc.width, desc.height, self.pitch, desc.pts));
        }
        Ok(out)
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

/// A CPU encoder fed by the GPU preprocess stage.
///
/// Host frames keep the CPU preprocess: at 1080p the scalar pass costs ~5 ms,
/// while a pageable upload of f32 RGB costs ~21 ms. Device frames are the case
/// this exists for: the model already has the pixels on the GPU, so only NV12
/// comes back over the bus.
pub struct GpuPreprocessEncoder {
    preprocess: GpuPreprocess,
    inner: Box<dyn crate::VideoEncoder>,
}

impl GpuPreprocessEncoder {
    pub fn new(inner: Box<dyn crate::VideoEncoder>) -> Result<Self, EncodeError> {
        Ok(Self {
            preprocess: GpuPreprocess::new()?,
            inner,
        })
    }
}

impl crate::VideoEncoder for GpuPreprocessEncoder {
    fn encode(
        &mut self,
        frame: &PreprocessedFrame,
    ) -> Result<Vec<blazar_core::EncodedUnit>, EncodeError> {
        self.inner.encode(frame)
    }

    fn encode_frame(
        &mut self,
        frame: blazar_core::FrameRef,
    ) -> Result<Vec<blazar_core::EncodedUnit>, EncodeError> {
        // Host frames stay on the CPU. End to end this is faster than the GPU
        // round trip (6.78 vs 7.98 ms/frame at 1080p with identical x264
        // settings). A microbenchmark that clones the frame first will claim
        // otherwise; the pipeline does not clone.
        self.inner.encode_frame(frame)
    }

    fn encode_device_batch(
        &mut self,
        ptr: u64,
        desc: FrameDesc,
        frames: u32,
        keep_alive: Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<Vec<blazar_core::EncodedUnit>, EncodeError> {
        let mut nv12 = self.preprocess.convert_device_frames(ptr, &desc, frames)?;
        // One descriptor covers the whole tensor, so the per-frame timestamps
        // have to be derived here. Without this every frame carries the first
        // frame's PTS and the encoder sees a non-increasing timeline.
        let step = (90_000.0 / desc.fps.max(1.0)).round() as u64;
        for (index, frame) in nv12.iter_mut().enumerate() {
            frame.pts = blazar_core::Pts(desc.pts.0 + index as u64 * step);
        }
        let mut units = Vec::new();
        for frame in &nv12 {
            units.extend(self.inner.encode(frame)?);
        }
        // The caller's allocation is no longer read at this point.
        drop(keep_alive);
        Ok(units)
    }

    fn flush(&mut self) -> Result<Vec<blazar_core::EncodedUnit>, EncodeError> {
        self.inner.flush()
    }

    fn finish(&mut self) -> Result<Vec<blazar_core::EncodedUnit>, EncodeError> {
        self.inner.finish()
    }

    fn request_keyframe(&mut self) {
        self.inner.request_keyframe()
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.inner.codec_config()
    }
}
