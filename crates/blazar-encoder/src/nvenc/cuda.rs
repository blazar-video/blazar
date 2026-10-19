//! GPU RGB→NV12 conversion for the NVENC input buffers.
//!
//! The CPU path costs ~3 ms/frame at 1080p. This module compiles a small CUDA
//! kernel with NVRTC at runtime (so the macOS build never needs CUDA headers)
//! and writes NV12 straight into the device buffer that NVENC reads, using the
//! driver's pitch.
//!
//! The kernel must agree with [`blazar_preprocess`]'s CPU conversion: same
//! BT.709/BT.601 matrix, same full-range quantisation, same 2x2 box filter.
//! `tests/nvenc_cuda.rs` compares the two byte-for-byte on hardware.

use std::sync::Arc;

use crate::EncodeError;
use blazar_core::{ColorRange, ColorSpace, Dtype, FrameDesc, Layout};
use cudarc::driver::{DevicePtr, 
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

/// CUDA C source. THWC = 0, BCTHW = 1; f32 = 0, bf16 = 1; ranges 0/1/2.
const KERNEL: &str = r#"
extern "C" __global__ void rgb_to_nv12(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int w, int h, int pitch,
    int layout, int fmt, int range,
    float kr, float kg, float kb)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cw = (w + 1) >> 1;
    int ch = (h + 1) >> 1;
    if (cx >= cw || cy >= ch) return;

    const int pixels = w * h;
    float cb_sum = 0.0f, cr_sum = 0.0f, n = 0.0f;

    for (int dy = 0; dy < 2; ++dy) {
        int sy = cy * 2 + dy;
        if (sy >= h) break;
        for (int dx = 0; dx < 2; ++dx) {
            int sx = cx * 2 + dx;
            if (sx >= w) break;
            int p = sy * w + sx;
            float r, g, b;
            if (fmt == 0) {
                const float* f = (const float*)src;
                if (layout == 0) { r = f[p*3]; g = f[p*3+1]; b = f[p*3+2]; }
                else { r = f[p]; g = f[pixels + p]; b = f[2*pixels + p]; }
            } else {
                const unsigned short* s = (const unsigned short*)src;
                unsigned short vr, vg, vb;
                if (layout == 0) { vr = s[p*3]; vg = s[p*3+1]; vb = s[p*3+2]; }
                else { vr = s[p]; vg = s[pixels + p]; vb = s[2*pixels + p]; }
                r = __uint_as_float(((unsigned)vr) << 16);
                g = __uint_as_float(((unsigned)vg) << 16);
                b = __uint_as_float(((unsigned)vb) << 16);
            }
            if (range == 1) { r = (r + 1.0f) * 0.5f; g = (g + 1.0f) * 0.5f; b = (b + 1.0f) * 0.5f; }
            else if (range == 2) { r *= 1.0f / 255.0f; g *= 1.0f / 255.0f; b *= 1.0f / 255.0f; }
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);

            float y = kr * r + kg * g + kb * b;
            float yq = roundf(y * 255.0f);
            dst[sy * pitch + sx] = (unsigned char)fminf(fmaxf(yq, 0.0f), 255.0f);

            float cb = (b - y) / (2.0f * (1.0f - kb)) + 0.5f;
            float cr = (r - y) / (2.0f * (1.0f - kr)) + 0.5f;
            cb_sum += cb;
            cr_sum += cr;
            n += 1.0f;
        }
    }

    float cb = cb_sum / n;
    float cr = cr_sum / n;
    int uv = pitch * h + cy * pitch + cx * 2;
    dst[uv] = (unsigned char)fminf(fmaxf(roundf(cb * 255.0f), 0.0f), 255.0f);
    dst[uv + 1] = (unsigned char)fminf(fmaxf(roundf(cr * 255.0f), 0.0f), 255.0f);
}
"#;

/// Compiled once per process, in the worker's CUDA context.
pub struct Nv12Converter {
    func: CudaFunction,
}

impl Nv12Converter {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, EncodeError> {
        // Compile to a cubin for this device instead of PTX. PTX is versioned
        // by the toolkit that produced it, and a driver older than the toolkit
        // rejects it with CUDA_ERROR_UNSUPPORTED_PTX_VERSION (toolkit 13.4 vs
        // driver 610.88 on Windows). A cubin for the device's own sm_XY sidesteps
        // that negotiation entirely.
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| EncodeError::Backend(format!("CUDA compute capability: {e:?}")))?;
        let arch = format!("--gpu-architecture=sm_{major}{minor}");
        let cubin = compile_cubin(KERNEL, &arch)?;
        let ptx = cudarc::nvrtc::Ptx::from_binary(cubin);
        let module = ctx
            .load_module(ptx)
            .map_err(|e| EncodeError::Backend(format!("load NV12 kernel: {e}")))?;
        let func = module
            .load_function("rgb_to_nv12")
            .map_err(|e| EncodeError::Backend(format!("find NV12 kernel: {e}")))?;
        Ok(Self { func })
    }

    /// Convert one RGB frame already on the device into `dst` as NV12 at `pitch`.
    #[allow(clippy::too_many_arguments)]
    pub fn convert(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        desc: &FrameDesc,
        pitch: u32,
    ) -> Result<(), EncodeError> {
        let (ptr, guard) = dst.device_ptr(stream);
        drop(guard);
        self.convert_raw(stream, src, ptr, desc, pitch)
    }

    /// Convert into a raw device pointer, e.g. a `cuMemAllocPitch` allocation
    /// that is registered with NVENC for zero-copy input.
    #[allow(clippy::too_many_arguments)]
    pub fn convert_raw(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<u8>,
        dst: u64,
        desc: &FrameDesc,
        pitch: u32,
    ) -> Result<(), EncodeError> {
        if !matches!(desc.dtype, Dtype::F32 | Dtype::Bf16) {
            return Err(EncodeError::Backend(format!(
                "GPU preprocess does not handle {:?}",
                desc.dtype
            )));
        }
        let layout = match desc.layout {
            Layout::Thwc => 0i32,
            Layout::Bcthw => 1i32,
        };
        let fmt = match desc.dtype {
            Dtype::F32 => 0i32,
            _ => 1i32,
        };
        let range = match desc.range {
            ColorRange::Unit01 => 0i32,
            ColorRange::SignedUnit => 1i32,
            ColorRange::FullU8 => 2i32,
        };
        let (kr, kb) = match desc.colorspace {
            ColorSpace::AssumedBt709 => (0.2126f32, 0.0722f32),
            ColorSpace::AssumedBt601 => (0.299f32, 0.114f32),
        };
        let kg = 1.0 - kr - kb;

        let w = desc.width as i32;
        let h = desc.height as i32;
        let pitch = pitch as i32;
        let cw = desc.width.div_ceil(2);
        let ch = desc.height.div_ceil(2);
        let cfg = LaunchConfig {
            grid_dim: (cw.div_ceil(16), ch.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut args = stream.launch_builder(&self.func);
        args.arg(src);
        args.arg(&dst);
        args.arg(&w);
        args.arg(&h);
        args.arg(&pitch);
        args.arg(&layout);
        args.arg(&fmt);
        args.arg(&range);
        args.arg(&kr);
        args.arg(&kg);
        args.arg(&kb);
        unsafe { args.launch(cfg) }
            .map_err(|e| EncodeError::Backend(format!("launch NV12 kernel: {e}")))?;
        Ok(())
    }
}

/// Compile `src` to a cubin for `arch` (e.g. `--gpu-architecture=sm_89`).
///
/// cudarc only wraps `nvrtcGetPTX`, so the cubin fetch goes through the raw
/// NVRTC entry points. The program is destroyed on every path.
fn compile_cubin(src: &str, arch: &str) -> Result<Vec<u8>, EncodeError> {
    use std::ffi::CString;
    let src = CString::new(src).expect("kernel source has no interior NUL");
    let prog = cudarc::nvrtc::result::create_program(&src, None)
        .map_err(|e| EncodeError::Backend(format!("create NVRTC program: {e:?}")))?;
    let result = (|| {
        let options = ["--cubin".to_string(), arch.to_string()];
        cudarc::nvrtc::result::compile_program(prog, &options)
            .map_err(|e| EncodeError::Backend(format!("compile NV12 kernel: {e:?}")))?;
        let mut size = 0usize;
        cudarc::nvrtc::sys::nvrtcGetCUBINSize(prog, &mut size)
            .result()
            .map_err(|e| EncodeError::Backend(format!("cubin size: {e:?}")))?;
        if size == 0 {
            return Err(EncodeError::Backend("NVRTC returned an empty cubin".into()));
        }
        let mut bytes = vec![0u8; size];
        cudarc::nvrtc::sys::nvrtcGetCUBIN(prog, bytes.as_mut_ptr().cast())
            .result()
            .map_err(|e| EncodeError::Backend(format!("cubin fetch: {e:?}")))?;
        Ok(bytes)
    })();
    // SAFETY: `prog` came from create_program and is destroyed exactly once.
    let _ = unsafe { cudarc::nvrtc::result::destroy_program(prog) };
    result
}
