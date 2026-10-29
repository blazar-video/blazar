// SPDX-License-Identifier: Apache-2.0
//! The driver-facing half of the NVENC backend.
//!
//! Everything here runs on a dedicated thread. That is not incidental:
//!
//! * `Encoder::start_session` consumes the `Encoder`, and `Buffer`/`Bitstream`
//!   borrow the `Encoder` stored inside `Session`, so a struct holding both is
//!   self-referential. Locals inside one function avoid that entirely.
//! * CUDA contexts are thread-affine; creating the shared context and binding
//!   it on the thread that uses it is correct by construction.
//! * NVENC's output pipeline is asynchronous (`NeedMoreInput`), so the natural
//!   shape is a worker owning the session, one input buffer, and the bitstream
//!   pool, with the `VideoEncoder` impl as a request/response handle.
//!
//! The handle is synchronous: each call sends one command and waits for the
//! reply. A channel round trip costs microseconds against a millisecond-scale
//! encode.

use std::sync::{mpsc, Arc, LazyLock};
use std::thread;
use std::time::Duration;

use blazar_core::{
    BufferOwner, EncodedUnit, FrameBuffer, FrameDesc, FrameRef, Pts, Timebase, TrackId, VideoCodec,
};
use blazar_preprocess::PreprocessedFrame;
use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream};
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        GUID, NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_CODEC_H264_GUID,
        NV_ENC_CODEC_HEVC_GUID, NV_ENC_CONFIG, NV_ENC_INPUT_RESOURCE_TYPE, NV_ENC_PARAMS_RC_MODE,
        NV_ENC_PIC_FLAGS, NV_ENC_PIC_TYPE, NV_ENC_PRESET_P4_GUID, NV_ENC_QP, NV_ENC_TUNING_INFO,
        NV_ENC_VUI_COLOR_PRIMARIES, NV_ENC_VUI_MATRIX_COEFFS, NV_ENC_VUI_TRANSFER_CHARACTERISTIC,
    },
    Bitstream, EncodePictureParams, Encoder, EncoderInitParams, EncoderInput, ErrorKind,
    RegisteredResource, Session,
};

use super::config::{NvencParams, NvencRateControl};
use super::cuda::Nv12Converter;
use super::keyframe::KeyframeSchedule;
use super::nv12::to_nv12;
use super::pool::SlotPool;
use crate::av1;
use crate::config::EncoderConfig;
use crate::nal::{self, AnnexB};
use crate::{EncodeError, VideoEncoder};

/// Output slots, and therefore the largest batch converted in one launch.
///
/// Every slot owns a pitched NV12 buffer that stays registered with NVENC, so
/// this is VRAM: roughly `3 * width * height` bytes per slot (1.6 MB at 480p,
/// 3.1 MB at 1080p). 32 allows a 30-frame chunk in one launch; the bitstream
/// side is only a few KiB per slot.
const POOL_CAPACITY: usize = 32;

/// Upper bound on the RGB staging one launch may claim. Staging is 24 bytes per
/// pixel per frame (f32x3), so 30 frames at 1080p would otherwise be ~750 MiB.
/// At 480p this still allows the full pool; at 1080p a chunk is ~10 frames.
const BATCH_STAGING_BUDGET: usize = 256 << 20;
/// `EncoderBusy` is recoverable; wait briefly and retry a bounded number of times.
const BUSY_RETRIES: u32 = 50;
const BUSY_SLEEP: Duration = Duration::from_millis(2);

/// Process-wide CUDA context, created once on the first NVENC worker thread.
/// CUDA contexts are reference-counted handles; each worker binds it to itself
/// before initializing NVENC. A failed creation is cached as an error so a
/// broken driver does not get retried per stream.
static CUDA_CONTEXT: LazyLock<Result<Arc<CudaContext>, String>> =
    LazyLock::new(|| CudaContext::new(0).map_err(|e| format!("CUDA context on device 0: {e:?}")));

/// One encoded frame plus any newly discovered codec configuration.
struct EncodeOutput {
    units: Vec<EncodedUnit>,
    codec_config: Option<Vec<u8>>,
}

enum Command {
    Encode {
        frame: PreprocessedFrame,
        force_keyframe: bool,
        reply: mpsc::Sender<Result<EncodeOutput, EncodeError>>,
    },
    /// Raw host frame: the worker converts it on the GPU.
    EncodeFrame {
        bytes: Vec<u8>,
        desc: FrameDesc,
        force_keyframe: bool,
        reply: mpsc::Sender<Result<EncodeOutput, EncodeError>>,
    },
    /// Batch whose frames already live on the GPU. `keep_alive` holds whatever
    /// owns that memory for the lifetime of the job.
    EncodeDevice {
        ptr: u64,
        desc: FrameDesc,
        frames: u32,
        force_keyframe: bool,
        keep_alive: Option<Box<dyn std::any::Any + Send>>,
        reply: mpsc::Sender<Result<EncodeOutput, EncodeError>>,
    },
    /// Raw host frames converted by one kernel launch.
    EncodeBatch {
        frames: Vec<(Vec<u8>, FrameDesc)>,
        force_keyframe: bool,
        reply: mpsc::Sender<Result<EncodeOutput, EncodeError>>,
    },
    Flush {
        reply: mpsc::Sender<Result<Vec<EncodedUnit>, EncodeError>>,
    },
    Finish {
        reply: mpsc::Sender<Result<Vec<EncodedUnit>, EncodeError>>,
    },
}

/// AV1 encoder backed by NVENC on the local GPU.
///
/// Constructing one starts a worker thread and opens the session, so hardware
/// or configuration problems surface here rather than on the first frame.
pub struct NvencEncoder {
    tx: Option<mpsc::Sender<Command>>,
    join: Option<thread::JoinHandle<()>>,
    codec_config: Option<Vec<u8>>,
    force_keyframe: bool,
    finished: bool,
}

impl NvencEncoder {
    /// Open an NVENC AV1 session for one stream.
    pub fn new(
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        fps: f64,
        timebase: Timebase,
        codec: VideoCodec,
    ) -> Result<Self, EncodeError> {
        // Load the driver here, on the caller's thread. Doing it in the worker
        // means a machine without a driver gets "the worker exited" instead of
        // "no NVIDIA driver", which is both true and useless.
        nvidia_video_codec_sdk::safe::available()
            .map_err(|e| EncodeError::Backend(e.to_string()))?;
        let params = NvencParams::from_encoder_config(cfg, width, height, fps)
            .map_err(|e| EncodeError::Backend(e.to_string()))?;
        let frame_duration = timebase.ticks_per_frame(fps);

        let (tx, rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name("blazar-nvenc".into())
            .spawn(move || worker_main(params, frame_duration, codec, rx, init_tx))
            .map_err(|e| EncodeError::Backend(format!("spawn NVENC worker: {e}")))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx: Some(tx),
                join: Some(join),
                codec_config: None,
                force_keyframe: false,
                finished: false,
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(EncodeError::Backend(
                    "NVENC worker exited during initialization".into(),
                ))
            }
        }
    }

    fn request<T>(
        &self,
        make: impl FnOnce(mpsc::Sender<Result<T, EncodeError>>) -> Command,
    ) -> Result<T, EncodeError> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| EncodeError::Backend("NVENC worker already stopped".into()))?;
        let (reply_tx, reply_rx) = mpsc::channel();
        tx.send(make(reply_tx))
            .map_err(|_| EncodeError::Backend("NVENC worker stopped".into()))?;
        reply_rx
            .recv()
            .map_err(|_| EncodeError::Backend("NVENC worker stopped before replying".into()))?
    }
}

impl VideoEncoder for NvencEncoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        let force_keyframe = std::mem::take(&mut self.force_keyframe);
        let owned = frame.clone();
        let out: EncodeOutput = self.request(move |reply| Command::Encode {
            frame: owned,
            force_keyframe,
            reply,
        })?;
        if out.codec_config.is_some() {
            self.codec_config = out.codec_config;
        }
        Ok(out.units)
    }

    fn encode_frame(&mut self, frame: FrameRef) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        let force_keyframe = std::mem::take(&mut self.force_keyframe);
        let FrameRef { buffer, desc } = frame;
        let bytes = match buffer {
            FrameBuffer::Host {
                owner: BufferOwner::Owned(bytes),
                ..
            } => bytes,
            other => other.host_bytes().map(|b| b.to_vec()).ok_or_else(|| {
                EncodeError::Backend("NVENC needs host frames (device ingest is Phase 2)".into())
            })?,
        };
        let out: EncodeOutput = self.request(move |reply| Command::EncodeFrame {
            bytes,
            desc,
            force_keyframe,
            reply,
        })?;
        if out.codec_config.is_some() {
            self.codec_config = out.codec_config;
        }
        Ok(out.units)
    }

    fn encode_batch(&mut self, frames: Vec<FrameRef>) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let force_keyframe = std::mem::take(&mut self.force_keyframe);
        let mut owned = Vec::with_capacity(frames.len());
        for frame in frames {
            let FrameRef { buffer, desc } = frame;
            let bytes = match buffer {
                FrameBuffer::Host {
                    owner: BufferOwner::Owned(bytes),
                    ..
                } => bytes,
                other => other.host_bytes().map(|b| b.to_vec()).ok_or_else(|| {
                    EncodeError::Backend(
                        "NVENC needs host frames (device ingest is Phase 2)".into(),
                    )
                })?,
            };
            owned.push((bytes, desc));
        }
        let out: EncodeOutput = self.request(move |reply| Command::EncodeBatch {
            frames: owned,
            force_keyframe,
            reply,
        })?;
        if out.codec_config.is_some() {
            self.codec_config = out.codec_config;
        }
        Ok(out.units)
    }

    fn encode_device_batch(
        &mut self,
        ptr: u64,
        desc: FrameDesc,
        frames: u32,
        keep_alive: Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        if frames == 0 {
            return Ok(Vec::new());
        }
        let force_keyframe = std::mem::take(&mut self.force_keyframe);
        let out: EncodeOutput = self.request(move |reply| Command::EncodeDevice {
            ptr,
            desc,
            frames,
            force_keyframe,
            keep_alive,
            reply,
        })?;
        if out.codec_config.is_some() {
            self.codec_config = out.codec_config;
        }
        Ok(out.units)
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        // NVENC cannot drain lookahead without end-of-stream. Segment
        // boundaries are handled by forcing a keyframe on the next frame
        // (`request_keyframe`), not by pretending to flush.
        self.request(|reply| Command::Flush { reply })
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        let units = self.request(|reply| Command::Finish { reply })?;
        self.finished = true;
        Ok(units)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.codec_config.clone()
    }
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's `recv` loop; join so the
        // session and its buffers are torn down before this handle disappears.
        self.tx = None;
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn worker_main(
    params: NvencParams,
    frame_duration: u64,
    codec: VideoCodec,
    rx: mpsc::Receiver<Command>,
    init: mpsc::Sender<Result<(), EncodeError>>,
) {
    match open_session(&params, codec) {
        Ok(session) => {
            if init.send(Ok(())).is_err() {
                return;
            }
            run_loop(&session, &params, frame_duration, codec, rx);
            // `session` drops here, after the buffers created inside
            // `run_loop` have already been dropped.
        }
        Err(e) => {
            let _ = init.send(Err(e));
        }
    }
}

fn guid_for(codec: VideoCodec) -> GUID {
    match codec {
        VideoCodec::Av1 => NV_ENC_CODEC_AV1_GUID,
        VideoCodec::H264 => NV_ENC_CODEC_H264_GUID,
        VideoCodec::Hevc => NV_ENC_CODEC_HEVC_GUID,
        // Never reached: `open_session` rejects VP9 before opening anything.
        // AV1 is returned only because the function must produce a GUID.
        VideoCodec::Vp9 => NV_ENC_CODEC_AV1_GUID,
    }
}

fn open_session(params: &NvencParams, codec: VideoCodec) -> Result<Session, EncodeError> {
    if matches!(codec, VideoCodec::Vp9) {
        return Err(EncodeError::Backend(
            "NVENC has no VP9 encoder; use the vp9 backend".into(),
        ));
    }
    let guid = guid_for(codec);
    let ctx = match &*CUDA_CONTEXT {
        Ok(ctx) => Arc::clone(ctx),
        Err(e) => return Err(EncodeError::Backend(e.clone())),
    };
    ctx.bind_to_thread()
        .map_err(|e| EncodeError::Backend(format!("bind CUDA context: {e:?}")))?;

    let encoder = Encoder::initialize_with_cuda(ctx).map_err(|e| {
        // INVALID_ENCODERDEVICE means the driver refused a session, and the
        // common cause by far is the concurrent-session cap on consumer cards.
        // The raw error names neither the cap nor what to do about it.
        let message = format!("NVENC init: {e:?}");
        if message.contains("InvalidEncoderDevice") {
            EncodeError::Backend(format!(
                "{}: the driver refused a new encoder session. Consumer GPUs cap concurrent \
                 NVENC sessions (measured: 9 on an RTX 4080 SUPER); datacenter GPUs do not. \
                 Close streams you are not using, or move to a datacenter GPU.",
                message
            ))
        } else {
            EncodeError::Backend(message)
        }
    })?;
    let guids = encoder
        .get_encode_guids()
        .map_err(|e| EncodeError::Backend(format!("NVENC encode GUIDs: {e:?}")))?;
    if !guids.contains(&guid) {
        return Err(EncodeError::Backend(format!(
            "this GPU does not advertise NVENC {codec:?}"
        )));
    }

    let mut preset = encoder
        .get_preset_config(
            guid,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )
        .map_err(|e| EncodeError::Backend(format!("NVENC preset config: {e:?}")))?;
    apply_config(&mut preset.presetCfg, params, codec);

    let mut init = EncoderInitParams::new(guid, params.width, params.height);
    init.preset_guid(NV_ENC_PRESET_P4_GUID)
        .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY)
        .framerate(params.fps_num, params.fps_den)
        .encode_config(&mut preset.presetCfg)
        // Required for AV1: with picture-type decision off, the driver coerces
        // every frame to intra. With it on, the driver chooses I/P and we force
        // IDRs explicitly via `NV_ENC_PIC_FLAG_FORCEIDR`.
        .enable_picture_type_decision();

    encoder
        .start_session(NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_NV12, init)
        .map_err(|e| EncodeError::Backend(format!("NVENC session: {e:?}")))
}

/// Apply calibration-derived settings to the preset configuration.
fn apply_config(cfg: &mut NV_ENC_CONFIG, params: &NvencParams, codec: VideoCodec) {
    cfg.gopLength = params.gop_length;
    cfg.frameIntervalP = params.frame_interval_p;

    match params.rate_control {
        NvencRateControl::ConstQp { qp } => {
            cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CONSTQP;
            cfg.rcParams.constQP = NV_ENC_QP {
                qpInterP: qp,
                qpInterB: qp,
                qpIntra: qp,
            };
        }
        NvencRateControl::Vbr {
            average_bitrate_bps,
            max_bitrate_bps,
            vbv_bits,
        } => {
            cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_VBR;
            cfg.rcParams.averageBitRate = average_bitrate_bps;
            cfg.rcParams.maxBitRate = max_bitrate_bps;
            cfg.rcParams.vbvBufferSize = vbv_bits;
        }
        NvencRateControl::Cbr {
            bitrate_bps,
            vbv_bits,
        } => {
            cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
            cfg.rcParams.averageBitRate = bitrate_bps;
            cfg.rcParams.maxBitRate = bitrate_bps;
            cfg.rcParams.vbvBufferSize = vbv_bits;
        }
    }

    // The preprocess stage emits YUV using the BT.709 matrix and never tags it.
    // Tag it here so players do not have to guess. Each codec keeps the colour
    // metadata in its own structure: AV1 in the codec config, H.264 and HEVC in
    // their VUI. Leaving the H.264/HEVC VUI unset makes decoders assume BT.601,
    // which is visibly wrong for BT.709 content.
    // SAFETY: the union variant matches the codec GUID of the open session.
    unsafe {
        let full_range = u32::from(params.color.full_range);
        match codec {
            // NVENC has no VP9 encoder; the session can never be opened for it.
            VideoCodec::Vp9 => {}
            VideoCodec::Av1 => {
                let av1 = &mut cfg.encodeCodecConfig.av1Config;
                if params.color.bt709 {
                    av1.colorPrimaries =
                        NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
                    av1.transferCharacteristics =
                        NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
                    av1.matrixCoefficients =
                        NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
                }
                av1.colorRange = full_range;
            }
            VideoCodec::H264 => {
                let vui = &mut cfg.encodeCodecConfig.h264Config.h264VUIParameters;
                vui.videoSignalTypePresentFlag = 1;
                vui.videoFullRangeFlag = full_range;
                vui.colourDescriptionPresentFlag = u32::from(params.color.bt709);
                if params.color.bt709 {
                    vui.colourPrimaries =
                        NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
                    vui.transferCharacteristics =
                        NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
                    vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
                }
            }
            VideoCodec::Hevc => {
                let vui = &mut cfg.encodeCodecConfig.hevcConfig.hevcVUIParameters;
                vui.videoSignalTypePresentFlag = 1;
                vui.videoFullRangeFlag = full_range;
                vui.colourDescriptionPresentFlag = u32::from(params.color.bt709);
                if params.color.bt709 {
                    vui.colourPrimaries =
                        NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
                    vui.transferCharacteristics =
                        NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
                    vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
                }
            }
        }
    }
}

fn run_loop(
    session: &Session,
    params: &NvencParams,
    frame_duration: u64,
    codec: VideoCodec,
    rx: mpsc::Receiver<Command>,
) {
    let ctx = match &*CUDA_CONTEXT {
        Ok(ctx) => Arc::clone(ctx),
        Err(e) => {
            fail_pending(&rx, EncodeError::Backend(e.clone()));
            return;
        }
    };
    let stream = ctx.default_stream();

    // Zero-copy input. NVENC only accepts a device pointer from a real
    // 2D-pitch allocation: `cuMemAlloc` returns ResourceRegisterFailed even
    // with the documented pitch == width, while `cuMemAllocPitch` plus the
    // pitch it returns registers. Verified on Windows and WSL.
    //
    // `device` is declared before `inputs` so the registered resources (which
    // borrow the session) are dropped before the allocations they point at.
    let device: Vec<PitchedInput> = match (0..POOL_CAPACITY)
        .map(|_| PitchedInput::new(params.width, params.height))
        .collect()
    {
        Ok(buffers) => buffers,
        Err(e) => {
            fail_pending(&rx, e);
            return;
        }
    };
    let pitch = device[0].pitch;
    if pitch < params.width {
        fail_pending(
            &rx,
            EncodeError::Backend(format!(
                "NVENC input stride {pitch} is smaller than width {}",
                params.width
            )),
        );
        return;
    }
    let mut inputs: Vec<RegisteredResource<'_, ()>> = match (0..POOL_CAPACITY)
        .map(|i| {
            session.register_generic_resource(
                (),
                NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                device[i].ptr as *mut std::ffi::c_void,
                pitch,
            )
        })
        .collect()
    {
        Ok(resources) => resources,
        Err(e) => {
            fail_pending(
                &rx,
                EncodeError::Backend(format!("register CUDA input: {e:?}")),
            );
            return;
        }
    };
    // Destination pointers for the batch kernel. One entry per pool slot, so a
    // batch never needs a fresh allocation per frame.
    let mut dst_ptrs: CudaSlice<u64> = match stream.alloc_zeros(POOL_CAPACITY) {
        Ok(buffer) => buffer,
        Err(e) => {
            fail_pending(
                &rx,
                EncodeError::Backend(format!("allocate dst pointer array: {e}")),
            );
            return;
        }
    };
    let staging_len = (params.width * params.height * 3 * 4) as usize;
    let mut staging: CudaSlice<u8> = match stream.alloc_zeros(staging_len) {
        Ok(buffer) => buffer,
        Err(e) => {
            fail_pending(
                &rx,
                EncodeError::Backend(format!("allocate RGB staging: {e}")),
            );
            return;
        }
    };
    let converter = match Nv12Converter::new(&ctx) {
        Ok(converter) => converter,
        Err(e) => {
            fail_pending(&rx, e);
            return;
        }
    };
    let mut bitstreams: Vec<Bitstream<'_>> = match (0..POOL_CAPACITY)
        .map(|_| session.create_output_bitstream())
        .collect()
    {
        Ok(buffers) => buffers,
        Err(e) => {
            fail_pending(&rx, EncodeError::Backend(format!("NVENC bitstream: {e:?}")));
            return;
        }
    };

    let mut pool = SlotPool::new(POOL_CAPACITY);
    let mut keyframes = KeyframeSchedule::new(params.gop_length);
    let mut codec_config: Option<Vec<u8>> = None;
    let mut decode_index: u64 = 0;
    let mut finished = false;

    while let Ok(command) = rx.recv() {
        let (source, force_keyframe, reply) = match command {
            Command::Encode {
                frame,
                force_keyframe,
                reply,
            } => (Source::Planar(frame), force_keyframe, reply),
            Command::EncodeFrame {
                bytes,
                desc,
                force_keyframe,
                reply,
            } => (
                Source::Rgb {
                    bytes: Box::new(bytes),
                    desc,
                },
                force_keyframe,
                reply,
            ),
            Command::EncodeDevice {
                ptr,
                desc,
                frames,
                force_keyframe,
                keep_alive,
                reply,
            } => {
                if finished {
                    let _ = reply.send(Err(EncodeError::AfterFinish));
                    continue;
                }
                let mut units = Vec::new();
                let idr: Vec<bool> = (0..frames as usize)
                    .map(|i| keyframes.next_is_keyframe(i == 0 && force_keyframe))
                    .collect();
                let result = encode_device_batch(
                    session,
                    &stream,
                    &device,
                    &mut inputs,
                    &mut bitstreams,
                    &mut dst_ptrs,
                    &converter,
                    &mut pool,
                    &mut codec_config,
                    &mut decode_index,
                    frame_duration,
                    codec,
                    pitch,
                    ptr,
                    desc,
                    frames,
                    &idr,
                    &mut units,
                );
                // Drop the caller's allocation only after the last encode that
                // reads it has been submitted.
                drop(keep_alive);
                let _ = reply.send(result.map(|()| EncodeOutput {
                    units,
                    codec_config: codec_config.clone(),
                }));
                continue;
            }
            Command::EncodeBatch {
                frames,
                force_keyframe,
                reply,
            } => {
                if finished {
                    let _ = reply.send(Err(EncodeError::AfterFinish));
                    continue;
                }
                let mut units = Vec::new();
                let idr: Vec<bool> = (0..frames.len())
                    .map(|i| keyframes.next_is_keyframe(i == 0 && force_keyframe))
                    .collect();
                let result = encode_batch_rgb(
                    session,
                    &stream,
                    &device,
                    &mut inputs,
                    &mut bitstreams,
                    &mut staging,
                    &mut dst_ptrs,
                    &converter,
                    &mut pool,
                    &mut codec_config,
                    &mut decode_index,
                    frame_duration,
                    codec,
                    pitch,
                    &frames,
                    &idr,
                    &mut units,
                );
                let _ = reply.send(result.map(|()| EncodeOutput {
                    units,
                    codec_config: codec_config.clone(),
                }));
                continue;
            }
            Command::Flush { reply } => {
                let _ = reply.send(Ok(Vec::new()));
                continue;
            }
            Command::Finish { reply } => {
                finished = true;
                let mut units = Vec::new();
                let result = finish(
                    session,
                    &mut bitstreams,
                    &mut pool,
                    &mut decode_index,
                    frame_duration,
                    codec,
                    &mut units,
                );
                let _ = reply.send(result.map(|()| units));
                continue;
            }
        };
        if finished {
            let _ = reply.send(Err(EncodeError::AfterFinish));
            continue;
        }
        let mut units = Vec::new();
        let force_idr = keyframes.next_is_keyframe(force_keyframe);
        let result = encode_one(
            session,
            &stream,
            &device,
            &mut inputs,
            &mut bitstreams,
            &mut staging,
            &mut dst_ptrs,
            &converter,
            &mut pool,
            &mut codec_config,
            &mut decode_index,
            frame_duration,
            codec,
            pitch,
            force_idr,
            &source,
            &mut units,
        );
        let reply_value = result.map(|()| EncodeOutput {
            units,
            codec_config: codec_config.clone(),
        });
        let _ = reply.send(reply_value);
    }
}

/// Where a frame comes from: already-planar (direct callers, tests) or raw RGB
/// that the worker converts on the GPU.
enum Source {
    Planar(PreprocessedFrame),
    Rgb {
        // Boxed so the enum stays small; the payload is moved, never cloned.
        #[allow(clippy::box_collection)]
        bytes: Box<Vec<u8>>,
        desc: FrameDesc,
    },
}

#[allow(clippy::too_many_arguments)]
fn encode_one(
    session: &Session,
    stream: &Arc<CudaStream>,
    device: &[PitchedInput],
    inputs: &mut [RegisteredResource<'_, ()>],
    bitstreams: &mut [Bitstream<'_>],
    staging: &mut CudaSlice<u8>,
    dst_ptrs: &mut CudaSlice<u64>,
    converter: &Nv12Converter,
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    pitch: u32,
    force_keyframe: bool,
    source: &Source,
    units: &mut Vec<EncodedUnit>,
) -> Result<(), EncodeError> {
    let pts = match source {
        Source::Planar(frame) => frame.pts.ticks(),
        Source::Rgb { desc, .. } => desc.pts.ticks(),
    };

    let slot = match pool.acquire(pts) {
        Some(slot) => slot,
        None => {
            // Every slot is in flight: wait for the oldest to complete. With a
            // 16-slot pool and no B-frames the driver must be able to emit the
            // oldest, so this is backpressure rather than a stall.
            drain_ready(
                bitstreams,
                pool,
                codec_config,
                decode_index,
                frame_duration,
                codec,
                units,
            )?;
            pool.acquire(pts)
                .ok_or_else(|| EncodeError::Backend("NVENC output slot pool exhausted".into()))?
        }
    };

    match source {
        Source::Planar(frame) => {
            let nv12 = to_nv12(frame, pitch).map_err(|e| EncodeError::Backend(e.to_string()))?;
            // SAFETY: `device[slot]` is a live pitched allocation of at least
            // `nv12.len()` bytes and the destination is not aliased.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    device[slot].ptr,
                    nv12.as_ptr().cast(),
                    nv12.len(),
                    stream.cu_stream(),
                )
            }
            .result()
            .map_err(|e| EncodeError::Backend(format!("upload NV12: {e:?}")))?;
        }
        Source::Rgb { bytes, desc } => {
            if bytes.len() > staging.len() {
                return Err(EncodeError::Backend(format!(
                    "RGB frame is {} bytes, staging is {}",
                    bytes.len(),
                    staging.len()
                )));
            }
            stream
                .memcpy_htod(bytes.as_slice(), staging)
                .map_err(|e| EncodeError::Backend(format!("upload RGB: {e}")))?;
            stream
                .memcpy_htod(&[device[slot].ptr], &mut dst_ptrs.slice_mut(0..1))
                .map_err(|e| EncodeError::Backend(format!("upload dst pointer: {e}")))?;
            converter.convert_batch(stream, staging, dst_ptrs, desc, 1, pitch)?;
        }
    }
    // NVENC reads the registered resource from the same CUDA context but is not
    // ordered against our stream; publish the writes first.
    stream
        .synchronize()
        .map_err(|e| EncodeError::Backend(format!("sync before encode: {e}")))?;

    let flags = if force_keyframe {
        NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
    } else {
        0
    };
    match encode_with_retry(
        session,
        &mut inputs[slot],
        &mut bitstreams[slot],
        pts,
        flags,
    )? {
        EncodeCall::Emitted => {}
        EncodeCall::NeedMoreInput => {
            // The frame is buffered; its output appears on a later call.
            return Ok(());
        }
    }
    drain_ready(
        bitstreams,
        pool,
        codec_config,
        decode_index,
        frame_duration,
        codec,
        units,
    )
}

fn fail_pending(rx: &mpsc::Receiver<Command>, error: EncodeError) {
    while let Ok(command) = rx.try_recv() {
        match command {
            Command::Encode { reply, .. }
            | Command::EncodeFrame { reply, .. }
            | Command::EncodeBatch { reply, .. }
            | Command::EncodeDevice { reply, .. } => {
                let _ = reply.send(Err(EncodeError::Backend(error.to_string())));
            }
            Command::Flush { reply } | Command::Finish { reply } => {
                let _ = reply.send(Err(EncodeError::Backend(error.to_string())));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drain_ready(
    bitstreams: &mut [Bitstream<'_>],
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    units: &mut Vec<EncodedUnit>,
) -> Result<(), EncodeError> {
    while let Some((slot, _)) = pool.in_flight().first().copied() {
        let lock = bitstreams[slot].lock().map_err(nvenc_error)?;
        if lock.data().is_empty() {
            return Err(EncodeError::Backend(
                "NVENC reported success but a pending output buffer is empty".into(),
            ));
        }
        let unit = unit_from_bitstream(&lock, codec, codec_config, decode_index, frame_duration)?;
        drop(lock);
        pool.release(slot);
        units.push(unit);
    }
    Ok(())
}

enum EncodeCall {
    Emitted,
    NeedMoreInput,
}

fn encode_with_retry(
    session: &Session,
    input: &mut impl EncoderInput,
    output: &mut Bitstream<'_>,
    timestamp: u64,
    flags: u32,
) -> Result<EncodeCall, EncodeError> {
    for _ in 0..BUSY_RETRIES {
        let params = EncodePictureParams {
            input_timestamp: timestamp,
            encode_pic_flags: flags,
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let result = session.encode_picture(input, output, params);
        ENCODE_MICROS.fetch_add(
            started.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        ENCODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match result {
            Ok(()) => return Ok(EncodeCall::Emitted),
            Err(e) if e.kind() == ErrorKind::NeedMoreInput => return Ok(EncodeCall::NeedMoreInput),
            Err(e) if e.kind() == ErrorKind::EncoderBusy => thread::sleep(BUSY_SLEEP),
            Err(e) => return Err(nvenc_error(e)),
        }
    }
    Err(EncodeError::Backend(
        "NVENC stayed busy across retries".into(),
    ))
}

/// Drain everything still in flight after end-of-stream.
fn finish(
    session: &Session,
    bitstreams: &mut [Bitstream<'_>],
    pool: &mut SlotPool,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    units: &mut Vec<EncodedUnit>,
) -> Result<(), EncodeError> {
    let mut eos = false;
    for _ in 0..BUSY_RETRIES {
        match session.end_of_stream() {
            Ok(()) => {
                eos = true;
                break;
            }
            Err(e) if e.kind() == ErrorKind::EncoderBusy => thread::sleep(BUSY_SLEEP),
            Err(e) => return Err(nvenc_error(e)),
        }
    }
    if !eos {
        return Err(EncodeError::Backend(
            "NVENC stayed busy while draining end-of-stream".into(),
        ));
    }

    for (slot, _) in pool.in_flight() {
        let lock = bitstreams[slot].lock().map_err(nvenc_error)?;
        // After end_of_stream every in-flight buffer is complete. Do not compare
        // the echoed timestamp with the slot's input PTS: under reordering the
        // driver reports a different frame's PTS, and the old guard silently
        // dropped the reorder window (two frames for H.264/HEVC with bframes=2).
        if lock.data().is_empty() {
            drop(lock);
            pool.release(slot);
            continue;
        }
        let mut no_config = None;
        let unit = unit_from_bitstream(&lock, codec, &mut no_config, decode_index, frame_duration)?;
        drop(lock);
        pool.release(slot);
        units.push(unit);
    }
    Ok(())
}

fn unit_from_bitstream(
    lock: &nvidia_video_codec_sdk::BitstreamLock<'_, '_>,
    codec: VideoCodec,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
) -> Result<EncodedUnit, EncodeError> {
    // With picture-type decision enabled, the driver reports the real type.
    // Only IDR is a random-access/sync sample; INTRA_ONLY (I) is not.
    let keyframe = matches!(lock.picture_type(), NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR);
    let raw = lock.data();
    let sample = match codec {
        // Unreachable: `open_session` rejects VP9 before this runs.
        VideoCodec::Vp9 => return Err(EncodeError::Backend("NVENC cannot encode VP9".into())),
        VideoCodec::Av1 => {
            if keyframe && codec_config.is_none() {
                *codec_config = Some(av1::av1c_from_keyframe(raw).map_err(|e| {
                    EncodeError::Backend(format!("derive av1C from keyframe: {e}"))
                })?);
            }
            raw.to_vec()
        }
        VideoCodec::H264 | VideoCodec::Hevc => {
            // Annex B in, length-prefixed sample plus avcC/hvcC out.
            let format = match codec {
                VideoCodec::H264 => AnnexB::H264,
                VideoCodec::Hevc => AnnexB::Hevc,
                VideoCodec::Av1 | VideoCodec::Vp9 => unreachable!("handled above"),
            };
            let parsed = nal::parse_access_unit(raw, format)
                .map_err(|e| EncodeError::Backend(format!("parse {codec:?} access unit: {e}")))?;
            if codec_config.is_none() {
                *codec_config = parsed.config;
            }
            parsed.sample
        }
    };

    let pts = Pts(lock.timestamp());
    let dts = Pts(decode_index.saturating_mul(frame_duration));
    *decode_index += 1;

    Ok(EncodedUnit {
        track_id: TrackId::Video,
        bytes: sample,
        pts,
        dts,
        keyframe,
        duration: frame_duration,
    })
}

fn nvenc_error(e: nvidia_video_codec_sdk::EncodeError) -> EncodeError {
    EncodeError::Backend(format!("NVENC: {e:?}"))
}

/// A device buffer allocated with `cuMemAllocPitch`, the allocation kind NVENC
/// accepts for `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`.
struct PitchedInput {
    ptr: u64,
    pitch: u32,
}

impl PitchedInput {
    /// `cuMemAllocPitch` the luma plane plus chroma rows, so the buffer holds a
    /// full NV12 frame at the driver's stride.
    fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        let rows = height + height.div_ceil(2);
        let mut ptr = 0u64;
        let mut pitch = 0usize;
        // SAFETY: out-pointers are valid; the current context is the worker's.
        let status = unsafe {
            sys::cuMemAllocPitch_v2(&mut ptr, &mut pitch, width as usize, rows as usize, 4)
        };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(EncodeError::Backend(format!(
                "cuMemAllocPitch {width}x{rows}: {status:?}"
            )));
        }
        Ok(Self {
            ptr,
            pitch: pitch as u32,
        })
    }
}

impl Drop for PitchedInput {
    fn drop(&mut self) {
        // SAFETY: the pointer came from cuMemAllocPitch and is freed once.
        unsafe { sys::cuMemFree_v2(self.ptr) };
    }
}

/// Convert a batch of RGB frames with one kernel launch, then encode each slot.
///
/// Batches larger than the slot pool (or larger than the staging budget) are
/// split into chunks of at most `POOL_CAPACITY` frames, one launch each. The
/// frames within a chunk are packed into staging in submission order, so the
/// kernel reads frame `i` from `i * frame_bytes`.
///
/// Chunks exist because every destination must be registered with NVENC and
/// alive at the same time: the pool is VRAM, and the RGB staging is 24 bytes per
/// pixel per frame. Splitting keeps both bounded without falling back to
/// one-frame submissions.
#[allow(clippy::too_many_arguments)]
fn encode_batch_rgb(
    session: &Session,
    stream: &Arc<CudaStream>,
    device: &[PitchedInput],
    inputs: &mut [RegisteredResource<'_, ()>],
    bitstreams: &mut [Bitstream<'_>],
    staging: &mut CudaSlice<u8>,
    dst_ptrs: &mut CudaSlice<u64>,
    converter: &Nv12Converter,
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    pitch: u32,
    frames: &[(Vec<u8>, FrameDesc)],
    idr: &[bool],
    units: &mut Vec<EncodedUnit>,
) -> Result<(), EncodeError> {
    let Some((first, desc0)) = frames.first() else {
        return Ok(());
    };
    let frame_bytes = first.len();
    // FrameDesc carries the timestamp, so only the format fields may match.
    let same_format = |a: &FrameDesc, b: &FrameDesc| {
        a.width == b.width
            && a.height == b.height
            && a.dtype == b.dtype
            && a.range == b.range
            && a.layout == b.layout
            && a.colorspace == b.colorspace
            && (a.fps - b.fps).abs() < 1e-9
    };
    if frames
        .iter()
        .any(|(bytes, desc)| bytes.len() != frame_bytes || !same_format(desc, desc0))
    {
        return Err(EncodeError::Backend(
            "batch frames must share geometry and format".into(),
        ));
    }

    // Chunk size: the pool is the hard limit, and staging is capped so a large
    // batch cannot reserve hundreds of megabytes of VRAM at 1080p.
    let budget_frames = (BATCH_STAGING_BUDGET / frame_bytes.max(1)).max(1);
    let max_per_chunk = POOL_CAPACITY.min(budget_frames).max(1);

    let mut start = 0;
    while start < frames.len() {
        let end = (start + max_per_chunk).min(frames.len());
        let consumed = encode_chunk(
            session,
            stream,
            device,
            inputs,
            bitstreams,
            staging,
            dst_ptrs,
            converter,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            pitch,
            &frames[start..end],
            &idr[start..end],
            units,
        )?;
        if consumed == 0 {
            // The pool had nothing free even after the helper drained. Encode
            // this one frame through the single-frame path, which acquires a
            // slot itself, then continue with the rest of the batch.
            let (bytes, desc) = &frames[start];
            let source = Source::Rgb {
                bytes: Box::new(bytes.clone()),
                desc: *desc,
            };
            encode_one(
                session,
                stream,
                device,
                inputs,
                bitstreams,
                staging,
                dst_ptrs,
                converter,
                pool,
                codec_config,
                decode_index,
                frame_duration,
                codec,
                pitch,
                idr[start],
                &source,
                units,
            )?;
            start += 1;
        } else {
            start += consumed;
        }
    }
    Ok(())
}

/// Convert and encode one chunk that fits the pool. Returns how many frames it
/// consumed; `0` means nothing was free and the caller should try one frame.
#[allow(clippy::too_many_arguments)]
fn encode_chunk(
    session: &Session,
    stream: &Arc<CudaStream>,
    device: &[PitchedInput],
    inputs: &mut [RegisteredResource<'_, ()>],
    bitstreams: &mut [Bitstream<'_>],
    staging: &mut CudaSlice<u8>,
    dst_ptrs: &mut CudaSlice<u64>,
    converter: &Nv12Converter,
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    pitch: u32,
    frames: &[(Vec<u8>, FrameDesc)],
    idr: &[bool],
    units: &mut Vec<EncodedUnit>,
) -> Result<usize, EncodeError> {
    let Some((_, desc0)) = frames.first() else {
        return Ok(0);
    };
    let frame_bytes = frames[0].0.len();

    let mut slots = Vec::with_capacity(frames.len());
    for _ in frames.iter() {
        match pool.reserve() {
            Some(slot) => slots.push(slot),
            None => break,
        }
    }
    if slots.is_empty() {
        // Nothing free: make room and let the caller retry this frame.
        drain_ready(
            bitstreams,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            units,
        )?;
        return Ok(0);
    }

    let batched = slots.len();
    let mut host = Vec::with_capacity(batched * frame_bytes);
    for (bytes, _) in frames.iter().take(batched) {
        host.extend_from_slice(bytes);
    }
    let ptrs: Vec<u64> = slots.iter().map(|slot| device[*slot].ptr).collect();
    // Staging starts one frame deep. Chunks are capped by BATCH_STAGING_BUDGET,
    // so this grows to the largest chunk and then stays there.
    if host.len() > staging.len() {
        *staging = stream
            .alloc_zeros(host.len())
            .map_err(|e| EncodeError::Backend(format!("grow batch staging: {e}")))?;
    }
    stream
        .memcpy_htod(&host, &mut staging.slice_mut(0..host.len()))
        .map_err(|e| EncodeError::Backend(format!("upload batch: {e}")))?;
    stream
        .memcpy_htod(&ptrs, &mut dst_ptrs.slice_mut(0..batched))
        .map_err(|e| EncodeError::Backend(format!("upload dst pointers: {e}")))?;
    converter.convert_batch(stream, staging, dst_ptrs, desc0, batched as u32, pitch)?;
    stream
        .synchronize()
        .map_err(|e| EncodeError::Backend(format!("sync before encode: {e}")))?;

    for (i, slot) in slots.iter().enumerate() {
        let (_, desc) = &frames[i];
        let flags = if idr[i] {
            NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
        } else {
            0
        };
        pool.submit(*slot, desc.pts.ticks());
        match encode_with_retry(
            session,
            &mut inputs[*slot],
            &mut bitstreams[*slot],
            desc.pts.ticks(),
            flags,
        )? {
            EncodeCall::Emitted => {}
            // The frame is buffered; its output appears on a later call.
            EncodeCall::NeedMoreInput => continue,
        }
        drain_ready(
            bitstreams,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            units,
        )?;
    }
    Ok(batched)
}

/// Bytes of one BCTHW frame in the caller's dtype.
fn device_frame_bytes(desc: &FrameDesc) -> usize {
    let elem = match desc.dtype {
        blazar_core::Dtype::F32 => 4,
        _ => 2,
    };
    desc.width as usize * desc.height as usize * 3 * elem
}

/// Convert and encode a batch that already lives on the GPU. The caller's
/// memory is read directly by the kernel; nothing is staged.
#[allow(clippy::too_many_arguments)]
fn encode_device_batch(
    session: &Session,
    stream: &Arc<CudaStream>,
    device: &[PitchedInput],
    inputs: &mut [RegisteredResource<'_, ()>],
    bitstreams: &mut [Bitstream<'_>],
    dst_ptrs: &mut CudaSlice<u64>,
    converter: &Nv12Converter,
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    pitch: u32,
    ptr: u64,
    desc: FrameDesc,
    frames: u32,
    idr: &[bool],
    units: &mut Vec<EncodedUnit>,
) -> Result<(), EncodeError> {
    let frame_bytes = device_frame_bytes(&desc);
    // A device frame advances by one plane in (C,T,H,W); THWC frames are whole.
    let pixels = desc.width as usize * desc.height as usize;
    let frame_step = match desc.layout {
        blazar_core::Layout::Thwc => {
            pixels
                * 3
                * if matches!(desc.dtype, blazar_core::Dtype::F32) {
                    4
                } else {
                    2
                }
        }
        blazar_core::Layout::Bcthw => {
            pixels
                * if matches!(desc.dtype, blazar_core::Dtype::F32) {
                    4
                } else {
                    2
                }
        }
    };
    let budget = (BATCH_STAGING_BUDGET / frame_bytes.max(1)).max(1);
    let max_per_chunk = POOL_CAPACITY.min(budget).max(1);

    let mut start = 0usize;
    let total = frames as usize;
    let mut stalls = 0;
    while start < total {
        let end = (start + max_per_chunk).min(total);
        let consumed = encode_device_chunk(
            session,
            stream,
            device,
            inputs,
            bitstreams,
            dst_ptrs,
            converter,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            pitch,
            ptr + (start as u64) * frame_step as u64,
            desc,
            (end - start) as u32,
            frames,
            &idr[start..end],
            units,
        )?;
        if consumed == 0 {
            // The helper drained; retry the same frame. A pool this starved
            // for several rounds is a bug, not backpressure.
            stalls += 1;
            if stalls > POOL_CAPACITY {
                return Err(EncodeError::Backend(
                    "NVENC output slot pool is not draining".into(),
                ));
            }
            continue;
        }
        stalls = 0;
        start += consumed;
    }
    Ok(())
}

/// One device-batch chunk: reserve slots, convert straight from `ptr`, encode.
#[allow(clippy::too_many_arguments)]
fn encode_device_chunk(
    session: &Session,
    stream: &Arc<CudaStream>,
    device: &[PitchedInput],
    inputs: &mut [RegisteredResource<'_, ()>],
    bitstreams: &mut [Bitstream<'_>],
    dst_ptrs: &mut CudaSlice<u64>,
    converter: &Nv12Converter,
    pool: &mut SlotPool,
    codec_config: &mut Option<Vec<u8>>,
    decode_index: &mut u64,
    frame_duration: u64,
    codec: VideoCodec,
    pitch: u32,
    ptr: u64,
    desc: FrameDesc,
    frames: u32,
    total_frames: u32,
    idr: &[bool],
    units: &mut Vec<EncodedUnit>,
) -> Result<usize, EncodeError> {
    let mut slots = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        match pool.reserve() {
            Some(slot) => slots.push(slot),
            None => break,
        }
    }
    if slots.is_empty() {
        drain_ready(
            bitstreams,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            units,
        )?;
        return Ok(0);
    }
    let batched = slots.len();
    let ptrs: Vec<u64> = slots.iter().map(|slot| device[*slot].ptr).collect();
    stream
        .memcpy_htod(&ptrs, &mut dst_ptrs.slice_mut(0..batched))
        .map_err(|e| EncodeError::Backend(format!("upload dst pointers: {e}")))?;
    converter.convert_batch_ptr(
        stream,
        ptr,
        dst_ptrs,
        &desc,
        batched as u32,
        total_frames,
        false,
        pitch,
    )?;
    stream
        .synchronize()
        .map_err(|e| EncodeError::Backend(format!("sync before encode: {e}")))?;

    for (i, slot) in slots.iter().enumerate() {
        // Every frame in a device batch shares one PTS list, so the caller sets
        // per-frame timestamps through `desc.pts` plus the frame index.
        let pts = desc.pts.ticks() + i as u64 * frame_duration;
        let flags = if idr[i] {
            NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32
        } else {
            0
        };
        pool.submit(*slot, pts);
        match encode_with_retry(
            session,
            &mut inputs[*slot],
            &mut bitstreams[*slot],
            pts,
            flags,
        )? {
            EncodeCall::Emitted => {}
            EncodeCall::NeedMoreInput => continue,
        }
        drain_ready(
            bitstreams,
            pool,
            codec_config,
            decode_index,
            frame_duration,
            codec,
            units,
        )?;
    }
    Ok(batched)
}

/// Time spent inside `nvEncEncodePicture`, process-wide.
///
/// The runtime's `stats.encode_us` covers the whole submit path — queueing,
/// conversion, encode and demux — so it cannot answer "is the encoder the
/// bottleneck". These counters measure only the driver call. They are cumulative
/// and never reset, so a caller samples them before and after a run and takes the
/// difference.
static ENCODE_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENCODE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(total microseconds inside the driver, number of driver calls)`.
///
/// Diagnostic only: read it around a benchmark to see how much of the wall clock
/// is the encoder rather than the code around it.
#[must_use]
pub fn encode_time_totals() -> (u64, u64) {
    (
        ENCODE_MICROS.load(std::sync::atomic::Ordering::Relaxed),
        ENCODE_CALLS.load(std::sync::atomic::Ordering::Relaxed),
    )
}
