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

/// Output bitstreams kept in flight. NVENC's lookahead is a handful of frames;
/// 16 leaves generous headroom and costs only a few KiB per slot.
const POOL_CAPACITY: usize = 16;
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
    }
}

fn open_session(params: &NvencParams, codec: VideoCodec) -> Result<Session, EncodeError> {
    let guid = guid_for(codec);
    let ctx = match &*CUDA_CONTEXT {
        Ok(ctx) => Arc::clone(ctx),
        Err(e) => return Err(EncodeError::Backend(e.clone())),
    };
    ctx.bind_to_thread()
        .map_err(|e| EncodeError::Backend(format!("bind CUDA context: {e:?}")))?;

    let encoder = Encoder::initialize_with_cuda(ctx)
        .map_err(|e| EncodeError::Backend(format!("NVENC init: {e:?}")))?;
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
    apply_config(&mut preset.presetCfg, params);

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
fn apply_config(cfg: &mut NV_ENC_CONFIG, params: &NvencParams) {
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

    // The preprocess stage emits full-range YUV using the BT.709 matrix and
    // never tags it. Tag it here so players do not have to guess.
    // SAFETY: the union variant matches the AV1 codec GUID used for the session.
    unsafe {
        let av1 = &mut cfg.encodeCodecConfig.av1Config;
        if params.color.bt709 {
            av1.colorPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
            av1.transferCharacteristics =
                NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
            av1.matrixCoefficients = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
        }
        av1.colorRange = u32::from(params.color.full_range);
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
            converter.convert_raw(stream, staging, device[slot].ptr, desc, pitch)?;
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
            Command::Encode { reply, .. } | Command::EncodeFrame { reply, .. } => {
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
        match session.encode_picture(input, output, params) {
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
                VideoCodec::Av1 => unreachable!("handled above"),
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
        let status = unsafe { sys::cuMemAllocPitch_v2(&mut ptr, &mut pitch, width as usize, rows as usize, 4) };
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
