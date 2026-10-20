//! `blazar-runtime` — the operator API and Phase-1 pipeline spine
//! (architecture §3, §4, §9).
//!
//! ```text
//! Encoder::new(identity, target, output, sink, index)   // per-deployment
//!   .open_stream(request_id) -> Stream                   // per-generation
//!     .submit_video(FrameRef, scene, boundary)
//!     .flush()
//!     .close()
//!     .into_sink()                                       // recover the sink
//! ```
//!
//! Phase 1 is synchronous and video-only: each `submit_video` runs
//! preprocess → encode → sink inline. The async scheduler / lookahead window
//! (architecture §4) is a Phase-1.x addition that slots in between submit and
//! encode without changing this surface — `submit` already returns immediately
//! with a sequence number, and output already goes to the sink, so making the
//! drain asynchronous is internal.
//!
//! Audio contracts exist in the types (TrackId, StreamParams.audio) but
//! `submit_audio` is not wired in Phase 1 (Opus backend is Phase 2); the method
//! is present and returns `Unsupported` so callers compile against the final
//! shape.

use blazar_calibration::{CalibrationIndex, Query};
use blazar_core::{
    FrameDesc, FrameRef, ModelIdentity, StreamParams, Target, Timebase, VideoCodec,
    VideoEncoderBackend, VideoTrackParams,
};
use blazar_encoder::{EncoderConfig, MockVideoEncoder, VideoEncoder};
use blazar_sink::Sink;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("no calibration resolved for the stream (index missing its default?)")]
    NoCalibration,
    #[error("preprocess error: {0}")]
    Preprocess(#[from] blazar_preprocess::PreprocessError),
    #[error("encode error: {0}")]
    Encode(#[from] blazar_encoder::EncodeError),
    #[error("sink error: {0}")]
    Sink(#[from] blazar_sink::SinkError),
    #[error("submit_video_batch was called with no frames")]
    EmptyBatch,
    #[error("audio is not supported in Phase 1 (Opus backend is Phase 2)")]
    AudioUnsupported,
    #[error("encoder backend {0:?} is not compiled in (enable its cargo feature)")]
    BackendUnavailable(&'static str),
    #[error(
        "no codec configuration: the encoder has not produced one and no override was supplied"
    )]
    MissingCodecConfig,
    #[error("stream already closed")]
    Closed,
    #[error("internal: encoder is not initialized (stream state machine violated)")]
    EncoderNotInitialized,
}

/// Output configuration for a deployment (architecture §3 `output`).
#[derive(Debug, Clone)]
pub struct Output {
    /// Wire/container codec.
    pub codec: VideoCodec,
    /// Backend/calibration identity that produces the wire codec.
    pub backend: VideoEncoderBackend,
    pub container: blazar_core::Container,
    /// Explicit codec configuration override (AV1 `av1C`, H.264 `avcC`, HEVC
    /// `hvcC` body). When `Some`, it is used instead of whatever the backend
    /// derives, which is how the mock (which has no codec) supplies one. Real
    /// encoders derive it from their first keyframe and should be left `None`:
    /// an override that disagrees with the bitstream produces an unplayable
    /// stream, and nothing here cross-checks it.
    pub video_extradata: Option<Vec<u8>>,
    /// Reserved for the Phase-2 Opus path.
    pub audio_extradata: Option<Vec<u8>>,
}

/// A hint the caller may attach to a submit (architecture §4). The only
/// legitimate caller control over scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// A scene cut: force a GOP boundary, don't lookahead across it.
    SceneCut,
}

/// Session-stable per deployment: identity, target, output, sink, calibrations.
/// Generic over the sink so file / WebSocket / WebRTC sinks all work.
pub struct Encoder<S: Sink> {
    identity: ModelIdentity,
    target: Target,
    output: Output,
    sink: S,
    index: CalibrationIndex,
    timebase: Timebase,
}

impl<S: Sink> Encoder<S> {
    pub fn new(
        identity: ModelIdentity,
        target: Target,
        output: Output,
        sink: S,
        index: CalibrationIndex,
    ) -> Self {
        Self {
            identity,
            target,
            output,
            sink,
            index,
            timebase: Timebase::MPEG_90K,
        }
    }

    fn codec_key(&self) -> &'static str {
        match self.output.backend {
            VideoEncoderBackend::SvtAv1 => "svtav1",
            VideoEncoderBackend::NvencAv1 => "nvenc_av1",
            VideoEncoderBackend::NvencH264 => "nvenc_h264",
            VideoEncoderBackend::NvencHevc => "nvenc_hevc",
            VideoEncoderBackend::Rav1e => "rav1e",
            VideoEncoderBackend::X264 => "x264",
            VideoEncoderBackend::X265 => "x265",
            VideoEncoderBackend::Vp9 => "vp9",
        }
    }

    /// Bind the runtime-owned request id once and return the per-generation
    /// handle (architecture §3). Blazar does not mint the id — the runtime owns
    /// it. The sink is opened lazily on the first submit, once frame dimensions
    /// and fps are known from the `FrameDesc` (the runtime does not expose them
    /// up front; architecture §8/§10).
    ///
    /// This consumes the encoder and the stream owns it: the encoder's only
    /// stateful purpose is to serve one stream, and ownership is what lets the
    /// Python binding (or any embedder) hold the stream without a self-borrow.
    /// Per-request setup that must be shared (calibrations) belongs to the
    /// caller; `CalibrationIndex` is `Clone`.
    pub fn open_stream(self, request_id: impl Into<String>) -> Stream<S> {
        Stream {
            enc: self,
            request_id: request_id.into(),
            encoder: None,
            seq: 0,
            encoder_open: false,
            sink_open: false,
            last_desc: None,
            closed: false,
        }
    }
}

/// Per-generation handle. Owns the encoder and holds the one stateful video
/// encoder for this stream.
pub struct Stream<S: Sink> {
    enc: Encoder<S>,
    request_id: String,
    encoder: Option<Box<dyn VideoEncoder>>,
    seq: u64,
    /// The encoder is built as soon as the first frame's descriptor is known;
    /// the sink is opened only once a codec configuration exists (after the
    /// first encode for real backends), then writes start.
    encoder_open: bool,
    sink_open: bool,
    /// Descriptor of the most recent frame, so a sink that only opens on the
    /// first emitted unit (lookahead encoders) can still be created at flush.
    last_desc: Option<FrameDesc>,
    closed: bool,
}

impl<S: Sink> Stream<S> {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Consume the stream and return its sink, for harnesses that inspect the
    /// output after `close()`.
    pub fn into_sink(self) -> S {
        self.enc.sink
    }

    /// Build the encoder from the first frame's descriptor. The sink is *not*
    /// opened here: a real encoder only reveals its codec configuration when it
    /// emits its first keyframe, so `ensure_sink` runs when the first unit is
    /// actually produced (which may be several frames in for lookahead encoders).
    fn ensure_encoder(&mut self, frame: &FrameRef) -> Result<(), RuntimeError> {
        self.ensure_encoder_for(&frame.desc)
    }

    fn ensure_encoder_for(&mut self, desc: &FrameDesc) -> Result<(), RuntimeError> {
        if self.encoder_open {
            return Ok(());
        }

        // Resolve the calibration. Scene defaults to agnostic ("*") in Phase 1;
        // a real classifier feeds this later (architecture §5).
        let query = Query::new(
            &self.enc.identity,
            "*",
            self.enc.codec_key(),
            self.enc.target,
        );
        let cal = self
            .enc
            .index
            .resolve(&query)
            .ok_or(RuntimeError::NoCalibration)?;
        tracing::info!(
            request_id = %self.request_id,
            source = ?cal.source,
            origin = %cal.origin,
            "resolved calibration"
        );
        let cfg = EncoderConfig::from_calibration(cal, self.enc.codec_key());
        self.encoder = Some(build_encoder(
            &cfg,
            desc,
            self.enc.output.codec,
            self.enc.output.backend,
            self.enc.timebase,
        )?);
        self.encoder_open = true;
        Ok(())
    }

    /// Resolve the container codec configuration: an explicit override wins,
    /// otherwise the backend must have produced one.
    fn resolve_codec_config(&self) -> Result<Vec<u8>, RuntimeError> {
        if let Some(extradata) = &self.enc.output.video_extradata {
            if extradata.is_empty() {
                return Err(RuntimeError::MissingCodecConfig);
            }
            return Ok(extradata.clone());
        }
        self.encoder
            .as_ref()
            .and_then(|enc| enc.codec_config())
            .filter(|extradata| !extradata.is_empty())
            .ok_or(RuntimeError::MissingCodecConfig)
    }

    /// Open the sink once the first frame has been encoded (or the override is
    /// available). Dimensions and fps come from the frame descriptor, because
    /// the runtime does not expose them up front (architecture §8/§10).
    /// Open the sink from the last frame's descriptor, if it is not open yet.
    fn open_sink_if_needed(&mut self) -> Result<(), RuntimeError> {
        if self.sink_open {
            return Ok(());
        }
        if let Some(desc) = self.last_desc {
            self.ensure_sink(&desc)?;
        }
        Ok(())
    }

    fn ensure_sink(&mut self, desc: &blazar_core::FrameDesc) -> Result<(), RuntimeError> {
        if self.sink_open {
            return Ok(());
        }
        let extradata = self.resolve_codec_config()?;
        let params = StreamParams {
            timebase: self.enc.timebase,
            video: VideoTrackParams {
                codec: self.enc.output.codec,
                width: desc.width,
                height: desc.height,
                fps: desc.fps,
                extradata,
            },
            audio: None, // video-only in Phase 1
            container: self.enc.output.container,
        };
        self.enc.sink.open(params)?;
        self.sink_open = true;
        Ok(())
    }

    /// Enqueue + (Phase 1) synchronously encode one video frame. Returns the
    /// submit sequence number (architecture §4: submit returns immediately with
    /// a seq for backpressure/accounting; output goes to the sink, not here).
    pub fn submit_video(
        &mut self,
        frame: FrameRef,
        _scene: Option<&str>,
        boundary: Option<Boundary>,
    ) -> Result<u64, RuntimeError> {
        if self.closed {
            return Err(RuntimeError::Closed);
        }
        self.ensure_encoder(&frame)?;

        if boundary == Some(Boundary::SceneCut) {
            if let Some(enc) = self.encoder.as_mut() {
                enc.request_keyframe();
            }
        }

        let desc = frame.desc;
        let units = {
            // A missing encoder here means the state machine was violated
            // ("open" without a built encoder). This must be an error, not a
            // panic: the runtime is embedded behind the Python FFI, where a
            // panic takes the interpreter with it.
            let encoder = self
                .encoder
                .as_mut()
                .ok_or(RuntimeError::EncoderNotInitialized)?;
            // Preprocessing happens inside the backend now (GPU for NVENC), so
            // this call does not do per-pixel work on the caller's thread.
            encoder.encode_frame(frame)?
        };
        // Open the sink only once there is something to write. Encoders with
        // lookahead or B-frames return nothing for the first frames; the
        // configuration arrives with their first packet, not with the first
        // submit.
        self.last_desc.get_or_insert(desc);
        if !units.is_empty() {
            self.ensure_sink(&desc)?;
            for u in &units {
                self.enc.sink.write(u)?;
            }
        }

        let seq = self.seq;
        self.seq += 1;
        Ok(seq)
    }

    /// Submit a batch whose frames already live on the GPU.
    ///
    /// `ptr` points at `frames` contiguous BCTHW frames described by `desc`.
    /// The backend reads it directly, and `keep_alive` is dropped once the
    /// encode has consumed it (the caller's tensor must outlive the encode).
    pub fn submit_video_batch_device(
        &mut self,
        ptr: u64,
        desc: FrameDesc,
        frames: u32,
        boundary: Option<Boundary>,
        keep_alive: Option<Box<dyn std::any::Any + Send>>,
    ) -> Result<u64, RuntimeError> {
        if self.closed {
            return Err(RuntimeError::Closed);
        }
        if frames == 0 {
            return Err(RuntimeError::EmptyBatch);
        }
        self.ensure_encoder_for(&desc)?;
        if boundary == Some(Boundary::SceneCut) {
            if let Some(enc) = self.encoder.as_mut() {
                enc.request_keyframe();
            }
        }
        let units = {
            let encoder = self
                .encoder
                .as_mut()
                .ok_or(RuntimeError::EncoderNotInitialized)?;
            encoder.encode_device_batch(ptr, desc, frames, keep_alive)?
        };
        self.last_desc.get_or_insert(desc);
        if !units.is_empty() {
            self.ensure_sink(&desc)?;
            for u in &units {
                self.enc.sink.write(u)?;
            }
        }
        let seq = self.seq;
        self.seq += 1;
        Ok(seq)
    }

    /// Submit several frames in one encoder submission. GPU backends convert
    /// them with a single kernel launch; the sink receives every unit in order.
    ///
    /// Returns one submission id for the batch. The frames must share geometry
    /// and format.
    pub fn submit_video_batch(
        &mut self,
        frames: Vec<FrameRef>,
        boundary: Option<Boundary>,
    ) -> Result<u64, RuntimeError> {
        if self.closed {
            return Err(RuntimeError::Closed);
        }
        let Some(first) = frames.first() else {
            return Err(RuntimeError::EmptyBatch);
        };
        let desc = first.desc;
        self.ensure_encoder(first)?;

        if boundary == Some(Boundary::SceneCut) {
            if let Some(enc) = self.encoder.as_mut() {
                enc.request_keyframe();
            }
        }

        let units = {
            let encoder = self
                .encoder
                .as_mut()
                .ok_or(RuntimeError::EncoderNotInitialized)?;
            encoder.encode_batch(frames)?
        };
        // Open the sink only once there is something to write. Encoders with
        // lookahead or B-frames return nothing for the first frames; the
        // configuration arrives with their first packet, not with the first
        // submit.
        self.last_desc.get_or_insert(desc);
        if !units.is_empty() {
            self.ensure_sink(&desc)?;
            for u in &units {
                self.enc.sink.write(u)?;
            }
        }

        let seq = self.seq;
        self.seq += 1;
        Ok(seq)
    }

    /// Audio submit — present for API shape; unsupported in Phase 1.
    pub fn submit_audio(&mut self) -> Result<u64, RuntimeError> {
        Err(RuntimeError::AudioUnsupported)
    }

    /// Drain buffered frames without ending the stream (segment boundary).
    pub fn flush(&mut self) -> Result<(), RuntimeError> {
        if let Some(enc) = self.encoder.as_mut() {
            let units = enc.flush()?;
            if !units.is_empty() {
                self.open_sink_if_needed()?;
            }
            for u in &units {
                self.enc.sink.write(u)?;
            }
        }
        if self.sink_open {
            self.enc.sink.flush()?;
        }
        Ok(())
    }

    /// Finalize this stream: drain the encoder, close the sink.
    pub fn close(&mut self) -> Result<(), RuntimeError> {
        if self.closed {
            return Ok(());
        }
        if let Some(enc) = self.encoder.as_mut() {
            let units = enc.finish()?;
            eprintln!("DEBUG close: {} units, sink_open={}", units.len(), self.sink_open);
            if !units.is_empty() {
                self.open_sink_if_needed()?;
            }
            for u in &units {
                self.enc.sink.write(u)?;
            }
        }
        if self.sink_open {
            self.enc.sink.close()?;
        }
        self.closed = true;
        Ok(())
    }
}

/// Build the encoder for a backend.
///
/// `SvtAv1`/`Rav1e` still use the mock: neither real backend has landed, and the
/// mock implements the full `VideoEncoder` contract. `NvencAv1` is real, but
/// only when the `nvenc` feature is enabled; otherwise the deployment gets an
/// explicit error rather than a silent mock.
/// Wrap a CPU encoder so device tensors are converted on the GPU.
#[cfg(feature = "gpu-preprocess")]
#[allow(dead_code)] // only called from the codec arms below
fn wrap_gpu(enc: Box<dyn VideoEncoder>) -> Result<Box<dyn VideoEncoder>, RuntimeError> {
    Ok(Box::new(
        blazar_encoder::gpu_preprocess::GpuPreprocessEncoder::new(enc)?,
    ))
}

#[cfg(not(feature = "gpu-preprocess"))]
#[allow(dead_code)] // only called from the codec arms below
fn wrap_gpu(enc: Box<dyn VideoEncoder>) -> Result<Box<dyn VideoEncoder>, RuntimeError> {
    Ok(enc)
}

fn build_encoder(
    cfg: &EncoderConfig,
    desc: &FrameDesc,
    codec: VideoCodec,
    backend: VideoEncoderBackend,
    timebase: Timebase,
) -> Result<Box<dyn VideoEncoder>, RuntimeError> {
    match backend {
        // Without the feature the mock keeps the pipeline contract testable on
        // machines that do not have libsvtav1.
        VideoEncoderBackend::SvtAv1 => {
            #[cfg(feature = "enc-svtav1")]
            {
                let frame_duration = timebase.ticks_per_frame(desc.fps);
                wrap_gpu(Box::new(blazar_encoder::svtav1::SvtAv1Encoder::new(
                    cfg,
                    desc.width,
                    desc.height,
                    desc.fps,
                    frame_duration,
                )?))
            }
            #[cfg(not(feature = "enc-svtav1"))]
            {
                Ok(Box::new(MockVideoEncoder::new(cfg, desc.fps, timebase)))
            }
        }
        VideoEncoderBackend::Rav1e => Ok(Box::new(MockVideoEncoder::new(cfg, desc.fps, timebase))),
        VideoEncoderBackend::X264 => {
            #[cfg(feature = "enc-x264")]
            {
                let frame_duration = timebase.ticks_per_frame(desc.fps);
                wrap_gpu(Box::new(blazar_encoder::x264::X264Encoder::new(
                    cfg,
                    desc.width,
                    desc.height,
                    desc.fps,
                    frame_duration,
                )?))
            }
            #[cfg(not(feature = "enc-x264"))]
            {
                Err(RuntimeError::BackendUnavailable("x264"))
            }
        }
        VideoEncoderBackend::X265 => {
            #[cfg(feature = "enc-x265")]
            {
                let frame_duration = timebase.ticks_per_frame(desc.fps);
                wrap_gpu(Box::new(blazar_encoder::x265::X265Encoder::new(
                    cfg,
                    desc.width,
                    desc.height,
                    desc.fps,
                    frame_duration,
                )?))
            }
            #[cfg(not(feature = "enc-x265"))]
            {
                Err(RuntimeError::BackendUnavailable("x265"))
            }
        }
        VideoEncoderBackend::Vp9 => {
            #[cfg(feature = "enc-vp9")]
            {
                let frame_duration = timebase.ticks_per_frame(desc.fps);
                wrap_gpu(Box::new(blazar_encoder::vp9::Vp9Encoder::new(
                    cfg,
                    desc.width,
                    desc.height,
                    desc.fps,
                    frame_duration,
                )?))
            }
            #[cfg(not(feature = "enc-vp9"))]
            {
                Err(RuntimeError::BackendUnavailable("vp9"))
            }
        }
        VideoEncoderBackend::NvencAv1
        | VideoEncoderBackend::NvencH264
        | VideoEncoderBackend::NvencHevc => {
            let expected = match backend {
                VideoEncoderBackend::NvencH264 => VideoCodec::H264,
                VideoEncoderBackend::NvencHevc => VideoCodec::Hevc,
                _ => VideoCodec::Av1,
            };
            if codec != expected {
                return Err(RuntimeError::Encode(blazar_encoder::EncodeError::Backend(
                    format!("{backend:?} produces {expected:?}, not {codec:?}"),
                )));
            }
            #[cfg(feature = "nvenc")]
            {
                Ok(Box::new(blazar_encoder::nvenc::NvencEncoder::new(
                    cfg,
                    desc.width,
                    desc.height,
                    desc.fps,
                    timebase,
                    codec,
                )?))
            }
            #[cfg(not(feature = "nvenc"))]
            {
                let _ = (cfg, desc, codec, timebase);
                Err(RuntimeError::BackendUnavailable("nvenc"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::{
        ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, Layout, Pts,
    };
    use blazar_sink::fmp4::DebugFileSink;

    #[test]
    fn submit_without_an_encoder_errors_instead_of_panicking() {
        let enc = Encoder::new(
            ModelIdentity {
                family: "wan".into(),
                version: "2.1".into(),
                quant: "bf16".into(),
                size: "1.3b".into(),
            },
            Target::Balanced,
            Output {
                codec: VideoCodec::Av1,
                backend: VideoEncoderBackend::SvtAv1,
                container: Container::FragmentedMp4,
                video_extradata: None,
                audio_extradata: None,
            },
            DebugFileSink::new(Vec::new()),
            CalibrationIndex::new(),
        );
        let mut stream = enc.open_stream("req-state-machine");

        // Simulate the violation: the flag claims an encoder exists, but it does
        // not. Python callers can drive this surface, so it must error cleanly.
        stream.encoder_open = true;

        let frame = FrameRef::new(
            FrameBuffer::host_owned(vec![0u8; 2 * 2 * 3 * 4]),
            FrameDesc {
                width: 2,
                height: 2,
                dtype: Dtype::F32,
                range: ColorRange::Unit01,
                layout: Layout::Thwc,
                colorspace: ColorSpace::AssumedBt709,
                fps: 24.0,
                pts: Pts(0),
            },
        );

        let err = stream
            .submit_video(frame, None, None)
            .expect_err("must not panic");
        assert!(matches!(err, RuntimeError::EncoderNotInitialized));
    }
}

#[cfg(test)]
mod send_assertions {
    use super::*;
    use blazar_sink::fmp4::Fmp4FileSink;
    use std::fs::File;
    use std::io::BufWriter;

    /// The Python binding moves a stream to a worker thread. If this stops
    /// compiling, encoding will block the caller again.
    #[test]
    fn stream_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Stream<Fmp4FileSink<BufWriter<File>>>>();
    }
}
