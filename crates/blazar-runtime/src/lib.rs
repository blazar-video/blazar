//! `blazar-runtime` — the operator API and Phase-1 pipeline spine
//! (architecture §3, §4, §9).
//!
//! ```text
//! Encoder::new(identity, target, output, sink, index)   // per-deployment
//!   .open_stream(request_id) -> Stream                   // per-generation
//!     .submit_video(FrameRef, scene, boundary)
//!     .flush()
//!     .close()
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
    FrameRef, ModelIdentity, StreamParams, Target, Timebase, VideoCodec, VideoTrackParams,
};
use blazar_encoder::{EncoderConfig, MockVideoEncoder, VideoEncoder};
use blazar_preprocess::{CpuPreprocess, PreprocessStage};
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
    #[error("audio is not supported in Phase 1 (Opus backend is Phase 2)")]
    AudioUnsupported,
    #[error("stream already closed")]
    Closed,
}

/// Output configuration for a deployment (architecture §3 `output`).
#[derive(Debug, Clone, Copy)]
pub struct Output {
    pub codec: VideoCodec,
    pub container: blazar_core::Container,
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
    preprocess: CpuPreprocess,
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
            preprocess: CpuPreprocess::new(),
        }
    }

    /// Consume the encoder and return its sink. Intended for tests/harnesses
    /// that need to inspect sink output after `close()`.
    pub fn into_sink_for_test(self) -> S {
        self.sink
    }

    fn codec_key(&self) -> &'static str {
        match self.output.codec {
            VideoCodec::SvtAv1 => "svtav1",
            VideoCodec::NvencAv1 => "nvenc_av1",
            VideoCodec::Rav1e => "rav1e",
        }
    }

    /// Bind the runtime-owned request id once and return a per-generation
    /// handle (architecture §3). Blazar does not mint the id — the runtime owns
    /// it. The sink is opened lazily on the first submit, once frame dimensions
    /// and fps are known from the `FrameDesc` (the runtime does not expose them
    /// up front; architecture §8/§10).
    pub fn open_stream(&mut self, request_id: impl Into<String>) -> Stream<'_, S> {
        Stream {
            enc: self,
            request_id: request_id.into(),
            encoder: None,
            seq: 0,
            opened: false,
            closed: false,
        }
    }
}

/// Per-generation handle. Holds the one stateful video encoder for this stream.
pub struct Stream<'a, S: Sink> {
    enc: &'a mut Encoder<S>,
    request_id: String,
    encoder: Option<MockVideoEncoder>,
    seq: u64,
    opened: bool,
    closed: bool,
}

impl<'a, S: Sink> Stream<'a, S> {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Open the sink + build the encoder from the first frame's descriptor.
    fn ensure_open(&mut self, frame: &FrameRef) -> Result<(), RuntimeError> {
        if self.opened {
            return Ok(());
        }
        let desc = &frame.desc;

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

        let params = StreamParams {
            timebase: self.enc.timebase,
            video: VideoTrackParams {
                codec: self.enc.output.codec,
                width: desc.width,
                height: desc.height,
                fps: desc.fps,
            },
            audio: None, // video-only in Phase 1
            container: self.enc.output.container,
        };
        self.enc.sink.open(params)?;
        self.encoder = Some(MockVideoEncoder::new(&cfg, desc.fps, self.enc.timebase));
        self.opened = true;
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
        self.ensure_open(&frame)?;

        if boundary == Some(Boundary::SceneCut) {
            if let Some(enc) = self.encoder.as_mut() {
                enc.request_keyframe();
            }
        }

        let pre = self.enc.preprocess.process(&frame)?;
        let units = self.encoder.as_mut().unwrap().encode(&pre)?;
        for u in &units {
            self.enc.sink.write(u)?;
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
            for u in enc.flush()? {
                self.enc.sink.write(&u)?;
            }
        }
        Ok(())
    }

    /// Finalize this stream: drain the encoder, close the sink.
    pub fn close(&mut self) -> Result<(), RuntimeError> {
        if self.closed {
            return Ok(());
        }
        if let Some(enc) = self.encoder.as_mut() {
            for u in enc.finish()? {
                self.enc.sink.write(&u)?;
            }
        }
        if self.opened {
            self.enc.sink.close()?;
        }
        self.closed = true;
        Ok(())
    }
}
