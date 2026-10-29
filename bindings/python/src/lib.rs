// SPDX-License-Identifier: Apache-2.0
//! Python bindings for the Blazar runtime.
//!
//! Two classes, deliberately thin:
//!
//! * `Calibrations` — a parsed calibration directory. Cheap to keep; reused
//!   across requests.
//! * `Encoder` — one stream, encoded on its own Rust worker thread. `submit`
//!   copies the frame and enqueues it; the worker preprocesses, encodes and
//!   writes. The caller keeps running and drains `poll()` for progress.
//!
//! Why a worker: releasing the GIL around the encode would not help the caller
//! (only other threads), and the caller is the decode loop. The worker never
//! touches Python; the GIL is released only where the calling thread must wait
//! (a full job queue, `flush`, `close`).
//!
//! Rules the binding enforces instead of guessing:
//! * `layout` and `range` are required on every submit; the runtimes never tag
//!   colorspace, so Blazar records the BT.709 assumption.
//! * Frames must be float32. numpy has no bf16 dtype; cast first.
//! * Width/height/fps are fixed by the first frame. A change is an error, not a
//!   new segment: the container's metadata is written once.
//!
//! Not here yet: audio, device tensors, and any sink other than a file.

use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use blazar_calibration::CalibrationIndex;
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, EncodedUnit, FrameBuffer, FrameDesc, FrameRef,
    Layout, ModelIdentity, Pts, StreamParams, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_runtime::{Boundary, Encoder as RuntimeEncoder, Output, Stream};
use blazar_sink::fmp4::Fmp4FileSink;
use blazar_sink::{Feedback, Sink, SinkError};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type FileSink = Fmp4FileSink<BufWriter<File>>;

/// Where encoded units go: a fragmented MP4 file, or back to the caller.
enum SinkTarget {
    File(Box<ReportingSink>),
    Memory(Box<blazar_sink::MemorySink>),
}

impl Sink for SinkTarget {
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError> {
        match self {
            SinkTarget::File(sink) => sink.open(params),
            SinkTarget::Memory(sink) => sink.open(params),
        }
    }

    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        match self {
            SinkTarget::File(sink) => sink.write(unit),
            SinkTarget::Memory(sink) => sink.write(unit),
        }
    }

    fn flush(&mut self) -> Result<(), SinkError> {
        match self {
            SinkTarget::File(sink) => sink.flush(),
            SinkTarget::Memory(sink) => sink.flush(),
        }
    }

    fn close(&mut self) -> Result<(), SinkError> {
        match self {
            SinkTarget::File(sink) => sink.close(),
            SinkTarget::Memory(sink) => sink.close(),
        }
    }

    fn poll_feedback(&mut self) -> Option<blazar_sink::Feedback> {
        match self {
            SinkTarget::File(sink) => sink.poll_feedback(),
            SinkTarget::Memory(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Worker plumbing: plain Rust types only, no Python objects.
// ---------------------------------------------------------------------------

struct FrameJob {
    seq: u64,
    bytes: Vec<u8>,
    desc: FrameDesc,
    boundary: Option<Boundary>,
}

enum Job {
    Frame(Box<FrameJob>),
    /// Several frames in one encoder submission: one GPU kernel launch.
    Batch(Vec<FrameJob>),
    /// Frames that are already on the GPU. `keep_alive` holds the tensor.
    Device {
        ptr: u64,
        desc: FrameDesc,
        frames: u32,
        boundary: Option<Boundary>,
        keep_alive: Option<Box<dyn std::any::Any + Send>>,
    },
    /// Interleaved float samples, encoded on the worker's stream.
    Audio {
        samples: Vec<f32>,
        sample_rate: u32,
        reply: SyncSender<Result<(), String>>,
    },
    Flush(SyncSender<Result<(), String>>),
    Close(SyncSender<Result<(), String>>),
}

/// One unit the sink accepted, recorded so the event can report size and
/// whether a fragment boundary was crossed.
struct WrittenUnit {
    keyframe: bool,
    bytes: usize,
    pts: u64,
    /// Decode timestamp, kept so flush/close events can report it too.
    dts: u64,
}

/// Sink decorator that records what was written on each `submit_video` call.
/// The runtime's submit only returns a sequence number; the Python caller needs
/// the encoded size and keyframe flag, and this is the least invasive place to
/// get them.
struct ReportingSink {
    inner: FileSink,
    written: Arc<Mutex<Vec<WrittenUnit>>>,
}

impl Sink for ReportingSink {
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError> {
        self.inner.open(params)
    }

    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        self.written
            .lock()
            .map_err(|_| SinkError::InvalidConfiguration("unit recorder poisoned".into()))?
            .push(WrittenUnit {
                keyframe: unit.keyframe,
                bytes: unit.bytes.len(),
                pts: unit.pts.0,
                dts: unit.dts.0,
            });
        self.inner.write(unit)
    }

    fn flush(&mut self) -> Result<(), SinkError> {
        self.inner.flush()
    }

    fn close(&mut self) -> Result<(), SinkError> {
        self.inner.close()
    }

    fn poll_feedback(&mut self) -> Option<Feedback> {
        self.inner.poll_feedback()
    }
}

enum WorkerEvent {
    Encoded {
        seq: u64,
        pts: u64,
        /// Decode timestamp. Together with `pts` this is what a muxer or a
        /// reordering consumer needs; with B-frames it is the monotonic one.
        dts: Option<u64>,
        keyframe: bool,
        units: usize,
        bytes: usize,
        encode_us: u64,
        queue_depth: usize,
        /// Encoded elementary-stream bytes, set when the stream writes to memory
        /// rather than to a file.
        data: Option<Vec<u8>>,
    },
    Failed {
        seq: Option<u64>,
        message: String,
    },
    Closed,
}

#[derive(Default)]
struct WorkerStats {
    encoded: u64,
    bytes: u64,
    last_encode_us: u64,
    max_encode_us: u64,
    failed: bool,
}

/// Identity of one session, logged by the worker so the encode can be traced
/// back to this process rather than to whatever wrote the file.
struct SessionInfo {
    request_id: String,
    backend: String,
    codec: String,
    target: String,
    calibrations: usize,
    queue_depth: usize,
}

struct Shared {
    events: Sender<WorkerEvent>,
    error: Mutex<Option<String>>,
    stats: Mutex<WorkerStats>,
    queued: AtomicUsize,
    /// Backend that actually answered, published by the worker once the stream
    /// opens. `backend="auto"` cannot know it before the first frame.
    resolved_backend: Mutex<Option<String>>,
    /// Layout fixed at open (`"thwc"`/`"bcthw"`), or `None` until the first
    /// frame decides. A stream's layout does not change: a container written
    /// with two interpretations of the same shape is unplayable.
    layout: Mutex<Option<String>>,
    /// `"host"` or `"device"`, decided by the first submit. Switching would put
    /// half a clip on each path, so it is refused instead.
    residency: Mutex<Option<String>>,
}

/// Hand each unit the memory sink has buffered to the caller.
///
/// Encoders with B-frames or a lookahead window emit nothing at submit time, so
/// this runs after every submit and again after flush/close, where the last units
/// appear. Draining is what removes them from the buffer, so nothing is delivered
/// twice.
fn drain_memory_units(shared: &Arc<Shared>, memory: &Option<blazar_sink::MemoryHandle>) {
    let Some(handle) = memory else {
        return;
    };
    for unit in handle.drain() {
        let bytes = unit.bytes.len();
        if let Ok(mut stats) = shared.stats.lock() {
            stats.bytes += bytes as u64;
        }
        let _ = shared.events.send(WorkerEvent::Encoded {
            seq: unit.dts,
            pts: unit.pts,
            dts: Some(unit.dts),
            keyframe: unit.keyframe,
            units: 1,
            bytes,
            encode_us: 0,
            queue_depth: shared.queued.load(Ordering::Relaxed),
            data: Some(unit.bytes),
        });
    }
}

/// Record which backend answered, once the stream has opened. `backend="auto"`
/// resolves during the first submit, so this is the earliest it is knowable.
fn publish_backend<S: blazar_sink::Sink>(stream: &blazar_runtime::Stream<S>, shared: &Shared) {
    if let Some(backend) = stream.resolved_backend() {
        let name = blazar_runtime::backend_selection::backend_name(backend).to_owned();
        if let Ok(mut slot) = shared.resolved_backend.lock() {
            if slot.as_deref() != Some(name.as_str()) {
                *slot = Some(name);
            }
        }
    }
}



/// Is this a `torch.Tensor`? torch may not be installed, in which case nothing
/// can be one.
fn is_torch_tensor(py: Python<'_>, frame: &Bound<'_, PyAny>) -> bool {
    match py.import("torch").and_then(|t| t.getattr("Tensor")) {
        Ok(class) => frame.is_instance(&class).unwrap_or(false),
        Err(_) => false,
    }
}

fn warn_deprecated(py: Python<'_>, message: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            message,
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
        ),
    )?;
    Ok(())
}

/// Resolve `backend` + `codec` into the pair the runtime builds.
///
/// `backend` names an implementation (`nvenc`, `videotoolbox`, `x264`, `x265`,
/// `svtav1`, `vp9`, `mock`) or `auto`. The old pair-shaped spellings
/// (`nvenc_h264`) still parse, with a DeprecationWarning: they restate `codec`,
/// and the two could disagree.
fn parse_backend_for(
    py: Python<'_>,
    backend: &str,
    codec: blazar_core::VideoCodec,
) -> PyResult<blazar_core::VideoEncoderBackend> {
    use blazar_core::{VideoCodec, VideoEncoderBackend as B};
    if backend.contains('_') {
        return Err(PyValueError::new_err(format!(
            "backend={backend:?} restates the codec; name the implementation and pass codec= \
             (for example backend=\"nvenc\", codec=\"h264\")"
        )));
    }
    let resolved = match (backend, codec) {
        ("nvenc", VideoCodec::H264) => B::NvencH264,
        ("nvenc", VideoCodec::Hevc) => B::NvencHevc,
        ("nvenc", VideoCodec::Av1) => B::NvencAv1,
        ("videotoolbox", VideoCodec::H264) => B::VideoToolboxH264,
        ("videotoolbox", VideoCodec::Hevc) => B::VideoToolboxHevc,
        ("x264", VideoCodec::H264) => B::X264,
        ("x265", VideoCodec::Hevc) => B::X265,
        ("svtav1", VideoCodec::Av1) => B::SvtAv1,
        ("vp9", VideoCodec::Vp9) => B::Vp9,
        ("mock", _) => B::Mock,
        (name, codec) => {
            return Err(PyValueError::new_err(format!(
                "backend {name:?} does not encode {codec:?}"
            )))
        }
    };
    Ok(resolved)
}

fn worker_main(
    mut stream: Stream<SinkTarget>,
    jobs: Receiver<Job>,
    shared: Arc<Shared>,
    written: Arc<Mutex<Vec<WrittenUnit>>>,
    memory: Option<blazar_sink::MemoryHandle>,
    info: SessionInfo,
) {
    tracing::info!(
        target: "blazar",
        request_id = %info.request_id,
        backend = %info.backend,
        codec = %info.codec,
        target = %info.target,
        calibrations = info.calibrations,
        queue_depth = info.queue_depth,
        "encoder worker started"
    );
    let mut finished = false;
    loop {
        let job = match jobs.recv() {
            Ok(job) => job,
            Err(_) => break, // sender dropped: finalize below
        };
        match job {
            Job::Audio {
                samples,
                sample_rate,
                reply,
            } => {
                shared.queued.fetch_sub(1, Ordering::Relaxed);
                let result = stream
                    .submit_audio(&samples, sample_rate)
                    .map(|_| ())
                    .map_err(|e| e.to_string());
                let _ = reply.send(result);
            }
            Job::Frame(job) => {
                shared.queued.fetch_sub(1, Ordering::Relaxed);
                publish_backend(&stream, &shared);
                let started = Instant::now();
                let frame = FrameRef::new(FrameBuffer::host_owned(job.bytes), job.desc);
                let result = stream.submit_video(frame, None, job.boundary);
                let elapsed = started.elapsed();
                match result {
                    Ok(seq) => {
                        let units: Vec<WrittenUnit> = written
                            .lock()
                            .map(|mut units| units.drain(..).collect())
                            .unwrap_or_default();
                        let bytes: usize = units.iter().map(|u| u.bytes).sum();
                        let keyframe = units.iter().any(|u| u.keyframe);
                        let pts = units.first().map(|u| u.pts).unwrap_or(job.desc.pts.0);
                        let encode_us = elapsed.as_micros() as u64;
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.encoded += 1;
                            stats.bytes += bytes as u64;
                            stats.last_encode_us = encode_us;
                            stats.max_encode_us = stats.max_encode_us.max(encode_us);
                        }
                        tracing::debug!(
                            target: "blazar",
                            seq,
                            pts,
                            keyframe,
                            units = units.len(),
                            bytes,
                            encode_us,
                            "frame encoded"
                        );
                        if let Ok(stats) = shared.stats.lock() {
                            if stats.encoded > 0 && stats.encoded % 24 == 0 {
                                tracing::info!(
                                    target: "blazar",
                                    frames = stats.encoded,
                                    bytes = stats.bytes,
                                    "encode progress"
                                );
                            }
                        }
                        let _ = shared.events.send(WorkerEvent::Encoded {
                            seq,
                            pts,
                            keyframe,
                            dts: units.first().map(|u| u.dts),
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                            data: None,
                        });
                        drain_memory_units(&shared, &memory);
                    }
                    Err(err) => {
                        let message = err.to_string();
                        if let Ok(mut slot) = shared.error.lock() {
                            *slot = Some(message.clone());
                        }
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.failed = true;
                        }
                        let _ = shared.events.send(WorkerEvent::Failed {
                            seq: Some(job.seq),
                            message,
                        });
                        break;
                    }
                }
            }
            Job::Batch(jobs) => {
                shared.queued.fetch_sub(1, Ordering::Relaxed);
                let started = Instant::now();
                let count = jobs.len();
                let first_seq = jobs.first().map(|job| job.seq).unwrap_or(0);
                let boundary = jobs.first().and_then(|job| job.boundary);
                let pts = jobs
                    .first()
                    .map(|job| job.desc.pts.0)
                    .unwrap_or_default();
                let frames: Vec<FrameRef> = jobs
                    .into_iter()
                    .map(|job| FrameRef::new(FrameBuffer::host_owned(job.bytes), job.desc))
                    .collect();
                let result = stream.submit_video_batch(frames, boundary);
                let elapsed = started.elapsed();
                match result {
                    Ok(seq) => {
                        let units: Vec<WrittenUnit> = written
                            .lock()
                            .map(|mut units| units.drain(..).collect())
                            .unwrap_or_default();
                        let bytes: usize = units.iter().map(|u| u.bytes).sum();
                        let keyframe = units.iter().any(|u| u.keyframe);
                        let encode_us = elapsed.as_micros() as u64;
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.encoded += count as u64;
                            stats.bytes += bytes as u64;
                            stats.last_encode_us = encode_us;
                            stats.max_encode_us = stats.max_encode_us.max(encode_us);
                        }
                        tracing::debug!(
                            target: "blazar",
                            seq,
                            frames = count,
                            keyframe,
                            units = units.len(),
                            bytes,
                            encode_us,
                            "batch encoded"
                        );
                        if let Ok(stats) = shared.stats.lock() {
                            if stats.encoded > 0 && stats.encoded % 24 == 0 {
                                tracing::info!(
                                    target: "blazar",
                                    frames = stats.encoded,
                                    bytes = stats.bytes,
                                    "encode progress"
                                );
                            }
                        }
                        let _ = shared.events.send(WorkerEvent::Encoded {
                            seq,
                            pts,
                            keyframe,
                            dts: units.first().map(|u| u.dts),
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                            data: None,
                        });
                        drain_memory_units(&shared, &memory);
                    }
                    Err(err) => {
                        let message = err.to_string();
                        if let Ok(mut slot) = shared.error.lock() {
                            *slot = Some(message.clone());
                        }
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.failed = true;
                        }
                        let _ = shared.events.send(WorkerEvent::Failed {
                            seq: Some(first_seq),
                            message,
                        });
                        break;
                    }
                }
            }
            Job::Device {
                ptr,
                desc,
                frames,
                boundary,
                keep_alive,
            } => {
                shared.queued.fetch_sub(1, Ordering::Relaxed);
                let started = Instant::now();
                let pts = desc.pts.0;
                let count = frames as u64;
                let result = stream.submit_video_batch_device(ptr, desc, frames, boundary, keep_alive);
                let elapsed = started.elapsed();
                match result {
                    Ok(seq) => {
                        let units: Vec<WrittenUnit> = written
                            .lock()
                            .map(|mut units| units.drain(..).collect())
                            .unwrap_or_default();
                        let bytes: usize = units.iter().map(|u| u.bytes).sum();
                        let keyframe = units.iter().any(|u| u.keyframe);
                        let encode_us = elapsed.as_micros() as u64;
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.encoded += count;
                            stats.bytes += bytes as u64;
                            stats.last_encode_us = encode_us;
                            stats.max_encode_us = stats.max_encode_us.max(encode_us);
                        }
                        tracing::debug!(
                            target: "blazar",
                            seq,
                            frames = count,
                            keyframe,
                            units = units.len(),
                            bytes,
                            encode_us,
                            "device batch encoded"
                        );
                        if let Ok(stats) = shared.stats.lock() {
                            if stats.encoded > 0 && stats.encoded % 24 == 0 {
                                tracing::info!(
                                    target: "blazar",
                                    frames = stats.encoded,
                                    bytes = stats.bytes,
                                    "encode progress"
                                );
                            }
                        }
                        let _ = shared.events.send(WorkerEvent::Encoded {
                            seq,
                            pts,
                            keyframe,
                            dts: units.first().map(|u| u.dts),
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                            data: None,
                        });
                        drain_memory_units(&shared, &memory);
                    }
                    Err(err) => {
                        let message = err.to_string();
                        if let Ok(mut slot) = shared.error.lock() {
                            *slot = Some(message.clone());
                        }
                        if let Ok(mut stats) = shared.stats.lock() {
                            stats.failed = true;
                        }
                        let _ = shared.events.send(WorkerEvent::Failed {
                            seq: None,
                            message,
                        });
                        break;
                    }
                }
            }
            Job::Flush(ack) => {
                let result = stream.flush().map_err(|err| err.to_string());
                drain_memory_units(&shared, &memory);
                let _ = ack.send(result);
            }
            Job::Close(ack) => {
                let result = stream.close().map_err(|err| err.to_string());
                // Closing drains the reorder window, so the last units may be
                // written here rather than during a submit. They still count.
                record_extra_bytes(&shared, &written);
                drain_memory_units(&shared, &memory);
                let _ = ack.send(result);
                finished = true;
                break;
            }
        }
    }
    if !finished {
        // Sender dropped or the worker failed: finalize best effort so the file
        // is playable even if the caller never called close().
        let _ = stream.close();
        record_extra_bytes(&shared, &written);
    }
    if let Ok(stats) = shared.stats.lock() {
        tracing::info!(
            target: "blazar",
            request_id = %info.request_id,
            frames = stats.encoded,
            bytes = stats.bytes,
            max_encode_us = stats.max_encode_us,
            failed = stats.failed,
            "encoder session finished"
        );
    }
    let _ = shared.events.send(WorkerEvent::Closed);
}

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// Parsed calibration files, reusable across encoders.
#[pyclass(name = "Calibrations", module = "blazar")]
struct PyCalibrations {
    index: CalibrationIndex,
}

#[pymethods]
impl PyCalibrations {
    /// Load every `*.toml` in `path`. Fails on any invalid file, naming it.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        CalibrationIndex::load_dir(path)
            .map(|index| Self { index })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __len__(&self) -> usize {
        self.index.len()
    }

    fn __repr__(&self) -> String {
        format!("<blazar.Calibrations: {} entries>", self.index.len())
    }
}

/// Process-wide cap on open NVENC sessions.
///
/// Every stream holds its own session (they are stateful and cannot be shared),
/// and consumer drivers refuse new ones past a cap — measured at 9 on an RTX 4080
/// SUPER. Refusing at the driver turns into a half-working server: some streams
/// encode, others die at startup. Queueing instead makes the limit a throughput
/// question rather than a correctness one.
struct SessionLimiter {
    open: std::sync::Mutex<usize>,
    freed: std::sync::Condvar,
    limit: usize,
}

impl SessionLimiter {
    fn new(limit: usize) -> Self {
        Self {
            open: std::sync::Mutex::new(0),
            freed: std::sync::Condvar::new(),
            limit,
        }
    }

    fn acquire(self: &'static Self, timeout: std::time::Duration) -> Option<SessionPermit> {
        let mut open = self.open.lock().ok()?;
        let deadline = std::time::Instant::now() + timeout;
        while *open >= self.limit {
            let remaining = deadline.checked_duration_since(std::time::Instant::now())?;
            let (guard, timed_out) = self.freed.wait_timeout(open, remaining).ok()?;
            open = guard;
            if timed_out.timed_out() && *open >= self.limit {
                return None;
            }
        }
        *open += 1;
        Some(SessionPermit { limiter: self })
    }
}

/// Releases its slot when the encoder is dropped, closed or fails to build.
struct SessionPermit {
    limiter: &'static SessionLimiter,
}

impl Drop for SessionPermit {
    fn drop(&mut self) {
        if let Ok(mut open) = self.limiter.open.lock() {
            *open = open.saturating_sub(1);
            self.limiter.freed.notify_one();
        }
    }
}

/// Sessions per process: eight is safe on consumer cards, which is where a
/// surprise cap hurts most. Datacenter deployments can raise it.
fn session_limit() -> usize {
    std::env::var("BLAZAR_MAX_NVENC_SESSIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|limit| *limit > 0)
        .unwrap_or(8)
}

fn session_limiter() -> &'static SessionLimiter {
    static LIMITER: std::sync::OnceLock<SessionLimiter> = std::sync::OnceLock::new();
    LIMITER.get_or_init(|| SessionLimiter::new(session_limit()))
}

/// One encode stream writing a fragmented MP4 on a worker thread.
#[pyclass(name = "Encoder", module = "blazar")]
struct PyEncoder {
    /// `None` after `close()`; dropping the sender closes the job channel.
    /// Mutex-wrapped because `SyncSender` is `Send` but not `Sync`, and a
    /// pyclass must be `Sync`.
    jobs: Mutex<Option<SyncSender<Job>>>,
    events: Mutex<Receiver<WorkerEvent>>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    request_id: String,
    timebase: Timebase,
    seq: u64,
    submitted: u64,
    frame_index: u64,
    frame_size: Option<(u32, u32)>,
    fps: Option<f64>,
    /// Held while a hardware session is open; `None` for software backends.
    _session: Option<SessionPermit>,
    /// Where the caller asked for RGB→NV12 conversion.
    preprocess: blazar_runtime::preprocess_placement::PreprocessPlacement,
    /// Which backend, so `auto` can be resolved for reporting.
    backend: blazar_core::VideoEncoderBackend,
}

impl PyEncoder {
    /// Fix the stream's residency on the first submit and refuse a change.
    fn claim_residency(&self, residency: &str) -> PyResult<()> {
        let mut slot = self
            .shared
            .residency
            .lock()
            .map_err(|_| PyRuntimeError::new_err("residency lock poisoned"))?;
        match slot.as_deref() {
            None => {
                *slot = Some(residency.to_owned());
                Ok(())
            }
            Some(current) if current == residency => Ok(()),
            Some(current) => Err(PyValueError::new_err(format!(
                "this stream is opened for {current} frames; got {residency} frames. \
                 Open a second Encoder for the other residency"
            ))),
        }
    }

    /// Fix the stream's layout on the first submit and refuse a change.
    fn claim_layout(&self, layout: blazar_core::Layout) -> PyResult<()> {
        let name = match layout {
            blazar_core::Layout::Thwc => "thwc",
            blazar_core::Layout::Bcthw => "bcthw",
        };
        let mut slot = self
            .shared
            .layout
            .lock()
            .map_err(|_| PyRuntimeError::new_err("layout lock poisoned"))?;
        match slot.as_deref() {
            None => {
                *slot = Some(name.to_owned());
                Ok(())
            }
            Some(current) if current == name => Ok(()),
            Some(current) => Err(PyValueError::new_err(format!(
                "layout={current:?} is fixed for this stream; got a {name} array. \
                 Pass layout={current:?} at Encoder(...)"
            ))),
        }
    }

}

#[pymethods]
impl PyEncoder {
    // Explicit keyword arguments mirror the Rust contract; folding them into a
    // config object would hide which fields are required.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        calibrations,
        identity,
        target,
        backend,
        output,
        codec = "av1",
        request_id = "stream",
        video_extradata = None,
        queue_depth = 4,
        max_buffered_units = 240,
        preprocess = "auto",
        backend_pool = "any",
        layout = None,
        audio = None,
    ))]
    fn new(
        py: Python<'_>,
        calibrations: &PyCalibrations,
        identity: &Bound<'_, PyDict>,
        target: &str,
        backend: &str,
        output: Option<&str>,
        codec: &str,
        request_id: &str,
        video_extradata: Option<Vec<u8>>,
        queue_depth: usize,
        max_buffered_units: usize,
        preprocess: &str,
        backend_pool: &str,
        layout: Option<&str>,
        audio: Option<&str>,
    ) -> PyResult<Self> {
        // Validated here so a typo fails before a frame is submitted.
        let initial_layout = match layout {
            Some(value) => {
                parse_layout(value)?;
                Some(value.to_owned())
            }
            None => None,
        };
        if queue_depth == 0 {
            return Err(PyValueError::new_err("queue_depth must be >= 1"));
        }
        let info = SessionInfo {
            request_id: request_id.to_string(),
            backend: backend.to_string(),
            codec: codec.to_string(),
            target: target.to_string(),
            calibrations: calibrations.index.len(),
            queue_depth,
        };
        let identity = model_identity(identity)?;
        let target = parse_target(target)?;
        let codec = parse_codec(codec)?;
        let (backend_auto, backend) = if backend == "auto" {
            // A concrete backend still has to be named for `Output`, but it is
            // never built: `with_backend_auto` makes the runtime choose.
            (true, blazar_core::VideoEncoderBackend::X264)
        } else {
            (false, parse_backend_for(py, backend, codec)?)
        };
        let pool: blazar_runtime::backend_selection::BackendPool = backend_pool
            .parse()
            .map_err(|e: blazar_runtime::backend_selection::SelectionError| {
                PyValueError::new_err(e.to_string())
            })?;

        // Queue rather than let the driver refuse: a server that half-starts is
        // worse than one that starts slowly.
        let session = match backend {
            VideoEncoderBackend::NvencAv1
            | VideoEncoderBackend::NvencH264
            | VideoEncoderBackend::NvencHevc => Some(
                // Waiting for a slot must not hold the GIL: every other thread
                // that would release one needs it.
                py.detach(|| session_limiter().acquire(Duration::from_secs(30)))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "no NVENC session available after 30s: {} already open in this \
                             process (BLAZAR_MAX_NVENC_SESSIONS). Close encoders you are done with.",
                            session_limit()
                        ))
                    })?,
            ),
            _ => None,
        };

        let written = Arc::new(Mutex::new(Vec::new()));
        let (sink, memory) = match output {
            Some(path) => {
                let file = File::create(path).map_err(|err| {
                    PyValueError::new_err(format!("cannot create output {path}: {err}"))
                })?;
                (
                    SinkTarget::File(Box::new(ReportingSink {
                        inner: Fmp4FileSink::new(BufWriter::new(file)),
                        written: Arc::clone(&written),
                    })),
                    None,
                )
            }
            // No file: the encoded units are handed back through poll().
            None => {
                let (sink, handle) = blazar_sink::MemorySink::new(max_buffered_units);
                (SinkTarget::Memory(Box::new(sink)), Some(handle))
            }
        };

        let runtime = RuntimeEncoder::new(
            identity,
            target,
            Output {
                codec,
                backend,
                container: Container::FragmentedMp4,
                video_extradata,
                audio_extradata: None,
            },
            sink,
            calibrations.index.clone(),
        );
        let placement: blazar_runtime::preprocess_placement::PreprocessPlacement =
            preprocess.parse().map_err(|e: blazar_runtime::preprocess_placement::PlacementError| {
                PyValueError::new_err(e.to_string())
            })?;
        #[allow(unused_mut)]
        let mut stream = runtime
            .open_stream(request_id)
            .with_preprocess(placement)
            .with_backend_auto(backend_auto, pool);
        match audio {
            None | Some("none") => {}
            #[cfg(feature = "opus")]
            Some("opus") => {
                stream = stream
                    .with_opus_audio(2, 96)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "audio must be \"opus\" or \"none\" (got {other:?}); AAC is not implemented"
                )))
            }
        }

        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(queue_depth);
        let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
        let shared = Arc::new(Shared {
            events: event_tx,
            error: Mutex::new(None),
            stats: Mutex::new(WorkerStats::default()),
            queued: AtomicUsize::new(0),
            resolved_backend: Mutex::new(None),
            layout: Mutex::new(initial_layout),
            residency: Mutex::new(None),
        });

        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("blazar-encode".into())
            .spawn(move || worker_main(stream, job_rx, worker_shared, written, memory, info))
            .map_err(|err| {
                PyRuntimeError::new_err(format!("cannot start encoder thread: {err}"))
            })?;

        Ok(Self {
            jobs: Mutex::new(Some(job_tx)),
            events: Mutex::new(event_rx),
            shared,
            worker: Some(worker),
            request_id: request_id.to_string(),
            timebase: Timebase::MPEG_90K,
            seq: 0,
            submitted: 0,
            frame_index: 0,
            frame_size: None,
            fps: None,
            _session: session,
            preprocess: placement,
            backend,
        })
    }

    /// Device frames: same entry point as host frames, dispatched by type.
    fn submit_device_frame(
        &mut self,
        py: Python<'_>,
        tensor: &Bound<'_, PyAny>,
        fps: f64,
        range: &str,
        pts_ticks: Option<u64>,
        scene_cut: bool,
    ) -> PyResult<u64> {
        self.claim_residency("device")?;
        self.claim_layout(blazar_core::Layout::Bcthw)?;
        self.submit_device_impl(py, tensor, fps, range, pts_ticks, scene_cut)
    }

    /// Copy one frame into the encoder's queue. Returns immediately (the encode
    /// happens on the worker). Blocks only if the queue is full, with the GIL
    /// released so other Python threads keep running.
    ///
    /// `frame` is a numpy array (host) or a CUDA tensor (device); the type picks
    /// the path, so there is one submit rather than one per residency.
    ///
    /// Host frames need `width`/`height` and a `layout` (from `Encoder(...)` or
    /// this call). Device tensors carry their shape: `(1, 3, T, H, W)` bcthw,
    /// contiguous, on CUDA, and are read in place.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        frame,
        fps,
        width = None,
        height = None,
        layout = None,
        range = "unit01",
        pts_ticks = None,
        scene_cut = false,
    ))]
    fn submit(
        &mut self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        fps: f64,
        width: Option<u32>,
        height: Option<u32>,
        layout: Option<&str>,
        range: &str,
        pts_ticks: Option<u64>,
        scene_cut: bool,
    ) -> PyResult<u64> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(PyValueError::new_err("fps must be > 0"));
        }
        // A sequence is a batch of host frames. It must not mix residencies:
        // half a batch on each path is a caller bug, not something to guess at.
        if frame.is_instance_of::<pyo3::types::PyList>() || frame.is_instance_of::<pyo3::types::PyTuple>() {
            let items: Vec<Bound<'_, PyAny>> = frame.extract()?;
            let mut saw_host = false;
            let mut saw_device = false;
            for item in &items {
                if is_torch_tensor(py, item) {
                    saw_device = true;
                } else {
                    saw_host = true;
                }
            }
            if saw_host && saw_device {
                return Err(PyValueError::new_err(
                    "a single submit cannot mix host and device frames; split the call or convert",
                ));
            }
            if saw_device {
                return Err(PyValueError::new_err(
                    "submit one device tensor, not a sequence of them: frames already batched \
                     belong in a single (1, 3, T, H, W) tensor",
                ));
            }
            let (mut last, mut count) = (0, 0);
            for item in &items {
                last = self.submit(
                    py,
                    item,
                    fps,
                    width,
                    height,
                    layout,
                    range,
                    pts_ticks,
                    scene_cut && count == 0,
                )?;
                count += 1;
            }
            if count == 0 {
                return Err(PyValueError::new_err("submit was given an empty sequence"));
            }
            return Ok(last);
        }

        // Geometry comes from the caller for host frames and from the tensor for
        // device frames, so these are optional here and checked per path.
        let (width, height) = match (width, height) {
            (Some(w), Some(h)) if w > 0 && h > 0 => (w, h),
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err("width and height must be non-zero"))
            }
            _ if is_torch_tensor(py, frame) => (0, 0),
            _ => {
                return Err(PyValueError::new_err(
                    "width and height are required for host frames",
                ))
            }
        };
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        let jobs = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;

        if is_torch_tensor(py, frame) {
            return self.submit_device_frame(py, frame, fps, range, pts_ticks, scene_cut);
        }

        let requested_layout: String = match layout {
            Some(value) => value.to_owned(),
            None => self
                .shared
                .layout
                .lock()
                .ok()
                .and_then(|slot| slot.clone())
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "layout is required for host frames; pass layout= at Encoder(...) or per call",
                    )
                })?,
        };
        let layout = parse_layout(&requested_layout)?;
        let range = parse_range(range)?;
        self.claim_residency("host")?;
        self.claim_layout(layout)?;

        // A stream's geometry is part of its container metadata (opened on the
        // first frame). Changing it mid-stream produces an unplayable file, so
        // reject it instead of letting the muxer write lies.
        if let Some((prev_w, prev_h)) = self.frame_size {
            if (prev_w, prev_h) != (width, height) {
                return Err(PyValueError::new_err(format!(
                    "frame size changed from {prev_w}x{prev_h} to {width}x{height}; \
                     open a new Encoder for a new stream"
                )));
            }
        }
        if let Some(prev_fps) = self.fps {
            if (prev_fps - fps).abs() > 1e-9 {
                return Err(PyValueError::new_err(format!(
                    "fps changed from {prev_fps} to {fps}; open a new Encoder for a new stream"
                )));
            }
        }

        // One copy under the GIL: the job owns its bytes, so the worker needs no
        // Python memory and can run without the GIL.
        let (bytes, dtype) = copy_frame(py, frame)?;

        let pts = pts_ticks
            .unwrap_or_else(|| Pts::from_frame_index(self.frame_index, fps, self.timebase).0);
        let desc = FrameDesc {
            width,
            height,
            dtype,
            range,
            layout,
            colorspace: ColorSpace::AssumedBt709,
            fps,
            pts: Pts(pts),
        };
        let job = Job::Frame(Box::new(FrameJob {
            seq: self.seq,
            bytes,
            desc,
            boundary: if scene_cut {
                Some(Boundary::SceneCut)
            } else {
                None
            },
        }));

        self.shared.queued.fetch_add(1, Ordering::Relaxed);
        let sent = py.detach(move || jobs.send(job));
        if sent.is_err() {
            self.shared.queued.fetch_sub(1, Ordering::Relaxed);
            return Err(PyRuntimeError::new_err(
                self.error_message()
                    .unwrap_or_else(|| "encoder worker stopped".to_string()),
            ));
        }

        let seq = self.seq;
        self.seq += 1;
        self.submitted += 1;
        self.frame_index += 1;
        self.frame_size.get_or_insert((width, height));
        self.fps.get_or_insert(fps);
        Ok(seq)
    }



    fn submit_device_impl(
        &mut self,
        py: Python<'_>,
        tensor: &Bound<'_, PyAny>,
        fps: f64,
        range: &str,
        pts_ticks: Option<u64>,
        scene_cut: bool,
    ) -> PyResult<u64> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(PyValueError::new_err("fps must be > 0"));
        }
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        {
            use blazar_runtime::preprocess_placement::{resolve, PreprocessSite, SourceLocation};
            let site = resolve(
                self.preprocess,
                SourceLocation::Device,
                blazar_runtime::backend_converts_on_device(self.backend),
                blazar_runtime::gpu_preprocess_available(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if site == PreprocessSite::Host {
                return Err(PyRuntimeError::new_err(
                    "preprocess=\"host\" cannot convert a device tensor yet: copy it to host \
                     first (tensor.cpu()) or use preprocess=\"auto\"",
                ));
            }
        }
        let is_cuda: bool = tensor.getattr("is_cuda")?.extract()?;
        if !is_cuda {
            return Err(PyValueError::new_err("tensor must be on CUDA"));
        }
        let contiguous: bool = tensor.call_method0("is_contiguous")?.extract()?;
        if !contiguous {
            return Err(PyValueError::new_err("tensor must be contiguous"));
        }
        let dtype_name = tensor.getattr("dtype")?.to_string();
        let dtype = match dtype_name.as_str() {
            "torch.bfloat16" => Dtype::Bf16,
            "torch.float32" => Dtype::F32,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported tensor dtype {other:?} (expected torch.bfloat16 or torch.float32)"
                )))
            }
        };
        let shape: Vec<i64> = tensor.getattr("shape")?.extract()?;
        if shape.len() != 5 || shape[0] != 1 || shape[1] != 3 || shape[2] < 1 {
            return Err(PyValueError::new_err(format!(
                "expected a (1, 3, T, H, W) tensor, got {shape:?}"
            )));
        }
        let (frames, height, width) = (shape[2] as u32, shape[3] as u32, shape[4] as u32);
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be non-zero"));
        }
        let ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;

        let jobs_out = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;
        let range = parse_range(range)?;

        if let Some((prev_w, prev_h)) = self.frame_size {
            if (prev_w, prev_h) != (width, height) {
                return Err(PyValueError::new_err(format!(
                    "frame size changed from {prev_w}x{prev_h} to {width}x{height}; \
                     open a new Encoder for a new stream"
                )));
            }
        }
        if let Some(prev_fps) = self.fps {
            if (prev_fps - fps).abs() > 1e-9 {
                return Err(PyValueError::new_err(format!(
                    "fps changed from {prev_fps} to {fps}; open a new Encoder for a new stream"
                )));
            }
        }

        let pts = pts_ticks
            .unwrap_or_else(|| Pts::from_frame_index(self.frame_index, fps, self.timebase).0);
        let desc = FrameDesc {
            width,
            height,
            dtype,
            range,
            layout: Layout::Bcthw,
            colorspace: ColorSpace::AssumedBt709,
            fps,
            pts: Pts(pts),
        };
        let keep_alive: Box<dyn std::any::Any + Send> = Box::new(tensor.clone().unbind());
        let job = Job::Device {
            ptr,
            desc,
            frames,
            boundary: if scene_cut {
                Some(Boundary::SceneCut)
            } else {
                None
            },
            keep_alive: Some(keep_alive),
        };

        self.shared.queued.fetch_add(1, Ordering::Relaxed);
        let sent = py.detach(move || jobs_out.send(job));
        if sent.is_err() {
            self.shared.queued.fetch_sub(1, Ordering::Relaxed);
            return Err(PyRuntimeError::new_err(
                self.error_message()
                    .unwrap_or_else(|| "encoder worker stopped".to_string()),
            ));
        }

        let seq = self.seq;
        self.seq += 1;
        self.submitted += frames as u64;
        self.frame_index += frames as u64;
        self.frame_size.get_or_insert((width, height));
        self.fps.get_or_insert(fps);
        Ok(seq)
    }

    /// Drain worker events as dicts. With a timeout, wait up to that many
    /// seconds for the first event; the GIL is released while waiting.
    #[pyo3(signature = (timeout = None))]
    fn poll(&mut self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Vec<Py<PyDict>>> {
        let mut events: Vec<WorkerEvent> = Vec::new();
        {
            let receiver = self
                .events
                .lock()
                .map_err(|_| PyRuntimeError::new_err("event queue lock poisoned"))?;
            while let Ok(event) = receiver.try_recv() {
                events.push(event);
            }
        }
        if events.is_empty() {
            if let Some(seconds) = timeout {
                let wait = Duration::from_secs_f64(seconds.max(0.0));
                let first = py.detach(|| {
                    self.events
                        .lock()
                        .ok()
                        .and_then(|receiver| receiver.recv_timeout(wait).ok())
                });
                if let Some(event) = first {
                    events.push(event);
                    let receiver = self
                        .events
                        .lock()
                        .map_err(|_| PyRuntimeError::new_err("event queue lock poisoned"))?;
                    while let Ok(event) = receiver.try_recv() {
                        events.push(event);
                    }
                }
            }
        }
        events
            .into_iter()
            .map(|event| event_to_dict(py, event))
            .collect()
    }

    /// Barrier: return once every frame submitted before this call has been
    /// encoded and written.
    #[pyo3(signature = (timeout = None))]
    fn flush(&mut self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        let jobs = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;
        let (ack_tx, ack_rx) = mpsc::sync_channel(1);
        py.detach(move || jobs.send(Job::Flush(ack_tx)))
            .map_err(|_| PyRuntimeError::new_err("encoder worker stopped"))?;
        wait_ack(py, ack_rx, timeout).map_err(|error| {
            PyRuntimeError::new_err(prefer_recorded(
                error.to_string(),
                self.error_message(),
            ))
        })
    }

    /// Drain the queue, finalize the container and stop the worker. Idempotent.
    /// Raises any encoder error that occurred on the worker.
    #[pyo3(signature = (timeout = None))]
    fn close(&mut self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        let mut outcome = Ok(());
        let jobs = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .take();
        if let Some(jobs) = jobs {
            let (ack_tx, ack_rx) = mpsc::sync_channel(1);
            match py.detach(move || jobs.send(Job::Close(ack_tx))) {
                Ok(()) => outcome = wait_ack(py, ack_rx, timeout),
                // The worker is already gone; its error state is read below.
                Err(_) => outcome = Ok(()),
            }
        }
        if let Some(worker) = self.worker.take() {
            let _ = py.detach(move || worker.join());
        }
        outcome = outcome.map_err(|error| {
            let recorded = self.error_message().or_else(|| self.failed_event_message());
            PyRuntimeError::new_err(prefer_recorded(error.to_string(), recorded))
        });
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        Ok(())
    }

    /// The backend that actually produced this stream.
    ///
    /// With `backend="auto"` this is `None` until the first frame, because the
    /// choice depends on what opens a session on this host.
    #[getter]
    fn resolved_backend(&self) -> PyResult<Option<String>> {
        Ok(self
            .shared
            .resolved_backend
            .lock()
            .map(|value| value.clone())
            .unwrap_or(None))
    }

    /// Queue interleaved float samples in `[-1, 1]` at 48 kHz.
    ///
    /// The encoder buffers to whole 20 ms frames, so any length is accepted; the
    /// remainder is flushed on `close()`. Requires `Encoder(audio="opus")`.
    #[pyo3(signature = (samples, sample_rate = 48000))]
    fn submit_audio(
        &mut self,
        py: Python<'_>,
        samples: &Bound<'_, PyAny>,
        sample_rate: u32,
    ) -> PyResult<()> {
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        let jobs = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;
        // A numpy array converts through the buffer protocol; a plain sequence
        // of floats converts directly. Either way the worker gets an owned
        // Vec<f32> and needs no reference to caller memory.
        let samples: Vec<f32> = if samples.hasattr("__array_interface__")? {
            samples
                .call_method1("__array__", ())?
                .call_method1("astype", ("float32",))?
                .extract::<Vec<f32>>()?
        } else {
            samples
                .extract::<Vec<f32>>()
                .map_err(|e| PyValueError::new_err(format!("samples must be floats: {e}")))?
        };
        let (tx, rx) = mpsc::sync_channel(1);
        jobs.send(Job::Audio {
            samples,
            sample_rate,
            reply: tx,
        })
        .map_err(|_| closed_error(&self.request_id))?;
        self.shared.queued.fetch_add(1, Ordering::Relaxed);
        rx.recv()
            .map_err(|_| PyRuntimeError::new_err("worker went away"))?
            .map_err(PyRuntimeError::new_err)
    }

    /// `"host"` or `"device"`: which path this stream takes, decided by the
    /// first submit. `None` before then.
    #[getter]
    fn residency(&self) -> PyResult<Option<String>> {
        Ok(self
            .shared
            .residency
            .lock()
            .map(|value| value.clone())
            .unwrap_or(None))
    }

    /// Where host frames are converted to NV12: `"host"` or `"device"`.
    ///
    /// `preprocess="auto"` resolves against the backend: a backend that
    /// converts into the surface it encodes from does it on the device, a CPU
    /// codec does it on the host. Device tensors always convert on the device
    /// unless `preprocess="host"` is rejected at submit time.
    #[getter]
    fn preprocess(&self) -> PyResult<String> {
        use blazar_runtime::preprocess_placement::{resolve, SourceLocation};
        let site = resolve(
            self.preprocess,
            SourceLocation::Host,
            blazar_runtime::backend_converts_on_device(self.backend),
            cfg!(feature = "gpu-preprocess"),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(site.as_str().to_owned())
    }

    /// Monotonic counters: `submitted`, `encoded`, `bytes`, `queued`,
    /// `last_encode_us`, `max_encode_us`, `failed`.
    #[getter]
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("submitted", self.submitted)?;
        dict.set_item("queued", self.shared.queued.load(Ordering::Relaxed))?;
        let stats = self
            .shared
            .stats
            .lock()
            .map_err(|_| PyRuntimeError::new_err("stats lock poisoned"))?;
        dict.set_item("encoded", stats.encoded)?;
        dict.set_item("bytes", stats.bytes)?;
        dict.set_item("last_encode_us", stats.last_encode_us)?;
        dict.set_item("max_encode_us", stats.max_encode_us)?;
        dict.set_item("failed", stats.failed)?;
        Ok(dict.unbind())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        slf.close(py, None)
    }

    fn __repr__(&self) -> String {
        format!(
            "<blazar.Encoder: request_id={:?}, submitted={}, closed={}>",
            self.request_id,
            self.submitted,
            self.jobs.lock().map(|jobs| jobs.is_none()).unwrap_or(true)
        )
    }
}

impl PyEncoder {
    /// A failure the worker reported as an event but never wrote to the error
    /// slot. The event carries the cause; the slot occasionally does not, and
    /// reporting the transport symptom then hides an actionable error.
    fn failed_event_message(&self) -> Option<String> {
        let receiver = self.events.lock().ok()?;
        let mut found = None;
        while let Ok(event) = receiver.try_recv() {
            if let WorkerEvent::Failed { message, .. } = &event {
                found = Some(message.clone());
            }
        }
        found
    }

    fn error_message(&self) -> Option<String> {
        self.shared.error.lock().ok().and_then(|slot| slot.clone())
    }
}

impl Drop for PyEncoder {
    fn drop(&mut self) {
        // Closing the job channel lets the worker drain and finalize; the join
        // handle is detached so dropping never blocks the interpreter.
        if let Ok(jobs) = self.jobs.get_mut() {
            jobs.take();
        }
        self.worker.take();
    }
}

/// Fold units the sink accepted outside a submit (the close-time reorder drain)
/// into the byte counter, so the reported bytes always equal the file's payload.
fn record_extra_bytes(shared: &Shared, written: &Mutex<Vec<WrittenUnit>>) {
    let extra: usize = written
        .lock()
        .map(|mut units| units.drain(..).map(|u| u.bytes).sum())
        .unwrap_or(0);
    if extra > 0 {
        if let Ok(mut stats) = shared.stats.lock() {
            stats.bytes += extra as u64;
        }
    }
}

/// Combine the transport symptom with the error the worker recorded.
///
/// A dead channel means the worker stopped; *why* it stopped is in the shared
/// error slot, and reporting only "worker stopped before replying" hides causes
/// that are entirely actionable, such as an NVENC session-limit failure.
fn prefer_recorded(symptom: String, recorded: Option<String>) -> String {
    match recorded {
        Some(reason) if !symptom.contains(&reason) => format!("{symptom}: {reason}"),
        Some(reason) => reason,
        None => symptom,
    }
}

fn wait_ack(
    py: Python<'_>,
    ack: Receiver<Result<(), String>>,
    timeout: Option<f64>,
) -> PyResult<()> {
    let received = py.detach(move || match timeout {
        Some(seconds) => ack
            .recv_timeout(Duration::from_secs_f64(seconds.max(0.0)))
            .map_err(|_| "timed out waiting for the encoder".to_string()),
        None => ack
            .recv()
            .map_err(|_| "encoder worker stopped before replying".to_string()),
    });
    match received {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(PyRuntimeError::new_err(message)),
        Err(message) => Err(PyRuntimeError::new_err(message)),
    }
}

fn event_to_dict(py: Python<'_>, event: WorkerEvent) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    match event {
        WorkerEvent::Encoded {
            seq,
            pts,
            dts,
            keyframe,
            units,
            bytes,
            encode_us,
            queue_depth,
            data,
        } => {
            dict.set_item("kind", "encoded")?;
            dict.set_item("seq", seq)?;
            dict.set_item("pts", pts)?;
            dict.set_item("dts", dts)?;
            dict.set_item("keyframe", keyframe)?;
            dict.set_item("units", units)?;
            dict.set_item("bytes", bytes)?;
            dict.set_item("encode_us", encode_us)?;
            dict.set_item("queue_depth", queue_depth)?;
            dict.set_item("data", data)?;
        }
        WorkerEvent::Failed { seq, message } => {
            dict.set_item("kind", "failed")?;
            dict.set_item("seq", seq)?;
            dict.set_item("message", message)?;
        }
        WorkerEvent::Closed => {
            dict.set_item("kind", "closed")?;
        }
    }
    Ok(dict.unbind())
}

/// Copy a numeric buffer-protocol object into owned bytes, reporting the dtype
/// the bytes are in.
///
/// `uint8` is taken as-is: the preprocess stage reads it at 1 byte per channel
/// instead of 4, which is the difference between 6.2 MB and 24.9 MB of traffic
/// per 1080p frame. Anything else must be float32. The order matters: `u8` is
/// tried first, and a float32 buffer fails that format check, so a frame can
/// never be misread as the other dtype.
fn copy_frame(py: Python<'_>, frame: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, Dtype)> {
    if let Ok(buffer) = PyBuffer::<u8>::get(frame) {
        let bytes = copy_buffer(py, &buffer, 1, Dtype::U8)?;
        return Ok((bytes, Dtype::U8));
    }

    let buffer = PyBuffer::<f32>::get(frame).map_err(|_| {
        PyValueError::new_err(
            "frame must be a uint8 or float32 array supporting the buffer protocol \
             (numpy, C order); convert with \
             numpy.ascontiguousarray(frame, dtype=numpy.uint8) or \
             tensor.cpu().float().numpy()",
        )
    })?;
    Ok((copy_buffer(py, &buffer, 4, Dtype::F32)?, Dtype::F32))
}

/// A borrowed buffer pointer that may cross into `detach`.
///
/// The buffer itself is not `Send` because it borrows Python memory; the pointer
/// says only "this range stays mapped for the duration of the copy", which the
/// caller guarantees by holding the export until the copy returns.
struct BufferPtr(*const u8);

// SAFETY: nothing is dereferenced while the pointer is merely moved, and the
// range is only read, never written.
unsafe impl Send for BufferPtr {}

/// Copy a buffer-protocol object to owned bytes **with the GIL released.**
///
/// This is the difference between a host frame costing interpreter time and
/// costing none: at 1088p the copy moves 6.3 MB, and doing that while holding the
/// GIL serialises the caller against every other Python thread in the process.
/// Measured before the change: 35.6 fps alone, 11.3 fps with one other Python
/// thread running. After: see `tests/test_gil.py`.
///
/// # Safety of releasing the GIL
///
/// Another Python thread could mutate the array being copied. That is a data race
/// the caller already has whenever it hands a buffer to an encoder that reads it
/// asynchronously, and it is the caller's to avoid; the alternative is holding the
/// interpreter for the whole copy, which is the thing this exists to prevent.
fn copy_buffer(
    py: Python<'_>,
    buffer: &PyBuffer<impl pyo3::buffer::Element>,
    element_size: usize,
    dtype: Dtype,
) -> PyResult<Vec<u8>> {
    // Little-endian is assumed, as it is on every target this ships for; the f32
    // path below is the only one where byte order could matter.
    if element_size == 4 && cfg!(target_endian = "big") {
        return Err(PyValueError::new_err(
            "float32 frames are only supported on little-endian targets",
        ));
    }
    // A strided view cannot be copied as one range. numpy hands us C-order
    // arrays and the error below says how to get one; refusing is cheaper than
    // element-wise copying 6 MB while the interpreter is idle anyway.
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "frame must be C-contiguous; convert with numpy.ascontiguousarray(frame)",
        ));
    }
    let addr = buffer.buf_ptr() as usize;
    let len = buffer.len_bytes();
    let bytes = unsafe {
        py.detach(move || {
            // SAFETY: the exporter's view is alive until this closure returns,
            // because `buffer` is still borrowed by the caller and no Python code
            // runs while the GIL is released. The address travels as an integer
            // because `detach` requires its closure to be `Send`.
            let slice = std::slice::from_raw_parts(addr as *const u8, len);
            slice.to_vec()
        })
    };
    let _ = dtype;
    Ok(bytes)
}

fn closed_error(request_id: &str) -> PyErr {
    PyRuntimeError::new_err(format!("stream {request_id:?} is closed"))
}

fn model_identity(dict: &Bound<'_, PyDict>) -> PyResult<ModelIdentity> {
    let field = |name: &str| -> PyResult<String> {
        match dict.get_item(name)? {
            Some(value) => value.extract::<String>(),
            None => Ok("*".to_string()),
        }
    };
    Ok(ModelIdentity {
        family: field("family")?,
        version: field("version")?,
        quant: field("quant")?,
        size: field("size")?,
    })
}

fn parse_target(value: &str) -> PyResult<Target> {
    match value {
        "low_latency" => Ok(Target::LowLatency),
        "balanced" => Ok(Target::Balanced),
        "quality" => Ok(Target::Quality),
        other => Err(PyValueError::new_err(format!(
            "unknown target {other:?} (expected low_latency|balanced|quality)"
        ))),
    }
}

/// Parse a pair-shaped backend name. Deprecated: `backend` should name the
/// implementation and `codec` the codec, resolved by `parse_backend_for`.
fn parse_backend(value: &str) -> PyResult<VideoEncoderBackend> {
    match value {
        "svtav1" => Ok(VideoEncoderBackend::SvtAv1),
        "nvenc_av1" => Ok(VideoEncoderBackend::NvencAv1),
        "nvenc_h264" => Ok(VideoEncoderBackend::NvencH264),
        "nvenc_hevc" => Ok(VideoEncoderBackend::NvencHevc),
        "mock" => Ok(VideoEncoderBackend::Mock),
        "x264" => Ok(VideoEncoderBackend::X264),
        "videotoolbox_h264" => Ok(VideoEncoderBackend::VideoToolboxH264),
        "videotoolbox_hevc" => Ok(VideoEncoderBackend::VideoToolboxHevc),
        "x265" => Ok(VideoEncoderBackend::X265),
        "vp9" => Ok(VideoEncoderBackend::Vp9),
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (expected nvenc|videotoolbox|svtav1|x264|x265|vp9|mock|auto, \
             or a deprecated pair name such as nvenc_h264)"
        ))),
    }
}

fn parse_codec(value: &str) -> PyResult<VideoCodec> {
    match value {
        "av1" => Ok(VideoCodec::Av1),
        "h264" => Ok(VideoCodec::H264),
        "hevc" => Ok(VideoCodec::Hevc),
        "vp9" => Ok(VideoCodec::Vp9),
        other => Err(PyValueError::new_err(format!(
            "unknown codec {other:?} (expected av1|h264|hevc|vp9)"
        ))),
    }
}

fn parse_layout(value: &str) -> PyResult<Layout> {
    match value {
        "thwc" => Ok(Layout::Thwc),
        "bcthw" => Ok(Layout::Bcthw),
        other => Err(PyValueError::new_err(format!(
            "unknown layout {other:?} (expected thwc for xDiT, bcthw for vLLM-Omni)"
        ))),
    }
}

fn parse_range(value: &str) -> PyResult<ColorRange> {
    match value {
        "unit01" => Ok(ColorRange::Unit01),
        "signed_unit" => Ok(ColorRange::SignedUnit),
        "full_u8" => Ok(ColorRange::FullU8),
        other => Err(PyValueError::new_err(format!(
            "unknown range {other:?} (expected unit01|signed_unit|full_u8)"
        ))),
    }
}

/// Log level from BLAZAR_LOG/RUST_LOG (error|warn|info|debug|trace|off),
/// defaulting to info. Logs go to stderr so they are visible next to ffmpeg's
/// and do not interleave with anything Python prints to stdout.
fn init_logging() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let raw = std::env::var("BLAZAR_LOG")
            .or_else(|_| std::env::var("RUST_LOG"))
            .unwrap_or_else(|_| "info".to_string());
        let level = match raw.trim().to_ascii_lowercase().as_str() {
            "off" => tracing::Level::ERROR,
            "error" => tracing::Level::ERROR,
            "warn" => tracing::Level::WARN,
            "debug" => tracing::Level::DEBUG,
            "trace" => tracing::Level::TRACE,
            _ => tracing::Level::INFO,
        };
        let _ = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_target(false)
            // Logs get captured to files; colour escapes make them unparseable.
            .with_ansi(false)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// Encoder implementations compiled into this wheel.
///
/// A release smoke test asserts this rather than "the import worked": a wheel
/// built with the wrong `--features` imports perfectly well and then fails at the
/// first frame, which is a worse way to find out.
#[pyfunction]
fn backends() -> Vec<String> {
    let mut names = Vec::new();
    if cfg!(feature = "nvenc") {
        names.push("nvenc".to_owned());
    }
    if cfg!(all(feature = "enc-videotoolbox", target_os = "macos")) {
        names.push("videotoolbox".to_owned());
    }
    if cfg!(feature = "enc-svtav1") {
        names.push("svtav1".to_owned());
    }
    if cfg!(feature = "enc-x264") {
        names.push("x264".to_owned());
    }
    if cfg!(feature = "enc-x265") {
        names.push("x265".to_owned());
    }
    if cfg!(feature = "enc-vp9") {
        names.push("vp9".to_owned());
    }
    names.push("mock".to_owned());
    names
}

#[pymodule]
fn _blazar(module: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logging();
    module.add_class::<PyCalibrations>()?;
    module.add_class::<PyEncoder>()?;
    module.add_function(wrap_pyfunction!(backends, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
