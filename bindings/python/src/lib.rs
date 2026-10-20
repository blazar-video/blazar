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
    Flush(SyncSender<Result<(), String>>),
    Close(SyncSender<Result<(), String>>),
}

/// One unit the sink accepted, recorded so the event can report size and
/// whether a fragment boundary was crossed.
struct WrittenUnit {
    keyframe: bool,
    bytes: usize,
    pts: u64,
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
        keyframe: bool,
        units: usize,
        bytes: usize,
        encode_us: u64,
        queue_depth: usize,
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
}

fn worker_main(
    mut stream: Stream<ReportingSink>,
    jobs: Receiver<Job>,
    shared: Arc<Shared>,
    written: Arc<Mutex<Vec<WrittenUnit>>>,
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
            Job::Frame(job) => {
                shared.queued.fetch_sub(1, Ordering::Relaxed);
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
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                        });
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
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                        });
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
                            units: units.len(),
                            bytes,
                            encode_us,
                            queue_depth: shared.queued.load(Ordering::Relaxed),
                        });
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
                let _ = ack.send(stream.flush().map_err(|err| err.to_string()));
            }
            Job::Close(ack) => {
                let result = stream.close().map_err(|err| err.to_string());
                // Closing drains the reorder window, so the last units may be
                // written here rather than during a submit. They still count.
                record_extra_bytes(&shared, &written);
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
    ))]
    fn new(
        calibrations: &PyCalibrations,
        identity: &Bound<'_, PyDict>,
        target: &str,
        backend: &str,
        output: &str,
        codec: &str,
        request_id: &str,
        video_extradata: Option<Vec<u8>>,
        queue_depth: usize,
    ) -> PyResult<Self> {
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
        let backend = parse_backend(backend)?;
        let codec = parse_codec(codec)?;

        let file = File::create(output).map_err(|err| {
            PyValueError::new_err(format!("cannot create output {output}: {err}"))
        })?;
        let written = Arc::new(Mutex::new(Vec::new()));
        let sink = ReportingSink {
            inner: Fmp4FileSink::new(BufWriter::new(file)),
            written: Arc::clone(&written),
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
        let stream = runtime.open_stream(request_id);

        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(queue_depth);
        let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
        let shared = Arc::new(Shared {
            events: event_tx,
            error: Mutex::new(None),
            stats: Mutex::new(WorkerStats::default()),
            queued: AtomicUsize::new(0),
        });

        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("blazar-encode".into())
            .spawn(move || worker_main(stream, job_rx, worker_shared, written, info))
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
        })
    }

    /// Copy one frame into the encoder's queue. Returns immediately (the encode
    /// happens on the worker). Blocks only if the queue is full, with the GIL
    /// released so other Python threads keep running.
    ///
    /// `frame` must be a C-contiguous float32 array (numpy), shaped per `layout`:
    /// THWC = `(height, width, 3)`, BCTHW = `(1, 3, 1, height, width)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        frame,
        width,
        height,
        fps,
        layout,
        range,
        pts_ticks = None,
        scene_cut = false,
    ))]
    fn submit(
        &mut self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        fps: f64,
        layout: &str,
        range: &str,
        pts_ticks: Option<u64>,
        scene_cut: bool,
    ) -> PyResult<u64> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(PyValueError::new_err("fps must be > 0"));
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be non-zero"));
        }
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        let jobs = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;

        let layout = parse_layout(layout)?;
        let range = parse_range(range)?;

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

    /// Queue several frames as one submission. GPU backends convert them with
    /// one kernel launch, then encode them one at a time. Returns the batch's
    /// submission id; `poll()` reports one event for the batch.
    ///
    /// Frames must share geometry and format. Use this for a tensor the model
    /// already produced as a chunk (BCTHW with T > 1) instead of a Python loop.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        frames,
        width,
        height,
        fps,
        layout,
        range,
        pts_ticks = None,
        scene_cut = false,
    ))]
    fn submit_batch(
        &mut self,
        py: Python<'_>,
        frames: Vec<Bound<'_, PyAny>>,
        width: u32,
        height: u32,
        fps: f64,
        layout: &str,
        range: &str,
        pts_ticks: Option<u64>,
        scene_cut: bool,
    ) -> PyResult<u64> {
        if frames.is_empty() {
            return Err(PyValueError::new_err("frames is empty"));
        }
        if !fps.is_finite() || fps <= 0.0 {
            return Err(PyValueError::new_err("fps must be > 0"));
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be non-zero"));
        }
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        let jobs_out = self
            .jobs
            .lock()
            .map_err(|_| PyRuntimeError::new_err("encoder state lock poisoned"))?
            .clone()
            .ok_or_else(|| closed_error(&self.request_id))?;

        let layout = parse_layout(layout)?;
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

        let mut jobs = Vec::with_capacity(frames.len());
        for (index, frame) in frames.iter().enumerate() {
            let (bytes, dtype) = copy_frame(py, frame)?;
            let frame_index = self.frame_index + index as u64;
            let pts = pts_ticks
                .map(|first| {
                    let step = self.timebase.ticks_per_frame(fps);
                    first + index as u64 * step
                })
                .unwrap_or_else(|| {
                    Pts::from_frame_index(frame_index, fps, self.timebase).0
                });
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
            jobs.push(FrameJob {
                seq: self.seq,
                bytes,
                desc,
                boundary: if index == 0 && scene_cut {
                    Some(Boundary::SceneCut)
                } else {
                    None
                },
            });
        }

        let count = jobs.len() as u64;
        self.shared.queued.fetch_add(1, Ordering::Relaxed);
        let sent = py.detach(move || jobs_out.send(Job::Batch(jobs)));
        if sent.is_err() {
            self.shared.queued.fetch_sub(1, Ordering::Relaxed);
            return Err(PyRuntimeError::new_err(
                self.error_message()
                    .unwrap_or_else(|| "encoder worker stopped".to_string()),
            ));
        }

        let seq = self.seq;
        self.seq += 1;
        self.submitted += count;
        self.frame_index += count;
        self.frame_size.get_or_insert((width, height));
        self.fps.get_or_insert(fps);
        Ok(seq)
    }

    /// Queue a batch whose frames are already on the GPU, shaped like the
    /// model's VAE output: `(1, 3, T, H, W)` bf16 (or f32), contiguous, CUDA.
    ///
    /// Nothing is copied to the host: the NV12 kernel reads the tensor's memory
    /// directly. The binding keeps a reference to `tensor` until the worker has
    /// encoded the batch, so the caller may drop it after this returns.
    #[pyo3(signature = (tensor, fps, range = "unit01", pts_ticks = None, scene_cut = false))]
    fn submit_device_batch(
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
        wait_ack(py, ack_rx, timeout)
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
        outcome?;
        if let Some(err) = self.error_message() {
            return Err(PyRuntimeError::new_err(err));
        }
        Ok(())
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
            keyframe,
            units,
            bytes,
            encode_us,
            queue_depth,
        } => {
            dict.set_item("kind", "encoded")?;
            dict.set_item("seq", seq)?;
            dict.set_item("pts", pts)?;
            dict.set_item("keyframe", keyframe)?;
            dict.set_item("units", units)?;
            dict.set_item("bytes", bytes)?;
            dict.set_item("encode_us", encode_us)?;
            dict.set_item("queue_depth", queue_depth)?;
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
        let bytes = buffer
            .to_vec(py)
            .map_err(|err| PyValueError::new_err(format!("cannot copy frame buffer: {err}")))?;
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
    let values = buffer
        .to_vec(py)
        .map_err(|err| PyValueError::new_err(format!("cannot copy frame buffer: {err}")))?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok((bytes, Dtype::F32))
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

fn parse_backend(value: &str) -> PyResult<VideoEncoderBackend> {
    match value {
        "svtav1" => Ok(VideoEncoderBackend::SvtAv1),
        "nvenc_av1" => Ok(VideoEncoderBackend::NvencAv1),
        "nvenc_h264" => Ok(VideoEncoderBackend::NvencH264),
        "nvenc_hevc" => Ok(VideoEncoderBackend::NvencHevc),
        "rav1e" => Ok(VideoEncoderBackend::Rav1e),
        "x264" => Ok(VideoEncoderBackend::X264),
        "x265" => Ok(VideoEncoderBackend::X265),
        "vp9" => Ok(VideoEncoderBackend::Vp9),
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (expected svtav1|nvenc_av1|nvenc_h264|nvenc_hevc|rav1e|x264|x265|vp9)"
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

#[pymodule]
fn _blazar(module: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logging();
    module.add_class::<PyCalibrations>()?;
    module.add_class::<PyEncoder>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
