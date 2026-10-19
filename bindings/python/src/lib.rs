//! Python bindings for the Blazar runtime.
//!
//! Two classes, deliberately thin:
//!
//! * `Calibrations` — a parsed calibration directory. Cheap to keep; reused
//!   across requests.
//! * `Encoder` — one stream. Owns the Rust stream (and therefore the encoder and
//!   the fMP4 sink). Create one per request; `close()` finalizes the file.
//!
//! Rules the binding enforces instead of guessing:
//! * `dtype`, `layout` and `range` are required on every submit. The runtimes
//!   never tag colorspace, so Blazar records the BT.709 assumption.
//! * The frame buffer must be C-contiguous. A non-contiguous array is rejected
//!   with the fix in the message rather than silently copied per pixel.
//! * Frames within a stream must keep width/height/fps. A change is an error, not
//!   a new segment.
//!
//! Not here yet: audio, device tensors, GIL release during encode (the stream is
//! intentionally `unsendable` for now), and any sink other than a file.

use std::fs::File;
use std::io::BufWriter;

use blazar_calibration::CalibrationIndex;
use blazar_core::{
    ColorRange, ColorSpace, Container, Dtype, FrameBuffer, FrameDesc, FrameRef, Layout,
    ModelIdentity, Pts, Target, Timebase, VideoCodec, VideoEncoderBackend,
};
use blazar_runtime::{Boundary, Encoder as RuntimeEncoder, Output, RuntimeError, Stream};
use blazar_sink::fmp4::Fmp4FileSink;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type FileSink = Fmp4FileSink<BufWriter<File>>;

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

/// One encode stream writing a fragmented MP4.
#[pyclass(name = "Encoder", module = "blazar", unsendable)]
struct PyEncoder {
    stream: Option<Stream<FileSink>>,
    request_id: String,
    timebase: Timebase,
    frame_index: u64,
    /// Fixed by the first submit; a change means a new request, not a segment.
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
    ) -> PyResult<Self> {
        let identity = model_identity(identity)?;
        let target = parse_target(target)?;
        let backend = parse_backend(backend)?;
        let codec = parse_codec(codec)?;

        let file = File::create(output).map_err(|err| {
            PyValueError::new_err(format!("cannot create output {output}: {err}"))
        })?;
        let sink = Fmp4FileSink::new(BufWriter::new(file));

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

        Ok(Self {
            stream: Some(runtime.open_stream(request_id)),
            request_id: request_id.to_string(),
            timebase: Timebase::MPEG_90K,
            frame_index: 0,
            frame_size: None,
            fps: None,
        })
    }

    /// Submit one frame. Returns the stream's sequence number.
    ///
    /// `frame` must be a C-contiguous float32 array (numpy), shaped per `layout`:
    /// THWC = `(height, width, 3)`, BCTHW = `(1, 3, 1, height, width)`.
    ///
    /// np.float32 is required because numpy has no bf16 dtype: cast a bf16 torch
    /// tensor with `.float().numpy()`/`.cpu().numpy()`. Non-contiguous views are
    /// copied in C order; a non-float32 buffer is rejected with the fix.
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

        // One copy: the preprocess/encode path then owns its bytes and does not
        // borrow Python memory (which also keeps the door open for releasing the
        // GIL here once the encoder handle is Send).
        let bytes = copy_contiguous(py, frame)?;

        let pts = pts_ticks
            .unwrap_or_else(|| Pts::from_frame_index(self.frame_index, fps, self.timebase).0);
        let desc = FrameDesc {
            width,
            height,
            dtype: Dtype::F32,
            range,
            layout,
            colorspace: ColorSpace::AssumedBt709,
            fps,
            pts: Pts(pts),
        };
        let boundary = if scene_cut {
            Some(Boundary::SceneCut)
        } else {
            None
        };

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| closed_error(&self.request_id))?;
        let seq = stream
            .submit_video(
                FrameRef::new(FrameBuffer::host_owned(bytes), desc),
                None,
                boundary,
            )
            .map_err(backend_error)?;
        self.frame_index += 1;
        self.frame_size.get_or_insert((width, height));
        self.fps.get_or_insert(fps);
        Ok(seq)
    }

    /// Drain what the encoder can emit without ending the stream.
    fn flush(&mut self) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| closed_error(&self.request_id))?;
        stream.flush().map_err(backend_error)
    }

    /// Finalize the stream and the container. Idempotent.
    fn close(&mut self) -> PyResult<()> {
        let Some(mut stream) = self.stream.take() else {
            return Ok(());
        };
        stream.close().map_err(backend_error)
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
        slf.close()
    }

    fn __repr__(&self) -> String {
        format!(
            "<blazar.Encoder: request_id={:?}, frames={}, closed={}>",
            self.request_id,
            self.frame_index,
            self.stream.is_none()
        )
    }
}

/// Copy a float32 buffer-protocol object into owned little-endian bytes.
fn copy_contiguous(py: Python<'_>, frame: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let buffer = PyBuffer::<f32>::get(frame).map_err(|_| {
        PyValueError::new_err(
            "frame must be a float32 array supporting the buffer protocol \
             (numpy float32, C order); convert with \
             numpy.ascontiguousarray(frame, dtype=numpy.float32) or \
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
    Ok(bytes)
}

fn closed_error(request_id: &str) -> PyErr {
    PyRuntimeError::new_err(format!("stream {request_id:?} is closed"))
}

fn backend_error(err: RuntimeError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
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
        "rav1e" => Ok(VideoEncoderBackend::Rav1e),
        other => Err(PyValueError::new_err(format!(
            "unknown backend {other:?} (expected svtav1|nvenc_av1|rav1e)"
        ))),
    }
}

fn parse_codec(value: &str) -> PyResult<VideoCodec> {
    match value {
        "av1" => Ok(VideoCodec::Av1),
        "h264" => Ok(VideoCodec::H264),
        "hevc" => Ok(VideoCodec::Hevc),
        other => Err(PyValueError::new_err(format!(
            "unknown codec {other:?} (expected av1|h264|hevc)"
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

#[pymodule]
fn _blazar(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCalibrations>()?;
    module.add_class::<PyEncoder>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
