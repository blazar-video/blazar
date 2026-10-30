// SPDX-License-Identifier: Apache-2.0
//! Span and counter instrumentation for the Blazar pipeline.
//!
//! The suite this exists for has one rule: numbers come from the code that runs,
//! not from a replica standing in for it. So the stages are recorded where they
//! happen, inside the worker and the submit path, and the Python side only reads
//! the result.
//!
//! # Cost
//!
//! A [`Tracer::disabled`] tracer costs one branch: [`Tracer::span`] returns a span
//! that touches nothing, and `add_counter` returns immediately. An enabled tracer
//! takes one mutex per recorded span, which is where the timestamps and the bounded
//! logs live. The overhead is measured rather than assumed — see
//! `examples/overhead.rs`.
//!
//! # What is exact and what is not
//!
//! `count`, `sum`, `min` and `max` cover every span. Percentiles come from a
//! bounded window of the most recent samples (a ring per stage), because holding
//! every timestamp for a long run is unbounded memory. A snapshot reports how many
//! samples that window held, so a percentile is always read with its basis.
//!
//! # Clocks
//!
//! Everything here is host time from [`std::time::Instant`]. Device-side intervals
//! need CUDA events and do not belong in these spans; counts of already-measured
//! values (driver microseconds, bytes moved) arrive through counters instead.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

/// A pipeline stage that can be timed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    /// Encoder/session construction, reported separately from per-frame figures.
    SessionOpen = 0,
    /// bf16/f32 to u8 on the device.
    Quantise = 1,
    /// Device to host copy.
    DeviceToHost = 2,
    /// Colour conversion into NV12 planes.
    Convert = 3,
    /// Host to device upload of the NV12 surface.
    Upload = 4,
    /// Handing the surface to the driver.
    Queue = 5,
    /// Draining encoded output.
    Poll = 6,
    /// Writing to the sink.
    MuxWrite = 7,
}

impl Stage {
    /// Every stage, in declaration order.
    pub const ALL: [Self; 8] = [
        Self::SessionOpen,
        Self::Quantise,
        Self::DeviceToHost,
        Self::Convert,
        Self::Upload,
        Self::Queue,
        Self::Poll,
        Self::MuxWrite,
    ];

    /// The name used in snapshots and traces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SessionOpen => "session_open",
            Self::Quantise => "quantise",
            Self::DeviceToHost => "d2h",
            Self::Convert => "convert",
            Self::Upload => "upload",
            Self::Queue => "queue",
            Self::Poll => "poll",
            Self::MuxWrite => "mux_write",
        }
    }
}

/// A value that is not a duration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Counter {
    /// Frames handed to the pipeline.
    FramesSubmitted = 0,
    /// Bytes copied device to host.
    BytesDeviceToHost = 1,
    /// Bytes uploaded host to device.
    BytesHostToDevice = 2,
    /// Microseconds spent inside the driver's encode call, as measured by the
    /// encoder itself rather than by a span around it.
    DriverMicros = 3,
}

impl Counter {
    /// Every counter, in declaration order.
    pub const ALL: [Self; 4] = [
        Self::FramesSubmitted,
        Self::BytesDeviceToHost,
        Self::BytesHostToDevice,
        Self::DriverMicros,
    ];

    /// The name used in snapshots.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FramesSubmitted => "frames_submitted",
            Self::BytesDeviceToHost => "bytes_d2h",
            Self::BytesHostToDevice => "bytes_h2d",
            Self::DriverMicros => "driver_us",
        }
    }
}

const NSTAGES: usize = Stage::ALL.len();
const NCOUNTERS: usize = Counter::ALL.len();

/// Default window of retained samples per stage.
pub const DEFAULT_SAMPLES: usize = 8192;
/// Default number of spans kept for the Chrome trace.
pub const DEFAULT_SPANS: usize = 65_536;

#[derive(Copy, Clone, Debug)]
struct SpanRecord {
    stage: Stage,
    start_ns: u64,
    dur_ns: u32,
    tid: u64,
}

struct State {
    /// Ring of the most recent samples per stage, in nanoseconds. Whole
    /// microseconds would collapse every sub-microsecond span to zero, which is
    /// exactly the range the cheap stages live in.
    samples: [Vec<u32>; NSTAGES],
    next: [usize; NSTAGES],
    counts: [u64; NSTAGES],
    sums: [u64; NSTAGES],
    mins: [u32; NSTAGES],
    maxs: [u32; NSTAGES],
    spans: VecDeque<SpanRecord>,
    spans_dropped: u64,
}

impl State {
    fn new(samples: usize, spans: usize) -> Self {
        Self {
            samples: std::array::from_fn(|_| Vec::with_capacity(samples)),
            next: [0; NSTAGES],
            counts: [0; NSTAGES],
            sums: [0; NSTAGES],
            mins: [u32::MAX; NSTAGES],
            maxs: [0; NSTAGES],
            spans: VecDeque::with_capacity(spans),
            spans_dropped: 0,
        }
    }
}

struct Inner {
    start: Instant,
    sample_cap: usize,
    span_cap: usize,
    state: Mutex<State>,
    counters: [AtomicU64; NCOUNTERS],
}

/// Records stage timings and counters.
///
/// Cheap to share: wrap in `Arc` and clone into the worker threads that need it.
pub struct Tracer {
    inner: Option<Inner>,
}

impl std::fmt::Debug for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer")
            .field("enabled", &self.inner.is_some())
            .finish()
    }
}

impl Tracer {
    /// An enabled tracer with the default window sizes.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacities(DEFAULT_SAMPLES, DEFAULT_SPANS)
    }

    /// An enabled tracer with explicit window sizes. Both bound memory: the first
    /// is samples retained per stage, the second is spans kept for the trace.
    #[must_use]
    pub fn with_capacities(samples: usize, spans: usize) -> Self {
        Self {
            inner: Some(Inner {
                start: Instant::now(),
                sample_cap: samples.max(1),
                span_cap: spans.max(1),
                state: Mutex::new(State::new(samples.max(1), spans.max(1))),
                counters: std::array::from_fn(|_| AtomicU64::new(0)),
            }),
        }
    }

    /// A tracer that records nothing. Every call to it is a no-op.
    #[must_use]
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Whether this tracer records anything.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Start timing `stage`. The sample is recorded when the returned span is
    /// dropped or finished.
    #[must_use]
    pub fn span(&self, stage: Stage) -> Span<'_> {
        let start = self.inner.as_ref().map(|_| Instant::now());
        Span {
            tracer: self,
            stage,
            start,
            done: false,
        }
    }

    /// Record an already-measured duration.
    pub fn record(&self, stage: Stage, dur: Duration) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let ns = u32::try_from(dur.as_nanos()).unwrap_or(u32::MAX);
        // A duration measured elsewhere has no start of its own; place it so it ends
        // now, which keeps the trace's timeline ordered.
        let now = offset_ns(inner);
        self.store(inner, stage, ns, now.saturating_sub(u64::from(ns)));
    }

    /// Record a span that began at `start`.
    fn record_span(&self, start: Instant, stage: Stage) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let ns = u32::try_from(start.elapsed().as_nanos()).unwrap_or(u32::MAX);
        self.store(
            inner,
            stage,
            ns,
            offset_ns(inner).saturating_sub(u64::from(ns)),
        );
    }

    /// Add to a counter.
    pub fn add_counter(&self, counter: Counter, value: u64) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        inner.counters[counter as usize].fetch_add(value, Ordering::Relaxed);
    }

    fn store(&self, inner: &Inner, stage: Stage, ns: u32, start_ns: u64) {
        let idx = stage as usize;
        let mut st = match inner.state.lock() {
            Ok(g) => g,
            // A panicking span must not poison the whole pipeline's instrumentation.
            Err(p) => p.into_inner(),
        };
        if st.samples[idx].len() < inner.sample_cap {
            st.samples[idx].push(ns);
        } else {
            let at = st.next[idx] % inner.sample_cap;
            st.samples[idx][at] = ns;
        }
        st.next[idx] = st.next[idx].wrapping_add(1);
        st.counts[idx] += 1;
        st.sums[idx] += u64::from(ns);
        st.mins[idx] = st.mins[idx].min(ns);
        st.maxs[idx] = st.maxs[idx].max(ns);

        if st.spans.len() == inner.span_cap {
            st.spans.pop_front();
            st.spans_dropped += 1;
        }
        st.spans.push_back(SpanRecord {
            stage,
            start_ns,
            dur_ns: ns,
            tid: thread_id(),
        });
    }

    /// A point-in-time summary. Cheap enough to call per frame from a harness.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        let Some(inner) = self.inner.as_ref() else {
            return Snapshot {
                spans_recorded: 0,
                spans_dropped: 0,
                stages: Vec::new(),
                counters: Vec::new(),
            };
        };
        let st = match inner.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut stages = Vec::new();
        for stage in Stage::ALL {
            let i = stage as usize;
            if st.counts[i] == 0 {
                continue;
            }
            let mut sorted = st.samples[i].clone();
            sorted.sort_unstable();
            let count = st.counts[i];
            stages.push(StageStat {
                stage: stage.as_str(),
                count,
                samples: sorted.len(),
                min_us: f64::from(st.mins[i]) / 1000.0,
                p50_us: percentile(&sorted, 0.50) / 1000.0,
                p95_us: percentile(&sorted, 0.95) / 1000.0,
                max_us: f64::from(st.maxs[i]) / 1000.0,
                mean_us: st.sums[i] as f64 / count as f64 / 1000.0,
            });
        }
        let counters = Counter::ALL
            .iter()
            .filter_map(|c| {
                let v = inner.counters[*c as usize].load(Ordering::Relaxed);
                (v > 0).then(|| CounterStat {
                    counter: c.as_str(),
                    value: v,
                })
            })
            .collect();
        Snapshot {
            spans_recorded: st.spans.len() as u64,
            spans_dropped: st.spans_dropped,
            stages,
            counters,
        }
    }

    /// Write the retained spans as a Chrome trace (also readable by Perfetto).
    ///
    /// # Errors
    ///
    /// Returns whatever the filesystem returns.
    pub fn write_chrome(&self, path: &Path) -> io::Result<()> {
        let Some(inner) = self.inner.as_ref() else {
            return Ok(());
        };
        let st = match inner.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut out = String::with_capacity(st.spans.len() * 96 + 32);
        out.push_str("{\"traceEvents\":[");
        for (n, r) in st.spans.iter().enumerate() {
            if n > 0 {
                out.push(',');
            }
            // Written by hand rather than through serde: this runs while the lock is
            // held and the shape is three fields.
            let _ = write!(
                out,
                "{{\"name\":\"{}\",\"cat\":\"stage\",\"ph\":\"X\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":{}}}",
                r.stage.as_str(),
                r.start_ns as f64 / 1000.0,
                f64::from(r.dur_ns) / 1000.0,
                r.tid
            );
        }
        out.push_str("]}");
        std::fs::write(path, out)
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// A stage timing in progress. Records one sample when dropped or finished.
#[derive(Debug)]
pub struct Span<'a> {
    tracer: &'a Tracer,
    stage: Stage,
    start: Option<Instant>,
    done: bool,
}

impl Span<'_> {
    /// Stop the clock now and record the sample.
    pub fn finish(mut self) {
        self.stop();
        self.done = true;
    }

    fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.tracer.record_span(start, self.stage);
        }
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        self.stop();
        let _ = self.done;
    }
}

/// A snapshot of one stage.
#[derive(Clone, Debug, Serialize)]
pub struct StageStat {
    /// Stage name.
    pub stage: &'static str,
    /// Spans recorded since construction (exact).
    pub count: u64,
    /// Samples the percentile window held; the basis for `p50_us` and `p95_us`.
    pub samples: usize,
    /// Fastest observed.
    pub min_us: f64,
    /// Median of the retained window.
    pub p50_us: f64,
    /// 95th percentile of the retained window.
    pub p95_us: f64,
    /// Slowest observed.
    pub max_us: f64,
    /// Mean over every span.
    pub mean_us: f64,
}

/// A counter in a snapshot.
#[derive(Clone, Debug, Serialize)]
pub struct CounterStat {
    /// Counter name.
    pub counter: &'static str,
    /// Accumulated value.
    pub value: u64,
}

/// A point-in-time summary of a [`Tracer`].
#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    /// Spans currently held in the trace log.
    pub spans_recorded: u64,
    /// Spans evicted from the log because it was full.
    pub spans_dropped: u64,
    /// Timings, one entry per stage that recorded anything.
    pub stages: Vec<StageStat>,
    /// Counters with a non-zero value.
    pub counters: Vec<CounterStat>,
}

impl Snapshot {
    /// The entry for `stage`, if it recorded anything.
    #[must_use]
    pub fn stage(&self, stage: Stage) -> Option<&StageStat> {
        self.stages.iter().find(|s| s.stage == stage.as_str())
    }

    /// The value of `counter`, if it was ever incremented.
    #[must_use]
    pub fn counter(&self, counter: Counter) -> Option<u64> {
        self.counters
            .iter()
            .find(|c| c.counter == counter.as_str())
            .map(|c| c.value)
    }
}

/// Nearest-rank percentile over a sorted slice, in nanoseconds.
fn percentile(sorted: &[u32], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    f64::from(sorted[rank - 1])
}

fn offset_ns(inner: &Inner) -> u64 {
    u64::try_from(inner.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn thread_id() -> u64 {
    // A per-thread identity is all a trace needs; the OS tid is not portable.
    thread_local! {
        static ID: u64 = {
            use std::sync::atomic::AtomicU64 as A;
            static NEXT: A = A::new(1);
            NEXT.fetch_add(1, Ordering::Relaxed)
        };
    }
    ID.with(|v| *v)
}
