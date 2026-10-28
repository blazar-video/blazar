// SPDX-License-Identifier: Apache-2.0
//! `blazar-sink` — the output sink contract (architecture §6).
//!
//! `write` takes structured [`EncodedUnit`]s (not raw bytes) so a file muxer
//! writes a container AND an RTP packetizer can mark keyframes. `track_id`
//! interleaves video/audio by PTS. A `poll_feedback` channel (sink→encoder) is
//! in the v0 interface even though only future streaming sinks use it: WebRTC
//! needs PLI→force-keyframe and RTCP-loss→bitrate-drop to reach rate control, so
//! a purely one-directional sink could not host WebRTC later.
//!
//! Build order (architecture §6): file sink → WebSocket fMP4 → str0m WebRTC.
//! The file sink is backed by the forked oxideav fMP4 implementation; the
//! transport sinks are the next delivery layers.

pub mod fmp4;

use blazar_core::{EncodedUnit, StreamParams, TrackId};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SinkError {
    #[error("write before open")]
    NotOpen,
    #[error("unit DTS {got} precedes last written DTS {last} on track {track:?}")]
    OutOfOrder { got: u64, last: u64, track: TrackId },
    #[error("io error: {0}")]
    Io(String),
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("invalid sink configuration: {0}")]
    InvalidConfiguration(String),
    #[error("muxer error: {0}")]
    Muxer(String),
    #[error("sink is closed")]
    Closed,
}

/// Feedback from a transport sink back to the encoder's rate control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Feedback {
    pub track_id: TrackId,
    pub force_keyframe: bool,
    pub bitrate_hint_kbps: Option<u32>,
}

/// The sink contract. Implemented by the file muxer, the WebSocket fMP4 sink,
/// and (later) the str0m WebRTC sink.
pub trait Sink {
    /// Declare both tracks + container + timebase. Called once before writes.
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError>;

    /// Write one encoded unit. The muxer interleaves tracks by PTS.
    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError>;

    /// Emit any pending segment/buffer boundary without finalizing the sink.
    /// File sinks may flush a fragment; transport sinks may publish a segment.
    fn flush(&mut self) -> Result<(), SinkError> {
        Ok(())
    }

    /// Finalize: file sinks write the index/moov; transports send EOS/drain.
    fn close(&mut self) -> Result<(), SinkError>;

    /// Poll optional sink→encoder feedback. File sinks always return `None`.
    fn poll_feedback(&mut self) -> Option<Feedback> {
        None
    }
}

/// Enforces the cross-track PTS invariant (architecture §3.1): within a track,
/// units are non-decreasing in PTS. Shared by sink implementations so the
/// ordering rule is defined once.
#[derive(Debug, Default)]
pub struct OrderGuard {
    last_video: Option<u64>,
    last_audio: Option<u64>,
}

impl OrderGuard {
    pub fn check(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        let slot = match unit.track_id {
            TrackId::Video => &mut self.last_video,
            TrackId::Audio => &mut self.last_audio,
        };
        // Decode order, not presentation order: with B-frames PTS is not
        // monotonic across the units a sink receives, but DTS is.
        let dts = unit.dts.ticks();
        if let Some(last) = *slot {
            if dts < last {
                return Err(SinkError::OutOfOrder {
                    got: dts,
                    last,
                    track: unit.track_id,
                });
            }
        }
        *slot = Some(dts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::Pts;

    fn vunit(pts: u64) -> EncodedUnit {
        EncodedUnit::video(vec![0], Pts(pts), Pts(pts), true, 3750)
    }

    fn reordered(pts: u64, dts: u64) -> EncodedUnit {
        EncodedUnit::video(vec![0], Pts(pts), Pts(dts), false, 3750)
    }

    #[test]
    fn order_guard_accepts_monotonic() {
        let mut g = OrderGuard::default();
        assert!(g.check(&vunit(0)).is_ok());
        assert!(g.check(&vunit(3750)).is_ok());
        assert!(g.check(&vunit(7500)).is_ok());
    }

    #[test]
    fn order_guard_rejects_regression() {
        let mut g = OrderGuard::default();
        g.check(&vunit(3750)).unwrap();
        let err = g.check(&vunit(0)).unwrap_err();
        assert!(matches!(err, SinkError::OutOfOrder { .. }));
    }

    #[test]
    fn order_guard_accepts_reordered_presentation_when_dts_is_monotonic() {
        // B-frames: the sink sees decode order, so PTS jumps around
        // (0, 2, 1, 4, 3) while DTS advances by one frame each time.
        let mut g = OrderGuard::default();
        for (pts, dts) in [(0, 0), (7500, 3750), (3750, 7500), (15000, 11250)] {
            assert!(g.check(&reordered(pts, dts)).is_ok(), "pts={pts} dts={dts}");
        }
    }

    #[test]
    fn order_guard_rejects_non_monotonic_dts() {
        let mut g = OrderGuard::default();
        g.check(&reordered(7500, 3750)).unwrap();
        assert!(matches!(
            g.check(&reordered(0, 0)).unwrap_err(),
            SinkError::OutOfOrder { .. }
        ));
    }

    #[test]
    fn order_guard_tracks_video_and_audio_independently() {
        // Audio and video interleave; a low audio PTS must not trip the video
        // ordering (separate per-track cursors).
        let mut g = OrderGuard::default();
        g.check(&vunit(7500)).unwrap();
        let a = EncodedUnit::audio(vec![0], Pts(1000), 480);
        assert!(g.check(&a).is_ok());
    }
}

/// Encoded unit kept in memory instead of being muxed to a file.
#[derive(Debug, Clone)]
pub struct StoredUnit {
    pub track: TrackId,
    pub pts: u64,
    pub dts: u64,
    pub keyframe: bool,
    pub bytes: Vec<u8>,
}

/// Shared handle to the units a [`MemorySink`] has written.
#[derive(Debug, Clone, Default)]
pub struct MemoryHandle(Arc<Mutex<VecDeque<StoredUnit>>>);

impl MemoryHandle {
    /// Take every unit written so far, oldest first.
    pub fn drain(&self) -> Vec<StoredUnit> {
        match self.0.lock() {
            Ok(mut q) => q.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Number of units waiting to be drained.
    pub fn len(&self) -> usize {
        self.0.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A [`Sink`] that keeps encoded units in memory for the caller to collect.
///
/// The buffer is bounded: when it is full `write` fails rather than dropping a
/// frame, so a caller that stops draining gets an error instead of silence.
pub struct MemorySink {
    handle: MemoryHandle,
    capacity: usize,
    params: Option<StreamParams>,
}

impl MemorySink {
    /// Returns the sink and a handle for draining it.
    pub fn new(capacity: usize) -> (Self, MemoryHandle) {
        let handle = MemoryHandle::default();
        (
            Self {
                handle: handle.clone(),
                capacity: capacity.max(1),
                params: None,
            },
            handle,
        )
    }
}

impl Sink for MemorySink {
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError> {
        self.params = Some(params);
        Ok(())
    }

    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        let mut queue = self
            .handle
            .0
            .lock()
            .map_err(|_| SinkError::InvalidConfiguration("memory sink poisoned".into()))?;
        if queue.len() >= self.capacity {
            return Err(SinkError::InvalidConfiguration(format!(
                "memory sink buffer full ({capacity} units): drain it or raise the capacity",
                capacity = self.capacity
            )));
        }
        queue.push_back(StoredUnit {
            track: unit.track_id,
            pts: unit.pts.0,
            dts: unit.dts.0,
            keyframe: unit.keyframe,
            bytes: unit.bytes.clone(),
        });
        Ok(())
    }

    fn close(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

#[cfg(test)]
mod memory_sink_tests {
    use super::*;
    use blazar_core::{Pts, Timebase};

    fn unit(bytes: &[u8], pts: u64) -> EncodedUnit {
        EncodedUnit {
            track_id: TrackId::Video,
            bytes: bytes.to_vec(),
            pts: Pts(pts),
            dts: Pts(pts),
            keyframe: true,
            duration: 3_000,
        }
    }

    fn params() -> StreamParams {
        StreamParams {
            timebase: Timebase::MPEG_90K,
            video: blazar_core::VideoTrackParams {
                codec: blazar_core::VideoCodec::H264,
                width: 64,
                height: 64,
                fps: 24.0,
                extradata: Vec::new(),
            },
            audio: None,
            container: blazar_core::Container::FragmentedMp4,
        }
    }

    #[test]
    fn keeps_units_in_order_and_drains_them() {
        let (mut sink, handle) = MemorySink::new(4);
        sink.open(params()).unwrap();
        sink.write(&unit(b"one", 1)).unwrap();
        sink.write(&unit(b"two", 2)).unwrap();
        assert_eq!(handle.len(), 2);

        let drained = handle.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].bytes, b"one");
        assert_eq!(drained[1].pts, 2);
        assert!(handle.is_empty());
    }

    #[test]
    fn a_full_buffer_is_an_error_not_a_dropped_frame() {
        let (mut sink, _handle) = MemorySink::new(1);
        sink.open(params()).unwrap();
        sink.write(&unit(b"one", 1)).unwrap();
        let err = sink.write(&unit(b"two", 2)).unwrap_err();
        assert!(format!("{err}").contains("buffer full"));
    }
}
