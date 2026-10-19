//! File sinks (architecture §6, build-order step 1).
//!
//! Two things live here:
//!
//! * [`DebugFileSink`] — a real, working file sink that writes each encoded
//!   unit in a deterministic, self-describing framing (not a standard
//!   container). It exists so the Phase-1 pipeline is runnable and verifiable
//!   end-to-end: ordering, interleave, and byte-exact payloads can be asserted
//!   without a conformant demuxer. This is the calibration-harness output and
//!   the integration-test target until the box muxer lands.
//!
//! * [`Fmp4FileSink`] — the real fragmented-MP4 muxer skeleton. It implements
//!   the [`Sink`] contract shape but the box-writing is deliberately
//!   unimplemented (returns [`SinkError::NotImplemented`]); the trait is
//!   designed so dropping the real muxer in is additive. This is the next
//!   concrete implementation task, called out rather than faked.

use std::io::Write;

use blazar_core::{EncodedUnit, StreamParams, TrackId};

use crate::{OrderGuard, Sink, SinkError};

/// Deterministic framed-dump file sink. Each unit is written as:
/// `[track:u8][keyframe:u8][pts:u64le][dts:u64le][duration:u64le][len:u32le][bytes]`.
/// A fixed 8-byte magic header `BLZRDBG1` precedes the stream.
pub struct DebugFileSink<W: Write> {
    writer: W,
    guard: OrderGuard,
    opened: bool,
    closed: bool,
    units_written: u64,
}

impl<W: Write> DebugFileSink<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            guard: OrderGuard::default(),
            opened: false,
            closed: false,
            units_written: 0,
        }
    }

    pub fn units_written(&self) -> u64 {
        self.units_written
    }

    /// Consume the sink and return the underlying writer (e.g. to inspect a
    /// `Vec<u8>` buffer in tests).
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn track_byte(t: TrackId) -> u8 {
        match t {
            TrackId::Video => 0,
            TrackId::Audio => 1,
        }
    }
}

impl<W: Write> Sink for DebugFileSink<W> {
    fn open(&mut self, _params: StreamParams) -> Result<(), SinkError> {
        self.writer
            .write_all(b"BLZRDBG1")
            .map_err(|e| SinkError::Io(e.to_string()))?;
        self.opened = true;
        Ok(())
    }

    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        if !self.opened {
            return Err(SinkError::NotOpen);
        }
        self.guard.check(unit)?;
        let mut rec = Vec::with_capacity(30 + unit.bytes.len());
        rec.push(Self::track_byte(unit.track_id));
        rec.push(unit.keyframe as u8);
        rec.extend_from_slice(&unit.pts.ticks().to_le_bytes());
        rec.extend_from_slice(&unit.dts.ticks().to_le_bytes());
        rec.extend_from_slice(&unit.duration.to_le_bytes());
        rec.extend_from_slice(&(unit.bytes.len() as u32).to_le_bytes());
        rec.extend_from_slice(&unit.bytes);
        self.writer
            .write_all(&rec)
            .map_err(|e| SinkError::Io(e.to_string()))?;
        self.units_written += 1;
        Ok(())
    }

    fn close(&mut self) -> Result<(), SinkError> {
        self.writer
            .flush()
            .map_err(|e| SinkError::Io(e.to_string()))?;
        self.closed = true;
        Ok(())
    }
}

/// Fragmented-MP4 muxer skeleton. The next real implementation task.
///
/// The box structure is well-defined (ftyp + moov with track descriptors, then
/// moof+mdat fragments per segment, mfra/index at close). The [`Sink`] contract
/// already carries everything a fragment needs (per-unit pts/dts/keyframe/
/// duration + track_id + StreamParams at open), so wiring a box writer here is
/// additive and does not touch the pipeline.
pub struct Fmp4FileSink<W: Write> {
    _writer: W,
    _params: Option<StreamParams>,
}

impl<W: Write> Fmp4FileSink<W> {
    pub fn new(writer: W) -> Self {
        Self {
            _writer: writer,
            _params: None,
        }
    }
}

impl<W: Write> Sink for Fmp4FileSink<W> {
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError> {
        self._params = Some(params);
        // TODO(phase1-muxer): write ftyp + moov (track descriptors from params).
        Err(SinkError::NotImplemented("fMP4 ftyp/moov box writing"))
    }

    fn write(&mut self, _unit: &EncodedUnit) -> Result<(), SinkError> {
        // TODO(phase1-muxer): accumulate a fragment; emit moof+mdat per segment.
        Err(SinkError::NotImplemented("fMP4 moof/mdat fragment writing"))
    }

    fn close(&mut self) -> Result<(), SinkError> {
        // TODO(phase1-muxer): flush last fragment + optional mfra index.
        Err(SinkError::NotImplemented("fMP4 finalization"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::{
        AudioCodec, AudioTrackParams, Container, Pts, Timebase, VideoCodec, VideoTrackParams,
    };

    fn params() -> StreamParams {
        StreamParams {
            timebase: Timebase::MPEG_90K,
            video: VideoTrackParams {
                codec: VideoCodec::SvtAv1,
                width: 2,
                height: 2,
                fps: 24.0,
            },
            audio: Some(AudioTrackParams {
                codec: AudioCodec::Opus,
                sample_rate: 48_000,
                channels: 2,
            }),
            container: Container::FragmentedMp4,
        }
    }

    #[test]
    fn debug_sink_writes_magic_and_counts_units() {
        let mut sink = DebugFileSink::new(Vec::new());
        sink.open(params()).unwrap();
        sink.write(&EncodedUnit::video(
            vec![1, 2, 3],
            Pts(0),
            Pts(0),
            true,
            3750,
        ))
        .unwrap();
        sink.write(&EncodedUnit::audio(vec![9], Pts(0), 480))
            .unwrap();
        sink.close().unwrap();
        assert_eq!(sink.units_written(), 2);
        let buf = sink.into_inner();
        assert_eq!(&buf[0..8], b"BLZRDBG1");
    }

    #[test]
    fn debug_sink_rejects_write_before_open() {
        let mut sink = DebugFileSink::new(Vec::new());
        let err = sink
            .write(&EncodedUnit::video(vec![0], Pts(0), Pts(0), true, 1))
            .unwrap_err();
        assert!(matches!(err, SinkError::NotOpen));
    }

    #[test]
    fn debug_sink_enforces_ordering() {
        let mut sink = DebugFileSink::new(Vec::new());
        sink.open(params()).unwrap();
        sink.write(&EncodedUnit::video(
            vec![0],
            Pts(3750),
            Pts(3750),
            false,
            3750,
        ))
        .unwrap();
        let err = sink
            .write(&EncodedUnit::video(vec![0], Pts(0), Pts(0), false, 3750))
            .unwrap_err();
        assert!(matches!(err, SinkError::OutOfOrder { .. }));
    }

    #[test]
    fn fmp4_skeleton_reports_not_implemented() {
        // Honest Phase-1 boundary: the real muxer is the next task.
        let mut sink = Fmp4FileSink::new(Vec::new());
        assert!(matches!(
            sink.open(params()),
            Err(SinkError::NotImplemented(_))
        ));
    }
}
