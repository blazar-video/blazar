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
//! * [`Fmp4FileSink`] — a real fragmented-MP4 file muxer backed by the
//!   upstream `oxideav-mp4` implementation. It writes AV1, H.264, and HEVC
//!   video plus Opus or AAC audio with explicit PTS/DTS/duration/keyframe
//!   metadata and exposes explicit fragment boundaries through [`Sink::flush`].

use std::io::{Seek, Write};

use blazar_core::{AudioCodec, Container, EncodedUnit, StreamParams, TrackId, VideoCodec};
use oxideav_core::{
    CodecId, CodecParameters, Muxer, Packet, StreamInfo, TimeBase as OxideTimeBase,
};
use oxideav_mp4::frag::{open_fragmented_typed, FragmentedMuxer};
use oxideav_mp4::options::{FragmentCadence, FragmentedOptions, Mp4MuxerOptions};

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

    fn flush(&mut self) -> Result<(), SinkError> {
        self.writer
            .flush()
            .map_err(|e| SinkError::Io(e.to_string()))
    }

    fn close(&mut self) -> Result<(), SinkError> {
        self.writer
            .flush()
            .map_err(|e| SinkError::Io(e.to_string()))?;
        self.closed = true;
        Ok(())
    }
}

/// Fragmented-MP4 file sink backed by `oxideav-mp4`.
///
/// The sink owns the muxer after `open`; `flush` publishes an explicit
/// `styp? + moof + mdat` fragment through the contribution in the forked
/// oxideav revision, and `close` writes the final trailer/index.
pub struct Fmp4FileSink<W: Write + Seek + Send + 'static> {
    writer: Option<W>,
    muxer: Option<FragmentedMuxer>,
    params: Option<StreamParams>,
    guard: OrderGuard,
    closed: bool,
}

impl<W: Write + Seek + Send + 'static> Fmp4FileSink<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: Some(writer),
            muxer: None,
            params: None,
            guard: OrderGuard::default(),
            closed: false,
        }
    }

    fn stream_infos(params: &StreamParams) -> Result<Vec<StreamInfo>, SinkError> {
        if params.container != Container::FragmentedMp4 {
            return Err(SinkError::InvalidConfiguration(
                "Fmp4FileSink requires Container::FragmentedMp4".into(),
            ));
        }
        if params.video.extradata.is_empty() {
            return Err(SinkError::InvalidConfiguration(
                "video codec configuration extradata is required".into(),
            ));
        }

        let time_base = OxideTimeBase::new(1, params.timebase.ticks_per_second() as i64);
        let video_codec = match params.video.codec {
            VideoCodec::Av1 => "av1",
            VideoCodec::H264 => "h264",
            VideoCodec::Hevc => "h265",
            VideoCodec::Vp9 => "vp9",
        };
        let mut video = CodecParameters::video(CodecId::new(video_codec));
        video.width = Some(params.video.width);
        video.height = Some(params.video.height);
        video.extradata = params.video.extradata.clone();

        let mut streams = vec![StreamInfo {
            index: 0,
            time_base,
            duration: None,
            start_time: Some(0),
            params: video,
        }];

        if let Some(audio) = &params.audio {
            let audio_codec = match audio.codec {
                AudioCodec::Opus => "opus",
                AudioCodec::Aac => "aac",
            };
            let mut audio_params = CodecParameters::audio(CodecId::new(audio_codec));
            audio_params.sample_rate = Some(audio.sample_rate);
            audio_params.channels = Some(audio.channels as u16);
            audio_params.extradata = audio.extradata.clone();
            streams.push(StreamInfo {
                index: 1,
                time_base,
                duration: None,
                start_time: Some(0),
                params: audio_params,
            });
        }

        Ok(streams)
    }

    fn packet_for(&self, unit: &EncodedUnit) -> Result<Packet, SinkError> {
        let params = self.params.as_ref().ok_or(SinkError::NotOpen)?;
        let stream_index = match unit.track_id {
            TrackId::Video => 0,
            TrackId::Audio => {
                if params.audio.is_none() {
                    return Err(SinkError::InvalidConfiguration(
                        "audio unit received on a video-only stream".into(),
                    ));
                }
                1
            }
        };
        let time_base = OxideTimeBase::new(1, params.timebase.ticks_per_second() as i64);
        let pts = i64::try_from(unit.pts.ticks()).map_err(|_| {
            SinkError::InvalidConfiguration("PTS exceeds oxideav i64 timestamp range".into())
        })?;
        let dts = i64::try_from(unit.dts.ticks()).map_err(|_| {
            SinkError::InvalidConfiguration("DTS exceeds oxideav i64 timestamp range".into())
        })?;
        let duration = i64::try_from(unit.duration).map_err(|_| {
            SinkError::InvalidConfiguration("duration exceeds oxideav i64 timestamp range".into())
        })?;

        let mut packet = Packet::new(stream_index, time_base, unit.bytes.clone());
        packet.pts = Some(pts);
        packet.dts = Some(dts);
        packet.duration = Some(duration);
        packet.flags.keyframe = unit.keyframe;
        Ok(packet)
    }
}

impl<W: Write + Seek + Send + 'static> Sink for Fmp4FileSink<W> {
    fn open(&mut self, params: StreamParams) -> Result<(), SinkError> {
        if self.muxer.is_some() || self.closed {
            return Err(SinkError::InvalidConfiguration(
                "fMP4 sink can only be opened once".into(),
            ));
        }
        let streams = Self::stream_infos(&params)?;
        let writer = self.writer.take().ok_or(SinkError::NotOpen)?;
        // Honour the packet timebase as the track's media timescale. The muxer
        // otherwise defaults video to 1/1000, which truncates frame durations
        // (3750 ticks at 90 kHz is 41.667 ms, not 41) and drifts over a stream.
        let video_timescale = params.timebase.ticks_per_second();
        let mut muxer = open_fragmented_typed(
            Box::new(writer),
            &streams,
            Mp4MuxerOptions {
                video_media_timescale: Some(video_timescale),
                ..Mp4MuxerOptions::default()
            },
            FragmentedOptions {
                cadence: FragmentCadence::EverySeconds(2.0),
                ..FragmentedOptions::default()
            },
        )
        .map_err(|e| SinkError::Muxer(e.to_string()))?;
        muxer
            .write_header()
            .map_err(|e| SinkError::Muxer(e.to_string()))?;
        self.params = Some(params);
        self.muxer = Some(muxer);
        Ok(())
    }

    fn write(&mut self, unit: &EncodedUnit) -> Result<(), SinkError> {
        if self.closed {
            return Err(SinkError::Closed);
        }
        self.guard.check(unit)?;
        let packet = self.packet_for(unit)?;
        self.muxer
            .as_mut()
            .ok_or(SinkError::NotOpen)?
            .write_packet(&packet)
            .map_err(|e| SinkError::Muxer(e.to_string()))
    }

    fn flush(&mut self) -> Result<(), SinkError> {
        if self.closed {
            return Err(SinkError::Closed);
        }
        self.muxer
            .as_mut()
            .ok_or(SinkError::NotOpen)?
            .flush_fragment()
            .map_err(|e| SinkError::Muxer(e.to_string()))
    }

    fn close(&mut self) -> Result<(), SinkError> {
        if self.closed {
            return Ok(());
        }
        self.muxer
            .as_mut()
            .ok_or(SinkError::NotOpen)?
            .write_trailer()
            .map_err(|e| SinkError::Muxer(e.to_string()))?;
        // The muxer owns the writer, and callers hand us a buffered one. Drop
        // it here so the buffer flushes before `close` returns: otherwise a
        // stream whose frames only arrive at finish (an encoder holding a
        // lookahead) leaves every byte in the buffer, and the file is empty
        // when the caller looks at it. Dropping is the only way to flush a
        // writer the muxer took ownership of; the trailer write above already
        // surfaces I/O errors.
        drop(self.muxer.take());
        self.closed = true;
        Ok(())
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
                codec: VideoCodec::Av1,
                width: 2,
                height: 2,
                fps: 24.0,
                extradata: vec![0x81, 0x04, 0x0c, 0x00],
            },
            audio: Some(AudioTrackParams {
                codec: AudioCodec::Opus,
                sample_rate: 48_000,
                channels: 2,
                extradata: b"OpusHead\x01\x01\x80\xbb\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00".to_vec(),
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

    /// Writer that records into a shared buffer, so a test can look at what
    /// the underlying device has *while the sink is still alive*.
    struct SharedWriter {
        buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        pos: u64,
    }

    impl std::io::Write for SharedWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            let mut buf = self.buf.lock().expect("shared buffer");
            let start = self.pos as usize;
            if buf.len() < start + data.len() {
                buf.resize(start + data.len(), 0);
            }
            buf[start..start + data.len()].copy_from_slice(data);
            self.pos += data.len() as u64;
            Ok(data.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl std::io::Seek for SharedWriter {
        fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
            self.pos = match from {
                std::io::SeekFrom::Start(n) => n,
                std::io::SeekFrom::Current(d) => (self.pos as i64 + d) as u64,
                std::io::SeekFrom::End(d) => {
                    let len = self.buf.lock().expect("shared buffer").len() as i64;
                    (len + d) as u64
                }
            };
            Ok(self.pos)
        }
    }

    /// The sink is handed a buffered writer in production. Closing must flush
    /// it before returning, otherwise a stream whose frames only arrive at
    /// finish leaves every byte in the buffer and the caller sees an empty
    /// file. (Found with x264 holding 24 frames in its lookahead.)
    #[test]
    fn close_flushes_a_buffered_writer_before_returning() {
        let shared = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = std::io::BufWriter::new(SharedWriter {
            buf: std::sync::Arc::clone(&shared),
            pos: 0,
        });
        let mut sink = Fmp4FileSink::new(writer);
        sink.open(params()).unwrap();
        for index in 0..4u64 {
            sink.write(&EncodedUnit::video(
                vec![0xff; 64],
                Pts(index * 3750),
                Pts(index * 3750),
                index == 0,
                3750,
            ))
            .unwrap();
        }
        sink.close().unwrap();

        // The sink object is deliberately still in scope: nothing may depend on
        // its destructor to make the file complete.
        let len = shared.lock().expect("shared buffer").len();
        assert!(len > 0, "writer still empty after close() returned");
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
    fn fmp4_sink_opens_and_closes_with_video_and_opus_tracks() {
        let mut sink = Fmp4FileSink::new(std::io::Cursor::new(Vec::new()));
        sink.open(params()).unwrap();
        sink.write(&EncodedUnit::video(
            vec![0x12, 0x00, 0xaa],
            Pts(0),
            Pts(0),
            true,
            3750,
        ))
        .unwrap();
        sink.close().unwrap();
    }
}
