use std::fs;
use std::path::PathBuf;

use blazar_core::{
    Container, EncodedUnit, Pts, StreamParams, Timebase, VideoCodec, VideoTrackParams,
};
use blazar_sink::fmp4::Fmp4FileSink;
use blazar_sink::{Sink, SinkError};

fn av1_params() -> StreamParams {
    StreamParams {
        timebase: Timebase::MPEG_90K,
        video: VideoTrackParams {
            codec: VideoCodec::Av1,
            width: 320,
            height: 240,
            fps: 24.0,
            // AV1DecoderConfigurationRecord body for the default 8-bit
            // 4:2:0 profile used by the scaffold encoder.
            extradata: vec![0x81, 0x04, 0x0c, 0x00],
        },
        audio: None,
        container: Container::FragmentedMp4,
    }
}

fn video_unit(index: u64, keyframe: bool) -> EncodedUnit {
    let pts = Pts(index * 3750);
    EncodedUnit::video(
        vec![0x12, 0x00, index as u8, 0xaa],
        pts,
        pts,
        keyframe,
        3750,
    )
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("blazar-{name}-{}.mp4", std::process::id()))
}

fn count_box(bytes: &[u8], fourcc: &[u8; 4]) -> usize {
    bytes.windows(4).filter(|window| *window == fourcc).count()
}

#[test]
fn open_write_flush_close_produces_fragmented_mp4() {
    let path = temp_path("fmp4-basic");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);

    sink.open(av1_params()).unwrap();
    sink.write(&video_unit(0, true)).unwrap();
    sink.flush().unwrap();
    sink.close().unwrap();
    drop(sink);

    let bytes = fs::read(&path).unwrap();
    assert_eq!(count_box(&bytes, b"ftyp"), 1);
    assert_eq!(count_box(&bytes, b"moov"), 1);
    assert_eq!(count_box(&bytes, b"mvex"), 1);
    assert_eq!(count_box(&bytes, b"moof"), 1);
    assert_eq!(count_box(&bytes, b"mdat"), 1);
    let _ = fs::remove_file(path);
}

#[test]
fn each_explicit_flush_creates_a_fragment() {
    let path = temp_path("fmp4-flush");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);

    sink.open(av1_params()).unwrap();
    sink.write(&video_unit(0, true)).unwrap();
    sink.flush().unwrap();
    sink.write(&video_unit(1, false)).unwrap();
    sink.flush().unwrap();
    sink.close().unwrap();
    drop(sink);

    let bytes = fs::read(&path).unwrap();
    assert_eq!(count_box(&bytes, b"moof"), 2);
    assert_eq!(count_box(&bytes, b"mdat"), 2);
    let _ = fs::remove_file(path);
}

#[test]
fn close_flushes_pending_final_fragment() {
    let path = temp_path("fmp4-close");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);

    sink.open(av1_params()).unwrap();
    sink.write(&video_unit(0, true)).unwrap();
    sink.close().unwrap();
    drop(sink);

    let bytes = fs::read(&path).unwrap();
    assert_eq!(count_box(&bytes, b"moof"), 1);
    assert_eq!(count_box(&bytes, b"mdat"), 1);
    let _ = fs::remove_file(path);
}

#[test]
fn missing_av1_configuration_is_rejected_at_open() {
    let path = temp_path("fmp4-missing-config");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);
    let mut params = av1_params();
    params.video.extradata.clear();

    let err = sink.open(params).unwrap_err();
    assert!(matches!(err, SinkError::InvalidConfiguration(_)));
    let _ = fs::remove_file(path);
}

#[test]
fn write_before_open_is_rejected() {
    let path = temp_path("fmp4-not-open");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);

    let err = sink.write(&video_unit(0, true)).unwrap_err();
    assert!(matches!(err, SinkError::NotOpen));
    let _ = fs::remove_file(path);
}

#[test]
fn h264_and_aac_map_to_mp4_sample_entries() {
    let path = temp_path("fmp4-h264-aac");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);
    let params = StreamParams {
        timebase: Timebase::MPEG_90K,
        video: VideoTrackParams {
            codec: VideoCodec::H264,
            width: 320,
            height: 240,
            fps: 24.0,
            extradata: vec![1, 0x64, 0, 0x1f, 0xff, 0xe1, 0, 4, 0x67, 0, 0, 1, 0x68, 0],
        },
        audio: Some(blazar_core::AudioTrackParams {
            codec: blazar_core::AudioCodec::Aac,
            sample_rate: 48_000,
            channels: 2,
            extradata: vec![0x11, 0x90],
        }),
        container: Container::FragmentedMp4,
    };

    sink.open(params).unwrap();
    sink.write(&video_unit(0, true)).unwrap();
    sink.write(&EncodedUnit::audio(vec![0x21, 0x10], Pts(0), 1800))
        .unwrap();
    sink.flush().unwrap();
    sink.close().unwrap();
    drop(sink);

    let bytes = fs::read(&path).unwrap();
    assert_eq!(count_box(&bytes, b"avc1"), 1);
    assert_eq!(count_box(&bytes, b"mp4a"), 1);
    let _ = fs::remove_file(path);
}

#[test]
fn hevc_maps_to_hvc1_sample_entry() {
    let path = temp_path("fmp4-hevc");
    let file = fs::File::create(&path).unwrap();
    let mut sink = Fmp4FileSink::new(file);
    let mut params = av1_params();
    params.video.codec = VideoCodec::Hevc;
    params.video.extradata = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    sink.open(params).unwrap();
    sink.write(&video_unit(0, true)).unwrap();
    sink.flush().unwrap();
    sink.close().unwrap();
    drop(sink);

    let bytes = fs::read(&path).unwrap();
    assert_eq!(count_box(&bytes, b"hvc1"), 1);
    let _ = fs::remove_file(path);
}
