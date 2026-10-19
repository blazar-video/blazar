use blazar_encoder::av1::{av1c_from_keyframe, Av1Error};

/// A real NVENC keyframe prefix captured on mosk (see `tests/fixtures/README.md`):
/// temporal delimiter, sequence header, start of the frame OBU. 832x480 NV12.
const KEYFRAME_HEAD: &[u8] = include_bytes!("fixtures/nvenc_av1_keyframe_head.bin");

#[test]
fn parses_the_sequence_header_of_a_real_nvenc_keyframe() {
    let (header, _obu) = blazar_encoder::av1::sequence_header_from_keyframe(KEYFRAME_HEAD)
        .expect("fixture is a valid AV1 keyframe prefix");

    assert_eq!(header.profile, 0, "8-bit 4:2:0 is profile 0");
    assert_eq!(header.level_idx, 4, "driver signalled level index 4");
    assert_eq!(header.tier, 0, "main tier");
    assert!(!header.high_bitdepth);
    assert!(!header.twelve_bit);
    assert!(!header.mono_chrome);
    assert!(header.subsampling_x && header.subsampling_y, "4:2:0");
    assert_eq!(header.chroma_sample_position, 0, "unspecified");
    // These two are the real cross-check on bit alignment: if any preceding
    // field were read at the wrong width, the dimensions would not match the
    // frame that produced the fixture.
    assert_eq!(header.max_frame_width, 832);
    assert_eq!(header.max_frame_height, 480);
}

#[test]
fn builds_av1c_from_the_real_sequence_header() {
    let av1c = av1c_from_keyframe(KEYFRAME_HEAD).expect("av1C is derivable");

    // marker+version, profile 0 / level 4, 8-bit 4:2:0 no chroma position,
    // no initial presentation delay.
    assert_eq!(&av1c[..4], &[0x81, 0x04, 0x0c, 0x00]);
    // configOBUs: the sequence header OBU verbatim, including its size field.
    assert_eq!(
        &av1c[4..],
        &[0x0a, 0x0b, 0x00, 0x00, 0x00, 0x24, 0xc6, 0x7f, 0xdf, 0x00, 0x86, 0x60, 0x10]
    );
    assert_eq!(av1c.len(), 4 + 13);
}

#[test]
fn rejects_input_without_a_sequence_header() {
    // Temporal delimiter (size 0) followed by a frame OBU (size 1).
    let obus = [0x12, 0x00, 0x32, 0x01, 0x00];
    assert_eq!(
        av1c_from_keyframe(&obus).unwrap_err(),
        Av1Error::NoSequenceHeader
    );
}

#[test]
fn rejects_an_obu_whose_size_runs_past_the_buffer() {
    // Sequence header OBU claiming 11 bytes of payload with only 2 present.
    let truncated = [0x0a, 0x0b, 0x00, 0x00];
    assert!(matches!(
        av1c_from_keyframe(&truncated),
        Err(Av1Error::TruncatedObu { .. })
    ));
}

#[test]
fn rejects_an_obu_without_a_size_field() {
    // OBU_SEQUENCE_HEADER with obu_has_size_field = 0. ISOBMFF samples and
    // av1C configOBUs require size fields, so this is not muxable.
    let no_size = [0x08, 0x00];
    assert!(matches!(
        av1c_from_keyframe(&no_size),
        Err(Av1Error::MissingSizeField { .. })
    ));
}

#[test]
fn rejects_a_truncated_sequence_header_payload() {
    // Sequence header OBU with only 3 payload bytes: valid framing, not enough
    // bits for profile/level.
    let short = [0x0a, 0x03, 0x00, 0x00, 0x00];
    assert_eq!(
        av1c_from_keyframe(&short).unwrap_err(),
        Av1Error::TruncatedSequenceHeader
    );
}

#[test]
fn rejects_an_empty_buffer() {
    assert_eq!(
        av1c_from_keyframe(&[]).unwrap_err(),
        Av1Error::NoSequenceHeader
    );
}
