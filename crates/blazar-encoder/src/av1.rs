// SPDX-License-Identifier: Apache-2.0
//! Minimal AV1 bitstream inspection, limited to what the container needs.
//!
//! The fMP4 sink requires an `av1C` codec configuration record *before* the
//! first sample is written, but an NVENC session only reveals its real
//! sequence header at the start of the first keyframe. Rather than guessing
//! profile/level from configuration, [`av1c_from_keyframe`] walks the
//! low-overhead OBUs of that keyframe, parses the sequence header, and builds
//! the record from the driver's own values.
//!
//! Only the fields that appear in `av1C` are extracted, plus the sequence
//! header OBU itself, which is carried verbatim in `configOBUs`. Everything
//! else in the keyframe is ignored. This is intentionally not a general AV1
//! parser: it does not decode tiles, frame headers, or metadata.
//!
//! References: AV1 spec §5.5.1 (`sequence_header_obu`) and the AV1 ISOBMFF
//! binding (`AV1CodecConfigurationRecord`).

use thiserror::Error;

/// `OBU_SEQUENCE_HEADER` per the AV1 spec.
pub const OBU_SEQUENCE_HEADER: u8 = 1;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Av1Error {
    #[error("OBU at offset {offset} declares a size that runs past the buffer")]
    TruncatedObu { offset: usize },
    #[error("OBU at offset {offset} has no size field; av1C configOBUs require one")]
    MissingSizeField { offset: usize },
    #[error("no sequence header OBU in the keyframe")]
    NoSequenceHeader,
    #[error("sequence header ended before all required fields were read")]
    TruncatedSequenceHeader,
    #[error("malformed sequence header field")]
    MalformedSequenceHeader,
}

/// The subset of the AV1 sequence header that the `av1C` record carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1SequenceHeader {
    pub profile: u8,
    pub level_idx: u8,
    pub tier: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    /// Full range when true. Not part of `av1C` (it travels in the sequence
    /// header OBU), but useful for cross-checking what the encoder signalled.
    pub color_range: bool,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
}

impl Av1SequenceHeader {
    /// True when the header describes 8-bit 4:2:0, the only format the Phase-1
    /// preprocess stage produces.
    pub fn is_8bit_420(&self) -> bool {
        !self.high_bitdepth && !self.mono_chrome && self.subsampling_x && self.subsampling_y
    }
}

/// Parse the sequence header of a low-overhead AV1 keyframe.
///
/// Returns the parsed header plus the raw sequence header OBU (header byte,
/// size field, payload), ready to be used as `av1C` `configOBUs`.
pub fn sequence_header_from_keyframe(
    bytes: &[u8],
) -> Result<(Av1SequenceHeader, Vec<u8>), Av1Error> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let start = offset;
        let header = bytes[offset];
        offset += 1;

        let obu_type = (header >> 3) & 0x0f;
        let has_extension = (header >> 2) & 1 == 1;
        let has_size = (header >> 1) & 1 == 1;

        if has_extension {
            offset = offset
                .checked_add(1)
                .ok_or(Av1Error::TruncatedObu { offset: start })?;
            if offset > bytes.len() {
                return Err(Av1Error::TruncatedObu { offset: start });
            }
        }
        if !has_size {
            return Err(Av1Error::MissingSizeField { offset: start });
        }

        let size = read_leb128(bytes, &mut offset, start)?;
        let end = offset
            .checked_add(size)
            .ok_or(Av1Error::TruncatedObu { offset: start })?;
        if end > bytes.len() {
            return Err(Av1Error::TruncatedObu { offset: start });
        }

        if obu_type == OBU_SEQUENCE_HEADER {
            let parsed = parse_sequence_header(&bytes[offset..end])?;
            return Ok((parsed, bytes[start..end].to_vec()));
        }
        offset = end;
    }
    Err(Av1Error::NoSequenceHeader)
}

/// Build an `AV1CodecConfigurationRecord` (the `av1C` box body) from a keyframe.
///
/// The record is `4` bytes of fixed fields followed by the sequence header OBU
/// as `configOBUs`. Values come from the parsed header; nothing is inferred
/// from the requested encoder configuration.
pub fn av1c_from_keyframe(bytes: &[u8]) -> Result<Vec<u8>, Av1Error> {
    let (header, sequence_header_obu) = sequence_header_from_keyframe(bytes)?;

    let mut out = Vec::with_capacity(4 + sequence_header_obu.len());
    // marker (1) + version (7) = 1
    out.push(0x81);
    // seq_profile (3) + seq_level_idx_0 (5)
    out.push(((header.profile & 0x07) << 5) | (header.level_idx & 0x1f));
    // seq_tier_0 (1) + high_bitdepth (1) + twelve_bit (1) + monochrome (1) +
    // chroma_subsampling_x (1) + chroma_subsampling_y (1) +
    // chroma_sample_position (2)
    out.push(
        ((header.tier & 0x01) << 7)
            | (u8::from(header.high_bitdepth) << 6)
            | (u8::from(header.twelve_bit) << 5)
            | (u8::from(header.mono_chrome) << 4)
            | (u8::from(header.subsampling_x) << 3)
            | (u8::from(header.subsampling_y) << 2)
            | (header.chroma_sample_position & 0x03),
    );
    // reserved (3) + initial_presentation_delay_present (1) = 0 + reserved (4)
    out.push(0x00);
    out.extend_from_slice(&sequence_header_obu);
    Ok(out)
}

/// Read an LEB128 size field, advancing `offset`.
fn read_leb128(bytes: &[u8], offset: &mut usize, obu_start: usize) -> Result<usize, Av1Error> {
    let mut size = 0usize;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(*offset)
            .ok_or(Av1Error::TruncatedObu { offset: obu_start })?;
        *offset += 1;
        if shift >= usize::BITS {
            return Err(Av1Error::TruncatedObu { offset: obu_start });
        }
        size |= usize::from(byte & 0x7f) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok(size);
        }
    }
}

/// MSB-first bit reader over an OBU payload.
struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn read_bit(&mut self) -> Result<u32, Av1Error> {
        let byte = *self
            .data
            .get(self.bit / 8)
            .ok_or(Av1Error::TruncatedSequenceHeader)?;
        let value = u32::from((byte >> (7 - (self.bit % 8))) & 1);
        self.bit += 1;
        Ok(value)
    }

    fn read_flag(&mut self) -> Result<bool, Av1Error> {
        Ok(self.read_bit()? == 1)
    }

    fn read_bits(&mut self, count: u32) -> Result<u32, Av1Error> {
        debug_assert!(count <= 32);
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// AV1's `uvlc()`: leading-zero count, then that many value bits.
    fn read_uvlc(&mut self) -> Result<u32, Av1Error> {
        let mut leading_zeros = 0u32;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(Av1Error::MalformedSequenceHeader);
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let rest = self.read_bits(leading_zeros)?;
        Ok(((1u32 << leading_zeros) - 1) + rest)
    }
}

/// Parse `sequence_header_obu()` far enough to reach `color_config()`.
fn parse_sequence_header(payload: &[u8]) -> Result<Av1SequenceHeader, Av1Error> {
    let mut r = BitReader::new(payload);

    let profile = r.read_bits(3)? as u8;
    let _still_picture = r.read_flag()?;
    let reduced_still_picture_header = r.read_flag()?;

    let mut level_idx = 0u8;
    let mut tier = 0u8;

    if reduced_still_picture_header {
        level_idx = r.read_bits(5)? as u8;
    } else {
        let timing_info_present = r.read_flag()?;
        let mut decoder_model_info_present = false;
        let mut buffer_delay_length = 0u32;
        if timing_info_present {
            let _num_units_in_display_tick = r.read_bits(32)?;
            let _time_scale = r.read_bits(32)?;
            if r.read_flag()? {
                let _num_ticks_per_picture_minus_1 = r.read_uvlc()?;
            }
            decoder_model_info_present = r.read_flag()?;
            if decoder_model_info_present {
                buffer_delay_length = r.read_bits(5)? + 1;
                let _num_units_in_decoding_tick = r.read_bits(32)?;
                let _buffer_removal_time_length = r.read_bits(5)? + 1;
                let _frame_presentation_time_length = r.read_bits(5)? + 1;
            }
        }
        let initial_display_delay_present = r.read_flag()?;
        let operating_points_cnt_minus_1 = r.read_bits(5)?;
        for i in 0..=operating_points_cnt_minus_1 {
            let _operating_point_idc = r.read_bits(12)?;
            let op_level_idx = r.read_bits(5)? as u8;
            let op_tier = if op_level_idx > 7 {
                r.read_bit()? as u8
            } else {
                0
            };
            if i == 0 {
                level_idx = op_level_idx;
                tier = op_tier;
            }
            if decoder_model_info_present && r.read_flag()? {
                let _decoder_buffer_delay = r.read_bits(buffer_delay_length)?;
                let _encoder_buffer_delay = r.read_bits(buffer_delay_length)?;
                let _low_delay_mode_flag = r.read_flag()?;
            }
            if initial_display_delay_present && r.read_flag()? {
                let _initial_display_delay_minus_1 = r.read_bits(4)?;
            }
        }
    }

    let frame_width_bits = r.read_bits(4)? + 1;
    let frame_height_bits = r.read_bits(4)? + 1;
    let max_frame_width = r.read_bits(frame_width_bits)? + 1;
    let max_frame_height = r.read_bits(frame_height_bits)? + 1;

    if r.read_flag()? {
        // frame_id_numbers_present_flag
        let _delta_frame_id_length_minus_2 = r.read_bits(4)?;
        let _additional_frame_id_length_minus_1 = r.read_bits(3)?;
    }
    let _use_128x128_superblock = r.read_flag()?;
    let _enable_filter_intra = r.read_flag()?;
    let _enable_intra_edge_filter = r.read_flag()?;
    let _enable_interintra_compound = r.read_flag()?;
    let _enable_masked_compound = r.read_flag()?;
    let _enable_warped_motion = r.read_flag()?;
    let _enable_dual_filter = r.read_flag()?;
    let enable_order_hint = r.read_flag()?;
    if enable_order_hint {
        let _enable_jnt_comp = r.read_flag()?;
        let _enable_ref_frame_mvs = r.read_flag()?;
    }
    let seq_choose_screen_content_tools = r.read_flag()?;
    let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
        2
    } else {
        r.read_bit()?
    };
    if seq_force_screen_content_tools > 0 && !seq_choose_screen_content_tools {
        let _seq_force_integer_mv = r.read_flag()?;
    }
    if enable_order_hint {
        let _order_hint_bits_minus_1 = r.read_bits(3)?;
    }
    let _enable_superres = r.read_flag()?;
    let _enable_cdef = r.read_flag()?;
    let _enable_restoration = r.read_flag()?;

    let color = parse_color_config(&mut r, profile)?;

    Ok(Av1SequenceHeader {
        profile,
        level_idx,
        tier,
        high_bitdepth: color.high_bitdepth,
        twelve_bit: color.twelve_bit,
        mono_chrome: color.mono_chrome,
        subsampling_x: color.subsampling_x,
        subsampling_y: color.subsampling_y,
        chroma_sample_position: color.chroma_sample_position,
        color_range: color.color_range,
        max_frame_width,
        max_frame_height,
    })
}

struct ColorConfig {
    high_bitdepth: bool,
    twelve_bit: bool,
    mono_chrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
    chroma_sample_position: u8,
    color_range: bool,
}

fn parse_color_config(r: &mut BitReader<'_>, profile: u8) -> Result<ColorConfig, Av1Error> {
    const CP_BT_709: u32 = 1;
    const TC_SRGB: u32 = 13;
    const MC_IDENTITY: u32 = 0;

    let high_bitdepth = r.read_flag()?;
    let twelve_bit = if profile == 2 && high_bitdepth {
        r.read_flag()?
    } else {
        false
    };
    let bit_depth = if profile == 2 && high_bitdepth {
        if twelve_bit {
            12
        } else {
            10
        }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let mono_chrome = if profile == 1 { false } else { r.read_flag()? };

    let color_description_present = r.read_flag()?;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present {
            (r.read_bits(8)?, r.read_bits(8)?, r.read_bits(8)?)
        } else {
            (2, 2, 2) // CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED
        };

    let mut chroma_sample_position = 0u8;
    let color_range;
    let subsampling_x;
    let subsampling_y;

    if mono_chrome {
        color_range = r.read_flag()?;
        subsampling_x = true;
        subsampling_y = true;
    } else if color_primaries == CP_BT_709
        && transfer_characteristics == TC_SRGB
        && matrix_coefficients == MC_IDENTITY
    {
        color_range = true;
        subsampling_x = false;
        subsampling_y = false;
    } else {
        color_range = r.read_flag()?;
        match profile {
            0 => {
                subsampling_x = true;
                subsampling_y = true;
            }
            1 => {
                subsampling_x = false;
                subsampling_y = false;
            }
            _ => {
                if bit_depth == 12 {
                    subsampling_x = r.read_flag()?;
                    subsampling_y = if subsampling_x { r.read_flag()? } else { false };
                } else {
                    subsampling_x = true;
                    subsampling_y = true;
                }
            }
        }
        if subsampling_x && subsampling_y {
            chroma_sample_position = r.read_bits(2)? as u8;
        }
    }
    let _separate_uv_delta_q = r.read_flag()?;

    Ok(ColorConfig {
        high_bitdepth,
        twelve_bit,
        mono_chrome,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        color_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_leb128_is_an_error() {
        // OBU header with has_size=1, then a continuation byte and EOF.
        let bytes = [0x0a, 0x80];
        assert!(matches!(
            sequence_header_from_keyframe(&bytes),
            Err(Av1Error::TruncatedObu { .. })
        ));
    }

    #[test]
    fn skips_padding_and_metadata_obus() {
        // Padding OBU (type 15, size 2), then a sequence header whose payload is
        // too short to parse: proves the walker advanced past the padding.
        let bytes = [0x7a, 0x02, 0x00, 0x00, 0x0a, 0x01, 0x00];
        assert_eq!(
            sequence_header_from_keyframe(&bytes).unwrap_err(),
            Av1Error::TruncatedSequenceHeader
        );
    }
}
