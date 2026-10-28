// SPDX-License-Identifier: Apache-2.0
//! Annex B (H.264/HEVC) to MP4 sample formats, plus `avcC`/`hvcC` records.
//!
//! NVENC emits H.264 and HEVC as Annex B: NAL units separated by `00 00 01` or
//! `00 00 00 01`, with SPS/PPS (and VPS for HEVC) in-band on keyframes. MP4
//! stores samples as length-prefixed NAL units and keeps the parameter sets in
//! a decoder configuration record (`avcC` / `hvcC`). oxideav treats those
//! records as opaque bytes, so this module builds them.
//!
//! The builders are tested byte-for-byte against records ffmpeg produced from
//! the same NVENC streams (`tests/fixtures/*.config`). That comparison is the
//! guard against a plausible-but-wrong SPS bit reader; reading the spec again
//! would not catch it.

use thiserror::Error;

/// Which Annex B flavour to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnexB {
    H264,
    Hevc,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NalError {
    #[error("access unit contains no NAL units")]
    NoNalUnits,
    #[error("access unit has no coded slice after removing parameter sets and AUDs")]
    NoCodedSlice,
    #[error("parameter sets are incomplete for {0:?}")]
    IncompleteParameterSets(AnnexB),
    #[error("malformed NAL: {0}")]
    Malformed(&'static str),
}

/// One access unit converted for MP4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAccessUnit {
    /// `avcC`/`hvcC` body, present when this unit carried the parameter sets.
    pub config: Option<Vec<u8>>,
    /// Length-prefixed NAL units, ready to be a sample.
    pub sample: Vec<u8>,
}

const H264_NAL_AUD: u8 = 9;
const H264_NAL_SPS: u8 = 7;
const H264_NAL_PPS: u8 = 8;
const HEVC_NAL_VPS: u8 = 32;
const HEVC_NAL_SPS: u8 = 33;
const HEVC_NAL_PPS: u8 = 34;
const HEVC_NAL_AUD: u8 = 35;

/// Split an Annex B buffer into NAL units (start codes removed).
///
/// A leading `00 00 00 01` is matched through its `00 00 01` suffix, so NAL
/// payloads are trimmed of trailing zero bytes, which encoders insert only as
/// part of the next start code.
pub fn split_annexb(bytes: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut cursor = 0usize;
    let mut start: Option<usize> = None;
    while cursor + 3 <= bytes.len() {
        if bytes[cursor] == 0 && bytes[cursor + 1] == 0 && bytes[cursor + 2] == 1 {
            if let Some(begin) = start {
                let mut end = cursor;
                while end > begin && bytes[end - 1] == 0 {
                    end -= 1;
                }
                if end > begin {
                    nals.push(&bytes[begin..end]);
                }
            }
            cursor += 3;
            start = Some(cursor);
        } else {
            cursor += 1;
        }
    }
    if let Some(begin) = start {
        let mut end = bytes.len();
        while end > begin && bytes[end - 1] == 0 {
            end -= 1;
        }
        if end > begin {
            nals.push(&bytes[begin..end]);
        }
    }
    nals
}

fn nal_type(nal: &[u8], codec: AnnexB) -> u8 {
    match codec {
        AnnexB::H264 => nal[0] & 0x1f,
        AnnexB::Hevc => (nal[0] >> 1) & 0x3f,
    }
}

fn is_parameter_set(nal_type: u8, codec: AnnexB) -> bool {
    match codec {
        AnnexB::H264 => nal_type == H264_NAL_SPS || nal_type == H264_NAL_PPS,
        AnnexB::Hevc => {
            nal_type == HEVC_NAL_VPS || nal_type == HEVC_NAL_SPS || nal_type == HEVC_NAL_PPS
        }
    }
}

fn is_aud(nal_type: u8, codec: AnnexB) -> bool {
    match codec {
        AnnexB::H264 => nal_type == H264_NAL_AUD,
        AnnexB::Hevc => nal_type == HEVC_NAL_AUD,
    }
}

/// Convert one access unit: build the config record if the parameter sets are
/// present, and always emit the sample with AUDs and parameter sets removed.
pub fn parse_access_unit(au: &[u8], codec: AnnexB) -> Result<ParsedAccessUnit, NalError> {
    let nals = split_annexb(au);
    if nals.is_empty() {
        return Err(NalError::NoNalUnits);
    }

    let mut config = match codec {
        AnnexB::H264 => h264_config(&nals)?,
        AnnexB::Hevc => hevc_config(&nals)?,
    };
    if config.is_none() {
        config = None;
    }

    let mut sample = Vec::with_capacity(au.len());
    let mut coded = 0usize;
    for nal in &nals {
        let kind = nal_type(nal, codec);
        if is_aud(kind, codec) || is_parameter_set(kind, codec) {
            continue;
        }
        let len = u32::try_from(nal.len()).map_err(|_| NalError::Malformed("NAL too large"))?;
        sample.extend_from_slice(&len.to_be_bytes());
        sample.extend_from_slice(nal);
        coded += 1;
    }
    if coded == 0 {
        return Err(NalError::NoCodedSlice);
    }
    Ok(ParsedAccessUnit { config, sample })
}

/// Build an `avcC`/`hvcC` body from parameter sets that came from a codec's own
/// configuration record rather than from the bitstream.
///
/// Backends that hand parameter sets out of band (VideoToolbox) use this so the
/// container sees exactly the same record the in-band path would produce.
pub fn config_from_parameter_sets(
    codec: AnnexB,
    sets: &[&[u8]],
) -> Result<Option<Vec<u8>>, NalError> {
    match codec {
        AnnexB::H264 => h264_config(sets),
        AnnexB::Hevc => hevc_config(sets),
    }
}

fn h264_config(nals: &[&[u8]]) -> Result<Option<Vec<u8>>, NalError> {
    let sps = nals
        .iter()
        .find(|n| nal_type(n, AnnexB::H264) == H264_NAL_SPS)
        .copied();
    let pps = nals
        .iter()
        .find(|n| nal_type(n, AnnexB::H264) == H264_NAL_PPS)
        .copied();
    let (Some(sps), Some(pps)) = (sps, pps) else {
        return match (sps, pps) {
            (None, None) => Ok(None),
            _ => Err(NalError::IncompleteParameterSets(AnnexB::H264)),
        };
    };
    if sps.len() < 4 {
        return Err(NalError::Malformed("H.264 SPS shorter than 4 bytes"));
    }
    let mut out = Vec::with_capacity(11 + sps.len() + pps.len());
    out.push(1); // configurationVersion
    out.extend_from_slice(&sps[1..4]); // profile, compat, level
    out.push(0xff); // reserved(6) | lengthSizeMinusOne = 3
    out.push(0xe1); // reserved(3) | numOfSequenceParameterSets = 1
    out.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    out.extend_from_slice(sps);
    out.push(1); // numOfPictureParameterSets
    out.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    out.extend_from_slice(pps);
    Ok(Some(out))
}

/// Fields `hvcC` needs from the SPS, in the order the record stores them.
struct HevcSps {
    profile_space: u8,
    tier: u8,
    profile_idc: u8,
    compatibility: u32,
    constraints: u64,
    level: u8,
    chroma_format_idc: u32,
    bit_depth_luma_minus8: u32,
    bit_depth_chroma_minus8: u32,
}

fn hevc_config(nals: &[&[u8]]) -> Result<Option<Vec<u8>>, NalError> {
    let find = |kind: u8| {
        nals.iter()
            .find(|n| nal_type(n, AnnexB::Hevc) == kind)
            .copied()
    };
    let vps = find(HEVC_NAL_VPS);
    let sps = find(HEVC_NAL_SPS);
    let pps = find(HEVC_NAL_PPS);
    let (Some(sps), Some(pps)) = (sps, pps) else {
        return match (vps, sps, pps) {
            (None, None, None) => Ok(None),
            _ => Err(NalError::IncompleteParameterSets(AnnexB::Hevc)),
        };
    };
    let fields = parse_hevc_sps(sps)?;

    let mut arrays: Vec<(u8, &[u8])> = Vec::with_capacity(3);
    if let Some(vps) = vps {
        arrays.push((HEVC_NAL_VPS, vps));
    }
    arrays.push((HEVC_NAL_SPS, sps));
    arrays.push((HEVC_NAL_PPS, pps));

    let mut out = Vec::with_capacity(23 + arrays.iter().map(|(_, n)| n.len() + 5).sum::<usize>());
    out.push(1); // configurationVersion
    out.push((fields.profile_space << 6) | (fields.tier << 5) | fields.profile_idc);
    out.extend_from_slice(&fields.compatibility.to_be_bytes());
    out.extend_from_slice(&fields.constraints.to_be_bytes()[2..]); // 6 bytes
    out.push(fields.level);
    out.extend_from_slice(&[0xf0, 0x00]); // min_spatial_segmentation_idc = 0
    out.push(0xfc); // parallelismType = 0
    out.push(0xfc | (fields.chroma_format_idc as u8 & 0x03));
    out.push(0xf8 | (fields.bit_depth_luma_minus8 as u8 & 0x07));
    out.push(0xf8 | (fields.bit_depth_chroma_minus8 as u8 & 0x07));
    out.extend_from_slice(&[0x00, 0x00]); // avgFrameRate
    out.push(0x0f); // constantFrameRate 0, numTemporalLayers 1, nested 1, lengthSize 3
    out.push(arrays.len() as u8);
    for (kind, nal) in arrays {
        out.push(kind); // array_completeness = 0, matching ffmpeg's record
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        out.extend_from_slice(nal);
    }
    Ok(Some(out))
}

/// Parse the SPS fields `hvcC` needs. `sps` still carries its 2-byte NAL header.
fn parse_hevc_sps(sps: &[u8]) -> Result<HevcSps, NalError> {
    if sps.len() < 3 {
        return Err(NalError::Malformed("HEVC SPS too short"));
    }
    // The SPS payload is EBSP: 0x03 after two zero bytes is an emulation
    // prevention byte, not syntax. Parsing it as syntax shifts every field.
    let rbsp = unescape_ebsp(&sps[2..]);
    let mut bits = BitReader::new(&rbsp);
    bits.skip(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = bits.u(3)? as usize;
    bits.skip(1)?; // sps_temporal_id_nesting_flag

    // profile_tier_level(1, max_sub_layers_minus1)
    let profile_space = bits.u(2)? as u8;
    let tier = bits.u(1)? as u8;
    let profile_idc = bits.u(5)? as u8;
    let compatibility = bits.u(32)? as u32;
    let constraints = bits.u(48)?;
    let level = bits.u(8)? as u8;

    let mut sub_profile = [false; 8];
    let mut sub_level = [false; 8];
    for slot in 0..max_sub_layers_minus1.min(8) {
        sub_profile[slot] = bits.u(1)? == 1;
        sub_level[slot] = bits.u(1)? == 1;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            bits.skip(2)?; // reserved alignment
        }
    }
    for slot in 0..max_sub_layers_minus1.min(8) {
        if sub_profile[slot] {
            bits.skip(2 + 1 + 5 + 32 + 48)?;
        }
        if sub_level[slot] {
            bits.skip(8)?;
        }
    }

    let _sps_id = bits.ue()?;
    let chroma_format_idc = bits.ue()?;
    if chroma_format_idc == 3 {
        bits.skip(1)?; // separate_colour_plane_flag
    }
    let _width = bits.ue()?;
    let _height = bits.ue()?;
    if bits.u(1)? == 1 {
        for _ in 0..4 {
            bits.ue()?; // conformance window offsets
        }
    }
    let bit_depth_luma_minus8 = bits.ue()?;
    let bit_depth_chroma_minus8 = bits.ue()?;

    Ok(HevcSps {
        profile_space,
        tier,
        profile_idc,
        compatibility,
        constraints,
        level,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
    })
}

/// Remove emulation prevention bytes (`00 00 03` -> `00 00`).
fn unescape_ebsp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0usize;
    for &byte in data {
        if zeros >= 2 && byte == 0x03 {
            zeros = 0;
            continue;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// MSB-first bit reader with Exp-Golomb, enough for the SPS prefix.
struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn u(&mut self, count: usize) -> Result<u64, NalError> {
        debug_assert!(count <= 64);
        let mut value = 0u64;
        for _ in 0..count {
            let byte = *self
                .data
                .get(self.bit / 8)
                .ok_or(NalError::Malformed("SPS ended mid-field"))?;
            let bit = (byte >> (7 - (self.bit % 8))) & 1;
            value = (value << 1) | u64::from(bit);
            self.bit += 1;
        }
        Ok(value)
    }

    fn skip(&mut self, count: usize) -> Result<(), NalError> {
        let end = self.bit + count;
        if end > self.data.len() * 8 {
            return Err(NalError::Malformed("SPS ended mid-field"));
        }
        self.bit = end;
        Ok(())
    }

    fn ue(&mut self) -> Result<u32, NalError> {
        let mut zeros = 0usize;
        while self.u(1)? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(NalError::Malformed("Exp-Golomb run too long"));
            }
        }
        let rest = if zeros == 0 { 0 } else { self.u(zeros)? };
        Ok(((1u64 << zeros) - 1 + rest) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H264_AU: &[u8] = include_bytes!("../tests/fixtures/h264.annexb");
    const HEVC_AU: &[u8] = include_bytes!("../tests/fixtures/hevc.annexb");

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0xaa, 0x00, 0x00, 0x01, 0x68, 0xbb, 0x00, 0x00, 0x00,
            0x01, 0x65, 0xcc,
        ];
        let nals = split_annexb(&data);
        assert_eq!(
            nals,
            vec![&[0x67, 0xaa][..], &[0x68, 0xbb][..], &[0x65, 0xcc][..]]
        );
    }

    #[test]
    fn ignores_bytes_before_the_first_start_code() {
        let data = [0xde, 0xad, 0x00, 0x00, 0x01, 0x65, 0x01];
        assert_eq!(split_annexb(&data), vec![&[0x65, 0x01][..]]);
    }

    #[test]
    fn empty_input_has_no_nals() {
        assert!(split_annexb(&[]).is_empty());
    }

    #[test]
    fn h264_config_embeds_the_streams_parameter_sets() {
        let parsed = parse_access_unit(H264_AU, AnnexB::H264).expect("parse");
        let config = parsed.config.expect("keyframe carries SPS/PPS");
        let sps = split_annexb(H264_AU)
            .into_iter()
            .find(|n| n[0] & 0x1f == H264_NAL_SPS)
            .expect("SPS in fixture")
            .to_vec();
        let pps = split_annexb(H264_AU)
            .into_iter()
            .find(|n| n[0] & 0x1f == H264_NAL_PPS)
            .expect("PPS in fixture")
            .to_vec();

        assert_eq!(config[0], 1, "configurationVersion");
        assert_eq!(
            &config[1..4],
            &sps[1..4],
            "profile/compat/level from the SPS"
        );
        assert_eq!(config[4], 0xff, "reserved + lengthSizeMinusOne = 3");
        assert_eq!(config[5], 0xe1, "one SPS");
        let sps_len = u16::from_be_bytes([config[6], config[7]]) as usize;
        assert_eq!(sps_len, sps.len());
        assert_eq!(&config[8..8 + sps_len], &sps[..]);
        let pps_count = config[8 + sps_len];
        assert_eq!(pps_count, 1);
        let pps_len = u16::from_be_bytes([config[9 + sps_len], config[10 + sps_len]]) as usize;
        assert_eq!(pps_len, pps.len());
        assert_eq!(&config[11 + sps_len..11 + sps_len + pps_len], &pps[..]);
        assert_eq!(config.len(), 11 + sps.len() + pps.len());
    }

    #[test]
    fn hevc_config_fields_come_from_the_sps() {
        let parsed = parse_access_unit(HEVC_AU, AnnexB::Hevc).expect("parse");
        let config = parsed.config.expect("keyframe carries VPS/SPS/PPS");
        // Values read off the fixture's SPS: Main profile, tier 0, level 60
        // (level_idc 0x3c), 4:2:0, 8-bit. An emulation-prevention slip changes
        // all of these, which is what this pins.
        assert_eq!(config[0], 1, "configurationVersion");
        assert_eq!(config[1] & 0x1f, 1, "general_profile_idc = Main");
        assert_eq!(config[1] >> 5 & 1, 0, "general_tier_flag");
        assert_eq!(config[12], 60, "general_level_idc");
        assert_eq!(config[16], 0xfd, "chroma_format_idc = 1");
        assert_eq!(config[17], 0xf8, "bit_depth_luma_minus8 = 0");
        assert_eq!(config[18], 0xf8, "bit_depth_chroma_minus8 = 0");
        assert_eq!(config[22], 3, "three parameter-set arrays");
        for (offset, kind) in [(23usize, HEVC_NAL_VPS), (23 + 5 + 24, HEVC_NAL_SPS)] {
            assert_eq!(config[offset] & 0x3f, kind, "array type");
            assert_eq!(
                u16::from_be_bytes([config[offset + 1], config[offset + 2]]),
                1
            );
        }
    }

    #[test]
    fn h264_sample_is_length_prefixed_without_parameter_sets() {
        let parsed = parse_access_unit(H264_AU, AnnexB::H264).expect("parse");
        let sample = parsed.sample;
        let mut offset = 0usize;
        let mut count = 0usize;
        while offset < sample.len() {
            let len = u32::from_be_bytes(sample[offset..offset + 4].try_into().unwrap()) as usize;
            let nal = &sample[offset + 4..offset + 4 + len];
            let kind = nal[0] & 0x1f;
            assert!(
                kind != H264_NAL_SPS && kind != H264_NAL_PPS && kind != H264_NAL_AUD,
                "parameter set or AUD leaked into the sample"
            );
            offset += 4 + len;
            count += 1;
        }
        assert_eq!(
            offset,
            sample.len(),
            "framing must consume the whole sample"
        );
        assert!(count > 0);
    }

    #[test]
    fn hevc_sample_is_length_prefixed_without_parameter_sets() {
        let parsed = parse_access_unit(HEVC_AU, AnnexB::Hevc).expect("parse");
        let sample = parsed.sample;
        let mut offset = 0usize;
        while offset < sample.len() {
            let len = u32::from_be_bytes(sample[offset..offset + 4].try_into().unwrap()) as usize;
            let nal = &sample[offset + 4..offset + 4 + len];
            let kind = (nal[0] >> 1) & 0x3f;
            assert!(
                kind != HEVC_NAL_VPS
                    && kind != HEVC_NAL_SPS
                    && kind != HEVC_NAL_PPS
                    && kind != HEVC_NAL_AUD,
                "parameter set or AUD leaked into the sample"
            );
            offset += 4 + len;
        }
        assert_eq!(offset, sample.len());
    }

    #[test]
    fn non_keyframe_has_no_config_but_still_converts() {
        // Start codes + a single IDR slice, no parameter sets.
        let au = [0x00, 0x00, 0x01, 0x65, 0x88, 0x84];
        let parsed = parse_access_unit(&au, AnnexB::H264).unwrap();
        assert!(parsed.config.is_none());
        assert_eq!(&parsed.sample[..4], &3u32.to_be_bytes());
        assert_eq!(&parsed.sample[4..], &[0x65, 0x88, 0x84]);
    }

    #[test]
    fn missing_pps_is_an_error_not_a_silent_partial_config() {
        let au = [0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0d];
        assert_eq!(
            parse_access_unit(&au, AnnexB::H264).unwrap_err(),
            NalError::IncompleteParameterSets(AnnexB::H264)
        );
    }

    #[test]
    fn empty_access_unit_is_rejected() {
        assert_eq!(
            parse_access_unit(&[], AnnexB::H264).unwrap_err(),
            NalError::NoNalUnits
        );
    }
}
