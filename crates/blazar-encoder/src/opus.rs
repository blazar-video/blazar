// SPDX-License-Identifier: Apache-2.0
//! Opus audio encoding.
//!
//! Opus is the only audio codec here because it is BSD-3 and can be shipped:
//! the usable AAC encoders are either non-free (`libfdk-aac`) or drag in
//! libavcodec.
//!
//! Two things this module exists to get right:
//!
//! * **Framing.** Opus codes 2.5–120 ms per packet, and the packet size must be
//!   one the encoder was configured for. Callers hand over arbitrary sample
//!   counts, so samples are buffered and emitted in whole 20 ms frames, with the
//!   remainder flushed at the end.
//! * **Pre-skip.** The encoder delays the signal by a fixed number of samples
//!   (`OPUS_GET_LOOKAHEAD`). That delay has to reach the container, or the
//!   decoder starts playing audio earlier than the timestamps claim and the
//!   result drifts against the picture.

use opus::{Application, Channels, Encoder as RawEncoder};

use crate::EncodeError;

/// One Opus packet, in presentation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusPacket {
    /// The compressed bytes.
    pub data: Vec<u8>,
    /// Presentation timestamp in 48 kHz samples from the start of the stream,
    /// already shifted by the pre-skip so sample zero plays where it should.
    pub pts: u64,
    /// Always `frame_size` samples: every Opus packet here is the same length.
    pub duration: u64,
}

/// Buffers interleaved float samples into fixed-size Opus packets.
#[derive(Debug)]
pub struct OpusAudioEncoder {
    encoder: RawEncoder,
    channels: usize,
    /// Samples per channel in one packet (960 for 20 ms at 48 kHz).
    frame_size: usize,
    /// Interleaved samples that do not yet make a whole frame.
    pending: Vec<f32>,
    /// Packets emitted so far, for timestamps.
    packets_emitted: u64,
    /// Encoder delay in samples, applied in reverse to the first timestamp.
    pre_skip: u64,
}

impl OpusAudioEncoder {
    /// 20 ms at 48 kHz. The safe default: valid for every Opus configuration.
    pub const FRAME_SAMPLES: usize = 960;

    /// Create an encoder for 48 kHz, 1 or 2 channels.
    ///
    /// 48 kHz because Opus supports 8/12/16/24/48 and fMP4 wants 48; anything
    /// else would need resampling that the caller is better placed to do.
    pub fn new(channels: u16, bitrate_kbps: u32) -> Result<Self, EncodeError> {
        let channel_config = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            other => {
                return Err(EncodeError::Backend(format!(
                    "Opus supports 1 or 2 channels here (got {other}); \
                     multistream is not implemented"
                )))
            }
        };
        let mut encoder = RawEncoder::new(48_000, channel_config, Application::Audio)
            .map_err(|e| EncodeError::Backend(format!("opus encoder: {e}")))?;
        // `Bitrate` is an alias in this version, but the typed constructor keeps
        // the unit explicit instead of a bare integer.
        encoder
            .set_bitrate(opus::Bitrate::Bits(bitrate_kbps.saturating_mul(1000) as i32))
            .map_err(|e| EncodeError::Backend(format!("opus bitrate: {e}")))?;
        let lookahead = encoder
            .get_lookahead()
            .map_err(|e| EncodeError::Backend(format!("opus lookahead: {e}")))?;
        let pre_skip = u64::try_from(lookahead).unwrap_or(0);
        Ok(Self {
            encoder,
            channels: channels as usize,
            frame_size: Self::FRAME_SAMPLES,
            pending: Vec::with_capacity(Self::FRAME_SAMPLES * channels as usize),
            packets_emitted: 0,
            pre_skip,
        })
    }

    /// The encoder's algorithmic delay, in samples. Written into `dOps` and
    /// subtracted from the first timestamp.
    #[must_use]
    pub fn pre_skip(&self) -> u64 {
        self.pre_skip
    }

    /// Samples per channel in one packet.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Queue interleaved float samples in `[-1, 1]`.
    ///
    /// Any length is accepted; whole 20 ms frames are encoded and the remainder
    /// is kept for the next call or the final [`Self::finish`].
    pub fn push(&mut self, interleaved: &[f32]) -> Result<Vec<OpusPacket>, EncodeError> {
        if interleaved.len() % self.channels != 0 {
            return Err(EncodeError::Backend(format!(
                "sample count {} is not a multiple of {} channels",
                interleaved.len(),
                self.channels
            )));
        }
        self.pending.extend_from_slice(interleaved);
        let frame_len = self.frame_size * self.channels;
        let mut packets = Vec::new();
        while self.pending.len() >= frame_len {
            let frame: Vec<f32> = self.pending.drain(..frame_len).collect();
            packets.push(self.encode_frame(&frame)?);
        }
        Ok(packets)
    }

    /// Encode the remainder, padding the last frame with silence.
    ///
    /// Padding rather than dropping is deliberate: a short tail is inaudible,
    /// a missing tail is a clipped word.
    pub fn finish(&mut self) -> Result<Vec<OpusPacket>, EncodeError> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let frame_len = self.frame_size * self.channels;
        let mut frame = std::mem::take(&mut self.pending);
        frame.resize(frame_len, 0.0);
        Ok(vec![self.encode_frame(&frame)?])
    }

    fn encode_frame(&mut self, frame: &[f32]) -> Result<OpusPacket, EncodeError> {
        // 4000 bytes is the documented maximum for one packet at any bitrate.
        let mut out = vec![0u8; 4000];
        let written = self
            .encoder
            .encode_float(frame, &mut out)
            .map_err(|e| EncodeError::Backend(format!("opus encode: {e}")))?;
        out.truncate(written);
        let pts = self.packets_emitted * self.frame_size as u64;
        self.packets_emitted += 1;
        Ok(OpusPacket {
            data: out,
            // The decoder will skip `pre_skip` samples, so the first packet is
            // presented that much earlier than the samples it carries.
            pts: pts.wrapping_sub(self.pre_skip),
            duration: self.frame_size as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(samples: usize, channels: usize) -> Vec<f32> {
        // 440 Hz at 48 kHz, one second by default.
        (0..samples)
            .flat_map(|n| {
                let value = (n as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5;
                std::iter::repeat_n(value, channels)
            })
            .collect()
    }

    #[test]
    fn one_second_is_fifty_packets_of_twenty_milliseconds() {
        let mut enc = OpusAudioEncoder::new(2, 96).expect("encoder opens");
        let mut packets = enc.push(&sine(48_000, 2)).expect("push encodes");
        packets.extend(enc.finish().expect("finish"));
        assert_eq!(packets.len(), 50, "48 kHz / 960 samples");
        assert!(packets.iter().all(|p| p.duration == 960));
    }

    #[test]
    fn every_packet_decodes_with_energy() {
        let mut enc = OpusAudioEncoder::new(2, 96).expect("encoder opens");
        let packets = enc.push(&sine(960 * 10, 2)).expect("push encodes");
        let mut dec = opus::Decoder::new(48_000, Channels::Stereo).expect("decoder opens");
        for packet in &packets {
            let mut out = vec![0f32; 960 * 2 * 6];
            let decoded = dec
                .decode_float(&packet.data, &mut out, false)
                .expect("packet decodes");
            let energy: f32 = out[..decoded * 2].iter().map(|s| s * s).sum();
            assert!(energy > 0.0, "decoded packet is silent");
        }
    }

    #[test]
    fn the_first_packet_starts_before_zero_by_the_pre_skip() {
        let mut enc = OpusAudioEncoder::new(2, 96).expect("encoder opens");
        let pre_skip = enc.pre_skip();
        assert!(pre_skip > 0, "an Opus encoder always delays the signal");
        let packets = enc.push(&sine(960, 2)).expect("push encodes");
        assert_eq!(packets[0].pts, 0u64.wrapping_sub(pre_skip));
    }

    #[test]
    fn a_partial_frame_is_flushed_not_dropped() {
        let mut enc = OpusAudioEncoder::new(1, 64).expect("encoder opens");
        let first = enc.push(&sine(500, 1)).expect("push buffers");
        assert!(first.is_empty(), "500 samples is less than one frame");
        let tail = enc.finish().expect("finish flushes");
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].duration, 960);
    }

    #[test]
    fn an_unrepresentable_channel_count_is_rejected() {
        match OpusAudioEncoder::new(6, 96) {
            Err(error) => assert!(format!("{error}").contains("1 or 2 channels")),
            Ok(_) => panic!("6 channels is out of scope"),
        }
    }

    #[test]
    fn a_sample_count_that_cannot_be_interleaved_is_rejected() {
        let mut enc = OpusAudioEncoder::new(2, 96).expect("encoder opens");
        let error = enc.push(&[0.0; 7]).expect_err("7 is not a multiple of 2");
        assert!(format!("{error}").contains("not a multiple"));
    }

    #[test]
    fn silence_is_much_smaller_than_a_tone() {
        let mut loud = OpusAudioEncoder::new(2, 64).expect("encoder opens");
        let mut quiet = OpusAudioEncoder::new(2, 64).expect("encoder opens");
        let loud_bytes: usize = loud
            .push(&sine(960 * 20, 2))
            .unwrap()
            .iter()
            .map(|p| p.data.len())
            .sum();
        let quiet_bytes: usize = quiet
            .push(&vec![0.0f32; 960 * 20 * 2])
            .unwrap()
            .iter()
            .map(|p| p.data.len())
            .sum();
        assert!(
            quiet_bytes * 2 < loud_bytes,
            "silence {quiet_bytes} should be far below a tone {loud_bytes}"
        );
    }
}
