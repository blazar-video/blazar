//! Software AV1 via SVT-AV1.
//!
//! Consumes the NV12 planes `GpuPreprocess` produces, so the pixel work stays
//! on the GPU and only NV12 crosses the bus. Compiled with `enc-svtav1`.

use std::ptr;

use blazar_core::{EncodedUnit, Pts};
use blazar_preprocess::PreprocessedFrame;
use blazar_svtav1_sys as sys;

use crate::av1::av1c_from_keyframe;
use crate::config::{EncoderConfig, RateControl};
use crate::{EncodeError, VideoEncoder};

/// SVT-AV1's `rate_control_mode` values (macros in the C header, not bindings).
const RC_CQP_CRF: u8 = 0;
const RC_VBR: u8 = 1;

pub struct SvtAv1Encoder {
    handle: *mut sys::EbComponentType,
    frame_duration: u64,
    codec_config: Option<Vec<u8>>,
    force_keyframe: bool,
    finished: bool,
    emitted: u64,
    width: u32,
    height: u32,
}

// SAFETY: the handle is owned by this struct and only used through `&mut`.
unsafe impl Send for SvtAv1Encoder {}

fn status(code: sys::EbErrorType::Type, what: &str) -> Result<(), EncodeError> {
    if code == sys::EbErrorType::EB_ErrorNone {
        Ok(())
    } else {
        Err(EncodeError::Backend(format!("{what} failed ({code})")))
    }
}

impl SvtAv1Encoder {
    pub fn new(
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        fps: f64,
        frame_duration: u64,
    ) -> Result<Self, EncodeError> {
        let width = width & !1;
        let height = height & !1;
        if width == 0 || height == 0 {
            return Err(EncodeError::Backend(
                "svt-av1 needs even, non-zero dimensions".into(),
            ));
        }

        let mut config: sys::EbSvtAv1EncConfiguration = unsafe { std::mem::zeroed() };
        let mut handle: *mut sys::EbComponentType = ptr::null_mut();
        // SAFETY: fresh handle + zeroed config, as the API expects.
        status(
            unsafe { sys::svt_av1_enc_init_handle(&mut handle, &mut config) },
            "svt_av1_enc_init_handle",
        )?;

        config.source_width = width;
        config.source_height = height;
        config.frame_rate_numerator = (fps * 1000.0).round() as u32;
        config.frame_rate_denominator = 1000;
        config.encoder_color_format = sys::EbColorFormat::EB_YUV420;
        config.intra_period_length = cfg.keyint.max(1) as i32;
        // Always random-access: in low-delay mode `svt_av1_enc_get_packet`
        // blocks until a packet is ready, which turns a request/response
        // encoder into a hang on the first frame.
        let _ = cfg.bframes;
        config.pred_structure = sys::PredStructure::RANDOM_ACCESS;
        config.enc_mode = preset_from(&cfg.codec_params).unwrap_or(8);
        match &cfg.rate_control {
            RateControl::Crf(crf) => {
                config.rate_control_mode = RC_CQP_CRF;
                config.qp = *crf;
            }
            RateControl::Vbr(bps) | RateControl::Cbr(bps) => {
                config.rate_control_mode = RC_VBR;
                config.target_bit_rate = *bps / 1000;
            }
        }

        // SAFETY: handle is valid and the config is fully populated.
        status(
            unsafe { sys::svt_av1_enc_set_parameter(handle, &mut config) },
            "svt_av1_enc_set_parameter",
        )?;
        status(unsafe { sys::svt_av1_enc_init(handle) }, "svt_av1_enc_init")?;

        Ok(Self {
            handle,
            frame_duration,
            codec_config: None,
            force_keyframe: false,
            finished: false,
            emitted: 0,
            width,
            height,
        })
    }

    /// Send one picture (or EOS) and collect every packet the encoder returns.
    fn drain(&mut self, frame: Option<&PreprocessedFrame>, eos: bool) -> Result<Vec<EncodedUnit>, EncodeError> {
        let mut io: sys::EbSvtIOFormat = unsafe { std::mem::zeroed() };
        let mut header: sys::EbBufferHeaderType = unsafe { std::mem::zeroed() };
        header.size = std::mem::size_of::<sys::EbBufferHeaderType>() as u32;

        if let Some(frame) = frame {
            if frame.width & !1 != self.width || frame.height & !1 != self.height {
                return Err(EncodeError::Backend(format!(
                    "svt-av1 frame is {}x{}, encoder is {}x{}",
                    frame.width, frame.height, self.width, self.height
                )));
            }
            io.luma = frame.y.as_ptr().cast_mut();
            io.cb = frame.u.as_ptr().cast_mut();
            io.cr = frame.v.as_ptr().cast_mut();
            io.y_stride = self.width;
            io.cb_stride = self.width / 2;
            io.cr_stride = self.width / 2;
            header.p_buffer = (&mut io as *mut sys::EbSvtIOFormat).cast();
            header.n_filled_len = frame.y.len() as u32 + frame.u.len() as u32 + frame.v.len() as u32;
            header.pts = frame.pts.0 as i64;
            if self.force_keyframe {
                header.pic_type = sys::EbAv1PictureType::EB_AV1_KEY_PICTURE;
                self.force_keyframe = false;
            } else {
                header.pic_type = sys::EbAv1PictureType::EB_AV1_INVALID_PICTURE;
            }
        }
        if eos {
            header.flags = sys::EB_BUFFERFLAG_EOS;
        }

        // SAFETY: the header describes a live picture (or EOS) for this handle.
        status(
            unsafe { sys::svt_av1_enc_send_picture(self.handle, &mut header) },
            "svt_av1_enc_send_picture",
        )?;

        let mut units = Vec::new();
        loop {
            let mut packet: *mut sys::EbBufferHeaderType = ptr::null_mut();
            // SAFETY: get_packet returns a library-owned buffer or null.
            let rc = unsafe { sys::svt_av1_enc_get_packet(self.handle, &mut packet, u8::from(eos)) };
            if rc != sys::EbErrorType::EB_ErrorNone {
                break;
            }
            if packet.is_null() {
                break;
            }
            // SAFETY: `packet` is non-null and owned by the library until
            // released. An EOS acknowledgement can carry a null payload.
            let (bytes, pic_type, pts) = unsafe {
                let packet_ref = &*packet;
                let len = packet_ref.n_filled_len as usize;
                if packet_ref.p_buffer.is_null() || len == 0 {
                    (Vec::new(), packet_ref.pic_type, packet_ref.pts)
                } else {
                    let data = std::slice::from_raw_parts(packet_ref.p_buffer, len).to_vec();
                    (data, packet_ref.pic_type, packet_ref.pts)
                }
            };
            unsafe { sys::svt_av1_enc_release_out_buffer(&mut packet) };

            if bytes.is_empty() {
                continue;
            }
            let keyframe = pic_type == sys::EbAv1PictureType::EB_AV1_KEY_PICTURE;
            if keyframe && self.codec_config.is_none() {
                if let Ok(config) = av1c_from_keyframe(&bytes) {
                    self.codec_config = Some(config);
                }
            }
            let dts = self.emitted * self.frame_duration;
            self.emitted += 1;
            units.push(EncodedUnit::video(
                bytes,
                Pts(pts.max(0) as u64),
                Pts(dts),
                keyframe,
                self.frame_duration,
            ));
        }
        Ok(units)
    }
}

/// `preset=N` from the calibration's codec params, clamped to SVT's range.
fn preset_from(params: &str) -> Option<i8> {
    let value = params
        .split(['\n', ';'])
        .find_map(|line| line.trim().strip_prefix("preset="))?;
    value.trim().parse::<i8>().ok().map(|p| p.clamp(-2, 13))
}

impl VideoEncoder for SvtAv1Encoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.drain(Some(frame), false)
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.finished = true;
        self.drain(None, true)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.codec_config.clone()
    }
}

impl Drop for SvtAv1Encoder {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was created by init_handle and is released once.
            unsafe {
                let _ = sys::svt_av1_enc_deinit(self.handle);
                let _ = sys::svt_av1_enc_deinit_handle(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}
