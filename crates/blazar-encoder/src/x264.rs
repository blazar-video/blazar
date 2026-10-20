//! Software H.264 via libx264.
//!
//! GPL: this module is only compiled with the `enc-x264` feature, and enabling
//! that feature makes the resulting binary GPL. It exists so a deployment can
//! encode on the CPU while the GPU decodes the next batch.
//!
//! The raw bindings are used rather than the `x264` wrapper crate because the
//! wrapper does not expose CRF, B-frames or colour metadata, all of which the
//! calibration model needs.

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

use blazar_core::{EncodedUnit, Pts};
use blazar_preprocess::PreprocessedFrame;
use x264_sys::*;

use crate::config::{EncoderConfig, RateControl};
use crate::nal::{parse_access_unit, AnnexB};
use crate::{EncodeError, VideoEncoder};

/// libx264 encoder context.
pub struct X264Encoder {
    enc: *mut x264_t,
    pic: x264_picture_t,
    width: u32,
    height: u32,
    frame_duration: u64,
    codec_config: Option<Vec<u8>>,
    force_keyframe: bool,
    finished: bool,
    /// Units emitted so far. Decode timestamps are derived from this rather
    /// than x264's `i_dts`, which is negative for the first frames when
    /// B-frames are on (the reorder delay has no negative timestamps on the
    /// wire).
    emitted: u64,
}

// SAFETY: the context is owned by this struct and only touched through `&mut`.
unsafe impl Send for X264Encoder {}

impl X264Encoder {
    /// Open an encoder for the given config. Fails if libx264 rejects the params.
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
                "x264 needs even, non-zero dimensions".into(),
            ));
        }

        let mut par: x264_param_t = unsafe { std::mem::zeroed() };
        let preset = CString::new("medium").expect("static");
        let tune = CString::new("zerolatency").expect("static");
        // SAFETY: `par` is fully initialised by the call.
        let rc = unsafe { x264_param_default_preset(&mut par, preset.as_ptr(), tune.as_ptr()) };
        if rc != 0 {
            return Err(EncodeError::Backend(format!(
                "x264_param_default_preset failed ({rc})"
            )));
        }

        par.i_csp = X264_CSP_I420 as c_int;
        par.i_width = width as c_int;
        par.i_height = height as c_int;
        par.i_fps_num = 24;
        par.i_fps_den = 1;
        par.i_timebase_num = 1;
        par.i_timebase_den = 90_000;
        par.b_vfr_input = 1;
        par.i_keyint_max = cfg.keyint.max(1) as c_int;
        par.i_keyint_min = cfg.keyint.max(1) as c_int;
        par.i_bframe = cfg.bframes as c_int;
        par.b_annexb = 1;
        par.b_repeat_headers = 1;
        par.i_log_level = X264_LOG_ERROR as c_int;
        // The preprocess stage emits full-range BT.709 and the stream should say
        // so, otherwise players assume BT.601 for SD sizes.
        par.vui.b_fullrange = 1;
        par.vui.i_colorprim = 1;
        par.vui.i_transfer = 1;
        par.vui.i_colmatrix = 1;

        match &cfg.rate_control {
            RateControl::Crf(crf) => {
                par.rc.i_rc_method = X264_RC_CRF as c_int;
                par.rc.f_rf_constant = *crf as f32;
            }
            // A single bitrate target maps to average bitrate; VBV differs only
            // in how strictly the buffer is enforced.
            RateControl::Vbr(bps) | RateControl::Cbr(bps) => {
                par.rc.i_rc_method = X264_RC_ABR as c_int;
                par.rc.i_bitrate = (*bps / 1000) as c_int;
            }
        }

        let profile = CString::new("high").expect("static");
        // SAFETY: `par` is a valid initialised struct.
        unsafe { x264_param_apply_profile(&mut par, profile.as_ptr()) };

        // SAFETY: `par` is valid; the returned handle is checked for null.
        let enc = unsafe { x264_encoder_open(&mut par) };
        if enc.is_null() {
            return Err(EncodeError::Backend("x264_encoder_open failed".into()));
        }

        let mut pic: x264_picture_t = unsafe { std::mem::zeroed() };
        // SAFETY: the context is valid and the dimensions match the encoder.
        let rc = unsafe { x264_picture_alloc(
            &mut pic,
            X264_CSP_I420 as c_int,
            width as c_int,
            height as c_int,
        ) };
        if rc != 0 {
            return Err(EncodeError::Backend(format!(
                "x264_picture_alloc failed ({rc})"
            )));
        }
        let _ = fps;

        Ok(Self {
            enc,
            pic,
            width,
            height,
            frame_duration,
            codec_config: None,
            force_keyframe: false,
            finished: false,
            emitted: 0,
        })
    }

    /// Copy one frame into the x264 picture and encode it.
    fn encode_picture(
        &mut self,
        frame: Option<&PreprocessedFrame>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedUnit>, EncodeError> {
        if let Some(frame) = frame {
            if frame.width & !1 != self.width || frame.height & !1 != self.height {
                return Err(EncodeError::Backend(format!(
                    "x264 frame is {}x{}, encoder is {}x{}",
                    frame.width, frame.height, self.width, self.height
                )));
            }
            let w = self.width as usize;
            let h = self.height as usize;
            let ch = h.div_ceil(2);
            // SAFETY: planes were allocated for exactly this geometry.
            unsafe {
                let y_dst = std::slice::from_raw_parts_mut(
                    self.pic.img.plane[0],
                    self.pic.img.i_stride[0] as usize * h,
                );
                for row in 0..h {
                    let dst = &mut y_dst[row * self.pic.img.i_stride[0] as usize
                        ..row * self.pic.img.i_stride[0] as usize + w];
                    dst.copy_from_slice(&frame.y[row * w..row * w + w]);
                }
                for (plane_index, src) in [(1usize, &frame.u), (2, &frame.v)] {
                    let stride = self.pic.img.i_stride[plane_index] as usize;
                    let dst = std::slice::from_raw_parts_mut(
                        self.pic.img.plane[plane_index],
                        stride * ch,
                    );
                    let row_bytes = src.len() / ch;
                    for row in 0..ch {
                        dst[row * stride..row * stride + row_bytes]
                            .copy_from_slice(&src[row * row_bytes..(row + 1) * row_bytes]);
                    }
                }
            }
            self.pic.i_pts = pts;
            let forced = if force_keyframe || self.force_keyframe {
                X264_TYPE_KEYFRAME as c_int
            } else {
                X264_TYPE_AUTO as c_int
            };
            self.pic.i_type = forced;
        }

        let mut nals: *mut x264_nal_t = ptr::null_mut();
        let mut nal_count: c_int = 0;
        let mut pic_out: x264_picture_t = unsafe { std::mem::zeroed() };
        let frame_ptr = if frame.is_some() {
            &mut self.pic as *mut x264_picture_t
        } else {
            ptr::null_mut()
        };
        // SAFETY: encoder, picture and output are valid for this call.
        let size = unsafe {
            x264_encoder_encode(self.enc, &mut nals, &mut nal_count, frame_ptr, &mut pic_out)
        };
        if size < 0 {
            return Err(EncodeError::Backend(format!("x264_encoder_encode ({size})")));
        }
        self.force_keyframe = false;
        if nal_count == 0 || nals.is_null() {
            return Ok(Vec::new());
        }

        // x264 emits Annex B (b_annexb = 1), so the payloads are start-code
        // prefixed and can be concatenated before the shared splitter runs.
        let mut au = Vec::new();
        for index in 0..nal_count as usize {
            // SAFETY: nals points at nal_count valid entries.
            let nal = unsafe { &*nals.add(index) };
            if nal.p_payload.is_null() || nal.i_payload <= 0 {
                continue;
            }
            // SAFETY: the payload has i_payload bytes.
            let payload = unsafe { std::slice::from_raw_parts(nal.p_payload, nal.i_payload as usize) };
            au.extend_from_slice(payload);
        }
        if au.is_empty() {
            return Ok(Vec::new());
        }
        let parsed = parse_access_unit(&au, AnnexB::H264)
            .map_err(|e| EncodeError::Backend(format!("x264 access unit: {e}")))?;
        if parsed.config.is_some() {
            self.codec_config = parsed.config.clone();
        }
        let keyframe = pic_out.i_type == X264_TYPE_IDR as c_int || pic_out.i_type == X264_TYPE_I as c_int;
        let dts = self.emitted * self.frame_duration;
        self.emitted += 1;
        Ok(vec![EncodedUnit::video(
            parsed.sample,
            Pts(pts.max(0) as u64),
            Pts(dts),
            keyframe,
            self.frame_duration,
        )])
    }
}

impl VideoEncoder for X264Encoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        let force = self.force_keyframe;
        self.encode_picture(Some(frame), frame.pts.0 as i64, force)
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        // x264 has no flush-without-EOS; B-frames are only held with bframes > 0,
        // and a segment boundary can wait for the next frame instead.
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.finished = true;
        let mut units = Vec::new();
        loop {
            // SAFETY: the context is valid; zero means nothing is buffered.
            let delayed = unsafe { x264_encoder_delayed_frames(self.enc) };
            if delayed <= 0 {
                break;
            }
            let mut nals: *mut x264_nal_t = ptr::null_mut();
            let mut nal_count: c_int = 0;
            let mut pic_out: x264_picture_t = unsafe { std::mem::zeroed() };
            let size = unsafe {
                x264_encoder_encode(
                    self.enc,
                    &mut nals,
                    &mut nal_count,
                    ptr::null_mut(),
                    &mut pic_out,
                )
            };
            if size < 0 || nal_count == 0 || nals.is_null() {
                break;
            }
            let mut au = Vec::new();
            for index in 0..nal_count as usize {
                let nal = unsafe { &*nals.add(index) };
                if nal.p_payload.is_null() || nal.i_payload <= 0 {
                    continue;
                }
                let payload =
                    unsafe { std::slice::from_raw_parts(nal.p_payload, nal.i_payload as usize) };
                au.extend_from_slice(payload);
            }
            let parsed = parse_access_unit(&au, AnnexB::H264)
                .map_err(|e| EncodeError::Backend(format!("x264 flush: {e}")))?;
            if parsed.config.is_some() {
                self.codec_config = parsed.config.clone();
            }
            let keyframe = pic_out.i_type == X264_TYPE_IDR as c_int || pic_out.i_type == X264_TYPE_I as c_int;
            let dts = self.emitted * self.frame_duration;
            self.emitted += 1;
            units.push(EncodedUnit::video(
                parsed.sample,
                Pts(pic_out.i_pts.max(0) as u64),
                Pts(dts),
                keyframe,
                self.frame_duration,
            ));
        }
        Ok(units)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.codec_config.clone()
    }
}

impl Drop for X264Encoder {
    fn drop(&mut self) {
        // SAFETY: both handles were created together and are released once.
        unsafe {
            if !self.pic.img.plane[0].is_null() {
                x264_picture_clean(&mut self.pic);
            }
            if !self.enc.is_null() {
                x264_encoder_close(self.enc);
                self.enc = ptr::null_mut();
            }
        }
    }
}

