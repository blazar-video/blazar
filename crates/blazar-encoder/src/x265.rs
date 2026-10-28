// SPDX-License-Identifier: Apache-2.0
//! Software HEVC via libx265.
//!
//! GPL, same as `x264.rs`: only compiled with `enc-x265`, and enabling that
//! feature makes the resulting binary GPL.
//!
//! The bindings are generated from the system header rather than hand-written,
//! because `x265.h` resolves `x265_encoder_open` through a macro to a
//! build-specific symbol (`x265_encoder_open_216` on the 4.1 library). A
//! hand-written declaration would link against a symbol that does not exist.

use std::ptr;

use blazar_core::{EncodedUnit, Pts};
use blazar_preprocess::PreprocessedFrame;
use blazar_x265_sys as x265;
use x265::*;

use crate::config::{EncoderConfig, RateControl};
use crate::nal::{parse_access_unit, AnnexB};
use crate::{EncodeError, VideoEncoder};

/// libx265 encoder context.
pub struct X265Encoder {
    enc: *mut x265_encoder,
    param: *mut x265_param,
    pic: *mut x265_picture,
    /// Packed I420 scratch: Y, then U, then V, each plane contiguous.
    buf: Vec<u8>,
    width: u32,
    height: u32,
    frame_duration: u64,
    codec_config: Option<Vec<u8>>,
    force_keyframe: bool,
    finished: bool,
    /// Units emitted so far. Decode timestamps are synthesised from this: x265
    /// reports `dts` on its output pictures, but the wire only ever needs a
    /// monotonic decode order and ours is the emission order.
    emitted: u64,
}

// SAFETY: the context is owned by this struct and only touched through `&mut`.
unsafe impl Send for X265Encoder {}

impl X265Encoder {
    /// Open an encoder for the given config. Fails if libx265 rejects the params.
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
                "x265 needs even, non-zero dimensions".into(),
            ));
        }

        // SAFETY: alloc has no preconditions and returns an owned param block.
        let param = unsafe { x265_param_alloc() };
        if param.is_null() {
            return Err(EncodeError::Backend("x265_param_alloc failed".into()));
        }
        // Every early return past this point has to free `param`, so the body
        // runs in a closure and the cleanup happens once at the end.
        let result = Self::configure_and_open(param, cfg, width, height, fps, frame_duration);
        if result.is_err() {
            // SAFETY: `param` is owned and not referenced by a closed encoder.
            unsafe { x265_param_free(param) };
        }
        result
    }

    fn configure_and_open(
        param: *mut x265_param,
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        fps: f64,
        frame_duration: u64,
    ) -> Result<Self, EncodeError> {
        let preset = std::ffi::CString::new("medium").expect("static");
        // SAFETY: `param` is a valid owned block; the strings outlive the call.
        let rc = unsafe { x265_param_default_preset(param, preset.as_ptr(), ptr::null()) };
        if rc < 0 {
            return Err(EncodeError::Backend(format!(
                "x265_param_default_preset failed ({rc})"
            )));
        }

        // SAFETY: `param` is valid and fully initialised by the call above.
        let p = unsafe { &mut *param };
        p.internalCsp = X265_CSP_I420 as i32;
        p.internalBitDepth = 8;
        p.sourceWidth = width as i32;
        p.sourceHeight = height as i32;
        p.fpsNum = 24;
        p.fpsDenom = 1;
        let _ = fps;
        p.keyframeMax = cfg.keyint.max(1) as i32;
        p.keyframeMin = cfg.keyint.max(1) as i32;
        p.bframes = cfg.bframes as i32;
        p.bRepeatHeaders = 1;
        p.bAnnexB = 1;
        p.logLevel = X265_LOG_ERROR as i32;

        match &cfg.rate_control {
            RateControl::Crf(crf) => {
                p.rc.rateControlMode = x265::X265_RC_CRF;
                p.rc.rfConstant = *crf as f64;
            }
            RateControl::Vbr(bps) | RateControl::Cbr(bps) => {
                p.rc.rateControlMode = x265::X265_RC_ABR;
                p.rc.bitrate = (*bps / 1000) as i32;
            }
        }

        // The preprocess stage emits full-range BT.709; without these the
        // decoder assumes limited-range BT.601 for SD sizes and PSNR drops by
        // ~8 dB. Same fix as the H.264 path.
        p.vui.bEnableVideoSignalTypePresentFlag = 1;
        p.vui.bEnableVideoFullRangeFlag = 1;
        p.vui.bEnableColorDescriptionPresentFlag = 1;
        p.vui.colorPrimaries = 1;
        p.vui.transferCharacteristics = 1;
        p.vui.matrixCoeffs = 1;

        // SAFETY: `param` is valid; the returned handle is checked for null.
        let enc = unsafe { x265_encoder_open(param) };
        if enc.is_null() {
            return Err(EncodeError::Backend("x265_encoder_open failed".into()));
        }

        // SAFETY: alloc has no preconditions.
        let pic = unsafe { x265_picture_alloc() };
        if pic.is_null() {
            // SAFETY: the encoder is open and owned here.
            unsafe { x265_encoder_close(enc) };
            return Err(EncodeError::Backend("x265_picture_alloc failed".into()));
        }
        // SAFETY: both pointers are valid and the param block outlives the
        // picture; this fills in colour space and bit depth defaults.
        unsafe { x265_picture_init(param, pic) };

        let w = width as usize;
        let h = height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let buf = vec![0u8; w * h + 2 * cw * ch];

        // SAFETY: encoder and out-parameters are valid.
        let mut codec_config = None;
        let mut nals: *mut x265_nal = ptr::null_mut();
        let mut nal_count: u32 = 0;
        let rc = unsafe { x265_encoder_headers(enc, &mut nals, &mut nal_count) };
        if rc >= 0 && !nals.is_null() && nal_count > 0 {
            // SAFETY: x265 handed back `nal_count` valid NALs for this call.
            let au = unsafe { collect_nals(nals, nal_count) };
            if let Ok(parsed) = parse_access_unit(&au, AnnexB::Hevc) {
                codec_config = parsed.config;
            }
        }

        tracing::info!(
            target: "blazar",
            rate_control = ?cfg.rate_control,
            bframes = cfg.bframes,
            keyint = cfg.keyint,
            preset = "medium",
            "x265 session opened"
        );

        Ok(Self {
            enc,
            param,
            pic,
            buf,
            width,
            height,
            frame_duration,
            codec_config,
            force_keyframe: false,
            finished: false,
            emitted: 0,
        })
    }

    /// Copy one frame into the picture and encode it.
    fn encode_picture(
        &mut self,
        frame: Option<&PreprocessedFrame>,
        pts: i64,
        force_keyframe: bool,
    ) -> Result<Vec<EncodedUnit>, EncodeError> {
        if let Some(frame) = frame {
            if frame.width & !1 != self.width || frame.height & !1 != self.height {
                return Err(EncodeError::Backend(format!(
                    "x265 frame is {}x{}, encoder is {}x{}",
                    frame.width, frame.height, self.width, self.height
                )));
            }
            let w = self.width as usize;
            let h = self.height as usize;
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            let (y_end, u_end) = (w * h, w * h + cw * ch);
            // Planes are given to x265 as separate pointers into one packed
            // buffer, which is the I420 layout it expects.
            self.buf[..w * h].copy_from_slice(&frame.y[..w * h]);
            self.buf[w * h..u_end].copy_from_slice(&frame.u[..cw * ch]);
            self.buf[u_end..].copy_from_slice(&frame.v[..cw * ch]);

            // SAFETY: the picture was allocated by x265 for this geometry and
            // the buffer outlives the encode call.
            unsafe {
                let pic = &mut *self.pic;
                pic.planes[0] = self.buf.as_mut_ptr() as *mut std::ffi::c_void;
                pic.planes[1] = self.buf.as_mut_ptr().add(y_end) as *mut std::ffi::c_void;
                pic.planes[2] = self.buf.as_mut_ptr().add(u_end) as *mut std::ffi::c_void;
                pic.stride[0] = w as i32;
                pic.stride[1] = cw as i32;
                pic.stride[2] = cw as i32;
                pic.pts = pts;
                pic.sliceType = if force_keyframe || self.force_keyframe {
                    X265_TYPE_IDR as i32
                } else {
                    X265_TYPE_AUTO as i32
                };
            }
        }

        let mut nals: *mut x265_nal = ptr::null_mut();
        let mut nal_count: u32 = 0;
        // SAFETY: zeroed output picture is the documented contract for a call
        // that may produce nothing.
        let mut pic_out: x265_picture = unsafe { std::mem::zeroed() };
        let pic_in = if frame.is_some() {
            self.pic
        } else {
            ptr::null_mut()
        };
        // SAFETY: encoder, input picture and output are valid for this call.
        let size = unsafe {
            x265_encoder_encode(self.enc, &mut nals, &mut nal_count, pic_in, &mut pic_out)
        };
        if size < 0 {
            return Err(EncodeError::Backend(format!(
                "x265_encoder_encode ({size})"
            )));
        }
        self.force_keyframe = false;
        if size == 0 || nal_count == 0 || nals.is_null() {
            return Ok(Vec::new());
        }

        self.unit_from(nals, nal_count, &pic_out)
    }

    fn unit_from(
        &mut self,
        nals: *mut x265_nal,
        nal_count: u32,
        pic_out: &x265_picture,
    ) -> Result<Vec<EncodedUnit>, EncodeError> {
        // SAFETY: x265 handed back `nal_count` valid NALs for this call.
        let au = unsafe { collect_nals(nals, nal_count) };
        if au.is_empty() {
            return Ok(Vec::new());
        }
        let parsed = parse_access_unit(&au, AnnexB::Hevc)
            .map_err(|e| EncodeError::Backend(format!("x265 access unit: {e}")))?;
        if parsed.config.is_some() {
            self.codec_config = parsed.config.clone();
        }
        let slice_type = pic_out.sliceType;
        let keyframe = slice_type == X265_TYPE_IDR as i32 || slice_type == X265_TYPE_I as i32;
        let dts = self.emitted * self.frame_duration;
        self.emitted += 1;
        // The packet returned here belongs to whichever frame x265 finished,
        // which with B-frames is not the frame just submitted. Using the
        // submitted PTS corrupts the timeline, so take the output picture's.
        Ok(vec![EncodedUnit::video(
            parsed.sample,
            Pts(pic_out.pts.max(0) as u64),
            Pts(dts),
            keyframe,
            self.frame_duration,
        )])
    }
}

/// Concatenate the NAL payloads of one packet.
///
/// # Safety
/// `nals` must point at `count` valid `x265_nal` entries whose payloads live
/// for the duration of the call.
unsafe fn collect_nals(nals: *mut x265_nal, count: u32) -> Vec<u8> {
    let mut au = Vec::new();
    for index in 0..count as usize {
        // SAFETY: guaranteed by the caller.
        let nal = unsafe { &*nals.add(index) };
        if nal.payload.is_null() || nal.sizeBytes == 0 {
            continue;
        }
        // SAFETY: the payload has sizeBytes bytes.
        let payload = unsafe { std::slice::from_raw_parts(nal.payload, nal.sizeBytes as usize) };
        au.extend_from_slice(payload);
    }
    au
}

impl VideoEncoder for X265Encoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        let force = self.force_keyframe;
        self.encode_picture(Some(frame), frame.pts.0 as i64, force)
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        // No flush-without-EOS in the API; B-frames are held only when the
        // caller asked for them, and a segment boundary can wait for the next
        // frame instead.
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.finished = true;
        let mut units = Vec::new();
        loop {
            let mut nals: *mut x265_nal = ptr::null_mut();
            let mut nal_count: u32 = 0;
            // SAFETY: zeroed output picture is the documented contract.
            let mut pic_out: x265_picture = unsafe { std::mem::zeroed() };
            // SAFETY: a null input picture tells x265 to drain.
            let size = unsafe {
                x265_encoder_encode(
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
            units.extend(self.unit_from(nals, nal_count, &pic_out)?);
            // A drain call that returns data may have more behind it; the loop
            // ends when one returns nothing.
            if size == 0 {
                break;
            }
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

impl Drop for X265Encoder {
    fn drop(&mut self) {
        // SAFETY: every handle was created together and is released once.
        unsafe {
            if !self.pic.is_null() {
                x265_picture_free(self.pic);
                self.pic = ptr::null_mut();
            }
            if !self.enc.is_null() {
                x265_encoder_close(self.enc);
                self.enc = ptr::null_mut();
            }
            if !self.param.is_null() {
                x265_param_free(self.param);
                self.param = ptr::null_mut();
            }
        }
    }
}
