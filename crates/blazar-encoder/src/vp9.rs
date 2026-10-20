//! Software VP9 via libvpx.
//!
//! Like the other CPU backends it consumes the NV12 planes produced on the GPU,
//! so only NV12 crosses the bus. Compiled with `enc-vp9`.

use std::ptr;

use blazar_core::{EncodedUnit, Pts};
use blazar_preprocess::PreprocessedFrame;
use blazar_vpx_sys as sys;

use crate::config::{EncoderConfig, RateControl};
use crate::{EncodeError, VideoEncoder};

/// `vpcC` body for profile 0, 8-bit 4:2:0, full-range BT.709, no
/// codec-initialization data. VP9 carries nothing else the muxer needs.
const VPCC: [u8; 12] = [
    1, 0, 0, 0, // version + flags
    0,  // profile 0 (8-bit 4:2:0)
    10, // level 1.0
    // bit_depth(4) | chroma_subsampling(3) | full_range(1)
    (8 << 4) | (1 << 1) | 1,
    1, // colour_primaries BT.709
    1, // transfer_characteristics BT.709
    1, // matrix_coefficients BT.709
    0, 0, // codec initialization data size
];

pub struct Vp9Encoder {
    ctx: sys::vpx_codec_ctx_t,
    image: sys::vpx_image,
    frame_duration: u64,
    force_keyframe: bool,
    finished: bool,
    emitted: u64,
    width: u32,
    height: u32,
}

// SAFETY: the context is owned by this struct and only used through `&mut`.
unsafe impl Send for Vp9Encoder {}

fn status(code: sys::vpx_codec_err_t::Type, what: &str) -> Result<(), EncodeError> {
    if code == sys::vpx_codec_err_t::VPX_CODEC_OK {
        Ok(())
    } else {
        Err(EncodeError::Backend(format!("{what} failed ({code})")))
    }
}

impl Vp9Encoder {
    pub fn new(
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        _fps: f64,
        frame_duration: u64,
    ) -> Result<Self, EncodeError> {
        let width = width & !1;
        let height = height & !1;
        if width == 0 || height == 0 {
            return Err(EncodeError::Backend(
                "vp9 needs even, non-zero dimensions".into(),
            ));
        }

        let mut enc_cfg: sys::vpx_codec_enc_cfg_t = unsafe { std::mem::zeroed() };
        // SAFETY: the config is written by libvpx for the VP9 encoder.
        status(
            unsafe { sys::vpx_codec_enc_config_default(sys::vpx_codec_vp9_cx(), &mut enc_cfg, 0) },
            "vpx_codec_enc_config_default",
        )?;
        enc_cfg.g_w = width;
        enc_cfg.g_h = height;
        enc_cfg.g_timebase.num = 1;
        enc_cfg.g_timebase.den = 90_000;
        enc_cfg.g_threads = 1;
        enc_cfg.g_pass = sys::vpx_enc_pass::VPX_RC_ONE_PASS;
        match &cfg.rate_control {
            RateControl::Crf(crf) => {
                // Constant quality: pin the quantizer range to the CRF value.
                enc_cfg.rc_end_usage = sys::vpx_rc_mode::VPX_Q;
                enc_cfg.rc_min_quantizer = (*crf).min(63);
                enc_cfg.rc_max_quantizer = (*crf).min(63);
            }
            RateControl::Vbr(bps) | RateControl::Cbr(bps) => {
                enc_cfg.rc_end_usage = sys::vpx_rc_mode::VPX_VBR;
                enc_cfg.rc_target_bitrate = bps / 1000;
            }
        }
        enc_cfg.kf_max_dist = cfg.keyint.max(1);
        enc_cfg.kf_mode = sys::vpx_kf_mode::VPX_KF_AUTO;

        let mut ctx: sys::vpx_codec_ctx_t = unsafe { std::mem::zeroed() };
        // SAFETY: interface, config and context are all valid.
        status(
            unsafe {
                sys::vpx_codec_enc_init_ver(
                    &mut ctx,
                    sys::vpx_codec_vp9_cx(),
                    &enc_cfg,
                    0,
                    sys::VPX_ENCODER_ABI_VERSION as i32,
                )
            },
            "vpx_codec_enc_init_ver",
        )?;

        let mut image: sys::vpx_image = unsafe { std::mem::zeroed() };
        image.fmt = sys::vpx_img_fmt::VPX_IMG_FMT_I420;
        image.range = sys::vpx_color_range::VPX_CR_FULL_RANGE;
        image.w = width;
        image.h = height;
        image.d_w = width;
        image.d_h = height;
        image.bit_depth = 8;
        // libvpx validates these against the format; `vpx_img_alloc` normally
        // sets them, and a zeroed image is rejected as an invalid parameter.
        image.x_chroma_shift = 1;
        image.y_chroma_shift = 1;
        // Strides are set per frame, when the plane pointers are known.

        Ok(Self {
            ctx,
            image,
            frame_duration,
            force_keyframe: false,
            finished: false,
            emitted: 0,
            width,
            height,
        })
    }

    /// Encode one frame (or flush when `frame` is None) and collect packets.
    fn encode_impl(&mut self, frame: Option<&PreprocessedFrame>) -> Result<Vec<EncodedUnit>, EncodeError> {
        let (img_ptr, pts, flags) = match frame {
            Some(frame) => {
                if frame.width & !1 != self.width || frame.height & !1 != self.height {
                    return Err(EncodeError::Backend(format!(
                        "vp9 frame is {}x{}, encoder is {}x{}",
                        frame.width, frame.height, self.width, self.height
                    )));
                }
                self.image.planes[0] = frame.y.as_ptr().cast_mut();
                self.image.planes[1] = frame.u.as_ptr().cast_mut();
                self.image.planes[2] = frame.v.as_ptr().cast_mut();
                self.image.stride[0] = self.width as i32;
                self.image.stride[1] = (self.width / 2) as i32;
                self.image.stride[2] = (self.width / 2) as i32;
                let flags: sys::vpx_enc_frame_flags_t = if self.force_keyframe {
                    sys::VPX_EFLAG_FORCE_KF as sys::vpx_enc_frame_flags_t
                } else {
                    0
                };
                self.force_keyframe = false;
                (&mut self.image as *mut sys::vpx_image, frame.pts.0 as i64, flags)
            }
            None => (ptr::null_mut(), 0, 0 as sys::vpx_enc_frame_flags_t),
        };

        // SAFETY: the image (or null for flush) is valid for this call.
        status(
            unsafe {
                sys::vpx_codec_encode(
                    &mut self.ctx,
                    img_ptr,
                    pts,
                    self.frame_duration,
                    flags,
                    sys::VPX_DL_REALTIME as u64,
                )
            },
            "vpx_codec_encode",
        )?;

        let mut units = Vec::new();
        let mut iter: sys::vpx_codec_iter_t = ptr::null();
        loop {
            // SAFETY: the iterator starts null and is advanced by the call.
            let packet = unsafe { sys::vpx_codec_get_cx_data(&mut self.ctx, &mut iter) };
            if packet.is_null() {
                break;
            }
            // SAFETY: a non-null packet points at a valid union for its kind.
            let (kind, data, packet_pts, keyframe) = unsafe {
                let p = &*packet;
                if p.kind != sys::vpx_codec_cx_pkt_kind::VPX_CODEC_CX_FRAME_PKT {
                    continue;
                }
                let frame = &p.data.frame;
                let bytes = if frame.buf.is_null() || frame.sz == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(frame.buf.cast::<u8>(), frame.sz as usize).to_vec()
                };
                (
                    p.kind,
                    bytes,
                    frame.pts,
                    frame.flags & sys::VPX_FRAME_IS_KEY != 0,
                )
            };
            let _ = kind;
            if data.is_empty() {
                continue;
            }
            let dts = self.emitted * self.frame_duration;
            self.emitted += 1;
            units.push(EncodedUnit::video(
                data,
                Pts(packet_pts.max(0) as u64),
                Pts(dts),
                keyframe,
                self.frame_duration,
            ));
        }
        Ok(units)
    }
}

impl VideoEncoder for Vp9Encoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.encode_impl(Some(frame))
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.finished {
            return Err(EncodeError::AfterFinish);
        }
        self.finished = true;
        let mut units = Vec::new();
        loop {
            let emitted = self.encode_impl(None)?;
            if emitted.is_empty() {
                break;
            }
            units.extend(emitted);
        }
        Ok(units)
    }

    fn request_keyframe(&mut self) {
        self.force_keyframe = true;
    }

    fn codec_config(&self) -> Option<Vec<u8>> {
        Some(VPCC.to_vec())
    }
}

impl Drop for Vp9Encoder {
    fn drop(&mut self) {
        // SAFETY: the context was initialised in `new` and is destroyed once.
        unsafe { sys::vpx_codec_destroy(&mut self.ctx) };
    }
}
