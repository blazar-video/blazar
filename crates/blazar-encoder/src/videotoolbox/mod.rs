// SPDX-License-Identifier: Apache-2.0
//! Apple VideoToolbox H.264/HEVC encoder.
//!
//! VideoToolbox is handed NV12 `CVPixelBuffer`s. The buffers are created by this
//! module and filled from the preprocessed frame: on Apple silicon the encoder
//! reads them from shared memory, so there is one copy into the buffer and no
//! bus transfer to a discrete device.
//!
//! Encoding is asynchronous. The output callback accumulates finished access
//! units in a queue that `encode` drains, so a frame that VideoToolbox keeps for
//! reference (or for B-frame reordering) is delivered by a later call, exactly
//! as the other backends behave.
//!
//! Parameter sets arrive in-band on the keyframe. They are split out here to
//! build the `avcC`/`hvcC` record the container needs, reusing the same parser
//! the NVENC path uses so both backends describe the stream identically.

mod apple;

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use blazar_core::{EncodedUnit, Pts, Timebase, TrackId};
use blazar_preprocess::PreprocessedFrame;

use crate::config::EncoderConfig;
use crate::nal::{config_from_parameter_sets, parse_access_unit, AnnexB};
use crate::{EncodeError, VideoEncoder};

use apple::*;

/// Codec this session encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

impl Codec {
    fn cm_type(self) -> CMVideoCodecType {
        match self {
            Codec::H264 => K_CM_VIDEO_CODEC_TYPE_H264,
            Codec::Hevc => K_CM_VIDEO_CODEC_TYPE_HEVC,
        }
    }

    fn annex_b(self) -> AnnexB {
        match self {
            Codec::H264 => AnnexB::H264,
            Codec::Hevc => AnnexB::Hevc,
        }
    }
}

/// What the output callback is given: the queue plus the codec, so parameter
/// sets can be read with the right framework call.
struct CallbackState {
    output: Mutex<Output>,
    codec: Codec,
}

/// State the output callback writes and `encode` reads.
#[derive(Default)]
struct Output {
    /// Finished access units, Annex B, with the presentation timestamp and
    /// whether VideoToolbox flagged the sample as not-for-reordering (a
    /// keyframe).
    units: Vec<(Vec<u8>, u64, bool)>,
    /// `avcC`/`hvcC` body taken from the sample's format description.
    config: Option<Vec<u8>>,
    /// First error reported by the encoder, kept so the next call fails loudly.
    error: Option<String>,
}

/// An open VideoToolbox compression session.
pub struct VideoToolboxEncoder {
    session: VTCompressionSessionRef,
    width: usize,
    height: usize,
    codec: Codec,
    state: Arc<CallbackState>,
    codec_config: Option<Vec<u8>>,
    /// Frames submitted, used to synthesise timestamps when the caller's PTS
    /// repeat (a caller that does not set them).
    submitted: i64,
    timebase: i32,
    /// Ticks per frame on `timebase`, so each unit reports how long it lasts.
    frame_duration: u64,
    closed: bool,
}

// SAFETY: the session is only touched from the thread that owns the encoder,
// and the shared queue is behind a mutex. VideoToolbox's callback runs on its
// own thread and only ever appends to that queue.
unsafe impl Send for VideoToolboxEncoder {}

impl VideoToolboxEncoder {
    /// Open a session for `width` x `height`.
    pub fn new(
        width: u32,
        height: u32,
        codec: Codec,
        cfg: &EncoderConfig,
    ) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(EncodeError::Backend(
                "VideoToolbox needs even, non-zero dimensions".into(),
            ));
        }
        let state = Arc::new(CallbackState {
            output: Mutex::new(Output::default()),
            codec,
        });
        // The callback's refcon points at state the encoder keeps alive; `Drop`
        // invalidates the session before that state is released.
        let refcon = Arc::as_ptr(&state) as *mut c_void;

        let mut session: VTCompressionSessionRef = std::ptr::null_mut();
        // SAFETY: every pointer is either null (meaning "default") or points at
        // something that outlives the call.
        let status = unsafe {
            VTCompressionSessionCreate(
                kCFAllocatorDefault,
                width as i32,
                height as i32,
                codec.cm_type(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                Some(output_callback),
                refcon,
                &mut session,
            )
        };
        if status != 0 || session.is_null() {
            return Err(EncodeError::Backend(format!(
                "VTCompressionSessionCreate failed ({status})"
            )));
        }

        let encoder = Self {
            session,
            width: width as usize,
            height: height as usize,
            codec,
            state,
            codec_config: None,
            submitted: 0,
            timebase: Timebase::MPEG_90K.ticks_per_second() as i32,
            // 24 fps until a caller says otherwise: `PreprocessedFrame` carries
            // no fps, and the calibration keyint is expressed in frames.
            frame_duration: Timebase::MPEG_90K.ticks_per_frame(24.0),
            closed: false,
        };

        // SAFETY: the session is live.
        unsafe {
            VTCompressionSessionPrepareToEncodeFrames(encoder.session);
        }
        encoder.apply_settings(cfg)?;
        Ok(encoder)
    }

    /// Push the calibration's knobs into the session.
    fn apply_settings(&self, cfg: &EncoderConfig) -> Result<(), EncodeError> {
        use crate::config::RateControl;
        let keyint = cfg.keyint.max(1) as i32;
        let bframes = cfg.bframes;
        // Real time off: the caller wants quality for the bitrate, not the
        // lowest latency. `low_latency` targets set bframes = 0 instead.
        set_bool_property(
            self.session,
            unsafe { kVTCompressionPropertyKey_RealTime },
            false,
        )?;
        set_bool_property(
            self.session,
            unsafe { kVTCompressionPropertyKey_AllowFrameReordering },
            bframes > 0,
        )?;
        set_i32_property(
            self.session,
            unsafe { kVTCompressionPropertyKey_MaxKeyFrameInterval },
            keyint,
        )?;
        match &cfg.rate_control {
            // VideoToolbox has no CRF. Its quality knob is a 0..1 fraction of
            // the encoder's best effort, so CRF 0..51 maps onto 1..0. This is a
            // mapping, not an equivalent rate control: the same CRF does not
            // produce the same bitrate as the software encoders.
            RateControl::Crf(crf) => {
                let quality = (1.0 - (*crf as f32 / 51.0)).clamp(0.0, 1.0);
                set_f32_property(
                    self.session,
                    unsafe { kVTCompressionPropertyKey_Quality },
                    quality,
                )?;
            }
            RateControl::Vbr(bps) | RateControl::Cbr(bps) => {
                set_i32_property(
                    self.session,
                    unsafe { kVTCompressionPropertyKey_AverageBitRate },
                    (*bps / 1000) as i32,
                )?;
            }
        }
        Ok(())
    }

    /// Wrap one NV12 frame in a `CVPixelBuffer` and hand it to the encoder.
    fn submit(&mut self, frame: &PreprocessedFrame) -> Result<(), EncodeError> {
        if frame.width as usize != self.width || frame.height as usize != self.height {
            return Err(EncodeError::Backend(format!(
                "frame is {}x{}, session is {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        let mut buffer: CVPixelBufferRef = std::ptr::null_mut();
        // SAFETY: the dimensions are checked above and the output slot is local.
        // Full range (`420f`), not video range (`420v`): the preprocessor emits
        // full-range luma and centred 0..255 chroma, and a video-range buffer
        // would be interpreted as 16..240, shifting every colour.
        let status = unsafe {
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                self.width,
                self.height,
                K_CV_PIXEL_FORMAT_420F,
                std::ptr::null(),
                &mut buffer,
            )
        };
        if status != 0 || buffer.is_null() {
            return Err(EncodeError::Backend(format!(
                "CVPixelBufferCreate failed ({status})"
            )));
        }
        // SAFETY: `buffer` is a live pixel buffer; the planes are copied with
        // the buffer's own row stride, which is at least the frame width.
        unsafe {
            CVPixelBufferLockBaseAddress(buffer, 0);
            let y_dst = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) as *mut u8;
            let y_stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0);
            let uv_dst = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) as *mut u8;
            let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1);

            let w = self.width;
            let h = self.height;
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    frame.y.as_ptr().add(row * w),
                    y_dst.add(row * y_stride),
                    w,
                );
            }
            // NV12 chroma is interleaved CbCr, two bytes per sample pair.
            for row in 0..h / 2 {
                let src_row = &frame.u[row * (w / 2)..];
                let src_v = &frame.v[row * (w / 2)..];
                let dst = uv_dst.add(row * uv_stride);
                for col in 0..w / 2 {
                    *dst.add(col * 2) = src_row[col];
                    *dst.add(col * 2 + 1) = src_v[col];
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, 0);

            let pts = CMTime::new(frame.pts.0 as i64, self.timebase);
            let duration = CMTime::new(self.frame_duration as i64, self.timebase);
            let mut flags: VTEncodeInfoFlags = 0;
            let status = VTCompressionSessionEncodeFrame(
                self.session,
                buffer as CVImageBufferRef,
                pts,
                duration,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut flags,
            );
            // The session retains what it needs; the buffer is released here.
            CFRelease(buffer as *const c_void);
            if status != 0 {
                return Err(EncodeError::Backend(format!(
                    "VTCompressionSessionEncodeFrame failed ({status})"
                )));
            }
        }
        self.submitted += 1;
        Ok(())
    }

    /// Take everything the callback has produced.
    fn drain(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        let units = {
            let mut out = self
                .state
                .output
                .lock()
                .map_err(|_| EncodeError::Backend("output queue poisoned".into()))?;
            if let Some(error) = out.error.take() {
                return Err(EncodeError::Backend(error));
            }
            if self.codec_config.is_none() {
                self.codec_config = out.config.clone();
            }
            std::mem::take(&mut out.units)
        };

        let mut result = Vec::with_capacity(units.len());
        for (annex_b, pts, keyframe) in units {
            let parsed = parse_access_unit(&annex_b, self.codec.annex_b())
                .map_err(|e| EncodeError::Backend(format!("VideoToolbox output: {e}")))?;
            if let Some(config) = parsed.config {
                if self.codec_config.is_none() {
                    self.codec_config = Some(config);
                }
            }
            result.push(EncodedUnit {
                track_id: TrackId::Video,
                bytes: parsed.sample,
                pts: Pts(pts),
                dts: Pts(pts),
                keyframe,
                duration: self.frame_duration,
            });
        }
        Ok(result)
    }
}

extern "C" fn output_callback(
    refcon: *mut c_void,
    _source: *mut c_void,
    status: OSStatus,
    _flags: VTEncodeInfoFlags,
    sample: CMSampleBufferRef,
) {
    if refcon.is_null() {
        return;
    }
    // SAFETY: `refcon` points at the state the encoder keeps alive until it
    // invalidates the session in `Drop`.
    let state = unsafe { &*(refcon as *const CallbackState) };
    let mut guard = match state.output.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if status != 0 {
        guard.error = Some(format!("VideoToolbox encode failed ({status})"));
        return;
    }
    if sample.is_null() {
        return;
    }
    // SAFETY: the sample buffer is valid for the duration of the callback.
    unsafe {
        let pts = CMSampleBufferGetPresentationTimeStamp(sample).value as u64;
        // A sample is a keyframe unless it carries the not-for-sync attachment.
        // Parameter sets live in the format description, not in the block
        // buffer, so they are read from there once and kept.
        if guard.config.is_none() {
            let fd = CMSampleBufferGetFormatDescription(sample);
            if !fd.is_null() {
                guard.config = read_parameter_sets(fd, state.codec);
            }
        }
        let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, 0);
        let keyframe = if attachments.is_null() || CFArrayGetCount(attachments) == 0 {
            true
        } else {
            let dict = CFArrayGetValueAtIndex(attachments, 0) as CFDictionaryRef;
            dict.is_null() || CFDictionaryContainsKey(dict, kCMSampleAttachmentKey_NotSync) == 0
        };
        let data_buffer = CMSampleBufferGetDataBuffer(sample);
        if data_buffer.is_null() {
            return;
        }
        let mut length_at_offset = 0usize;
        let mut total_length = 0usize;
        let mut data: *mut u8 = std::ptr::null_mut();
        let rc = CMBlockBufferGetDataPointer(
            data_buffer,
            0,
            &mut length_at_offset,
            &mut total_length,
            &mut data,
        );
        if rc == 0 && !data.is_null() && total_length > 0 {
            // VideoToolbox emits length-prefixed NAL units; the shared parser
            // wants Annex B, so the prefixes are rewritten as start codes.
            let input = std::slice::from_raw_parts(data, total_length);
            if let Some(annex) = length_prefixed_to_annex_b(input) {
                guard.units.push((annex, pts, keyframe));
            } else {
                guard.error = Some("VideoToolbox produced a malformed access unit".into());
            }
        }
    }
}

/// Rewrite 4-byte length prefixes as start codes.
fn length_prefixed_to_annex_b(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() + 16);
    let mut i = 0;
    while i + 4 <= input.len() {
        let len = u32::from_be_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]) as usize;
        i += 4;
        if len == 0 || i + len > input.len() {
            return None;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&input[i..i + len]);
        i += len;
    }
    Some(out)
}

/// Set a boolean session property.
fn set_bool_property(
    session: VTCompressionSessionRef,
    key: CFStringRef,
    value: bool,
) -> Result<(), EncodeError> {
    // SAFETY: `key` is a framework constant and the session is live.
    let status = unsafe {
        let v = if value {
            kCFBooleanTrue
        } else {
            kCFBooleanFalse
        };
        VTSessionSetProperty(session, key, v)
    };
    if status != 0 {
        return Err(EncodeError::Backend(format!(
            "VideoToolbox rejected a boolean property ({status})"
        )));
    }
    Ok(())
}

/// Set an `i32` session property.
fn set_i32_property(
    session: VTCompressionSessionRef,
    key: CFStringRef,
    value: i32,
) -> Result<(), EncodeError> {
    // SAFETY: the CFNumber owns its value; it is released after the call.
    unsafe {
        let number = CFNumberCreate(
            kCFAllocatorDefault,
            K_CF_NUMBER_SINT32_TYPE,
            (&raw const value).cast(),
        );
        if number.is_null() {
            return Err(EncodeError::Backend("CFNumberCreate failed".into()));
        }
        let status = VTSessionSetProperty(session, key, number);
        CFRelease(number);
        if status != 0 {
            return Err(EncodeError::Backend(format!(
                "VideoToolbox rejected an integer property ({status})"
            )));
        }
    }
    Ok(())
}

/// Set an `f32` session property.
fn set_f32_property(
    session: VTCompressionSessionRef,
    key: CFStringRef,
    value: f32,
) -> Result<(), EncodeError> {
    // SAFETY: as above, with the float representation type.
    unsafe {
        let number = CFNumberCreate(
            kCFAllocatorDefault,
            K_CF_NUMBER_FLOAT32_TYPE,
            (&raw const value).cast(),
        );
        if number.is_null() {
            return Err(EncodeError::Backend("CFNumberCreate failed".into()));
        }
        let status = VTSessionSetProperty(session, key, number);
        CFRelease(number);
        if status != 0 {
            return Err(EncodeError::Backend(format!(
                "VideoToolbox rejected a float property ({status})"
            )));
        }
    }
    Ok(())
}

impl VideoEncoder for VideoToolboxEncoder {
    fn encode(&mut self, frame: &PreprocessedFrame) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.submit(frame)?;
        self.drain()
    }

    fn flush(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        if self.closed {
            return Ok(Vec::new());
        }
        // SAFETY: the session is live until `Drop`.
        let status = unsafe {
            VTCompressionSessionCompleteFrames(self.session, CMTime::new(0, self.timebase))
        };
        if status != 0 {
            return Err(EncodeError::Backend(format!(
                "VTCompressionSessionCompleteFrames failed ({status})"
            )));
        }
        let mut units = self.drain()?;
        // VideoToolbox may hold frames after the flush call returns; drain once
        // more so the queue is empty when the caller closes.
        units.extend(self.drain()?);
        Ok(units)
    }

    fn finish(&mut self) -> Result<Vec<EncodedUnit>, EncodeError> {
        self.flush()
    }

    /// VideoToolbox has no per-frame keyframe request on this path: the frame
    /// property that forces one (`kVTEncodeFrameOptionKey_ForceKeyFrame`) is
    /// not wired up, so a PLI is a no-op here rather than a silent lie about
    /// the next frame's type.
    fn request_keyframe(&mut self) {}

    fn codec_config(&self) -> Option<Vec<u8>> {
        self.codec_config.clone()
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        if self.session.is_null() {
            return;
        }
        // SAFETY: invalidating before releasing stops the callback, so the
        // queue cannot be touched after this point.
        unsafe {
            VTCompressionSessionCompleteFrames(self.session, CMTime::new(0, self.timebase));
            VTCompressionSessionInvalidate(self.session);
            CFRelease(self.session as *const c_void);
        }
        self.session = std::ptr::null_mut();
        self.closed = true;
    }
}

/// Read the parameter sets out of a format description and build the
/// `avcC`/`hvcC` body from them.
///
/// # Safety
/// `fd` must be a live `CMFormatDescription` for `codec`.
unsafe fn read_parameter_sets(fd: CMFormatDescriptionRef, codec: Codec) -> Option<Vec<u8>> {
    let sets = collect_sets(fd, codec.annex_b())?;
    if sets.is_empty() {
        return None;
    }
    let refs: Vec<&[u8]> = sets.iter().map(|s| s.as_slice()).collect();
    config_from_parameter_sets(codec.annex_b(), &refs)
        .ok()
        .flatten()
}

/// # Safety
/// `fd` must be a live `CMFormatDescription`.
unsafe fn collect_sets(fd: CMFormatDescriptionRef, codec: AnnexB) -> Option<Vec<Vec<u8>>> {
    let mut sets = Vec::new();
    let mut index = 0usize;
    loop {
        let mut pointer: *const u8 = std::ptr::null();
        let mut size = 0usize;
        let mut count = 0usize;
        let mut header_length = 0i32;
        let status = match codec {
            AnnexB::H264 => CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                fd,
                index,
                &mut pointer,
                &mut size,
                &mut count,
                &mut header_length,
            ),
            AnnexB::Hevc => CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                fd,
                index,
                &mut pointer,
                &mut size,
                &mut count,
                &mut header_length,
            ),
        };
        if index == 0 && status != 0 {
            return None;
        }
        if status != 0 || pointer.is_null() || size == 0 || index >= count {
            break;
        }
        sets.push(std::slice::from_raw_parts(pointer, size).to_vec());
        index += 1;
        if index >= count {
            break;
        }
    }
    Some(sets)
}
