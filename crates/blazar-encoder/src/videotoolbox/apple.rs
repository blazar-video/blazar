//! Minimal VideoToolbox bindings, declared here rather than pulled from a
//! crate so the dependency surface stays small and auditable.
//!
//! Only the calls this encoder makes are declared. Link failures surface as
//! `OSStatus` values from the create call rather than as missing symbols,
//! because the frameworks ship with macOS.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::c_void;

pub type OSStatus = i32;
pub type CVReturn = i32;
pub type CFAllocatorRef = *const c_void;
pub type CFDictionaryRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFArrayRef = *const c_void;
pub type CFTypeRef = *const c_void;
pub type Boolean = u8;
pub type OSType = u32;
pub type CMVideoCodecType = u32;
pub type CVOptionFlags = u64;
pub type VTEncodeInfoFlags = u32;
pub type VTCompressionSessionRef = *mut c_void;
pub type CMSampleBufferRef = *mut c_void;
pub type CMBlockBufferRef = *mut c_void;
pub type CMFormatDescriptionRef = *mut c_void;
pub type CVImageBufferRef = *mut c_void;
pub type CVPixelBufferRef = *mut c_void;

/// `CMTime` as laid out by CoreMedia.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

pub const K_CM_TIME_FLAG_VALID: u32 = 1;

impl CMTime {
    pub fn new(value: i64, timescale: i32) -> Self {
        Self {
            value,
            timescale,
            flags: K_CM_TIME_FLAG_VALID,
            epoch: 0,
        }
    }
}

/// `'420v'`: 8-bit NV12, video range.
pub const K_CV_PIXEL_FORMAT_420V: OSType = u32::from_be_bytes(*b"420v");
/// `'avc1'` and `'hvc1'` codec types.
pub const K_CM_VIDEO_CODEC_TYPE_H264: CMVideoCodecType = u32::from_be_bytes(*b"avc1");
pub const K_CM_VIDEO_CODEC_TYPE_HEVC: CMVideoCodecType = u32::from_be_bytes(*b"hvc1");

pub type VTCompressionOutputCallback = unsafe extern "C" fn(
    output_callback_ref_con: *mut c_void,
    source_frame_ref_con: *mut c_void,
    status: OSStatus,
    info_flags: VTEncodeInfoFlags,
    sample_buffer: CMSampleBufferRef,
);

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    pub fn VTCompressionSessionCreate(
        allocator: CFAllocatorRef,
        width: i32,
        height: i32,
        codec_type: CMVideoCodecType,
        encoder_specification: CFDictionaryRef,
        source_image_buffer_attributes: CFDictionaryRef,
        compressed_data_allocator_attributes: CFDictionaryRef,
        output_callback: Option<VTCompressionOutputCallback>,
        output_callback_ref_con: *mut c_void,
        compression_session_out: *mut VTCompressionSessionRef,
    ) -> OSStatus;

    pub fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;

    pub fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image_buffer: CVImageBufferRef,
        presentation_time_stamp: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut VTEncodeInfoFlags,
    ) -> OSStatus;

    pub fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        complete_until_presentation_time_stamp: CMTime,
    ) -> OSStatus;

    pub fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);

    pub fn VTSessionSetProperty(
        session: VTCompressionSessionRef,
        property_key: CFStringRef,
        property_value: CFTypeRef,
    ) -> OSStatus;

    pub static kVTCompressionPropertyKey_RealTime: CFStringRef;
    pub static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    pub static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
    pub static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;
    pub static kVTCompressionPropertyKey_Quality: CFStringRef;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    pub fn CVPixelBufferCreate(
        allocator: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format_type: OSType,
        pixel_buffer_attributes: CFDictionaryRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> CVReturn;

    pub fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: CVOptionFlags) -> CVReturn;
    pub fn CVPixelBufferUnlockBaseAddress(
        buffer: CVPixelBufferRef,
        flags: CVOptionFlags,
    ) -> CVReturn;
    pub fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVPixelBufferRef, plane: usize)
        -> *mut c_void;
    pub fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    pub fn CMSampleBufferGetDataBuffer(sample_buffer: CMSampleBufferRef) -> CMBlockBufferRef;
    pub fn CMSampleBufferGetFormatDescription(
        sample_buffer: CMSampleBufferRef,
    ) -> CMFormatDescriptionRef;
    pub fn CMSampleBufferGetPresentationTimeStamp(sample_buffer: CMSampleBufferRef) -> CMTime;
    pub fn CMSampleBufferGetSampleAttachmentsArray(
        sample_buffer: CMSampleBufferRef,
        create_if_necessary: Boolean,
    ) -> CFArrayRef;

    pub fn CMBlockBufferGetDataPointer(
        block_buffer: CMBlockBufferRef,
        offset: usize,
        length_at_offset_out: *mut usize,
        total_length_out: *mut usize,
        data_pointer_out: *mut *mut u8,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        format_description: CMFormatDescriptionRef,
        parameter_set_index: usize,
        parameter_set_pointer_out: *mut *const u8,
        parameter_set_size_out: *mut usize,
        parameter_set_count_out: *mut usize,
        nal_unit_header_length_out: *mut i32,
    ) -> OSStatus;

    pub fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        format_description: CMFormatDescriptionRef,
        parameter_set_index: usize,
        parameter_set_pointer_out: *mut *const u8,
        parameter_set_size_out: *mut usize,
        parameter_set_count_out: *mut usize,
        nal_unit_header_length_out: *mut i32,
    ) -> OSStatus;

    pub static kCMSampleAttachmentKey_NotSync: CFStringRef;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub static kCFAllocatorDefault: CFAllocatorRef;
    pub static kCFBooleanTrue: CFTypeRef;
    pub static kCFBooleanFalse: CFTypeRef;
    pub static kCFTypeDictionaryKeyCallBacks: c_void;
    pub static kCFTypeDictionaryValueCallBacks: c_void;

    pub fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const *const c_void,
        values: *const *const c_void,
        num_values: isize,
        key_call_backs: *const c_void,
        value_call_backs: *const c_void,
    ) -> CFDictionaryRef;

    pub fn CFDictionaryCreateMutable(
        allocator: CFAllocatorRef,
        capacity: isize,
        key_call_backs: *const c_void,
        value_call_backs: *const c_void,
    ) -> CFDictionaryRef;

    pub fn CFDictionarySetValue(dict: CFDictionaryRef, key: *const c_void, value: *const c_void);
    pub fn CFDictionaryContainsKey(dict: CFDictionaryRef, key: *const c_void) -> Boolean;

    pub fn CFNumberCreate(
        allocator: CFAllocatorRef,
        the_type: isize,
        value_ptr: *const c_void,
    ) -> CFTypeRef;

    pub fn CFArrayGetCount(array: CFArrayRef) -> isize;
    pub fn CFArrayGetValueAtIndex(array: CFArrayRef, idx: isize) -> *const c_void;
    pub fn CFRelease(cf: *const c_void);
}

/// `kCFNumberSInt32Type`.
pub const K_CF_NUMBER_SINT32_TYPE: isize = 3;
/// `kCFNumberFloat32Type`.
pub const K_CF_NUMBER_FLOAT32_TYPE: isize = 5;
