// SPDX-License-Identifier: Apache-2.0
//! Bindings to the SVT-AV1 encoder API (`EbSvtAv1Enc.h`).
//!
//! Generated at build time from the header found by pkg-config, so the struct
//! layout always matches the library that will be linked.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// bindgen reproduces the upstream header's doc comments, which contain
// bracket notation that rustdoc reads as intra-doc links.
#[allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::bare_urls
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use generated::*;

// ---------------------------------------------------------------------------
// Version differences, normalised here so the encoder does not need cfgs.
//
// 2.x: `svt_av1_enc_init_handle(handle, app_data, config)`, `pred_structure` is
// a uint8_t with macro constants, `rate_control_mode` is uint32_t.
// 3.x+: two-argument init, `PredStructure` enum, uint8_t rate control.
// ---------------------------------------------------------------------------

/// `svt_av1_enc_init_handle` with the argument list the installed version takes.
///
/// # Safety
/// `handle` and `config` must be valid for the call.
pub unsafe fn enc_init_handle(
    handle: *mut *mut EbComponentType,
    config: *mut EbSvtAv1EncConfiguration,
) -> EbErrorType::Type {
    #[cfg(svt_av1_2)]
    {
        unsafe { svt_av1_enc_init_handle(handle, ::core::ptr::null_mut(), config) }
    }
    #[cfg(not(svt_av1_2))]
    {
        unsafe { svt_av1_enc_init_handle(handle, config) }
    }
}

/// Random-access prediction (B-frames), without the caller naming the constant.
pub fn set_random_access(config: &mut EbSvtAv1EncConfiguration) {
    #[cfg(svt_av1_2)]
    {
        config.pred_structure = 2; // SVT_AV1_RANDOM_ACCESS
    }
    #[cfg(not(svt_av1_2))]
    {
        config.pred_structure = PredStructure::RANDOM_ACCESS;
    }
}

/// Constant-QP/CRF rate control.
pub fn set_rate_control_cqp(config: &mut EbSvtAv1EncConfiguration, qp: u32) {
    #[cfg(svt_av1_2)]
    {
        config.rate_control_mode = 0; // SVT_AV1_RC_MODE_CQP_OR_CRF
    }
    #[cfg(not(svt_av1_2))]
    {
        config.rate_control_mode = 0;
    }
    config.qp = qp;
}

/// Variable-bitrate rate control.
pub fn set_rate_control_vbr(config: &mut EbSvtAv1EncConfiguration, kbps: u32) {
    #[cfg(svt_av1_2)]
    {
        config.rate_control_mode = 1; // SVT_AV1_RC_MODE_VBR
    }
    #[cfg(not(svt_av1_2))]
    {
        config.rate_control_mode = 1;
    }
    config.target_bit_rate = kbps;
}
