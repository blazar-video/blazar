//! Raw bindings to `libx265`.
//!
//! Generated from the system header at build time. Bindgen is not optional
//! here: `x265.h` resolves `x265_encoder_open` and friends through macros to
//! build-specific symbols (`x265_encoder_open_216` on the 4.1 library), so a
//! hand-written declaration would link against a symbol that does not exist.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// bindgen reproduces the upstream header's doc comments, which contain
// bracket notation that rustdoc reads as intra-doc links.
#[allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::bare_urls,
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use generated::*;

// `x265_encoder_open` is a macro to a build-specific symbol in the header; the
// build script resolves the generated name so callers can use the plain one.
include!(concat!(env!("OUT_DIR"), "/encoder_open.rs"));

/// Rate-control methods (`X265_RC_METHODS` in the header).
///
/// The header declares these in an anonymous-enum typedef, which bindgen's
/// `allowlist_var` does not emit. The values are ABI and cannot move without
/// breaking every encoded stream, so they are spelled out here.
pub const X265_RC_ABR: i32 = 0;
pub const X265_RC_CQP: i32 = 1;
pub const X265_RC_CRF: i32 = 2;
