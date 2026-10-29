// SPDX-License-Identifier: Apache-2.0
//! Raw bindings to `libx265`.
//!
//! The types are checked in (`src/pregenerated.rs`), generated from libx265 4.2
//! headers: generating them per build depended on the local clang, and clang 19
//! with bindgen 0.69 produced an opaque `x265_param`.
//!
//! The `x265_encoder_open` symbol is different — `x265.h` resolves it through a
//! macro to a build-specific name — so the build script reads it from the header
//! that is actually linked and emits the alias.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// bindgen reproduces the upstream header's doc comments, which contain
// bracket notation that rustdoc reads as intra-doc links.
#[allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::bare_urls
)]
mod generated {
    include!("pregenerated.rs");
    include!(concat!(env!("OUT_DIR"), "/encoder_open.rs"));
}
pub use generated::*;

/// Rate-control methods (`X265_RC_METHODS` in the header).
///
/// The header declares these in an anonymous-enum typedef, which bindgen's
/// `allowlist_var` does not emit. The values are ABI and cannot move without
/// breaking every encoded stream, so they are spelled out here.
pub const X265_RC_ABR: i32 = 0;
pub const X265_RC_CQP: i32 = 1;
pub const X265_RC_CRF: i32 = 2;
