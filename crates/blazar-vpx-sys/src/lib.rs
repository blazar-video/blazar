//! Bindings to libvpx for VP9 encoding.
//!
//! Generated at build time from the header pkg-config reports, so the struct
//! layout matches the library that will be linked.

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
    rustdoc::bare_urls,
)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub use generated::*;

