// SPDX-License-Identifier: Apache-2.0
//! Link the extension module with chained fixups on macOS.
//!
//! rustc 1.95 + the Command Line Tools linker (ld-27) emit a classic
//! `LC_DYLD_INFO_ONLY` LINKEDIT whose string table is only 4-byte aligned. The
//! macOS 26/27 dyld rejects that with "mis-aligned LINKEDIT string pool" at
//! import. `-fixup_chains` switches the output to the chained-fixups format,
//! whose LINKEDIT is laid out correctly, so the module loads.
//!
//! `cargo:rustc-link-arg` scopes the flag to *this* crate's final link only, so
//! proc-macro and build-script host artifacts (which the linker handles fine) are
//! untouched — unlike a blanket `RUSTFLAGS`, which breaks those builds.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-fixup_chains");
    }
}
