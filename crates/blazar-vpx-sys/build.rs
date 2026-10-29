// SPDX-License-Identifier: Apache-2.0
//! Locate libvpx and, if asked, regenerate the bindings.
//!
//! The bindings in `src/pregenerated.rs` are checked in. Generating them at
//! build time made the build depend on the local clang and on whichever vpx
//! headers pkg-config resolved first: on Ubuntu 24.04 that produced an opaque
//! `vpx_codec_enc_cfg` and fourteen compile errors in the encoder. Version 1.16
//! of the API is stable, so pinning the bindings removes a moving part without
//! costing anything.
//!
//! Building with `--features regenerate` writes fresh bindings over that file;
//! commit them when the target libvpx version changes on purpose.

fn main() {
    let lib = pkg_config::Config::new()
        .probe("vpx")
        .expect("libvpx development files (pkg-config vpx) are required");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BLAZAR_REGENERATE_VPX");

    #[cfg(feature = "regenerate")]
    regenerate(&lib);
}

#[cfg(feature = "regenerate")]
fn regenerate(lib: &pkg_config::Library) {
    let mut builder = bindgen::Builder::default()
        .header_contents(
            "vpx.h",
            "#include <vpx/vpx_encoder.h>
             #include <vpx/vp8cx.h>
             #include <vpx/vpx_codec.h>
             #include <vpx/vpx_image.h>",
        )
        .allowlist_function("vpx_.*")
        .allowlist_type("vpx_.*")
        .allowlist_type("vpx_codec_.*")
        .allowlist_type("vpx_image_.*")
        .allowlist_type("vpx_rational.*")
        .allowlist_type("vpx_fixed_buf.*")
        .allowlist_type("vpx_enc.*")
        .allowlist_var("VP8.*")
        .allowlist_var("VP9.*")
        .allowlist_var("VPX_.*")
        .default_enum_style(bindgen::EnumVariation::ModuleConsts);
    for include in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    let bindings = builder.generate().expect("generate libvpx bindings");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pregenerated.rs");
    bindings
        .write_to_file(&path)
        .expect("write bindings");
    println!("cargo:warning=rewrote {}", path.display());
}
