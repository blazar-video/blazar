//! Generate libvpx bindings for the VP9 encoder from the system headers.

use std::path::PathBuf;

fn main() {
    let lib = pkg_config::Config::new()
        .probe("vpx")
        .expect("libvpx development files (pkg-config vpx) are required");
    println!("cargo:rerun-if-changed=build.rs");

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
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}
