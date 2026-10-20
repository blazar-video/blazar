//! Generate SVT-AV1 bindings from the header installed by the package manager.
//!
//! The published `svt-av1-sys` crate is unbuildable with current bindgen, so
//! the bindings are generated here from the same header the linker will use.

use std::path::PathBuf;

fn main() {
    let lib = pkg_config::Config::new()
        .probe("SvtAv1Enc")
        .expect("SVT-AV1 development files (pkg-config SvtAv1Enc) are required");
    println!("cargo:rerun-if-changed=build.rs");

    let mut builder = bindgen::Builder::default()
        .header_contents("svt.h", "#include <EbSvtAv1Enc.h>\n")
        .allowlist_function("svt_av1_.*")
        .allowlist_type("EbSvtAv1.*")
        .allowlist_type("EbComponentType")
        .allowlist_type("EbBufferHeaderType")
        .allowlist_type("EbErrorType")
        .allowlist_type("EbSvtIOFormat")
        .allowlist_var("EB_.*")
        .default_enum_style(bindgen::EnumVariation::ModuleConsts);
    for include in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }

    let bindings = builder.generate().expect("generate SVT-AV1 bindings");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}
