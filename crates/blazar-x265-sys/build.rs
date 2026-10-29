// SPDX-License-Identifier: Apache-2.0
//! Locate libx265 and work out which `x265_encoder_open_<build>` symbol to use.
//!
//! Two things are generated in different places, for different reasons.
//!
//! The *types* live in `src/pregenerated.rs` and are checked in. Generating them
//! at build time made the build depend on the local clang: on Ubuntu 24.04 with
//! clang 19, bindgen 0.69 emitted an opaque `x265_param` (`_address` only) and
//! the encoder failed to compile with 22 field errors. Version 1.16 of the
//! approach used for `blazar-vpx-sys` applies here too.
//!
//! The *symbol* cannot be pinned: `x265.h` resolves `x265_encoder_open` through
//! a macro to a build-specific name (`x265_encoder_open_216` on 4.1, 217 on
//! other builds), so it has to be read from the header that is actually linked.
//! That is a one-line macro expansion, so it does not care whether clang can
//! parse the struct bodies.

fn main() {
    let lib = pkg_config::Config::new()
        .probe("x265")
        .expect("libx265 development files (pkg-config x265) are required");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=X265_ENCODER_OPEN_BUILD");

    #[cfg(feature = "regenerate")]
    regenerate(&lib);

    // Parse only the function, so this stays fast and independent of whether the
    // struct definitions come out opaque.
    let mut builder = bindgen::Builder::default()
        .header_contents("alias.h", "#include <x265.h>")
        .allowlist_function("^x265_encoder_open_\\d+$")
        .generate_comments(false)
        .layout_tests(false);
    for include in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    let bindings = builder
        .generate()
        .expect("resolve x265_encoder_open symbol");
    let text = bindings.to_string();
    let build = text
        .split("pub fn x265_encoder_open_")
        .nth(1)
        .and_then(|rest| rest.split('(').next())
        .map(|digits| digits.trim().to_string())
        .expect("x265_encoder_open_<build> missing from the header");

    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(
        out.join("encoder_open.rs"),
        format!(
            "// Generated: the symbol libx265 exports for x265_encoder_open.\n\
             extern \"C\" {{\n    \
             pub fn x265_encoder_open_{build}(param: *mut x265_param) -> *mut x265_encoder;\n}}\n\
             pub use x265_encoder_open_{build} as x265_encoder_open;\n"
        ),
    )
    .expect("write encoder_open alias");
}

#[cfg(feature = "regenerate")]
fn regenerate(lib: &pkg_config::Library) {
    let mut builder = bindgen::Builder::default()
        .header_contents("x265.h", "#include <x265.h>")
        .allowlist_function("x265_.*")
        .allowlist_type("x265_.*")
        .allowlist_var("X265_.*")
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .layout_tests(false);
    for include in &lib.include_paths {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    let bindings = builder.generate().expect("generate libx265 bindings");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pregenerated.rs");
    bindings.write_to_file(&path).expect("write bindings");
    println!("cargo:warning=rewrote {}", path.display());
}
