// SPDX-License-Identifier: Apache-2.0
//! Generate libx265 bindings for the HEVC encoder from the system headers.

use std::path::PathBuf;

fn main() {
    let lib = pkg_config::Config::new()
        .probe("x265")
        .expect("libx265 development files (pkg-config x265) are required");
    println!("cargo:rerun-if-changed=build.rs");

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
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    // `x265.h` defines `x265_encoder_open` as a macro over a build-specific
    // symbol (`x265_encoder_open_216` on 4.1). Bindgen emits the expanded name,
    // so record whatever it produced under the plain name. Hardcoding 216 here
    // would break on every other libx265 build.
    let text = bindings.to_string();
    let build = text
        .split("pub fn x265_encoder_open_")
        .nth(1)
        .and_then(|rest| rest.split('(').next())
        .map(|digits| digits.trim().to_string())
        .expect("x265_encoder_open_<build> missing from the generated bindings");
    std::fs::write(
        out.join("encoder_open.rs"),
        format!("pub use crate::x265_encoder_open_{build} as x265_encoder_open;\n"),
    )
    .expect("write encoder_open alias");

    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}
