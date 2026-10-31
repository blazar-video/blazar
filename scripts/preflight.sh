#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build everything the CI matrix builds, on this machine, before pushing.
#
# CI is not a compiler. Every environment-specific failure so far — a distro
# libvpx too old for the bindings, apt package names that differ between Ubuntu
# releases, a container that had no CMake — was reproducible on a local Linux box
# in under two minutes. This script is the local equivalent of the matrix, so
# those failures are found here instead.
#
# Usage: scripts/preflight.sh [--quick]
#   --quick   skip the Python wheel and its tests (Rust surfaces only)
#
# Requires: cargo, and for the Python parts a wheel-buildable environment
# (maturin, a Python with headers). Skips with a message when a codec library is
# absent rather than pretending it passed.

set -uo pipefail

cd "$(dirname "$0")/.."
root=$(pwd)
failed=()

step() {
    local name=$1
    shift
    printf '\n=== %s\n' "$name"
    if "$@"; then
        printf '    ok\n'
    else
        printf '    FAILED\n'
        failed+=("$name")
    fi
}

rust_fmt() { cargo fmt --all -- --check; }

rust_clippy() {
    # The -sys crates are excluded from the workspace lint for the same reason
    # CI excludes them: their build scripts need codec headers.
    cargo clippy --workspace --exclude blazar-vpx-sys --exclude blazar-x265-sys \
        --exclude blazar-svtav1-sys --all-targets -- -D warnings
}

rust_tests() {
    cargo test --workspace --exclude blazar-vpx-sys --exclude blazar-x265-sys \
        --exclude blazar-svtav1-sys
    cargo test -p blazar-encoder -p blazar-runtime --features enc-x264
}

rust_docs() {
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps \
        --exclude blazar-vpx-sys --exclude blazar-x265-sys --exclude blazar-svtav1-sys
}

# The pkg-config name each feature needs, if any. A codec whose library is not
# installed on this machine is reported as skipped, not failed: that is an
# environment gap, and calling it a failure teaches people to ignore the script.
codes_need() {
    case "$1" in
        enc-x264) echo x264 ;;
        enc-x265) echo x265 ;;
        enc-svtav1) echo SvtAv1Enc ;;
        enc-vp9) echo vpx ;;
        *) echo "" ;;
    esac
}

# Each optional encoder is built on its own, so a feature that only works in
# combination with others is caught.
feature_sets() {
    local status=0 features code need missing
    for features in "" "enc-x264" "enc-svtav1" "enc-vp9" "enc-x265" "enc-opus" \
                    "nvenc" "gpu-preprocess" "nvenc,gpu-preprocess" \
                    "enc-svtav1,enc-vp9,enc-x264,enc-opus"; do
        # The NVENC SDK fork ships no Apple sys module, so anything with `nvenc` is
        # unbuildable on macOS regardless of what pkg-config reports.
        if [[ ($features == *nvenc* || $features == *gpu-preprocess*) && $(uname -s) == Darwin ]]; then
            printf -- '--- features: %s  SKIPPED (no NVENC sys module on macOS)\n' "$features"
            continue
        fi
        missing=""
        for code in ${features//,/ }; do
            need=$(codes_need "$code")
            [[ -z $need ]] && continue
            pkg-config --exists "$need" 2>/dev/null || missing="$missing $need"
        done
        if [[ -n $missing ]]; then
            printf -- '--- features: %s  SKIPPED (missing:%s)\n' "${features:-<default>}" "$missing"
            continue
        fi
        printf -- '--- features: %s\n' "${features:-<default>}"
        if ! cargo build --release -p blazar-encoder -p blazar-runtime \
            ${features:+--features "$features"} 2>&1 | tail -20; then
            status=1
        fi
    done
    return $status
}

python_wheel() {
    command -v maturin >/dev/null || { echo "maturin not installed, skipping"; return 0; }
    local python
    python=${PYO3_PYTHON:-python3}
    ( cd bindings/python && PYO3_PYTHON=$python maturin build --release --out dist \
        --features nvenc,gpu-preprocess,enc-svtav1,enc-vp9,opus ) || return 1
    local wheel
    wheel=$(ls -t bindings/python/dist/*.whl | head -1)
    "$python" -m pip install --force-reinstall --no-deps "$wheel" || return 1
    ( cd bindings/python && "$python" -m pytest tests/test_acceptance.py -q ) || return 1
}

step fmt rust_fmt
step clippy rust_clippy
step "feature matrix" feature_sets
step "rust tests" rust_tests
step "rustdoc" rust_docs

if [[ "${1:-}" != "--quick" ]]; then
    step "python wheel + acceptance" python_wheel
fi

printf '\n'
if ((${#failed[@]})); then
    printf 'FAILED: %s\n' "${failed[*]}"
    exit 1
fi
printf 'preflight passed in %s\n' "$root"
