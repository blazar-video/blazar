# Publishing to PyPI

What has to be true before `twine upload` is not a mistake.

## 1. Licensing: wheels cannot carry x264 or x265

`crates/blazar-encoder/Cargo.toml` says it plainly: *"GPL: enabling this makes
the resulting binary GPL."* This project is Apache-2.0, so a wheel built with
`enc-x264` or `enc-x265` is a licensing conflict, not a grey area.

Wheel feature sets are therefore permissive only:

| platform | features |
|---|---|
| `manylinux_2_28_x86_64` | `nvenc,gpu-preprocess,enc-svtav1,enc-vp9` |
| `manylinux_2_28_aarch64` | `enc-svtav1,enc-vp9` |
| `macosx_11_0_arm64` | `enc-videotoolbox` |

NVENC (driver runtime, no bundled code), SVT-AV1 (BSD-2-Clause-Patent), libvpx
(BSD-3-Clause) and VideoToolbox (system framework) are all fine to ship.
Anyone who wants x264 or x265 builds from source with `--features enc-x264`;
that is documented rather than convenient, which is the correct trade.

If H.264/HEVC on CPU is a requirement for wheel users, the honest options are a
separate `blazar-video-gpl` distribution or a licence change — not slipping the
GPL codecs into the Apache wheel and hoping nobody reads the metadata.

## 2. The wheel must be self-contained

`enc-svtav1` and `enc-vp9` link shared libraries. Inside the manylinux container
maturin runs `auditwheel repair`, which bundles them into the wheel; that only
works if the libraries are installed in the container when it builds, which is
why `release.yml` builds inside `PyO3/maturin-action` with `manylinux: 2_28`.

Check before publishing, on a machine without those dev packages:

```sh
python -m venv /tmp/venv && . /tmp/venv/bin/activate
pip install dist/*.whl
python -c "import blazar; print(blazar.__version__)"
ldd $(python -c "import blazar._blazar as m; print(m.__file__)") | grep "not found"
```

`CUDA` must not appear as a hard link: `cudarc` is configured with
`dynamic-loading`, so the wheel loads the driver at runtime and installs on a
machine with no GPU.

## 3. Metadata

`pyproject.toml` carries name, description, readme, keywords, classifiers and
project URLs. Still to do before the first upload:

- **Version**: `0.1.0` in `Cargo.toml` (`[workspace.package]`),
  `bindings/python/Cargo.toml` and `bindings/python/pyproject.toml`. All three
  must agree, and `CHANGELOG.md` must have the matching section.
- **`twine check dist/*`** — PyPI rejects a wheel with an unparsable README, and
  it is cheaper to find out locally.
- **PEP 639** (`license = "Apache-2.0"` plus `license-files`) instead of the
  deprecated `license = { text = ... }`. Deferred because the toolchain support
  varies; do it before 1.0.

## 4. Name and account

- `blazar-video` must be available, or the project name changes. Check at
  <https://pypi.org/project/blazar-video/>.
- **TestPyPI first**: `twine upload --repository testpypi dist/*`, then install
  from it in a clean venv. A first upload to the real index cannot be undone
  (the version number is burned even if the files are yanked).

## 5. Trusted publishing

No long-lived token. On PyPI: project → Publishing → add a GitHub publisher with
owner/repo, workflow filename **`release.yml`**, environment **`pypi`**. Both
sides must match exactly or the OIDC exchange fails with a 403 that does not say
why.

Also create the `pypi` environment in the repository settings (Settings →
Environments) — a required reviewer there is the last human gate before an upload.

For TestPyPI, a second publisher pointing at the same workflow with environment
`testpypi`, and a `--repository testpypi` job gated on `workflow_dispatch`.

## 6. What must be green

`ci.yml` on the release commit. That covers the CPU surface only — no NVENC, no
VideoToolbox. While the self-hosted runners do not exist, the release notes must
say "validated on the CPU path only", not imply otherwise.

## 7. Sequence

```sh
# locally, once green
cargo build --release --workspace           # the crates still build
cd bindings/python && maturin build --release --features enc-svtav1,enc-vp9
twine check dist/*
python -m venv /tmp/v && . /tmp/v/bin/activate && pip install dist/*.whl
python -c "import blazar; print(blazar.__version__)"   # no "not found" in ldd

git tag -a v0.1.0 -m "v0.1.0" && git push origin v0.1.0   # release.yml takes over
```

## Known limitations of the current automation

- **The smoke test does not check the backend list.** `release.yml` imports the
  wheel and touches `Encoder` but never asserts which backends were compiled in,
  so a wheel built with the wrong features would publish. Needs a
  `blazar.backends()` accessor (or an equivalent) exposed from the extension.
- **No sdist.** Only wheels are published. `pip install blazar-video` on a
  platform without a wheel fails rather than falling back to a source build,
  which is the safer default while the crate publishes separate SDK forks, but
  it should be a deliberate choice rather than an omission.
