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
| `manylinux_2_28_x86_64` (**`-gpl`**) | the above plus `enc-x264,enc-x265` |
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

## The GPL distribution

`blazar-video-gpl` is a second distribution, built from the same crate with
`enc-x264,enc-x265` added, published by `.github/workflows/release-gpl.yml`
under the `pypi-gpl` environment.

**How it is built.** maturin cannot build it from `packaging/gpl`: its
`python-source` handling copies the Python package only when the crate being
built lives inside the project, and that project has to point `manifest-path` at
`bindings/python`. Every layout tried — a copy of the package, a symlinked
directory, a symlinked crate — produced an extension with no
`blazar/__init__.py`, which imports only by accident.

So the wheel is built once, from `bindings/python`, with the GPL features on and
`--auditwheel=repair` so x264/x265 travel inside it, and
`scripts/gpl_wheel.py` rewrites exactly what makes it a different distribution:
the filename, the `.dist-info` directory, `Name`, the licence expression and
classifier, the `NOTICE.md`, and a recomputed `RECORD`. One crate, one Python
package, no second tree to drift.

**Source builds** need none of that:

```sh
MATURIN_PEP517_ARGS="--features nvenc,gpu-preprocess,enc-svtav1,enc-vp9,enc-x264,enc-x265,opus" \
  pip install ./bindings/python
```

**Why this is legal:** x264 and x265 are *GPL-2.0-or-later* (verified in their
source headers). The Apache-2.0 core can be combined into a **GPLv3** work,
because Apache-2.0 is GPLv3-compatible. It could not be combined with a
GPLv2-only work, which is why this axis matters and why the wheel declares
`GPL-3.0-or-later` rather than inheriting anything from x264's GPLv2 default.

**What we do differently from the precedent:** PyAV ships libx264 and libx265 in
a wheel whose metadata says `BSD-3-Clause`. We ship the GPL wheel under its own
name, with a GPL licence expression and a `NOTICE.md` in every wheel naming the
bundled libraries and their sources. A scanner reading our metadata gets the
truth; a scanner reading PyAV's does not.

**Obligations this creates**, none of them optional:

- corresponding source for the combined work — the `v<version>` tag, which is the
  same source the `blazar-video` sdist is built from, plus the links in
  `packaging/gpl/NOTICE.md` for the bundled libraries;
- the licence text in the wheel;
- no added restrictions on the code we distribute;
- `v<version>` tags on the repository matching the published version.

**What users take on** is stated in `NOTICE.md`, and it is worth repeating in any
sales conversation: installing this pulls copyleft into their process, and
shipping a product that includes it brings source obligations with it. The
permissive wheel plus a commercial x264/x265 licence is the way out, and that
route also avoids the H.264/HEVC **patent** licensing that applies regardless of
which encoder ran.

## 4. Name and account

- `blazar-video` must be available, or the project name changes. Check at
  <https://pypi.org/project/blazar-video/>.
- **Install the wheel from a local build first**, then publish: `pip install dist/*.whl`, then install
  from it in a clean venv. A first upload to the real index cannot be undone
  (the version number is burned even if the files are yanked).

## 4a. Rehearsing

A manual run (`workflow_dispatch`) builds every artefact and runs both smoke jobs,
and **publishes nothing**. Only a tag publishes.

There is no rehearsal index in the pipeline. A TestPyPI run only ever exercised a
second, independent publisher configuration — it could not tell you anything about
the one that matters, and a misconfigured rehearsal costs a round trip just like a
misconfigured upload would. The upload itself is the one step with no local
equivalent; the version number is the thing it costs if the publisher is wrong.

What a rehearsal must cover instead: install the built wheel in a clean venv and
run the acceptance suite against it. The smoke jobs do exactly that on the runner
of each platform.

`blazar.backends()` is what the smoke test asserts: the list of encoder
implementations compiled into the wheel. Importing the package proves nothing —
a wheel built with the wrong `--features` imports fine and fails at the first
frame, after publishing.

## 5. Trusted publishing

No long-lived token. On PyPI: project → Publishing → add a GitHub publisher with
owner/repo, workflow filename **`release.yml`**, environment **`pypi`**. Both
sides must match exactly or the OIDC exchange fails with a 403 that does not say
why.

Also create the `pypi` environment in the repository settings (Settings →
Environments) — a required reviewer there is the last human gate before an upload.

The GPL distribution needs its own publisher: project `blazar-video-gpl`,
environment `pypi-gpl`.

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

- **Environment pinning.** The PyPI publisher here is configured with
  environment name *Any*, so it is not bound to a specific workflow file. The
  GitHub-side environment protection (`pypi`, `pypi-gpl` with required reviewers)
  is therefore the only gate between a workflow edit and a live upload.
- **No arm64 Linux wheel for the GPL distribution.** Only x86_64 is built there;
  the Apache wheel has both. Deliberate for now — the manylinux aarch64 image
  needs x264/x265 built from source.
