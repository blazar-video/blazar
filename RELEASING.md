# Releasing

Two artifacts are versioned and released together: the Rust crates, and the
Python wheel. The calibrations live in their own repository and are versioned
separately, because a calibration changes what a stream *means* without changing
any code.

## Versioning

`0.x` until the encoder API stops moving. Within `0.x`, **the minor version may
break**; the patch version never does.

- `0.1.0` → `0.2.0`: any change to the public Python surface, calibration keys, or
  the meaning of an existing setting.
- `0.1.0` → `0.1.1`: fixes only. No API change, no bitstream change at identical
  settings.

A **bitstream change at identical settings** (different QP mapping, different VUI,
different parameter sets) is a minor bump even though no signature changed: a
caller comparing outputs across versions must be able to see it in the number.

The calibrations repository carries its own tag, `calibrations-vX.Y.Z`, and the
release notes state which calibration tag the wheel was validated against. A
wheel is never shipped pointing at "whatever calibrations main was that day".

## Deprecation

One minor release of overlap, minimum, and the replacement is named in the
warning. `Deprecated` is not a resting state: the removal version goes in the
docstring at the moment the deprecation is added, and the release that removes it
says so in the changelog.

There is no backlog of deprecations today — the pre-release API churn was folded
in rather than accumulated (see the `[Feat]` commits that removed
`submit_device_batch` and the pair-shaped backend names).

## What CI gates

| Workflow | Runs on | Covers |
|---|---|---|
| `ci.yml` | every push and PR | fmt, clippy `-D warnings`, workspace tests, rustdoc `-D warnings`, x264 round trip, mock-backend Python tests on 3.11/3.13 × Linux/macOS, license headers, no committed artifacts |
| `hardware.yml` | manual | NVENC and VideoToolbox suites on self-hosted runners |

Hosted runners have no GPU, so the NVENC and VideoToolbox suites **cannot** run
there. They are a separate workflow on purpose: a flaky hardware box must not be
able to block a merge, and a red hardware job should be visibly a hardware
problem rather than a code one.

A release requires a green `ci.yml` on the release commit. That is only the CPU
surface, so it is a floor, not a guarantee: nothing in it opens an NVENC or
VideoToolbox session.

**While the self-hosted runners are not yet running**, the hardware suites are
manual (`workflow_dispatch`). A release in that state must say so in its notes —
"validated on the CPU path only; hardware suites not run" — rather than imply
coverage it does not have. Once the runners exist, the gate becomes: green
`ci.yml` **and** a green `hardware.yml` within the previous 24 hours, and a
release waits for hardware rather than shipping an encoder nobody executed.

## Cutting a release

1. **Bump the version** in three places, which must agree:
   - `Cargo.toml` → `[workspace.package] version`
   - `bindings/python/Cargo.toml` → `[package] version`
   - `bindings/python/pyproject.toml` → `[project] version`
2. **Update `CHANGELOG.md`**: an `## [X.Y.Z] - YYYY-MM-DD` section with
   `Added` / `Changed` / `Fixed` / `Removed`. Breaking changes get their own
   subsection with the migration in one line each.
3. **`cargo update -w`** if any pinned dependency moved, then commit the lock
   alongside the version bump. A release with an uncommitted lock is not
   reproducible.
4. **Land it on `main`** through the normal review, wait for green `ci.yml`.
5. **Tag**: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
6. **Wheels**: the tag triggers `release.yml`, which builds
   `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64` and
   publishes to PyPI via trusted publishing (no long-lived token). Attach the
   same wheels to the GitHub release.
7. **Crates**: `cargo publish -p blazar-core -p blazar-calibration -p
   blazar-sink -p blazar-preprocess -p blazar-encoder -p blazar-runtime`, in
   dependency order, after the PyPI publish succeeds.
8. **Release notes**: build matrix, calibration tag validated against, the
   measured performance table if it changed, and any known gap from `state.md`
   that users should know about.

## What must not happen in a release

- A wheel built without the feature set the notes claim. The wheel matrix is
  explicit (`nvenc,gpu-preprocess,enc-x264,enc-x265,enc-svtav1,enc-vp9` on Linux,
  `enc-videotoolbox` on macOS) and the smoke test in `release.yml` asserts the
  backend list the wheel reports.
- A release whose notes cite an unmeasured number. Throughput claims come from
  `benchmark-nvenc-*.md` in the state directory, with the method stated.
- A silent behaviour change. `backend="auto"` can pick a different encoder than a
  pinned name; anything in that family goes in the changelog under `Changed`.

## Rollback

Yank the PyPI release (`twine yank`), revert the tag's changelog entry, and
publish a patch from the previous tag rather than force-pushing `main`. Crates
cannot be yanked usefully once depended upon, so a bad crate release is fixed
forward.
