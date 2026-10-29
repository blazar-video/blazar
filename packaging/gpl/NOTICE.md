# Source availability — `blazar-video-gpl`

This wheel is licensed **GPL-3.0-or-later**, not Apache-2.0. It is distributed
under the GPL because it is linked with the GPL encoders x264 and x265.

## Our source

The complete corresponding source of Blazar is the repository at
<https://github.com/blazar-video/blazar>, tag `v<version>` — the same source the
`blazar-video` sdist of that version is built from. This distribution differs
only in which encoders the extension is compiled against, so building that
source with the GPL features reproduces this wheel:

```sh
git clone --branch v<version> https://github.com/blazar-video/blazar
MATURIN_PEP517_ARGS="--features nvenc,gpu-preprocess,enc-svtav1,enc-vp9,enc-x264,enc-x265,opus" \
  pip install ./blazar/bindings/python
```

There is no separate `blazar-video-gpl` sdist, because there is no separate
source: a copy would be a second tree that could drift from the one that built
the wheel. Naming the tag is a stronger statement of what this binary was built
from than a tarball with the same contents.

The GPL components themselves are unmodified upstream releases, linked
dynamically and bundled by `auditwheel`:

| component | licence | source |
|---|---|---|
| x264 | GPL-2.0-or-later | <https://code.videolan.org/videolan/x264> |
| x265 | GPL-2.0-or-later | <https://bitbucket.org/multicoreware/x265_git> |
| SVT-AV1 | BSD-2-Clause-Patent | <https://gitlab.com/AOMediaCodec/SVT-AV1> |
| libvpx | BSD-3-Clause | <https://chromium.googlesource.com/webm/libvpx> |

Exact versions are those installed by the manylinux/macOS build image and are
recorded in the release notes for each tag.

Where we are required to provide corresponding source rather than point at
upstream, write to the address in the repository's `README.md` and it will be
supplied on the same medium the binary was distributed, as GPL §6 permits.

## What this means for you

- Installing `blazar-video-gpl` puts copyleft code in your process. Redistributing
  a product that includes it, or a binary built with `enc-x264`/`enc-x265`, brings
  the GPL's source obligations with it.
- If you need the encoders without those obligations, x264 and x265 are available
  commercially from their owners; that route lets you build with the GPL-free
  `blazar-video` and link your own licensed copies.
- H.264 and HEVC **patent** licensing (MPEG-LA / Access Advance) is a separate
  matter from the GPL and applies to commercial distribution regardless of which
  encoder produced the stream.
