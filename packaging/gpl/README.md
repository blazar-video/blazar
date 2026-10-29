# blazar-video-gpl

The same library as `blazar-video`, built with the GPL encoders included:

| | `blazar-video` | `blazar-video-gpl` |
|---|---|---|
| Licence | Apache-2.0 | GPL-3.0-or-later |
| NVENC (H.264, HEVC, AV1) | yes | yes |
| SVT-AV1, VP9 | yes | yes |
| Opus audio | yes | yes |
| x264 (H.264) | no | yes |
| x265 (HEVC) | no | yes |

Install this one when you need x264 or x265 and can accept the GPL. Install
`blazar-video` when you cannot: it never pulls copyleft transitively, and the two
distributions cannot both be installed at once.

x264 and x265 are GPL-2.0-or-later. Apache-2.0 code can be combined into a
GPLv3 work but not a GPLv2-only one, which is why this distribution is
GPL-3.0-or-later.

The usage documentation is the main project's:
https://github.com/blazar-video/blazar

The Python API is identical; only `blazar.backends()` differs.
