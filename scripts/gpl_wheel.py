#!/usr/bin/env python3
"""Rewrite an Apache wheel into the GPL distribution.

maturin will not build the GPL distribution directly. Its `python-source`
handling copies the Python package only when the crate being built lives inside
the project; `packaging/gpl` points `manifest-path` at `bindings/python`, and in
that case maturin emits the extension and silently omits `blazar/__init__.py`.
Three layouts were tried — a copy of the package, a symlinked directory, a
symlinked crate — and all three produced an extension with no package around it.

So the GPL wheel is built once, from the same crate and the same Python package,
and this script rewrites what makes it a different distribution:

  * the filename and the ``.dist-info`` directory,
  * ``Name``, ``License-Expression`` and the licence classifier in ``METADATA``,
  * ``NOTICE.md``, which names the GPL libraries and where their source lives,
  * ``RECORD``, recomputed because every hash above changed.

Nothing else differs: the same code, the same encoder set plus x264/x265. That
is the point of the distribution, and doing it as a rename keeps one crate and
one Python package rather than two that drift.

Usage:
    scripts/gpl_wheel.py build/blazar_video-0.1.0-cp312-...whl
    scripts/gpl_wheel.py dist/*.whl        # rewrites each file in place

Exits non-zero if a wheel already looks like the GPL distribution, so a rerun
cannot double-rename.
"""

from __future__ import annotations

import base64
import hashlib
import sys
import zipfile
from pathlib import Path

APACHE_NAME = "blazar-video"
GPL_NAME = "blazar-video-gpl"
NOTICE = Path(__file__).resolve().parent.parent / "packaging" / "gpl" / "NOTICE.md"


def digest(data: bytes) -> str:
    """The `RECORD` hash form: sha256, urlsafe base64, no padding."""
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    )


def rewrite_metadata(text: str, version: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("Name: "):
            out.append(f"Name: {GPL_NAME}")
        elif line.startswith("License-Expression: "):
            out.append("License-Expression: GPL-3.0-or-later")
        elif line.startswith("License :: "):
            continue
        elif line.startswith("License: "):
            # The legacy single-string field: maturin writes the Apache licence
            # here even when a License-Expression is present, and a wheel that
            # says both Apache and GPL is worse than one that says neither.
            continue
        elif line.startswith("Classifier: License :: "):
            out.append("Classifier: License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)")
        else:
            out.append(line)
    body = "\n".join(out)
    if "License-Expression:" not in body:
        body = body.replace(
            f"Name: {GPL_NAME}\n",
            f"Name: {GPL_NAME}\nLicense-Expression: GPL-3.0-or-later\n",
            1,
        )
    return body + "\n"


def convert(wheel: Path) -> Path:
    if GPL_NAME.replace("-", "_") in wheel.name:
        raise SystemExit(f"{wheel.name} is already the GPL distribution")

    target = wheel.with_name(wheel.name.replace("blazar_video-", "blazar_video_gpl-", 1))
    version = wheel.name.split("-")[1]

    with zipfile.ZipFile(wheel) as src:
        entries = [(i, src.read(i.filename)) for i in src.infolist()]

    renamed: list[tuple[str, bytes]] = []
    dist_info = None
    for info, data in entries:
        name = info.filename
        if name.endswith(".dist-info/METADATA"):
            dist_info = name.rsplit("/METADATA", 1)[0]
            data = rewrite_metadata(data.decode(), version).encode()
        name = name.replace("blazar_video-", "blazar_video_gpl-", 1)
        renamed.append((name, data))

    if dist_info is None:
        raise SystemExit(f"{wheel.name} has no METADATA")
    new_dist_info = dist_info.replace("blazar_video-", "blazar_video_gpl-", 1)

    renamed.append((f"{new_dist_info}/NOTICE.md", NOTICE.read_bytes()))

    # RECORD lists every other file with its hash; rebuild it from the data we
    # are about to write, and give it an empty hash of its own, as the format
    # requires.
    record_path = f"{new_dist_info}/RECORD"
    lines = [
        f"{name},{digest(data)},{len(data)}"
        for name, data in renamed
        if name != record_path
    ]
    lines.append(f"{record_path},,")
    renamed = [(n, d) for n, d in renamed if n != record_path]
    renamed.append((record_path, ("\n".join(lines) + "\n").encode()))

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in renamed:
            out.writestr(name, data)

    wheel.unlink()
    return target


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().split("Usage:")[1].strip(), file=sys.stderr)
        return 2
    for arg in argv[1:]:
        produced = convert(Path(arg))
        print(produced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
