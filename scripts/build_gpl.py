#!/usr/bin/env python3
"""Build `blazar-video-gpl` with maturin, from a generated project.

maturin copies `python-source` only when the crate being built lives inside the
project, and it rejects `..` in that path. `packaging/gpl` therefore could not
build this distribution: it got the extension and no `blazar/__init__.py`.

Rewriting an Apache wheel afterwards works, but the archive PyPI received was
rejected with "ZIP archive not accepted: Trailing data" while the same file
passed `zipfile.testzip()` locally. Rather than argue with a rewriter, this
script assembles a project maturin is happy to build:

  * a scratch directory under `target/` (never `dist/`, never the source tree),
  * `bindings/python`'s crate, pyproject and Python package copied into it, so
    there is exactly one copy of each file and no drift,
  * the crate made standalone (`[workspace]`) with absolute paths to `crates/`,
    and the distribution's name and licence written into its pyproject.

The wheel maturin produces is then a normal wheel: no post-processing, no
recomputed `RECORD`, nothing for PyPI to disagree with.

Usage:
    scripts/build_gpl.py [--out dist] [--features a,b,c]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "bindings" / "python"
SCRATCH = ROOT / "target" / "gpl-build"
GPL_NAME = "blazar-video-gpl"
GPL_LICENSE = "GPL-3.0-or-later"
GPL_CLASSIFIER = "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
DEFAULT_FEATURES = "nvenc,gpu-preprocess,enc-svtav1,enc-vp9,enc-x264,enc-x265,opus"


def standalone_cargo_toml(text: str) -> str:
    """Make the copied crate independent of the workspace it came from.

    `bindings/python` is a workspace member; a copy elsewhere would either join
    the parent workspace by accident or refuse to build. Path dependencies are
    rewritten to absolute paths into this repository, so the copy needs no
    relative geometry of its own.
    """
    text = re.sub(
        r'path = "\.\./([^"]+)"',
        lambda m: f'path = "{(ROOT / "crates" / m.group(1)).as_posix()}"',
        text,
    )
    if "[workspace]" not in text:
        text += "\n# Standalone: this copy is not a member of the repository workspace.\n[workspace]\n"
    # The workspace's release profile does not travel with a copied member, and
    # a standalone build of this crate produced a Mach-O that dyld rejected
    # ("mis-aligned LINKEDIT string pool"). Build it the way the workspace does.
    profile = (ROOT / "Cargo.toml").read_text()
    if "[profile.release]" in profile:
        block = profile[profile.index("[profile.release]") :]
        block = block[: block.index("\n[")] if "\n[" in block else block
        text += "\n" + block.rstrip() + "\n"
    return text


def gpl_pyproject(text: str) -> str:
    text = text.replace('name = "blazar-video"', f'name = "{GPL_NAME}"')
    # PEP 639 form: a bare string is a licence expression, and maturin then
    # writes `License-Expression` instead of the legacy free-text `License:`
    # field, which is what a scanner reads.
    text = text.replace(
        'license = { text = "Apache-2.0" }', f'license = "{GPL_LICENSE}"'
    )
    text = text.replace(
        '"License :: OSI Approved :: Apache Software License",',
        f'"{GPL_CLASSIFIER}",',
    )
    # The notice has to travel in the wheel: it names the GPL libraries, their
    # licences and where their source lives.
    text = text.replace(
        "include = [",
        'include = [\n    { path = "NOTICE.md", format = "wheel" },',
        1,
    )
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(SOURCE / "dist"))
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--repair",
        action="store_true",
        default=sys.platform.startswith("linux"),
        help="bundle the GPL libraries into the wheel (default: on Linux only)",
    )
    args = parser.parse_args(argv[1:])

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "src").mkdir(parents=True)

    shutil.copy2(SOURCE / "Cargo.toml", SCRATCH / "Cargo.toml")
    (SCRATCH / "Cargo.toml").write_text(
        standalone_cargo_toml((SCRATCH / "Cargo.toml").read_text())
    )
    shutil.copytree(SOURCE / "src", SCRATCH / "src", dirs_exist_ok=True)
    shutil.copytree(SOURCE / "python", SCRATCH / "python", dirs_exist_ok=True)
    shutil.copy2(SOURCE / "pyproject.toml", SCRATCH / "pyproject.toml")
    (SCRATCH / "pyproject.toml").write_text(
        gpl_pyproject((SCRATCH / "pyproject.toml").read_text())
    )
    shutil.copy2(ROOT / "packaging" / "gpl" / "NOTICE.md", SCRATCH / "NOTICE.md")
    # The Apache pyproject points `readme` at README.md next to it; the GPL
    # distribution gets its own, which says what it is and when to install it.
    shutil.copy2(ROOT / "packaging" / "gpl" / "README.md", SCRATCH / "README.md")
    # Stale bytecode from a local run must not end up in a wheel.
    for cache in SCRATCH.rglob("__pycache__"):
        shutil.rmtree(cache)

    # Absolute, because maturin runs with cwd=SCRATCH: a relative --out would
    # write the wheel inside the scratch directory and the cleanup would take it
    # with it. That is exactly what the first container run did.
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "maturin", "build", "--release",
        "--features", args.features,
        "--out", str(out),
    ]
    if args.repair:
        # Bundling the GPL libraries is what makes the wheel self-contained, and
        # on Linux auditwheel does it without touching the objects. Delocate's
        # macOS equivalent rewrites the Mach-O paths in place and produced a
        # binary that dyld rejects ("mis-aligned LINKEDIT string pool"), so on
        # macOS the wheel is left as built and the libraries come from the user's
        # own install.
        command += ["--auditwheel", "repair"]
    print("+", " ".join(command), f"(in {SCRATCH})", file=sys.stderr)
    result = subprocess.run(command, cwd=SCRATCH)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    if result.returncode == 0:
        wheels = sorted(out.glob("*.whl"))
        if not wheels:
            print(f"maturin reported success but {out} has no wheel", file=sys.stderr)
            return 1
        for wheel in wheels:
            print(wheel)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
