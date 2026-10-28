#!/usr/bin/env python3
"""Check that every place carrying a version agrees.

Four files state it — the cargo workspace, the binding crate, and two
distributions — plus the changelog must have a section for it. They are edited
by hand at release time, which is exactly when nobody wants to debug a wheel
whose filename disagrees with its metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def table_version(path: Path, table: str) -> str:
    """First `version = "..."` inside `[table]`.

    Parsed by hand rather than with tomllib: CI images here still carry Python
    3.10, and this script is the check that must never be the reason a release
    workflow cannot run.
    """
    text = path.read_text(encoding="utf-8")
    header = f"[{table}]"
    start = text.index(header) + len(header)
    end = text.find("\n[", start)
    section = text[start : end if end != -1 else len(text)]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit(f"{path}: no version under {header}")
    return match.group(1)


def main() -> int:
    sources: dict[str, str] = {
        "Cargo.toml [workspace.package]": table_version(
            ROOT / "Cargo.toml", "workspace.package"
        ),
        "bindings/python/Cargo.toml": table_version(
            ROOT / "bindings/python/Cargo.toml", "package"
        ),
        "bindings/python/pyproject.toml": table_version(
            ROOT / "bindings/python/pyproject.toml", "project"
        ),
        "packaging/gpl/pyproject.toml": table_version(
            ROOT / "packaging/gpl/pyproject.toml", "project"
        ),
    }

    failures: list[str] = []
    versions = set(sources.values())
    if len(versions) != 1:
        failures.append("versions disagree:\n" + "\n".join(
            f"  {name}: {version}" for name, version in sources.items()
        ))
    version = next(iter(versions))

    if version == "0.0.0":
        # 0.0.0 is the not-yet-released sentinel: every file agrees, but there is
        # no release to describe. Bumping to 0.1.0 is what turns the changelog
        # check on.
        print("version 0.0.0 agrees (unreleased); no changelog section required")
        return 0

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    linked = re.search(rf"^## \[{re.escape(version)}\]", changelog, re.MULTILINE)
    if not linked:
        failures.append(
            f"CHANGELOG.md has no '## [{version}]' section; "
            "the release workflow uses it for the notes"
        )
    elif "TBD" in changelog[linked.start() : changelog.find("\n## ", linked.start() + 1)]:
        # Not a failure: tagging with TBD left in is a choice, but it should be
        # loud rather than a date nobody replaced.
        print(f"note: CHANGELOG.md section [{version}] still says TBD")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"version {version} agrees across {len(sources)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
