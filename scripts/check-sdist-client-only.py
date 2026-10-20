#!/usr/bin/env python3
"""Assert a built sdist has no bundled proxy and find_binary explains why."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def sdist_has_bundled_bin(names: list[str]) -> bool:
    return any(
        "/llmentary/bin/" in name or name.endswith("/llmentary/bin") for name in names
    )


def assert_sdist_has_no_bin(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    if sdist_has_bundled_bin(names):
        bundled = [name for name in names if "llmentary/bin" in name]
        raise SystemExit(f"{sdist} unexpectedly contains bundled proxy files: {bundled}")
    return names


def assert_sdist_find_binary_message(sdist: Path, work: Path) -> str:
    target = work / "site"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(target),
            "--no-deps",
            "--quiet",
            str(sdist),
        ],
        check=True,
    )
    script = (
        "from llmentary.binary import BinaryResolutionError, find_binary\n"
        "try:\n"
        "    find_binary(env={'PATH': ''})\n"
        "except BinaryResolutionError as exc:\n"
        "    print(exc)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('expected BinaryResolutionError')\n"
    )
    child_env = os.environ.copy()
    child_env.pop("LLMENTARY_BIN", None)
    child_env["PYTHONPATH"] = str(target)
    child_env["PATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=child_env,
    )
    text = completed.stdout
    if completed.returncode != 0:
        raise SystemExit(
            f"sdist find_binary check failed:\n{completed.stdout}\n{completed.stderr}"
        )
    for needle in ("no bundled proxy", "--only-binary=:all:"):
        if needle not in text:
            raise SystemExit(f"sdist find_binary message missing {needle!r}:\n{text}")
    return text


def find_sdist(dist: Path) -> Path:
    matches = sorted(dist.glob("*.tar.gz"))
    if not matches:
        raise SystemExit(f"no sdist in {dist}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", nargs="?", default="dist", type=Path)
    parser.add_argument(
        "--install",
        action="store_true",
        help="pip install the sdist and assert find_binary() message",
    )
    args = parser.parse_args(argv)
    sdist = find_sdist(args.dist)
    assert_sdist_has_no_bin(sdist)
    print(f"{sdist.name} has no bundled proxy bin/")
    if args.install:
        work = Path(os.environ.get("TMPDIR", "/tmp")) / "llmentary-sdist-check"
        work.mkdir(parents=True, exist_ok=True)
        message = assert_sdist_find_binary_message(sdist, work)
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
