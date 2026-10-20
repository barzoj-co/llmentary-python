#!/usr/bin/env python3
"""Download the PROXY_VERSION GoReleaser binary after verifying checksums.txt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_release_wheels import install_release_binary, read_proxy_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the extracted llmentary binary",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Release version without a v prefix (default: PROXY_VERSION)",
    )
    args = parser.parse_args()
    version = args.version or read_proxy_version()
    dest = Path(args.output).expanduser()
    installed = install_release_binary(dest=dest, version=version)
    print(f"installed {version} -> {installed}")


if __name__ == "__main__":
    main()
