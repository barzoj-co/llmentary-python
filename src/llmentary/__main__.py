from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Optional, TextIO

from .doctor import doctor, version_report_line

_USAGE = "usage: python -m llmentary doctor | --version | version\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--version"], ["version"]):
        return print_version()
    if args == ["doctor"]:
        return int(doctor())
    sys.stderr.write(_USAGE)
    return 2


def print_version(*, out: Optional[TextIO] = None) -> int:
    """Print the client and bundled proxy versions. Always exits 0."""

    stream = sys.stdout if out is None else out
    stream.write(version_report_line() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
