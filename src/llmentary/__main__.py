from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Optional

from .doctor import doctor


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["doctor"]:
        return doctor()
    sys.stderr.write("usage: python -m llmentary doctor\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
