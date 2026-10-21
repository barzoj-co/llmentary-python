#!/usr/bin/env python3
"""Print the coverage fail-under threshold stored in pyproject.toml.

Regular PRs use ``[tool.llmentary.coverage] fail_under``. release-please PRs
use ``release_fail_under`` (Denis: releases require >=90).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def table_int(text: str, table: str, key: str) -> int:
    in_table = False
    header = f"[{table}]"
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_table = stripped == header
            continue
        if not in_table:
            continue
        name, sep, value = stripped.partition("=")
        if sep and name.strip() == key:
            return int(value.strip())
    raise SystemExit(f"missing [{table}] {key} in {PYPROJECT}")


def load_thresholds(path: Path = PYPROJECT) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    fail_under = table_int(text, "tool.llmentary.coverage", "fail_under")
    release_fail_under = table_int(text, "tool.llmentary.coverage", "release_fail_under")
    return fail_under, release_fail_under


def is_release_please(
    *,
    title: str = "",
    labels: str = "",
    head_ref: str = "",
    flag: str = "",
) -> bool:
    if flag.strip() in {"1", "true", "yes"}:
        return True
    if "autorelease: pending" in labels:
        return True
    if title.startswith("chore(main): release"):
        return True
    return head_ref.startswith("release-please--")


def fail_under_for(
    *,
    title: str = "",
    labels: str = "",
    head_ref: str = "",
    flag: str = "",
    path: Path = PYPROJECT,
) -> int:
    regular, release = load_thresholds(path)
    if is_release_please(title=title, labels=labels, head_ref=head_ref, flag=flag):
        return max(regular, release)
    return regular


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT,
        help="pyproject.toml to read (default: repository root)",
    )
    args = parser.parse_args(argv)
    env = os.environ
    value = fail_under_for(
        title=env.get("COVERAGE_PR_TITLE", ""),
        labels=env.get("COVERAGE_PR_LABELS", ""),
        head_ref=env.get("COVERAGE_HEAD_REF", ""),
        flag=env.get("COVERAGE_RELEASE_PLEASE", ""),
        path=args.pyproject,
    )
    sys.stdout.write(f"{value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
