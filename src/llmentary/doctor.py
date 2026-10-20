from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, TextIO

from . import __version__
from .binary import (
    BinaryResolutionError,
    bundled_proxy_version,
    detected_platform_tag,
    find_binary,
    install_kind,
)


def doctor(*, env: Optional[dict[str, str]] = None, out: Optional[TextIO] = None) -> int:
    """Print install diagnostics. Exit 1 when no proxy binary can be resolved."""

    import sys

    stream = sys.stdout if out is None else out
    resolved: Optional[Path] = None
    error: Optional[BinaryResolutionError] = None
    try:
        resolved = find_binary(env=env)
    except BinaryResolutionError as exc:
        error = exc

    lines = [
        f"llmentary {__version__}",
        f"bundled proxy: {bundled_proxy_version()}",
        f"binary: {resolved if resolved is not None else 'not found'}",
        f"binary --version: {binary_version_output(resolved)}",
        f"platform tag: {detected_platform_tag()}",
        f"install: {install_kind()}",
    ]
    stream.write("\n".join(lines) + "\n")
    if error is not None:
        stream.write("\n" + str(error) + "\n")
        return 1
    return 0


def binary_version_output(path: Optional[Path]) -> str:
    if path is None:
        return "unknown"
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not text:
        return "unknown"
    return text.splitlines()[0]
