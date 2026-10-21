from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, TextIO

from .binary import (
    BinaryResolutionError,
    bundled_proxy_version,
    contract_path,
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

    from . import __version__

    lines = [
        f"llmentary {__version__}",
        f"bundled proxy: {bundled_proxy_version()}",
        f"binary: {resolved if resolved is not None else 'not found'}",
        f"binary --version: {binary_version_output(resolved)}",
        f"platform tag: {detected_platform_tag()}",
        f"install: {install_kind()}",
        f"contract: {contract_path(env=env) or 'not bundled'}",
    ]
    stream.write("\n".join(lines) + "\n")
    if error is not None:
        stream.write("\n" + str(error) + "\n")
        return 1
    return 0


def proxy_binary_path(*, env: Optional[dict[str, str]] = None) -> Optional[Path]:
    """Resolved proxy binary, or ``None`` when none can be found."""

    try:
        return find_binary(env=env)
    except BinaryResolutionError:
        return None


def proxy_version(*, env: Optional[dict[str, str]] = None) -> str:
    """First line of ``llmentary --version``, or ``not found``."""

    path = proxy_binary_path(env=env)
    if path is None:
        return "not found"
    return binary_version_output(path)


def proxy_report_header(*, env: Optional[dict[str, str]] = None) -> str:
    """Single-line pytest header: client version, proxy version, binary path."""

    from . import __version__

    path = proxy_binary_path(env=env)
    if path is None:
        return f"llmentary {__version__}; proxy: not found; binary: not found"
    return (
        f"llmentary {__version__}; proxy: {binary_version_output(path)}; binary: {path}"
    )


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
