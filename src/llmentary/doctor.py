from __future__ import annotations

import re
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

_SEMVER_TOKEN = re.compile(
    r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
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


def proxy_version_line(*, env: Optional[dict[str, str]] = None) -> Optional[str]:
    """First line of ``llmentary --version``, or ``None`` when unresolved."""

    path = proxy_binary_path(env=env)
    if path is None:
        return None
    text = binary_version_output(path)
    if text in {"", "unknown"}:
        return None
    return text


def proxy_version(*, env: Optional[dict[str, str]] = None) -> Optional[str]:
    """Proxy semver from ``llmentary --version``, or ``None`` when unresolved.

    The full ``--version`` line stays on :func:`proxy_version_line`.
    """

    return parse_proxy_semver(proxy_version_line(env=env))


def parse_proxy_semver(line: Optional[str]) -> Optional[str]:
    """Return the version token from a ``--version`` line, or ``None``."""

    if line is None:
        return None
    first = line.splitlines()[0].strip() if line else ""
    if not first or first in {"unknown", "not found"}:
        return None
    tokens = first.split()
    if tokens and tokens[0] == "llmentary":
        tokens = tokens[1:]
    if not tokens:
        return None
    token = tokens[0].lstrip("v").strip("(),")
    if not _SEMVER_TOKEN.fullmatch(token):
        return None
    return token


def _version_report(env: Optional[dict[str, str]]) -> tuple[str, Optional[str]]:
    from . import __version__

    path = proxy_binary_path(env=env)
    if path is None:
        return f"llmentary {__version__}; proxy not found", None
    line = binary_version_output(path)
    semver = parse_proxy_semver(line) or "unknown"
    prefix = f"llmentary {__version__}; proxy {semver} ({path})"
    if line in {"", "unknown", "not found", semver}:
        return prefix, None
    return prefix, line


def version_report_line(*, env: Optional[dict[str, str]] = None) -> str:
    """Shared version line used by ``--version`` and the pytest header.

    Found: ``llmentary <ver>; proxy <semver> (<path>)``
    Missing: ``llmentary <ver>; proxy not found``
    """

    prefix, _ = _version_report(env)
    return prefix


def proxy_report_header(*, env: Optional[dict[str, str]] = None) -> str:
    """Single-line pytest header: shared version line, then the full binary line."""

    prefix, line = _version_report(env)
    return f"{prefix}; {line}" if line is not None else prefix


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
