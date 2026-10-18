from __future__ import annotations

import os
import platform
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Optional


class BinaryResolutionError(RuntimeError):
    """Raised when the llm-test-lab proxy binary cannot be located."""


def find_binary(*, env: Optional[dict[str, str]] = None) -> Path:
    """Resolve the llm-test-lab Go proxy binary.

    Lookup order:

    1. ``LLMTESTLAB_BIN``
    2. Binary bundled in a platform-specific wheel
    3. ``llmtestlab`` on ``PATH``

    Runtime downloads are disabled. Platform wheels ship a version-pinned copy of
    the Go proxy that matches this Python package version.
    """

    env_map: Mapping[str, str] = env if env is not None else os.environ
    configured = env_map.get("LLMTESTLAB_BIN")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return ensure_executable(path)
        raise BinaryResolutionError(f"LLMTESTLAB_BIN points to a missing file: {path}")

    bundled = bundled_binary()
    if bundled is not None:
        return ensure_executable(bundled)

    path_value = shutil.which("llmtestlab", path=env_map.get("PATH"))
    if path_value:
        return Path(path_value)

    raise BinaryResolutionError(
        "Could not find llm-test-lab proxy binary. Set LLMTESTLAB_BIN, install a "
        "platform-specific llm-test-lab wheel that bundles the proxy, or put "
        "'llmtestlab' on PATH. Runtime downloads are disabled."
    )


def bundled_binary() -> Optional[Path]:
    platform_tag = bundled_platform_tag()
    if platform_tag is None:
        return None
    candidate = Path(__file__).resolve().parent / "bin" / platform_tag / binary_name()
    return candidate if candidate.is_file() else None


def bundled_platform_tag() -> Optional[str]:
    goos, goarch = goreleaser_platform()
    mapping = {
        ("linux", "amd64"): "manylinux2014_x86_64",
        ("linux", "arm64"): "manylinux2014_aarch64",
        ("darwin", "amd64"): "macosx_11_0_x86_64",
        ("darwin", "arm64"): "macosx_11_0_arm64",
        ("windows", "amd64"): "win_amd64",
    }
    return mapping.get((goos, goarch))


def binary_name() -> str:
    return "llmtestlab.exe" if os.name == "nt" else "llmtestlab"


def goreleaser_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    goos = {
        "darwin": "darwin",
        "linux": "linux",
        "windows": "windows",
    }.get(system, system)
    goarch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    return goos, goarch


def ensure_executable(path: Path) -> Path:
    if os.name == "nt":
        return path
    mode = path.stat().st_mode
    if mode & stat.S_IXUSR:
        return path
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
