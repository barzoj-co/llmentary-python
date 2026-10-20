from __future__ import annotations

import os
import platform
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

BINARIES_DOC = "https://github.com/barzoj-co/llmentary/blob/main/docs/binaries.md"
SUPPORTED_WHEEL_TAGS: tuple[str, ...] = (
    "manylinux2014_x86_64",
    "manylinux2014_aarch64",
    "macosx_11_0_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
)


class BinaryResolutionError(RuntimeError):
    """Raised when the llmentary proxy binary cannot be located."""


def find_binary(*, env: Optional[dict[str, str]] = None) -> Path:
    """Resolve the llmentary Go proxy binary.

    Lookup order:

    1. ``LLMENTARY_BIN``
    2. Binary bundled in a platform-specific wheel
    3. ``llmentary`` on ``PATH``

    Runtime downloads are disabled. Platform wheels ship a version-pinned copy of
    the Go proxy. The bundled proxy version is recorded in the wheel ``VERSION``
    file and pinned via ``PROXY_VERSION``; it is not required to equal this
    Python package version.
    """

    env_map: Mapping[str, str] = env if env is not None else os.environ
    configured = env_map.get("LLMENTARY_BIN")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return ensure_executable(path)
        raise BinaryResolutionError(f"LLMENTARY_BIN points to a missing file: {path}")

    bundled = bundled_binary()
    if bundled is not None:
        return ensure_executable(bundled)

    path_value = shutil.which("llmentary", path=env_map.get("PATH"))
    if path_value:
        return Path(path_value)

    raise BinaryResolutionError(missing_binary_message())


def missing_binary_message() -> str:
    tag = detected_platform_tag()
    if bundled_platform_tag() is None:
        situation = f"no wheel for platform `{tag}`"
    else:
        situation = (
            "this llmentary install has no bundled proxy "
            "(installed from sdist or a source tree)"
        )
    tags = ", ".join(SUPPORTED_WHEEL_TAGS)
    return (
        "Could not find the llmentary proxy binary.\n"
        f"{situation}\n"
        f"detected platform tag: {tag}\n"
        f"published wheel tags: {tags}\n"
        "\n"
        "Install a platform wheel, or point at a binary:\n"
        "1. pip install --only-binary=:all: llmentary\n"
        "2. LLMENTARY_BIN=/path/to/llmentary\n"
        "3. Get the binary from npm @llmentary/<os>-<cpu> or the GitHub Release:\n"
        f"   {BINARIES_DOC}\n"
        "\n"
        "Runtime downloads are disabled."
    )


def package_bin_dir() -> Path:
    return Path(__file__).resolve().parent / "bin"


def bundled_binary() -> Optional[Path]:
    platform_tag = bundled_platform_tag()
    if platform_tag is None:
        return None
    candidate = package_bin_dir() / platform_tag / binary_name()
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


def detected_platform_tag() -> str:
    tag = bundled_platform_tag()
    if tag is not None:
        return tag
    goos, goarch = goreleaser_platform()
    return f"{goos}_{goarch}"


def install_kind() -> str:
    if bundled_binary() is not None:
        return "wheel"
    return "sdist or source tree"


def bundled_proxy_version() -> str:
    bundled = bundled_binary()
    if bundled is None:
        return "unknown"
    version_file = bundled.parent / "VERSION"
    if not version_file.is_file():
        return "unknown"
    text = version_file.read_text(encoding="utf-8").strip()
    return text or "unknown"


def binary_name() -> str:
    return "llmentary.exe" if os.name == "nt" else "llmentary"


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


def sdist_has_bundled_bin(names: Sequence[str]) -> bool:
    return any(
        "/llmentary/bin/" in name or name.endswith("/llmentary/bin") for name in names
    )
