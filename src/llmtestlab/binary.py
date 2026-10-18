from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BinaryResolutionError(RuntimeError):
    """Raised when the llm-test-lab proxy binary cannot be located."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str


def find_binary(
    *,
    env: Optional[dict[str, str]] = None,
    cache_dir: Optional[Path] = None,
    download: bool = False,
    version: Optional[str] = None,
    github_token: Optional[str] = None,
) -> Path:
    """Find the llm-test-lab Go proxy binary.

    Resolution order mirrors the Go client documentation requested for Python:
    ``LLMTESTLAB_BIN``, then ``PATH``, then the local cache. Release download is
    opt-in because the upstream repository is private.
    """

    env_map: Mapping[str, str] = env if env is not None else os.environ
    configured = env_map.get("LLMTESTLAB_BIN")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        raise BinaryResolutionError(f"LLMTESTLAB_BIN points to a missing file: {path}")

    path_value = shutil.which("llmtestlab", path=env_map.get("PATH"))
    if path_value:
        return Path(path_value)

    root = cache_dir or default_cache_dir()
    cached = cached_binary(root, version=version)
    if cached:
        return cached

    if download:
        return download_binary(
            cache_dir=root,
            version=version or env_map.get("LLMTESTLAB_VERSION") or "latest",
            github_token=github_token or env_map.get("GITHUB_TOKEN"),
        )

    raise BinaryResolutionError(
        "Could not find llm-test-lab proxy binary. Set LLMTESTLAB_BIN, put "
        "'llmtestlab' on PATH, pre-populate the cache directory "
        f"({root}), or enable opt-in download with LLMTESTLAB_DOWNLOAD=1. "
        "Downloads use GitHub Releases from private repo barzoj-co/llm-test-lab "
        "and require credentials with read access."
    )


def default_cache_dir() -> Path:
    override = os.environ.get("LLMTESTLAB_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "llm-test-lab"


def cached_binary(cache_dir: Path, *, version: Optional[str]) -> Optional[Path]:
    candidates: Iterable[Path]
    if version:
        candidates = [cache_dir / version / binary_name()]
    else:
        candidates = sorted(cache_dir.glob(f"*/{binary_name()}"), reverse=True)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def download_binary(
    *,
    cache_dir: Path,
    version: str = "latest",
    github_token: Optional[str] = None,
) -> Path:
    release = fetch_release(version=version, github_token=github_token)
    tag = str(release.get("tag_name") or version)
    raw_assets = release.get("assets")
    asset_items = raw_assets if isinstance(raw_assets, list) else []
    assets = [
        ReleaseAsset(name=str(asset["name"]), url=str(asset["browser_download_url"]))
        for asset in asset_items
        if is_asset_mapping(asset)
    ]
    checksum_asset = next((asset for asset in assets if "checksum" in asset.name.lower()), None)
    if checksum_asset is None:
        raise BinaryResolutionError(
            f"Release {tag} has no checksum asset. Refusing to download llm-test-lab without "
            "checksum verification."
        )

    checksums = parse_checksums(download_bytes(checksum_asset.url, github_token=github_token))
    wanted = select_archive_asset(assets)
    expected = checksums.get(wanted.name)
    if not expected:
        raise BinaryResolutionError(
            f"Release {tag} checksum file does not contain selected asset {wanted.name}."
        )

    archive_bytes = download_bytes(wanted.url, github_token=github_token)
    actual = hashlib.sha256(archive_bytes).hexdigest()
    if actual.lower() != expected.lower():
        raise BinaryResolutionError(
            f"Checksum mismatch for {wanted.name}: expected {expected}, got {actual}."
        )

    install_dir = cache_dir / tag
    install_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        extracted = extract_binary(archive_bytes, asset_name=wanted.name, destination=Path(tmp))
        target = install_dir / binary_name()
        shutil.copy2(extracted, target)
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return target


def fetch_release(*, version: str, github_token: Optional[str]) -> dict[str, object]:
    suffix = "latest" if version == "latest" else f"tags/{version}"
    url = f"https://api.github.com/repos/barzoj-co/llm-test-lab/releases/{suffix}"
    try:
        data = download_bytes(url, github_token=github_token)
    except BinaryResolutionError:
        raise
    import json

    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise BinaryResolutionError("Unexpected GitHub release response while downloading binary.")
    return parsed


def download_bytes(url: str, *, github_token: Optional[str]) -> bytes:
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "llm-test-lab-python",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return cast(bytes, response.read())
    except HTTPError as exc:
        if exc.code in {401, 403, 404}:
            raise BinaryResolutionError(
                "Could not download llm-test-lab from GitHub Releases. The upstream "
                "repository is private; provide GITHUB_TOKEN with read access or set "
                "LLMTESTLAB_BIN."
            ) from exc
        raise BinaryResolutionError(
            f"GitHub release download failed with HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise BinaryResolutionError(f"GitHub release download failed: {exc.reason}") from exc


def parse_checksums(data: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, name = parts
        checksums[name.lstrip("*")] = checksum
    return checksums


def is_asset_mapping(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("name") is not None
        and value.get("browser_download_url") is not None
    )


def select_archive_asset(assets: Iterable[ReleaseAsset]) -> ReleaseAsset:
    goos, goarch = goreleaser_platform()
    patterns = [
        re.compile(rf"llmtestlab.*{goos}.*{goarch}.*\.(tar\.gz|zip)$", re.IGNORECASE),
        re.compile(rf"llm-test-lab.*{goos}.*{goarch}.*\.(tar\.gz|zip)$", re.IGNORECASE),
        re.compile(rf".*{goos}.*{goarch}.*\.(tar\.gz|zip)$", re.IGNORECASE),
    ]
    candidates = [asset for asset in assets if not asset.name.lower().endswith(".txt")]
    for pattern in patterns:
        for candidate in candidates:
            if pattern.fullmatch(candidate.name):
                return candidate
    names = ", ".join(asset.name for asset in candidates) or "<none>"
    raise BinaryResolutionError(
        f"No llm-test-lab release asset matched platform {goos}/{goarch}. Assets: {names}"
    )


def extract_binary(data: bytes, *, asset_name: str, destination: Path) -> Path:
    archive = destination / asset_name
    archive.write_bytes(data)
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    else:
        with tarfile.open(archive) as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(destination, filter="data")
            else:
                tf.extractall(destination)
    matches = [path for path in destination.rglob(binary_name()) if path.is_file()]
    if not matches:
        raise BinaryResolutionError(f"Archive {asset_name} did not contain {binary_name()}.")
    return matches[0]


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
