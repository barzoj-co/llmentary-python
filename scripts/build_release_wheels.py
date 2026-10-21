"""Build version-pinned platform wheels that bundle verified GoReleaser binaries.

The Python package version is expected to already match the Go proxy release
tag (see .github/workflows/release.yml). Each wheel contains exactly one
platform binary copied from GitHub Release assets after sha256 verification
against checksums.txt. Runtime clients never download binaries.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform as py_platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

REPO = "barzoj-co/llmentary"
REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_BIN_DIR = Path("src/llmentary/bin")
AGPL_LICENSE_FILE = REPO_ROOT / "licenses" / "AGPL-3.0.txt"
PLATFORM_WHEEL_LICENSE = "MIT AND AGPL-3.0-only"
AGPL_LICENSE_FILE_FIELD = "License-File: licenses/AGPL-3.0.txt"


@dataclass(frozen=True)
class Target:
    goos: str
    arch_aliases: tuple[str, ...]
    wheel_tag: str
    binary_name: str


TARGETS = [
    Target("linux", ("amd64", "x86_64"), "manylinux2014_x86_64", "llmentary"),
    Target("linux", ("arm64", "aarch64"), "manylinux2014_aarch64", "llmentary"),
    Target("darwin", ("amd64", "x86_64"), "macosx_11_0_x86_64", "llmentary"),
    Target("darwin", ("arm64", "aarch64"), "macosx_11_0_arm64", "llmentary"),
    Target("windows", ("amd64", "x86_64"), "win_amd64", "llmentary.exe"),
]


@dataclass(frozen=True)
class Asset:
    name: str
    url: str


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise HTTPError(req.full_url, code, msg, headers, fp)


def main() -> None:
    version = os.environ.get("LLMENTARY_RELEASE_VERSION")
    if not version:
        raise SystemExit("LLMENTARY_RELEASE_VERSION is required")
    token = os.environ.get("LLMENTARY_REPO_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("LLMENTARY_REPO_TOKEN or GITHUB_TOKEN is required")

    release = fetch_release(version, token)
    assets = release_assets(release)
    checksum_asset = select_checksum_asset(assets)
    checksums = parse_checksums(download_asset(checksum_asset.url, token))

    dist = Path("dist")
    raw = dist / "_raw"
    shutil.rmtree(dist, ignore_errors=True)
    shutil.rmtree(PACKAGE_BIN_DIR, ignore_errors=True)
    dist.mkdir()

    subprocess.run([sys.executable, "-m", "build", "--sdist", "--outdir", str(dist)], check=True)

    for target in TARGETS:
        asset = select_asset(assets, target)
        archive = download_asset(asset.url, token)
        verify_asset_checksum(asset.name, archive, checksums)

        shutil.rmtree(PACKAGE_BIN_DIR, ignore_errors=True)
        installed = install_binary(archive, asset.name, target, version=version)
        if host_can_execute(target):
            assert_clean_release_version(installed, version)
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(raw)], check=True)
        wheel = one(raw.glob("*.whl"))
        tagged = retag_wheel(wheel, target.wheel_tag, dist)
        apply_platform_wheel_license(tagged)

    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(PACKAGE_BIN_DIR, ignore_errors=True)
    assert_platform_wheels(dist)
    print_built_artifacts(dist)


def fetch_release(version: str, token: str) -> dict[str, Any]:
    tag = version if version.startswith("v") else f"v{version}"
    data = download_bytes(
        f"https://api.github.com/repos/{REPO}/releases/tags/{tag}",
        token=token,
        accept="application/vnd.github+json",
        follow_redirects=False,
    )
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit("Unexpected GitHub release response")
    return parsed


def release_assets(release: dict[str, Any]) -> list[Asset]:
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        return []
    assets = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            continue
        name = raw_asset.get("name")
        url = raw_asset.get("url")
        if isinstance(name, str) and isinstance(url, str):
            assets.append(Asset(name=name, url=url))
    return assets


def select_checksum_asset(assets: list[Asset]) -> Asset:
    matches = [asset for asset in assets if Path(asset.name).name.lower() == "checksums.txt"]
    if not matches:
        raise SystemExit("Release has no checksums.txt asset")
    return matches[0]


def select_asset(assets: list[Asset], target: Target) -> Asset:
    candidates = [
        asset
        for asset in assets
        if re.search(r"\.(tar\.gz|zip)$", asset.name, flags=re.IGNORECASE)
    ]
    for asset in candidates:
        name = Path(asset.name).name.lower()
        if not name.startswith("llmentary_"):
            continue
        if target.goos in name and any(alias in name for alias in target.arch_aliases):
            return asset
    names = ", ".join(asset.name for asset in candidates)
    raise SystemExit(f"No release asset matched {target.goos}/{target.arch_aliases}: {names}")


def download_asset(url: str, token: str) -> bytes:
    return download_bytes(
        url,
        token=token,
        accept="application/octet-stream",
        follow_redirects=False,
    )


def download_bytes(url: str, *, token: Optional[str], accept: str, follow_redirects: bool) -> bytes:
    headers = {"Accept": accept, "User-Agent": "llmentary-python-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        opener = None if follow_redirects else build_opener(NoRedirectHandler)
        open_url = urlopen if opener is None else opener.open
        with open_url(request, timeout=60) as response:
            return cast(bytes, response.read())
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308} and not follow_redirects:
            location = exc.headers.get("Location")
            if not location:
                raise SystemExit("GitHub release asset redirect had no Location") from exc
            next_token = token if is_github_api_url(location) else None
            return download_bytes(
                location,
                token=next_token,
                accept="application/octet-stream",
                follow_redirects=not next_token,
            )
        raise SystemExit(f"Download failed for {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise SystemExit(f"Download failed for {url}: {exc.reason}") from exc


def parse_checksums(data: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("*")] = parts[0]
    return checksums


def install_binary(archive: bytes, asset_name: str, target: Target, *, version: str) -> Path:
    destination = PACKAGE_BIN_DIR / target.wheel_tag
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    target_path = destination / target.binary_name
    extract_binary(archive, asset_name, target_path, target.binary_name)
    if not AGPL_LICENSE_FILE.is_file():
        raise SystemExit("Missing licenses/AGPL-3.0.txt")
    shutil.copy2(AGPL_LICENSE_FILE, destination / "AGPL-3.0.txt")
    (destination / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    return target_path


def extract_binary(archive: bytes, asset_name: str, dest: Path, binary_name: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / Path(asset_name).name
        archive_path.write_bytes(archive)
        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                safe_extract_zip(zf, tmp_path)
        else:
            with tarfile.open(archive_path) as tf:
                safe_extract_tar(tf, tmp_path)
        binary = find_extracted_binary(tmp_path, binary_name)
        if not binary.resolve().is_relative_to(tmp_path.resolve()):
            raise SystemExit("Refusing to install a binary outside the extract directory")
        shutil.copy2(binary, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        contract = find_extracted_named_file(tmp_path, "openapi.yaml")
        if contract is not None:
            if not contract.resolve().is_relative_to(tmp_path.resolve()):
                raise SystemExit("Refusing to install openapi.yaml outside the extract directory")
            shutil.copy2(contract, dest.parent / "openapi.yaml")
    return dest


def read_proxy_version(path: Path = REPO_ROOT / "PROXY_VERSION") -> str:
    version = path.read_text(encoding="utf-8").strip()
    if not version:
        raise SystemExit("PROXY_VERSION is empty")
    return version


def host_platform() -> tuple[str, str]:
    system = py_platform.system().lower()
    machine = py_platform.machine().lower()
    goos = {"darwin": "darwin", "linux": "linux", "windows": "windows"}.get(system, system)
    goarch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    return goos, goarch


def target_for_platform(goos: str, goarch: str) -> Target:
    for target in TARGETS:
        if target.goos == goos and goarch in target.arch_aliases:
            return target
    raise SystemExit(f"No bundled wheel target for {goos}/{goarch}")


def host_can_execute(target: Target) -> bool:
    goos, goarch = host_platform()
    return target.goos == goos and goarch in target.arch_aliases


def assert_clean_release_version(binary: Path, version: str) -> None:
    """Fail unless ``binary --version`` reports ``version`` with no ``+dirty``."""

    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"failed to run {binary} --version: {exc}") from exc
    text = (completed.stdout or completed.stderr).strip()
    first = text.splitlines()[0] if text else ""
    if completed.returncode != 0 or not first:
        raise SystemExit(f"{binary} --version failed: {text or completed.returncode}")
    if "+dirty" in first:
        raise SystemExit(f"{binary} --version is dirty: {first}")
    expected = version[1:] if version.startswith("v") else version
    tokens = first.split()
    reported = tokens[1] if len(tokens) > 1 else ""
    if reported.lstrip("v") != expected:
        raise SystemExit(f"{binary} --version {first!r} does not report {expected}")


def verify_asset_checksum(name: str, data: bytes, checksums: dict[str, str]) -> None:
    expected = checksums.get(name)
    if not expected:
        raise SystemExit(f"Checksum file has no entry for {name}")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected.lower():
        raise SystemExit(f"Checksum mismatch for {name}: expected {expected}, got {actual}")


def install_release_binary(
    *,
    dest: Path,
    version: Optional[str] = None,
    token: Optional[str] = None,
    goos: Optional[str] = None,
    goarch: Optional[str] = None,
) -> Path:
    pin = version or read_proxy_version()
    secret = token or os.environ.get("LLMENTARY_REPO_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not secret:
        raise SystemExit("LLMENTARY_REPO_TOKEN or GITHUB_TOKEN is required")
    if goos is None or goarch is None:
        host_goos, host_goarch = host_platform()
        goos = goos or host_goos
        goarch = goarch or host_goarch
    target = target_for_platform(goos, goarch)
    release = fetch_release(pin, secret)
    assets = release_assets(release)
    checksums = parse_checksums(download_asset(select_checksum_asset(assets).url, secret))
    asset = select_asset(assets, target)
    archive = download_asset(asset.url, secret)
    verify_asset_checksum(asset.name, archive, checksums)
    extract_binary(archive, asset.name, dest, target.binary_name)
    if host_can_execute(target):
        assert_clean_release_version(dest, pin)
    return dest


def is_github_api_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() == "api.github.com"


def safe_extract_zip(zf: zipfile.ZipFile, destination: Path) -> None:
    dest = destination.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not target.is_relative_to(dest):
            raise SystemExit(f"Unsafe path in archive: {info.filename}")
    zf.extractall(destination)


def safe_extract_tar(tf: tarfile.TarFile, destination: Path) -> None:
    dest = destination.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not target.is_relative_to(dest):
            raise SystemExit(f"Unsafe path in archive: {member.name}")
        if not (member.isdir() or member.isreg()):
            raise SystemExit(f"Unsupported member in archive: {member.name}")
    if sys.version_info >= (3, 12):
        tf.extractall(destination, filter="data")
    else:
        tf.extractall(destination)


def find_extracted_binary(root: Path, binary_name: str) -> Path:
    matches = [path for path in root.rglob(binary_name) if path.is_file()]
    if not matches:
        raise SystemExit(f"Archive did not contain {binary_name}")
    return matches[0]


def find_extracted_named_file(root: Path, name: str) -> Optional[Path]:
    matches = [path for path in root.rglob(name) if path.is_file() and path.name == name]
    return matches[0] if matches else None


def rewrite_wheel_license_metadata(metadata: str) -> str:
    if re.search(r"^License-Expression:", metadata, flags=re.MULTILINE):
        metadata = re.sub(
            r"^License-Expression:.*$",
            f"License-Expression: {PLATFORM_WHEEL_LICENSE}",
            metadata,
            count=1,
            flags=re.MULTILINE,
        )
    elif re.search(r"^License:", metadata, flags=re.MULTILINE):
        metadata = re.sub(
            r"^License:.*$",
            f"License: {PLATFORM_WHEEL_LICENSE}",
            metadata,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        metadata = metadata.rstrip() + f"\nLicense-Expression: {PLATFORM_WHEEL_LICENSE}\n"

    metadata = "".join(
        line
        for line in metadata.splitlines(keepends=True)
        if not line.startswith("Classifier: License ::")
    )
    if AGPL_LICENSE_FILE_FIELD not in metadata:
        prefixes = (
            ("License-File:",)
            if "License-File:" in metadata
            else ("License-Expression:", "License:")
        )
        metadata = _insert_metadata_line(metadata, AGPL_LICENSE_FILE_FIELD, prefixes)
    return metadata


def _insert_metadata_line(metadata: str, line: str, prefixes: tuple[str, ...]) -> str:
    lines = metadata.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    indexes = [
        index for index, current in enumerate(lines) if current.startswith(prefixes)
    ]
    if indexes:
        insert_at = indexes[-1] + 1
    else:
        insert_at = next(
            (index for index, current in enumerate(lines) if current.strip() == ""),
            len(lines),
        )
    lines.insert(insert_at, line + "\n")
    return "".join(lines)


def wheel_record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def wheel_license_path(dist_info: str, license_file: str) -> str:
    """Return the PEP 639 wheel path for a License-File entry.

    Core metadata stores the project-root-relative path (``licenses/AGPL-3.0.txt``).
    In a wheel that file must live at ``<dist-info>/licenses/<path>``, so the
    AGPL text is ``<dist-info>/licenses/licenses/AGPL-3.0.txt``.
    """
    relative = license_file.replace("\\", "/").lstrip("/")
    if not relative or relative.startswith("../") or "/../" in f"/{relative}/":
        raise SystemExit(f"Refusing unsafe License-File path: {license_file}")
    return f"{dist_info}/licenses/{relative}"


def apply_platform_wheel_license(wheel_path: Path) -> None:
    if not AGPL_LICENSE_FILE.is_file():
        raise SystemExit("Missing licenses/AGPL-3.0.txt")
    license_files = {
        "licenses/AGPL-3.0.txt": AGPL_LICENSE_FILE.read_bytes(),
    }
    contents: dict[str, bytes] = {}
    dist_info: Optional[str] = None
    with zipfile.ZipFile(wheel_path) as zin:
        for info in zin.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.endswith(".dist-info/RECORD"):
                dist_info = name.rsplit("/", 1)[0]
                continue
            data = zin.read(info.filename)
            if name.endswith(".dist-info/METADATA"):
                dist_info = name.rsplit("/", 1)[0]
                data = rewrite_wheel_license_metadata(data.decode("utf-8")).encode("utf-8")
            contents[name] = data
    if dist_info is None:
        raise SystemExit(f"{wheel_path.name} has no METADATA")
    for relative, payload in license_files.items():
        contents.setdefault(wheel_license_path(dist_info, relative), payload)
    record_name = f"{dist_info}/RECORD"
    record_lines = [
        f"{name},{wheel_record_digest(data)},{len(data)}"
        for name, data in sorted(contents.items())
    ]
    record_lines.append(f"{record_name},,")
    contents[record_name] = ("\n".join(record_lines) + "\n").encode("utf-8")

    tmp_path = wheel_path.with_name(wheel_path.name + ".relicense")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)
    tmp_path.replace(wheel_path)


def one(paths: Iterable[Path]) -> Path:
    items = list(paths)
    if len(items) != 1:
        raise SystemExit(f"Expected exactly one wheel, found {len(items)}")
    return items[0]


def retag_wheel(wheel: Path, platform_tag: str, dest_dir: Path) -> Path:
    """Retag a wheel with the current `wheel tags` CLI and move it into dest_dir.

    `wheel tags` writes next to the source file and has no `--wheel-dir`.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wheel",
            "tags",
            "--platform-tag",
            platform_tag,
            "--remove",
            str(wheel),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        raise SystemExit(f"wheel tags should print one filename, got {names!r}")
    produced = wheel.parent / Path(names[0]).name
    if not produced.is_file():
        raise SystemExit(f"wheel tags reported {names[0]!r} but {produced} is missing")
    destination = dest_dir / produced.name
    if produced.resolve() != destination.resolve():
        shutil.move(str(produced), str(destination))
    return destination


def read_wheel_file_tags(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        name = next(
            (
                entry.replace("\\", "/")
                for entry in zf.namelist()
                if entry.replace("\\", "/").endswith(".dist-info/WHEEL")
            ),
            None,
        )
        if name is None:
            raise SystemExit(f"{wheel_path.name} has no WHEEL metadata")
        return [
            line.split(":", 1)[1].strip()
            for line in zf.read(name).decode("utf-8").splitlines()
            if line.startswith("Tag:")
        ]


def platform_tag_from_wheel_name(name: str) -> str:
    if not name.endswith(".whl"):
        raise SystemExit(f"not a wheel: {name}")
    return name[:-4].rsplit("-", 1)[-1]


def assert_platform_wheels(dist: Path) -> None:
    wheels = sorted(path for path in dist.glob("*.whl"))
    expected = {target.wheel_tag for target in TARGETS}
    found = {platform_tag_from_wheel_name(wheel.name): wheel for wheel in wheels}
    missing = expected - found.keys()
    extra = set(found) - expected
    if missing or extra:
        raise SystemExit(
            f"platform wheels mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for platform, wheel in sorted(found.items()):
        tags = read_wheel_file_tags(wheel)
        if not any(tag.endswith(f"-{platform}") for tag in tags):
            raise SystemExit(
                f"{wheel.name} tags {tags} do not include platform {platform}"
            )


def print_built_artifacts(dist: Path) -> None:
    print("Built:")
    for path in sorted(item for item in dist.iterdir() if item.is_file()):
        if path.suffix == ".whl":
            tags = ",".join(read_wheel_file_tags(path))
            print(f"  {path.name}  tags={tags}")
        else:
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
