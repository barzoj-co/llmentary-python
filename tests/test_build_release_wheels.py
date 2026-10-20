from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_release_wheels", ROOT / "scripts" / "build_release_wheels.py"
)
assert SPEC is not None and SPEC.loader is not None
wheels: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wheels
SPEC.loader.exec_module(wheels)

CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_dist_license_files", ROOT / "scripts" / "check-dist-license-files.py"
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker: Any = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)


def test_safe_extract_tar_rejects_symlink(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        link = tarfile.TarInfo(name="evil-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/outside"
        tar.addfile(link)
    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
        with pytest.raises(SystemExit, match="Unsupported member"):
            wheels.safe_extract_tar(tar, tmp_path)


def test_parse_goreleaser_checksums() -> None:
    checksums = wheels.parse_checksums(
        b"abc123  llmentary_0.1.0_linux_amd64.tar.gz\ndef456 *llmentary_0.1.0_darwin_arm64.tar.gz\n"
    )
    assert checksums == {
        "llmentary_0.1.0_linux_amd64.tar.gz": "abc123",
        "llmentary_0.1.0_darwin_arm64.tar.gz": "def456",
    }


def test_select_checksum_asset_ignores_signing_artifacts() -> None:
    assets = [
        wheels.Asset("checksums.txt.sig", "https://example/checksums.txt.sig"),
        wheels.Asset("checksums.txt.pem", "https://example/checksums.txt.pem"),
        wheels.Asset("checksums.txt", "https://example/checksums.txt"),
        wheels.Asset("checksums.txt.sigstore.json", "https://example/checksums.txt.sigstore.json"),
    ]
    assert wheels.select_checksum_asset(assets).name == "checksums.txt"


def _record_line(name: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    hashed = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{name},{hashed},{len(data)}"


def test_apply_platform_wheel_license_rewrites_metadata(tmp_path: Path) -> None:
    wheel_path = tmp_path / "llmentary-0.1.0-py3-none-any.whl"
    metadata = (
        b"Metadata-Version: 2.4\nName: llmentary\nLicense-Expression: MIT\n"
        b"License-File: LICENSE\n"
        b"Classifier: License :: OSI Approved :: MIT License\n"
    )
    init = b"x = 1\n"
    dist_info = "llmentary-0.1.0.dist-info"
    stale_record = (
        "\n".join(
            [
                _record_line(f"{dist_info}/METADATA", metadata),
                _record_line("llmentary/__init__.py", init),
                f"{dist_info}/RECORD,,",
            ]
        )
        + "\n"
    )
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr("llmentary/__init__.py", init)
        zf.writestr(f"{dist_info}/RECORD", stale_record)
    wheels.apply_platform_wheel_license(wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
        record = zf.read(f"{dist_info}/RECORD").decode("utf-8")
        files = {
            name: zf.read(name)
            for name in (
                f"{dist_info}/METADATA",
                "llmentary/__init__.py",
                f"{dist_info}/licenses/licenses/AGPL-3.0.txt",
            )
        }
        entries = {
            path: (digest, size)
            for path, digest, size in (line.split(",") for line in record.splitlines() if line)
        }
    rewritten_metadata = files[f"{dist_info}/METADATA"]
    agpl_bytes = files[f"{dist_info}/licenses/licenses/AGPL-3.0.txt"]
    assert b"License-Expression: MIT AND AGPL-3.0-only" in rewritten_metadata
    assert b"License-File: licenses/AGPL-3.0.txt" in rewritten_metadata
    assert b"Classifier: License ::" not in rewritten_metadata
    assert f"{dist_info}/licenses/licenses/AGPL-3.0.txt" in names
    assert f"{dist_info}/licenses/AGPL-3.0.txt" not in names
    assert b"GNU AFFERO GENERAL PUBLIC LICENSE" in agpl_bytes
    for name, data in files.items():
        _, digest, size = _record_line(name, data).split(",")
        assert entries[name] == (digest, size)
    assert entries[f"{dist_info}/RECORD"] == ("", "")
    assert _record_line(f"{dist_info}/METADATA", metadata) not in record


def test_rewrite_wheel_license_metadata_sets_combined_spdx() -> None:
    rewritten = wheels.rewrite_wheel_license_metadata(
        "Metadata-Version: 2.4\nName: llmentary\nLicense-Expression: MIT\n"
        "License-File: LICENSE\n"
        "Classifier: License :: OSI Approved :: MIT License\n"
    )
    assert "License-Expression: MIT AND AGPL-3.0-only" in rewritten
    assert "License-Expression: MIT\n" not in rewritten
    assert "License-File: LICENSE" in rewritten
    assert wheels.AGPL_LICENSE_FILE_FIELD in rewritten
    assert "Classifier: License ::" not in rewritten


def test_rewrite_wheel_license_metadata_inserts_before_description_body() -> None:
    rewritten = wheels.rewrite_wheel_license_metadata(
        "Metadata-Version: 2.4\nName: llmentary\nLicense-Expression: MIT\n"
        "\nLong description that is not a header.\n"
    )
    header, separator, body = rewritten.partition("\n\n")
    assert separator == "\n\n"
    assert "License-Expression: MIT AND AGPL-3.0-only" in header
    assert wheels.AGPL_LICENSE_FILE_FIELD in header
    assert "Classifier: License ::" not in rewritten
    assert body.startswith("Long description that is not a header.")


def test_select_asset_matches_platform() -> None:
    assets = [
        wheels.Asset("checksums.txt", "https://example/checksums.txt"),
        wheels.Asset("llmentary_0.1.0_linux_amd64.tar.gz", "https://example/linux"),
        wheels.Asset("llmentary_0.1.0_darwin_arm64.tar.gz", "https://example/darwin"),
        wheels.Asset("llmentary_0.1.0_windows_amd64.zip", "https://example/windows"),
    ]
    linux = next(target for target in wheels.TARGETS if target.wheel_tag == "manylinux2014_x86_64")
    darwin = next(target for target in wheels.TARGETS if target.wheel_tag == "macosx_11_0_arm64")
    windows = next(target for target in wheels.TARGETS if target.wheel_tag == "win_amd64")
    assert wheels.select_asset(assets, linux).name == "llmentary_0.1.0_linux_amd64.tar.gz"
    assert wheels.select_asset(assets, darwin).name == "llmentary_0.1.0_darwin_arm64.tar.gz"
    assert wheels.select_asset(assets, windows).name == "llmentary_0.1.0_windows_amd64.zip"
    with pytest.raises(SystemExit, match="No release asset matched"):
        wheels.select_asset(
            [wheels.Asset("other_0.1.0_linux_amd64.tar.gz", "https://example/other")],
            linux,
        )


def test_install_binary_extracts_and_records_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="llmentary")
        payload = b"#!/bin/sh\necho proxy\n"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    target = wheels.TARGETS[0]
    wheels.PACKAGE_BIN_DIR = tmp_path / "src" / "llmentary" / "bin"
    wheels.install_binary(
        archive.getvalue(),
        "llmentary_0.1.0_linux_amd64.tar.gz",
        target,
        version="1.2.3",
    )
    installed = wheels.PACKAGE_BIN_DIR / target.wheel_tag / "llmentary"
    assert installed.is_file()
    dest = wheels.PACKAGE_BIN_DIR / target.wheel_tag
    assert (dest / "VERSION").read_text(encoding="utf-8") == "1.2.3\n"
    agpl_text = (dest / "AGPL-3.0.txt").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in agpl_text


def test_asset_download_strips_authorization_on_redirect() -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/asset":
                self.send_response(302)
                address = self.server.server_address
                assert isinstance(address, tuple)
                port = address[1]
                self.send_header("Location", f"http://127.0.0.1:{port}/object")
                self.end_headers()
                return
            if self.path == "/object":
                if self.headers.get("Authorization"):
                    self.send_error(403, "signed URL rejected Authorization")
                    return
                body = b"verified-bytes"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        data = wheels.download_bytes(
            f"http://127.0.0.1:{server.server_address[1]}/asset",
            token="secret-token",
            accept="application/octet-stream",
            follow_redirects=False,
        )
        assert data == b"verified-bytes"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_download_bytes_follows_http_error_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        headers = getattr(request, "headers", {})
        url = getattr(request, "full_url", "")
        token = headers.get("Authorization")
        calls.append((url, token))
        if url.endswith("/asset"):
            headers = Message()
            headers["Location"] = "https://objects.example/file"
            raise HTTPError(url, 302, "Found", headers, None)
        return FakeResponse()

    monkeypatch.setattr(wheels, "urlopen", fake_urlopen)
    opener = type("O", (), {"open": staticmethod(fake_urlopen)})()
    monkeypatch.setattr(wheels, "build_opener", lambda handler: opener)
    data = wheels.download_bytes(
        "https://api.github.com/asset",
        token="secret-token",
        accept="application/octet-stream",
        follow_redirects=False,
    )
    assert data == b"ok"
    assert calls[0][1] == "Bearer secret-token"
    assert calls[1] == ("https://objects.example/file", None)


def test_hatch_wheel_includes_bundled_binary(tmp_path: Path) -> None:
    platform_tag = "manylinux2014_x86_64"
    project_root = tmp_path / "project"
    project_root.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project_root / name)
    shutil.copytree(ROOT / "src", project_root / "src")
    bin_dir = project_root / "src" / "llmentary" / "bin" / platform_tag
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "llmentary"
    binary.write_bytes(b"dummy-proxy")
    binary.chmod(0o755)
    shutil.copy2(ROOT / "licenses" / "AGPL-3.0.txt", bin_dir / "AGPL-3.0.txt")
    outdir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=project_root,
        check=True,
    )
    built = list(outdir.glob("*.whl"))
    assert len(built) == 1
    wheels.apply_platform_wheel_license(built[0])
    with zipfile.ZipFile(built[0]) as zf:
        names = {name.replace("\\", "/") for name in zf.namelist()}
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        agpl_name = next(
            name
            for name in names
            if name.endswith(".dist-info/licenses/licenses/AGPL-3.0.txt")
        )
        metadata = zf.read(metadata_name)
        agpl = zf.read(agpl_name)
    assert f"llmentary/bin/{platform_tag}/llmentary" in names
    assert f"llmentary/bin/{platform_tag}/AGPL-3.0.txt" in names
    assert agpl_name in names
    assert not any(name.endswith(".dist-info/licenses/AGPL-3.0.txt") for name in names)
    assert b"License-Expression: MIT AND AGPL-3.0-only" in metadata
    assert b"License-File: licenses/AGPL-3.0.txt" in metadata
    assert b"Classifier: License ::" not in metadata
    assert b"GNU AFFERO GENERAL PUBLIC LICENSE" in agpl


def _write_minimal_wheel(path: Path, *, platform_tag: str = "any") -> None:
    dist_info = "llmentary-0.1.0.dist-info"
    wheel_meta = (
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
        f"Tag: py3-none-{platform_tag}\n"
    ).encode()
    metadata = b"Metadata-Version: 2.4\nName: llmentary\nVersion: 0.1.0\n"
    payload = b"x = 1\n"
    record = (
        "\n".join(
            [
                _record_line(f"{dist_info}/WHEEL", wheel_meta),
                _record_line(f"{dist_info}/METADATA", metadata),
                _record_line("llmentary/__init__.py", payload),
                f"{dist_info}/RECORD,,",
            ]
        )
        + "\n"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist_info}/WHEEL", wheel_meta)
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr("llmentary/__init__.py", payload)
        zf.writestr(f"{dist_info}/RECORD", record)


def test_retag_wheel_uses_current_cli_without_wheel_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "raw"
    dest_dir = tmp_path / "dist"
    src_dir.mkdir()
    dest_dir.mkdir()
    source = src_dir / "llmentary-0.1.0-py3-none-any.whl"
    _write_minimal_wheel(source)
    commands: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd: list[str], **kwargs: Any) -> Any:
        commands.append(list(cmd))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(wheels.subprocess, "run", spy)
    tagged = wheels.retag_wheel(source, "manylinux2014_x86_64", dest_dir)
    assert tagged == dest_dir / "llmentary-0.1.0-py3-none-manylinux2014_x86_64.whl"
    assert tagged.is_file()
    assert not source.exists()
    assert commands, "wheel tags was not invoked"
    assert "--wheel-dir" not in commands[0]
    assert commands[0][1:4] == ["-m", "wheel", "tags"]
    assert "--platform-tag" in commands[0]
    assert wheels.read_wheel_file_tags(tagged) == ["py3-none-manylinux2014_x86_64"]


def test_assert_platform_wheels_requires_every_target(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    for target in wheels.TARGETS:
        _write_minimal_wheel(
            dist / f"llmentary-0.1.0-py3-none-{target.wheel_tag}.whl",
            platform_tag=target.wheel_tag,
        )
    wheels.assert_platform_wheels(dist)
    extra = dist / "llmentary-0.1.0-py3-none-any.whl"
    _write_minimal_wheel(extra)
    with pytest.raises(SystemExit, match="extra="):
        wheels.assert_platform_wheels(dist)
    extra.unlink()
    missing = dist / f"llmentary-0.1.0-py3-none-{wheels.TARGETS[0].wheel_tag}.whl"
    missing.unlink()
    with pytest.raises(SystemExit, match="missing="):
        wheels.assert_platform_wheels(dist)


def test_print_built_artifacts_lists_filenames_and_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "llmentary-0.1.0.tar.gz").write_bytes(b"sdist")
    wheel_path = dist / "llmentary-0.1.0-py3-none-manylinux2014_x86_64.whl"
    _write_minimal_wheel(wheel_path, platform_tag="manylinux2014_x86_64")
    wheels.print_built_artifacts(dist)
    output = capsys.readouterr().out
    assert "llmentary-0.1.0.tar.gz" in output
    assert "llmentary-0.1.0-py3-none-manylinux2014_x86_64.whl" in output
    assert "tags=py3-none-manylinux2014_x86_64" in output


def _workflow_pins(text: str, name: str) -> list[str]:
    return re.findall(rf'{name}: "([^"]+)"', text)


def test_publish_dry_run_matches_release_wheel_build() -> None:
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    hygiene = (ROOT / ".github/workflows/hygiene.yml").read_text(encoding="utf-8")
    for name in ("BUILD_VERSION", "WHEEL_VERSION", "TWINE_VERSION"):
        release_pins = _workflow_pins(release, name)
        hygiene_pins = _workflow_pins(hygiene, name)
        assert release_pins, f"release.yml is missing {name}"
        assert hygiene_pins, f"hygiene.yml is missing {name}"
        assert set(release_pins) == set(hygiene_pins)
        assert len(set(release_pins)) == 1
    for needle in (
        "python scripts/build_release_wheels.py",
        "twine check --strict dist/*",
        "python scripts/check-dist-license-files.py dist",
        "bash scripts/inspect-release-wheels.sh",
        (
            'python -m pip install "build==${BUILD_VERSION}" '
            '"wheel==${WHEEL_VERSION}" "twine==${TWINE_VERSION}"'
        ),
    ):
        assert needle in release
        assert needle in hygiene
    assert "name: publish (dry run)" in hygiene
    assert "name: Check License-File paths" in release
    assert "name: Check License-File paths" in hygiene
    assert 'TWINE_VERSION: "7.0.0"' in release
    assert 'TWINE_VERSION: "7.0.0"' in hygiene


def test_wheel_license_path_nests_under_dist_info_licenses() -> None:
    assert (
        wheels.wheel_license_path("llmentary-0.1.1.dist-info", "licenses/AGPL-3.0.txt")
        == "llmentary-0.1.1.dist-info/licenses/licenses/AGPL-3.0.txt"
    )
    assert (
        wheels.wheel_license_path("llmentary-0.1.1.dist-info", "LICENSE")
        == "llmentary-0.1.1.dist-info/licenses/LICENSE"
    )
    with pytest.raises(SystemExit, match="unsafe"):
        wheels.wheel_license_path("llmentary-0.1.1.dist-info", "../LICENSE")


def test_retag_then_license_rewrite_is_pep639_and_record_valid(tmp_path: Path) -> None:
    src_dir = tmp_path / "raw"
    dest_dir = tmp_path / "dist"
    src_dir.mkdir()
    dest_dir.mkdir()
    source = src_dir / "llmentary-0.1.0-py3-none-any.whl"
    _write_minimal_wheel(source)
    tagged = wheels.retag_wheel(source, "macosx_11_0_arm64", dest_dir)
    wheels.apply_platform_wheel_license(tagged)
    checker.check_wheel(tagged)
    with zipfile.ZipFile(tagged) as zf:
        names = {_norm_zip(name) for name in zf.namelist()}
        metadata = zf.read("llmentary-0.1.0.dist-info/METADATA")
    assert b"License-File: licenses/AGPL-3.0.txt" in metadata
    assert "llmentary-0.1.0.dist-info/licenses/licenses/AGPL-3.0.txt" in names
    assert "llmentary-0.1.0.dist-info/licenses/AGPL-3.0.txt" not in names


def test_license_file_checker_rejects_unprefixed_agpl_path(tmp_path: Path) -> None:
    dist_info = "llmentary-0.1.1.dist-info"
    metadata = (
        b"Metadata-Version: 2.4\nName: llmentary\nVersion: 0.1.1\n"
        b"License-File: LICENSE\nLicense-File: licenses/AGPL-3.0.txt\n"
    )
    license_text = b"MIT\n"
    agpl_text = b"AGPL\n"
    payload = b"x = 1\n"
    files = {
        f"{dist_info}/METADATA": metadata,
        "llmentary/__init__.py": payload,
        f"{dist_info}/licenses/LICENSE": license_text,
        f"{dist_info}/licenses/AGPL-3.0.txt": agpl_text,
    }
    record = (
        "\n".join(_record_line(name, data) for name, data in sorted(files.items()))
        + f"\n{dist_info}/RECORD,,\n"
    )
    wheel_path = tmp_path / "llmentary-0.1.1-py3-none-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr(f"{dist_info}/RECORD", record)
    with pytest.raises(SystemExit, match=r"licenses/licenses/AGPL-3\.0\.txt"):
        checker.check_wheel(wheel_path)


def test_license_file_checker_self_test() -> None:
    checker.self_test()


def _norm_zip(name: str) -> str:
    return name.replace("\\", "/")
