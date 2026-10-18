from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
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


def test_parse_goreleaser_checksums() -> None:
    checksums = wheels.parse_checksums(
        b"abc123  llmtestlab_Linux_x86_64.tar.gz\ndef456 *llmtestlab_Darwin_arm64.tar.gz\n"
    )
    assert checksums == {
        "llmtestlab_Linux_x86_64.tar.gz": "abc123",
        "llmtestlab_Darwin_arm64.tar.gz": "def456",
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
    wheel_path = tmp_path / "llm_test_lab-0.1.0-py3-none-any.whl"
    metadata = (
        b"Metadata-Version: 2.4\nName: llm-test-lab\nLicense-Expression: MIT\n"
        b"Classifier: License :: OSI Approved :: MIT License\n"
    )
    init = b"x = 1\n"
    dist_info = "llm_test_lab-0.1.0.dist-info"
    stale_record = (
        "\n".join(
            [
                _record_line(f"{dist_info}/METADATA", metadata),
                _record_line("llmtestlab/__init__.py", init),
                f"{dist_info}/RECORD,,",
            ]
        )
        + "\n"
    )
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr("llmtestlab/__init__.py", init)
        zf.writestr(f"{dist_info}/RECORD", stale_record)
    wheels.apply_platform_wheel_license(wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
        record = zf.read(f"{dist_info}/RECORD").decode("utf-8")
        files = {
            name: zf.read(name)
            for name in (
                f"{dist_info}/METADATA",
                "llmtestlab/__init__.py",
                f"{dist_info}/licenses/AGPL-3.0.txt",
                f"{dist_info}/licenses/PROXY-NOTICE.txt",
            )
        }
        entries = {
            path: (digest, size)
            for path, digest, size in (line.split(",") for line in record.splitlines() if line)
        }
    assert b"License-Expression: MIT AND AGPL-3.0-only" in files[f"{dist_info}/METADATA"]
    assert f"{dist_info}/licenses/AGPL-3.0.txt" in names
    assert f"{dist_info}/licenses/PROXY-NOTICE.txt" in names
    for name, data in files.items():
        _, digest, size = _record_line(name, data).split(",")
        assert entries[name] == (digest, size)
    assert entries[f"{dist_info}/RECORD"] == ("", "")
    assert _record_line(f"{dist_info}/METADATA", metadata) not in record


def test_rewrite_wheel_license_metadata_sets_combined_spdx() -> None:
    rewritten = wheels.rewrite_wheel_license_metadata(
        "Metadata-Version: 2.4\nName: llm-test-lab\nLicense-Expression: MIT\n"
        "Classifier: License :: OSI Approved :: MIT License\n"
    )
    assert "License-Expression: MIT AND AGPL-3.0-only" in rewritten
    assert "License-Expression: MIT\n" not in rewritten
    assert wheels.AGPL_CLASSIFIER in rewritten


def test_select_asset_matches_platform() -> None:
    assets = [
        wheels.Asset("checksums.txt", "https://example/checksums.txt"),
        wheels.Asset("llmtestlab_Linux_x86_64.tar.gz", "https://example/linux"),
        wheels.Asset("llmtestlab_Darwin_arm64.tar.gz", "https://example/darwin"),
        wheels.Asset("llmtestlab_Windows_x86_64.zip", "https://example/windows"),
    ]
    linux = next(target for target in wheels.TARGETS if target.wheel_tag == "manylinux2014_x86_64")
    darwin = next(target for target in wheels.TARGETS if target.wheel_tag == "macosx_11_0_arm64")
    windows = next(target for target in wheels.TARGETS if target.wheel_tag == "win_amd64")
    assert wheels.select_asset(assets, linux).name == "llmtestlab_Linux_x86_64.tar.gz"
    assert wheels.select_asset(assets, darwin).name == "llmtestlab_Darwin_arm64.tar.gz"
    assert wheels.select_asset(assets, windows).name == "llmtestlab_Windows_x86_64.zip"


def test_install_binary_extracts_and_records_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="llmtestlab")
        payload = b"#!/bin/sh\necho proxy\n"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    target = wheels.TARGETS[0]
    wheels.PACKAGE_BIN_DIR = tmp_path / "src" / "llmtestlab" / "bin"
    wheels.install_binary(
        archive.getvalue(),
        "llmtestlab_linux_amd64.tar.gz",
        target,
        version="1.2.3",
    )
    installed = wheels.PACKAGE_BIN_DIR / target.wheel_tag / "llmtestlab"
    assert installed.is_file()
    dest = wheels.PACKAGE_BIN_DIR / target.wheel_tag
    assert (dest / "VERSION").read_text(encoding="utf-8") == "1.2.3\n"
    agpl_text = (dest / "AGPL-3.0.txt").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in agpl_text
    proxy_license = (dest / "PROXY-LICENSE.txt").read_text(encoding="utf-8")
    assert "https://github.com/barzoj-co/llm-test-lab" in proxy_license
    assert "Bundled proxy version: 1.2.3" in proxy_license


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
    bin_dir = project_root / "src" / "llmtestlab" / "bin" / platform_tag
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "llmtestlab"
    binary.write_bytes(b"dummy-proxy")
    binary.chmod(0o755)
    outdir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)],
        cwd=project_root,
        check=True,
    )
    built = list(outdir.glob("*.whl"))
    assert len(built) == 1
    with zipfile.ZipFile(built[0]) as zf:
        names = zf.namelist()
    assert any(
        name.replace("\\", "/").endswith(f"llmtestlab/bin/{platform_tag}/llmtestlab")
        for name in names
    )
