from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmentary.binary import (
    SUPPORTED_WHEEL_TAGS,
    BinaryResolutionError,
    bundled_platform_tag,
    find_binary,
)


def test_find_binary_uses_env_path(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    assert find_binary(env={"LLMENTARY_BIN": str(binary), "PATH": ""}) == binary


def test_find_binary_env_overrides_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "bundled"
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    override = tmp_path / "override"
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: bundled)
    assert find_binary(env={"LLMENTARY_BIN": str(override), "PATH": ""}) == override


def test_find_binary_uses_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundled = tmp_path / "llmentary"
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: bundled)
    assert find_binary(env={"PATH": ""}) == bundled


def test_find_binary_errors_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    with pytest.raises(BinaryResolutionError, match="LLMENTARY_BIN"):
        find_binary(env={"PATH": ""})


def test_find_binary_uses_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / ("llmentary.exe" if os.name == "nt" else "llmentary")
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_binary(env={"PATH": str(tmp_path)}) == binary


def test_bundled_platform_tag_linux_amd64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmentary.binary.goreleaser_platform", lambda: ("linux", "amd64"))
    assert bundled_platform_tag() == "manylinux2014_x86_64"


def test_bundled_platform_tag_darwin_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmentary.binary.goreleaser_platform", lambda: ("darwin", "arm64"))
    assert bundled_platform_tag() == "macosx_11_0_arm64"


def test_sdist_install_error_mentions_no_bundled_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llmentary.binary.package_bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr("llmentary.binary.goreleaser_platform", lambda: ("linux", "amd64"))
    with pytest.raises(BinaryResolutionError) as caught:
        find_binary(env={"PATH": ""})
    message = str(caught.value)
    assert "no bundled proxy" in message
    assert "manylinux2014_x86_64" in message
    assert "--only-binary=:all:" in message


def test_unsupported_platform_error_mentions_no_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmentary.binary.goreleaser_platform", lambda: ("plan9", "mips"))
    with pytest.raises(BinaryResolutionError) as caught:
        find_binary(env={"PATH": ""})
    message = str(caught.value)
    assert "no wheel for platform `plan9_mips`" in message
    for tag in SUPPORTED_WHEEL_TAGS:
        assert tag in message
