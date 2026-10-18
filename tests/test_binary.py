from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmtestlab.binary import (
    BinaryResolutionError,
    find_binary,
    parse_checksums,
    select_archive_asset,
)


def test_find_binary_uses_env_path(tmp_path: Path) -> None:
    binary = tmp_path / "llmtestlab"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    assert find_binary(env={"LLMTESTLAB_BIN": str(binary), "PATH": ""}) == binary


def test_find_binary_errors_clearly_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmtestlab.binary.default_cache_dir", lambda: tmp_path)
    with pytest.raises(BinaryResolutionError, match="LLMTESTLAB_BIN"):
        find_binary(env={"PATH": ""})


def test_parse_goreleaser_checksums() -> None:
    checksums = parse_checksums(b"abc123  llmtestlab_linux_amd64.tar.gz\n")
    assert checksums == {"llmtestlab_linux_amd64.tar.gz": "abc123"}


def test_select_archive_asset_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmtestlab import binary as binary_module

    monkeypatch.setattr(binary_module, "goreleaser_platform", lambda: ("linux", "amd64"))
    selected = select_archive_asset(
        [
            binary_module.ReleaseAsset("checksums.txt", "https://example/checksums.txt"),
            binary_module.ReleaseAsset("llmtestlab_linux_amd64.tar.gz", "https://example/archive"),
        ]
    )
    assert selected.name == "llmtestlab_linux_amd64.tar.gz"


def test_find_binary_uses_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / ("llmtestlab.exe" if os.name == "nt" else "llmtestlab")
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_binary(env={"PATH": str(tmp_path)}) == binary
