from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from llmentary import __version__
from llmentary.__main__ import main, print_version


def test_doctor_wheel_case_prints_path(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho llmentary unknown\n", encoding="utf-8")
    binary.chmod(0o755)
    env = os.environ.copy()
    env["LLMENTARY_BIN"] = str(binary)
    env["PATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-m", "llmentary", "doctor"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert str(binary) in completed.stdout
    assert "llmentary " in completed.stdout
    assert "platform tag:" in completed.stdout
    assert "contract:" in completed.stdout


def test_doctor_binless_case_exits_1() -> None:
    env = os.environ.copy()
    env.pop("LLMENTARY_BIN", None)
    env["PATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-m", "llmentary", "doctor"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 1
    assert "not found" in completed.stdout
    assert "no bundled proxy" in completed.stdout or "no wheel for platform" in completed.stdout
    assert "--only-binary=:all:" in completed.stdout


def test_cli_version_prints_package_and_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.1+dirty'\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out == f"llmentary {__version__} ; proxy 0.4.1+dirty ({binary})\n"
    assert main(["version"]) == 0
    assert capsys.readouterr().out == out


def test_cli_version_missing_binary_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLMENTARY_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("llmentary.doctor.find_binary", _raise_missing)
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"llmentary {__version__} ; proxy: not found\n"
    assert print_version() == 0
    assert capsys.readouterr().out == f"llmentary {__version__} ; proxy: not found\n"


def test_cli_unknown_args_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bogus"]) == 2
    assert main(["--version", "doctor"]) == 2
    err = capsys.readouterr().err
    assert err.count("usage: python -m llmentary doctor | --version | version") == 2


def _raise_missing(env: object = None) -> None:
    from llmentary.binary import BinaryResolutionError

    raise BinaryResolutionError("no binary")


def test_cli_version_subprocess_matches_inprocess(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.1'\n", encoding="utf-8")
    binary.chmod(0o755)
    env = os.environ.copy()
    env["LLMENTARY_BIN"] = str(binary)
    env["PATH"] = ""
    completed = subprocess.run(
        [sys.executable, "-m", "llmentary", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == f"llmentary {__version__} ; proxy 0.4.1 ({binary})\n"
