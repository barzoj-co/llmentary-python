from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmentary import __version__
from llmentary.doctor import (
    proxy_report_header,
    proxy_version,
    proxy_version_line,
    version_report_line,
)


def test_pytest_report_header_includes_proxy_version(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text(
        "#!/bin/sh\necho 'llmentary 0.2.0 (d3698347ec0ac297a45fd531e4011b60b2039324)'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    pytester.makepyfile(
        """
        def test_ok():
            assert True
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    text = result.stdout.str()
    prefix = f"llmentary {__version__}; proxy 0.2.0 ({binary})"
    assert prefix in text
    assert "llmentary 0.2.0 (d3698347ec0ac297a45fd531e4011b60b2039324)" in text
    header = proxy_report_header()
    assert header.count("\n") == 0
    assert header == (
        f"{prefix}; llmentary 0.2.0 (d3698347ec0ac297a45fd531e4011b60b2039324)"
    )
    assert header.startswith(version_report_line())
    assert str(binary) in header
    assert proxy_version() == "0.2.0"
    assert proxy_version_line() == (
        "llmentary 0.2.0 (d3698347ec0ac297a45fd531e4011b60b2039324)"
    )


def test_pytest_report_header_uses_llmentary_bin_option(
    pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "from-option"
    binary.write_text(
        "#!/bin/sh\necho 'llmentary 0.2.0 (from-option)'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.delenv("LLMENTARY_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    pytester.makepyfile(
        """
        def test_ok():
            assert True
        """
    )
    result = pytester.runpytest(f"--llmentary-bin={binary}")
    result.assert_outcomes(passed=1)
    text = result.stdout.str()
    assert f"proxy 0.2.0 ({binary})" in text
    assert str(binary) in text
    assert "from-option" in text


def test_pytest_report_header_without_binary(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMENTARY_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    pytester.makepyfile(
        """
        def test_ok():
            assert True
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    text = result.stdout.str()
    assert "proxy not found" in text
    header = proxy_report_header(env={"PATH": "", **{
        key: value for key, value in os.environ.items() if key != "LLMENTARY_BIN"
    }})
    assert header.count("\n") == 0
    assert header == f"llmentary {__version__}; proxy not found"
    assert proxy_version(env={"PATH": ""}) is None
    assert proxy_version_line(env={"PATH": ""}) is None
