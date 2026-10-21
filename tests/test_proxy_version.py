from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read_proxy_pin() -> str:
    return (ROOT / "PROXY_VERSION").read_text(encoding="utf-8").strip()


def test_proxy_version_pin_is_0_2_2() -> None:
    assert read_proxy_pin() == "0.2.2"


@pytest.mark.integration
def test_installed_proxy_version_is_clean_0_2_2() -> None:
    pin = read_proxy_pin()
    binary = os.environ.get("LLMENTARY_BIN")
    assert binary, "LLMENTARY_BIN is required for this integration test"
    completed = subprocess.run(
        [binary, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    text = (completed.stdout or completed.stderr).strip()
    first = text.splitlines()[0] if text else ""
    tokens = first.split()
    reported = tokens[1] if len(tokens) > 1 else ""
    assert tokens[:1] == ["llmentary"], first
    assert reported.lstrip("v") == pin, first
    assert "+dirty" not in first, first
