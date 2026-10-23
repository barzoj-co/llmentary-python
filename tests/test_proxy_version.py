from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PIN = "0.5.1"
PIN_DOC_PATHS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs" / "api.md",
    ROOT / "docs" / "binaries.md",
    ROOT / "docs" / "pytest.md",
    ROOT / "docs" / "releasing.md",
    ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
    ROOT / "tests" / "test_proxy_version.py",
)
CURRENT_PIN_PHRASES = (
    "Published platform wheels bundle proxy v{pin}",
    "currently {pin}",
    "currently `{pin}`",
    "The current pin is `{pin}`",
    "proxy v{pin}",
    'placeholder: "{pin} or /path/to/llmentary"',
)


def read_proxy_pin() -> str:
    return (ROOT / "PROXY_VERSION").read_text(encoding="utf-8").strip()


def test_proxy_version_pin_is_0_5_1() -> None:
    pin = read_proxy_pin()
    assert pin == CURRENT_PIN
    assert not pin.startswith("v")
    assert pin == pin.strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin)


def test_current_pin_docs_match_proxy_version() -> None:
    pin = read_proxy_pin()
    texts = {path: path.read_text(encoding="utf-8") for path in PIN_DOC_PATHS}
    for path, text in texts.items():
        assert pin in text, f"{path.relative_to(ROOT)} should mention pin {pin}"

    joined = "\n".join(texts.values())
    for template in CURRENT_PIN_PHRASES:
        phrase = template.format(pin=pin)
        assert phrase in joined, f"missing current-pin phrase: {phrase}"

    for stale in ("0.4.1", "0.5.0"):
        if pin == stale:
            continue
        stale_current = (
            f"bundle proxy v{stale}",
            f"currently {stale}",
            f"currently `{stale}`",
            f"The current pin is `{stale}`",
            f"proxy v{stale}",
            f'placeholder: "{stale} or /path/to/llmentary"',
        )
        for path, text in texts.items():
            for phrase in stale_current:
                assert phrase not in text, (
                    f"{path.relative_to(ROOT)} still claims stale pin {stale}: {phrase}"
                )


@pytest.mark.integration
def test_installed_proxy_version_is_clean_0_5_1() -> None:
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
