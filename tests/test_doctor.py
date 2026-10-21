from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
