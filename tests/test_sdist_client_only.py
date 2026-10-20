from __future__ import annotations

import importlib.util
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_sdist_client_only", ROOT / "scripts" / "check-sdist-client-only.py"
)
assert SPEC is not None and SPEC.loader is not None
checker: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_built_sdist_has_no_bin_and_find_binary_message(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        check=True,
        cwd=ROOT,
    )
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert sdists, "python -m build --sdist produced no tarball"
    names = checker.assert_sdist_has_no_bin(sdists[0])
    assert names
    with tarfile.open(sdists[0]) as archive:
        assert archive.getnames() == names

    message = checker.assert_sdist_find_binary_message(sdists[0], tmp_path / "work")
    assert "no bundled proxy" in message
    assert "--only-binary=:all:" in message
