from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_fail_under", ROOT / "scripts" / "coverage_fail_under.py"
)
assert SPEC is not None and SPEC.loader is not None
threshold: Any = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = threshold
SPEC.loader.exec_module(threshold)

CI_YML = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_thresholds_live_in_one_pyproject_table() -> None:
    regular, release = threshold.load_thresholds()
    assert regular == 92
    assert release == 90
    assert regular >= release


def test_regular_prs_use_floored_fail_under() -> None:
    assert (
        threshold.fail_under_for(title="fix: return proxy semver", labels="bug") == 92
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "chore(main): release 0.6.0"},
        {"labels": "autorelease: pending,automerge"},
        {"head_ref": "release-please--branches--main"},
        {"flag": "1"},
    ],
)
def test_release_please_uses_release_fail_under(kwargs: dict[str, str]) -> None:
    assert threshold.fail_under_for(**kwargs) == 92


def test_release_please_never_lowers_ratchet(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.llmentary.coverage]\nfail_under = 92\nrelease_fail_under = 90\n",
        encoding="utf-8",
    )
    assert threshold.fail_under_for(flag="1", path=pyproject) == 92


def test_missing_key_exits(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.llmentary.coverage]\nfail_under = 83\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="release_fail_under"):
        threshold.load_thresholds(pyproject)


def test_cli_prints_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COVERAGE_RELEASE_PLEASE", raising=False)
    monkeypatch.delenv("COVERAGE_PR_TITLE", raising=False)
    monkeypatch.delenv("COVERAGE_PR_LABELS", raising=False)
    monkeypatch.delenv("COVERAGE_HEAD_REF", raising=False)
    assert threshold.main([]) == 0
    assert capsys.readouterr().out == "92\n"

    monkeypatch.setenv("COVERAGE_PR_TITLE", "chore(main): release 0.6.0")
    assert threshold.main([]) == 0
    assert capsys.readouterr().out == "92\n"


def test_ci_job_names_are_byte_identical() -> None:
    names = re.findall(r"^    name: (.+)$", CI_YML, flags=re.MULTILINE)
    assert names == [
        "Python ${{ matrix.python-version }}",
        "coverage",
        "OpenAI E2E",
        "e2e-gate",
    ]


def test_coverage_job_runs_coverage_run_and_fail_under() -> None:
    assert "coverage run -m pytest -m \"not e2e\" -rs" in CI_YML
    assert 'coverage report --fail-under="$(python scripts/coverage_fail_under.py)"' in CI_YML
    assert "--cov " not in CI_YML
    assert "--cov-report" not in CI_YML
