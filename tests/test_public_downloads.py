from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
HYGIENE_YML = ROOT / ".github" / "workflows" / "hygiene.yml"
RELEASE_YML = ROOT / ".github" / "workflows" / "release.yml"
FETCH_SPEC = importlib.util.spec_from_file_location(
    "fetch_isolation_vectors", ROOT / "scripts" / "fetch_isolation_vectors.py"
)
assert FETCH_SPEC is not None and FETCH_SPEC.loader is not None
fetch: Any = importlib.util.module_from_spec(FETCH_SPEC)
sys.modules[FETCH_SPEC.name] = fetch
FETCH_SPEC.loader.exec_module(fetch)


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_workflows_drop_private_repo_token() -> None:
    for path in (CI_YML, HYGIENE_YML, RELEASE_YML):
        text = _load(path)
        assert "LLMTESTLAB_REPO_TOKEN" not in text
        assert "LLMENTARY_REPO_TOKEN" not in text
        assert "Fail clearly if llmentary repo token is missing" not in text


def test_ci_job_names_stay_byte_identical() -> None:
    names = [
        line.split("name: ", 1)[1]
        for line in _load(CI_YML).splitlines()
        if line.startswith("    name: ")
    ]
    assert names == [
        "Python ${{ matrix.python-version }}",
        "coverage",
        "OpenAI E2E",
        "e2e-gate",
    ]


def test_openai_e2e_still_requires_openai_key() -> None:
    text = _load(CI_YML)
    assert "HAS_OPENAI_KEY: ${{ secrets.OPENAI_API_KEY != '' }}" in text
    assert "Add secrets.OPENAI_API_KEY for real OpenAI E2E tests." in text


def test_e2e_gate_self_test() -> None:
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "e2e-gate.sh"), "--self-test"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "skipped (fork, no secrets)" in completed.stdout
    assert "self-test: ok" in completed.stdout


@pytest.mark.parametrize(
    ("result", "event", "fork", "want_rc", "want_msg"),
    [
        ("success", "pull_request", "false", 0, "live e2e succeeded"),
        ("skipped", "pull_request", "true", 0, "skipped (fork, no secrets)"),
        (
            "skipped",
            "pull_request",
            "false",
            1,
            "live e2e did not run (event=pull_request, result=skipped)",
        ),
        ("failure", "pull_request", "true", 1, "live e2e did not succeed (result=failure)"),
    ],
)
def test_e2e_gate_fork_and_same_repo_corners(
    result: str, event: str, fork: str, want_rc: int, want_msg: str
) -> None:
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "e2e-gate.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "E2E_RESULT": result,
            "EVENT_NAME": event,
            "IS_FORK": fork,
        },
    )
    assert completed.returncode == want_rc, completed.stdout + completed.stderr
    assert want_msg in completed.stdout + completed.stderr


def test_fetch_vectors_token_is_optional_github_token_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("LLMTESTLAB_REPO_TOKEN", raising=False)
    monkeypatch.delenv("LLMENTARY_REPO_TOKEN", raising=False)
    assert fetch._token() == ""
    monkeypatch.setenv("LLMTESTLAB_REPO_TOKEN", "old-crutch")
    monkeypatch.setenv("LLMENTARY_REPO_TOKEN", "also-old")
    assert fetch._token() == ""
    monkeypatch.setenv("GITHUB_TOKEN", " rate-limit ")
    assert fetch._token() == "rate-limit"


def test_fetch_vectors_anonymous_when_no_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("LLMTESTLAB_REPO_TOKEN", raising=False)
    seen: list[str | None] = []

    def fake_get(url: str, token: str) -> tuple[object | None, int | None]:
        seen.append(token)
        if url.endswith("testdata/isolation?ref=main"):
            return [{"name": "empty-other.json", "type": "file"}], 200
        if url.endswith("empty-other.json?ref=main"):
            encoded = __import__("base64").b64encode(b'{"id":"empty-other"}').decode("ascii")
            return {"name": "empty-other.json", "content": encoded}, 200
        return None, 404

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_isolation_vectors.py", "--output", str(tmp_path / "out"), "--root", str(ROOT)],
    )
    assert fetch.main() == 0
    assert seen[0] == ""
    assert (tmp_path / "out" / "empty-other.json").is_file()
    assert "isolation vector drift check" in capsys.readouterr().out


def test_fetch_vectors_sends_authorization_when_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str | None] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        headers = getattr(request, "headers", {})
        captured.append(headers.get("Authorization"))
        return FakeResponse()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    payload, status = fetch._get("https://api.github.com/example", "rate-limit")
    assert status == 200
    assert payload == []
    assert captured == ["Bearer rate-limit"]


def test_fetch_vectors_omits_authorization_when_token_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"[]"

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        captured.append(dict(getattr(request, "headers", {})))
        return FakeResponse()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    fetch._get("https://api.github.com/example", "")
    assert "Authorization" not in captured[0]


def test_fetch_vectors_download_failure_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_get(url: str, token: str) -> tuple[object | None, int | None]:
        return None, 404

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_isolation_vectors.py", "--output", str(tmp_path / "out"), "--root", str(ROOT)],
    )
    assert fetch.main() == 1
    assert "failed: HTTP 404" in capsys.readouterr().out


def test_fetch_vectors_transport_failure_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_get(url: str, token: str) -> tuple[object | None, int | None]:
        return None, None

    monkeypatch.setattr(fetch, "_get", fake_get)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_isolation_vectors.py", "--output", str(tmp_path / "out"), "--root", str(ROOT)],
    )
    assert fetch.main() == 1
    assert "transport error" in capsys.readouterr().out


def test_fetch_vectors_http_error_other_than_404_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: int = 0) -> object:
        raise HTTPError(
            getattr(request, "full_url", ""),
            500,
            "Server Error",
            Message(),
            None,
        )

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(HTTPError):
        fetch._get("https://api.github.com/example", "")


def test_docs_drop_token_crutch_and_once_public() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "LLMTESTLAB_REPO_TOKEN" not in contributing
    assert "TODO: remove once llmentary is public" not in contributing
    assert "skipped (fork, no secrets)" in contributing
    assert "LLMTESTLAB_REPO_TOKEN" not in security
    assert "once this repository is public" not in security
    assert "skipped (fork, no secrets)" in security


# Split so this file itself is not a hit for the public-tree scan.
_PRIVATE_PROJECT_NEEDLES = ("heal" + "sens", "live" + "kit", "chat" + "service")
_SCAN_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "htmlcov",
    "dist",
    "build",
    ".tox",
}


def test_repo_has_no_private_project_names() -> None:
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in _SCAN_SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lowered = text.lower()
            rel = path.relative_to(ROOT).as_posix()
            for needle in _PRIVATE_PROJECT_NEEDLES:
                if needle in lowered:
                    hits.append(f"{rel}: {needle}")
    assert hits == []
