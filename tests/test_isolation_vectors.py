from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "tests" / "testdata" / "isolation"
PROXY_VERSION = (ROOT / "PROXY_VERSION").read_text(encoding="utf-8").strip()
PROXY_VECTOR_API = (
    f"https://api.github.com/repos/barzoj-co/llmentary/contents/testdata/isolation"
    f"?ref=v{PROXY_VERSION}"
)


def _vector_files() -> list[Path]:
    return sorted(path for path in VECTOR_DIR.glob("*.json") if path.is_file())


@pytest.mark.parametrize("path", _vector_files(), ids=lambda path: path.stem)
def test_isolation_vector(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    left = SessionTrace.from_payload(payload["self"]["id"], payload["self"])
    right = SessionTrace.from_payload(payload["other"]["id"], payload["other"])
    expectation = Expectation(left)
    if payload.get("allow_empty"):
        expectation.allow_empty()
    if payload.get("expect") == "pass":
        expectation.isolated_from(right)
        return
    with pytest.raises(AssertionError) as caught:
        expectation.isolated_from(right)
    message = str(caught.value)
    for fragment in payload.get("message_contains") or ():
        assert fragment in message, message


def test_isolation_vectors_match_proxy_pin() -> None:
    """Fail when the proxy pin publishes testdata that drifted from ours."""

    request = urllib.request.Request(PROXY_VECTOR_API)
    token = os.environ.get("LLMTESTLAB_REPO_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            listing = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 403}:
            return
        raise

    if not isinstance(listing, list):
        return
    remote = {
        item["name"]: item.get("sha")
        for item in listing
        if isinstance(item, dict) and str(item.get("name", "")).endswith(".json")
    }
    if not remote:
        return
    local = {path.name: _git_blob_sha(path) for path in _vector_files()}
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    changed = sorted(
        name for name, sha in remote.items() if name in local and local[name] != sha
    )
    assert not missing and not extra and not changed, (
        f"vendored isolation vectors drifted from proxy v{PROXY_VERSION}: "
        f"missing={missing} extra={extra} changed={changed}"
    )


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def test_vector_dir_is_not_empty() -> None:
    assert _vector_files(), "expected testdata/isolation/*.json"
