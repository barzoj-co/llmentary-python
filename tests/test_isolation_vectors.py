from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "tests" / "testdata" / "isolation"
PROXY_VERSION = (ROOT / "PROXY_VERSION").read_text(encoding="utf-8").strip()
# Proxy DOG-29 PR #35 publishes testdata/isolation on this branch first;
# the pin tag is the long-term source after that PR merges.
PROXY_VECTOR_REFS = (
    "cursor/dog-29-isolation-spec-f2b6",
    f"v{PROXY_VERSION}",
)
PROXY_VECTOR_PATH = "testdata/isolation"
PROXY_VECTOR_REPO = "barzoj-co/llmentary"
OFFICIAL_DIR = Path(
    os.environ.get("LLMENTARY_ISOLATION_VECTORS", "/tmp/llmentary-isolation-vectors")
)


def _vector_files() -> list[Path]:
    return sorted(path for path in VECTOR_DIR.glob("*.json") if path.is_file())


def _session_pair(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for left_key, right_key in (
        ("x", "y"),
        ("self", "other"),
        ("left", "right"),
        ("a", "b"),
    ):
        left = payload.get(left_key)
        right = payload.get(right_key)
        if isinstance(left, dict) and isinstance(right, dict):
            return left, right
    raise KeyError(f"vector {payload.get('id') or payload.get('name')!r} has no session pair")


def _session_id(session: dict[str, Any]) -> str:
    for key in ("id", "session_id", "sessionId"):
        value = session.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"session payload has no id: {list(session)}")


def _expect_isolated(payload: dict[str, Any]) -> bool:
    raw = payload.get("expect")
    if isinstance(raw, dict) and isinstance(raw.get("isolated"), bool):
        return raw["isolated"]
    value = payload.get("expect") or payload.get("verdict") or payload.get("want")
    if value == "pass":
        return True
    if value == "fail":
        return False
    raise KeyError(f"vector {payload.get('id') or payload.get('name')!r} has no isolated expect")


def _message_fragments(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("expect")
    if isinstance(raw, dict):
        raw = raw.get("message_contains")
    else:
        raw = payload.get("message_contains")
        if raw is None:
            raw = payload.get("message")
        if raw is None:
            raw = payload.get("error_contains")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(item) for item in raw]


def _ignore_texts(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("ignore_texts")
    if raw is None:
        raw = payload.get("ignoreTexts")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _min_chars(payload: dict[str, Any]) -> int:
    raw = payload.get("min_chars", payload.get("minChars", 8))
    return int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 8


@pytest.mark.parametrize("path", _vector_files(), ids=lambda path: path.stem)
def test_isolation_vector(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    left_raw, right_raw = _session_pair(payload)
    left = SessionTrace.from_payload(_session_id(left_raw), left_raw)
    right = SessionTrace.from_payload(_session_id(right_raw), right_raw)
    expectation = Expectation(left)
    if payload.get("allow_empty") or payload.get("allowEmpty"):
        expectation.allow_empty()
    kwargs: dict[str, Any] = {"min_chars": _min_chars(payload)}
    ignored = _ignore_texts(payload)
    if ignored:
        kwargs["ignore_texts"] = ignored
    if _expect_isolated(payload):
        expectation.isolated_from(right, **kwargs)
        return
    with pytest.raises(AssertionError) as caught:
        expectation.isolated_from(right, **kwargs)
    message = str(caught.value)
    for fragment in _message_fragments(payload):
        assert fragment in message, message


def test_isolation_vectors_match_proxy_pin() -> None:
    """Fail when testdata/isolation drifted from the proxy DOG-29 copy."""

    remote_files = _official_files()
    if remote_files is None:
        token = os.environ.get("LLMTESTLAB_REPO_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            return
        listing, ref = _fetch_proxy_listing(token)
        if listing is None or ref is None:
            return
        remote = {
            item["name"]: item
            for item in listing
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and str(item["name"]).endswith(".json")
        }
        if not remote:
            return
        local = {path.name: _git_blob_sha(path) for path in _vector_files()}
        missing = sorted(set(remote) - set(local))
        extra = sorted(set(local) - set(remote))
        changed = sorted(
            name
            for name, item in remote.items()
            if name in local and local[name] != item.get("sha")
        )
        if missing or extra or changed:
            snippets: list[str] = []
            for name in missing + changed:
                try:
                    body = _download_file(token, ref, name).decode("utf-8")
                except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
                    body = f"<download failed: {type(exc).__name__}>"
                snippets.append(f"----- {name} -----\n{body}")
            raise AssertionError(
                f"vendored isolation vectors drifted from {PROXY_VECTOR_REPO} "
                f"{PROXY_VECTOR_PATH}@{ref}: missing={missing} extra={extra} "
                f"changed={changed}\n" + "\n".join(snippets)
            )
        return

    local = {path.name: path.read_bytes() for path in _vector_files()}
    missing = sorted(set(remote_files) - set(local))
    extra = sorted(set(local) - set(remote_files))
    changed = sorted(
        name for name in set(remote_files) & set(local) if local[name] != remote_files[name]
    )
    if missing or extra or changed:
        snippets = []
        for name in missing + changed:
            body = remote_files.get(name, b"<missing>").decode("utf-8", errors="replace")
            snippets.append(f"----- {name} -----\n{body}")
        raise AssertionError(
            f"vendored isolation vectors drifted from {PROXY_VECTOR_PATH}: "
            f"missing={missing} extra={extra} changed={changed}\n" + "\n".join(snippets)
        )


def _official_files() -> dict[str, bytes] | None:
    if not OFFICIAL_DIR.is_dir():
        return None
    files = {
        path.name: path.read_bytes()
        for path in OFFICIAL_DIR.glob("*.json")
        if path.is_file()
    }
    return files or None


def _fetch_proxy_listing(token: str) -> tuple[list[Any] | None, str | None]:
    for ref in PROXY_VECTOR_REFS:
        url = (
            f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
            f"{PROXY_VECTOR_PATH}?ref={ref}"
        )
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Accept", "application/vnd.github+json")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                listing = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 403}:
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            return None, None
        if isinstance(listing, list):
            return listing, ref
    return None, None


def _download_file(token: str, ref: str, name: str) -> bytes:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}/{name}?ref={ref}"
    )
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    encoded = payload.get("content")
    if not isinstance(encoded, str):
        raise ValueError(f"{name} had no content")
    return base64.b64decode(encoded)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def test_vector_dir_is_not_empty() -> None:
    assert _vector_files(), "expected testdata/isolation/*.json"
