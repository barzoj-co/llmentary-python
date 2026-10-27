from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
import warnings
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "tests" / "testdata" / "isolation"
PROXY_VERSION = (ROOT / "PROXY_VERSION").read_text(encoding="utf-8").strip()
# The proxy isolation-vector change (PR #35) landed on main as bb39133; the pin tag is the
# long-term source once a release includes testdata/isolation.
PROXY_VECTOR_REFS = (
    "main",
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
    isolated = raw.get("isolated") if isinstance(raw, dict) else None
    if isinstance(isolated, bool):
        return isolated
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


def _log_compared(count: int, source: str) -> None:
    message = f"isolation vector drift check: compared {count} vector(s) against {source}"
    print(message, flush=True)
    warnings.warn(message, UserWarning, stacklevel=2)


def test_isolation_vectors_match_proxy_pin() -> None:
    """Fail when testdata/isolation drifted from the proxy isolation copy."""

    official = _official_json_bytes()
    if official:
        source = _official_source()
        _assert_bytes_match(official, source)
        _log_compared(len(official), source)
        return

    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    listing, ref, error = _fetch_proxy_listing(token)
    if error is not None:
        raise AssertionError(error)
    assert listing is not None and ref is not None
    remote = {
        str(item["name"]): item
        for item in listing
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and str(item["name"]).endswith(".json")
    }
    local_sha = {path.name: _git_blob_sha(path) for path in _vector_files()}
    missing = sorted(set(remote) - set(local_sha))
    extra = sorted(set(local_sha) - set(remote))
    changed = sorted(
        name
        for name, item in remote.items()
        if name in local_sha and local_sha[name] != item.get("sha")
    )
    source = f"{PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH}@{ref}"
    if missing or extra or changed:
        snippets: list[str] = []
        for name in missing + changed:
            try:
                body = _download_file(token, ref, name).decode("utf-8")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
                body = f"<download failed: {type(exc).__name__}>"
            snippets.append(f"----- {name} -----\n{body}")
        raise AssertionError(
            f"vendored isolation vectors drifted from {source}: "
            f"missing={missing} extra={extra} changed={changed}\n" + "\n".join(snippets)
        )
    _log_compared(len(remote), source)


def _official_json_bytes() -> dict[str, bytes]:
    if not OFFICIAL_DIR.is_dir():
        return {}
    return {
        path.name: path.read_bytes()
        for path in OFFICIAL_DIR.glob("*.json")
        if path.is_file()
    }


def _official_source() -> str:
    ref_path = OFFICIAL_DIR / ".ref"
    ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else "unknown"
    return f"{PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH}@{ref}"


def _assert_bytes_match(remote_files: dict[str, bytes], source: str) -> None:
    local_bytes = {path.name: path.read_bytes() for path in _vector_files()}
    missing = sorted(set(remote_files) - set(local_bytes))
    extra = sorted(set(local_bytes) - set(remote_files))
    changed = sorted(
        name
        for name in set(remote_files) & set(local_bytes)
        if local_bytes[name] != remote_files[name]
    )
    if missing or extra or changed:
        snippets = []
        for name in missing + changed:
            body = remote_files.get(name, b"<missing>").decode("utf-8", errors="replace")
            snippets.append(f"----- {name} -----\n{body}")
        raise AssertionError(
            f"vendored isolation vectors drifted from {source}: "
            f"missing={missing} extra={extra} changed={changed}\n" + "\n".join(snippets)
        )


def _fetch_proxy_listing(token: str) -> tuple[list[Any] | None, str | None, str | None]:
    last_http: int | None = None
    transport_error = False
    for ref in PROXY_VECTOR_REFS:
        url = (
            f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
            f"{PROXY_VECTOR_PATH}?ref={ref}"
        )
        request = urllib.request.Request(url)
        request.add_header("Accept", "application/vnd.github+json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                listing = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 403}:
                last_http = exc.code
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            transport_error = True
            continue
        if isinstance(listing, list):
            return listing, ref, None
    joined = ", ".join(PROXY_VECTOR_REFS)
    if last_http is not None:
        return (
            None,
            None,
            "isolation vector drift check failed: HTTP "
            f"{last_http} fetching {PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH} from {joined}",
        )
    if transport_error:
        return (
            None,
            None,
            "isolation vector drift check failed: transport error fetching "
            f"{PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH}",
        )
    return (
        None,
        None,
        "isolation vector drift check failed: "
        f"{PROXY_VECTOR_PATH} not found on {joined}",
    )


def _download_file(token: str, ref: str, name: str) -> bytes:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}/{name}?ref={ref}"
    )
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
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


def test_fetch_proxy_listing_omits_authorization_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'[{"name":"empty-other.json"}]'

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        captured.append(dict(getattr(request, "headers", {})))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    listing, ref, error = _fetch_proxy_listing("")
    assert error is None
    assert ref == "main"
    assert listing == [{"name": "empty-other.json"}]
    assert "Authorization" not in captured[0]


def test_fetch_proxy_listing_sends_optional_github_token(
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

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    listing, _ref, error = _fetch_proxy_listing("rate-limit")
    assert error is None
    assert listing == []
    assert captured[0] == "Bearer rate-limit"


def test_fetch_proxy_listing_http_failure_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: int = 0) -> object:
        raise urllib.error.HTTPError(
            getattr(request, "full_url", ""),
            404,
            "Not Found",
            Message(),
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    listing, ref, error = _fetch_proxy_listing("")
    assert listing is None
    assert ref is None
    assert error is not None
    assert "HTTP 404" in error


def test_download_file_omits_authorization_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            encoded = base64.b64encode(b'{"id":"x"}').decode("ascii")
            return json.dumps({"content": encoded}).encode()

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        captured.append(dict(getattr(request, "headers", {})))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _download_file("", "main", "empty-other.json") == b'{"id":"x"}'
    assert "Authorization" not in captured[0]
