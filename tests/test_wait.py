from __future__ import annotations

import argparse
from typing import Optional, cast

import httpx
import pytest
from test_go_session_schema import chat_tool_request, chat_tool_response, go_call

from llmentary import WaitTimeoutError
from llmentary.client import ControlClient
from llmentary.pytest_plugin import _cli_duration
from llmentary.types import SessionTrace
from llmentary.wait import (
    DEFAULT_WAIT_QUIET,
    DEFAULT_WAIT_TIMEOUT,
    TIMEOUT_ENV,
    count_active_calls,
    default_wait_quiet,
    default_wait_timeout,
    format_go_duration,
    parse_duration,
    wait_query_string,
    wait_timeout_error,
)


@pytest.fixture(autouse=True)
def _clear_wait_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLMENTARY_WAIT_TIMEOUT", "LLMENTARY_WAIT_QUIET", "LLMENTARY_QUIET_FOR"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (30.0, "30s"),
        (90.0, "90s"),
        (0.5, "500ms"),
        (0.2, "200ms"),
        (0.1, "100ms"),
        (1.5, "1500ms"),
    ],
)
def test_format_go_duration(seconds: float, expected: str) -> None:
    assert format_go_duration(seconds) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("45", 45.0),
        ("45s", 45.0),
        ("0.2", 0.2),
        ("200ms", 0.2),
    ],
)
def test_parse_duration_table(raw: str, expected: float) -> None:
    assert parse_duration(raw, name=TIMEOUT_ENV) == pytest.approx(expected)


def test_parse_duration_invalid_names_variable() -> None:
    with pytest.raises(ValueError, match="LLMENTARY_WAIT_QUIET"):
        parse_duration("nope", name="LLMENTARY_WAIT_QUIET")


def test_cli_duration_keeps_variable_name() -> None:
    parse = _cli_duration("LLMENTARY_WAIT_TIMEOUT")
    with pytest.raises(argparse.ArgumentTypeError, match="LLMENTARY_WAIT_TIMEOUT"):
        parse("nope")


def test_duration_from_env_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMENTARY_WAIT_TIMEOUT", raising=False)
    monkeypatch.delenv("LLMENTARY_WAIT_QUIET", raising=False)
    monkeypatch.delenv("LLMENTARY_QUIET_FOR", raising=False)
    assert default_wait_timeout() == DEFAULT_WAIT_TIMEOUT
    assert default_wait_quiet() == DEFAULT_WAIT_QUIET

    monkeypatch.setenv("LLMENTARY_WAIT_TIMEOUT", "45s")
    monkeypatch.setenv("LLMENTARY_WAIT_QUIET", "200ms")
    assert default_wait_timeout() == 45.0
    assert default_wait_quiet() == pytest.approx(0.2)


def test_wait_query_string_defaults_and_configured() -> None:
    assert wait_query_string(30.0, 0.5) == "timeout=30s&quiet=500ms"
    assert wait_query_string(90.0, 0.2) == "timeout=90s&quiet=200ms"
    assert wait_query_string(5.0, 0.1, min_calls=2) == "timeout=5s&quiet=100ms&min_calls=2"


def _wait_payload(session_id: str, *, ended: bool = True, seq: int = 1) -> dict[str, object]:
    call = go_call(
        seq=seq,
        api_family="chat.completions",
        request_body=chat_tool_request(),
        response_body=chat_tool_response(),
    )
    if not ended:
        call["ended_at"] = ""
    return {"id": session_id, "calls": [call]}


@pytest.mark.parametrize(
    ("kwargs", "query"),
    [
        ({}, "timeout=30s&quiet=500ms"),
        ({"timeout": 90, "quiet": 0.2}, "timeout=90s&quiet=200ms"),
        ({"timeout": 90, "quiet_for": 0.2}, "timeout=90s&quiet=200ms"),
    ],
)
def test_wait_session_sends_timeout_and_quiet_query_string(
    kwargs: dict[str, float],
    query: str,
) -> None:
    session_id = "sess-query"
    recorded: list[str] = []
    payload = _wait_payload(session_id)

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(_query_string(request))
        assert "quiet_for" not in str(request.url)
        assert "timeout_ms" not in str(request.url)
        assert "idle_ms" not in str(request.url)
        if request.url.path.endswith("/wait"):
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        client.wait_session(
            session_id,
            timeout=kwargs.get("timeout"),
            quiet=kwargs.get("quiet"),
            quiet_for=kwargs.get("quiet_for"),
        )

    assert recorded == [query]


def test_wait_session_sends_min_calls() -> None:
    session_id = "sess-min"
    recorded: list[str] = []
    payload = _wait_payload(session_id)

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(_query_string(request))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        client.wait_session(session_id, timeout=5, quiet=0.1, min_calls=2)

    assert recorded == ["timeout=5s&quiet=100ms&min_calls=2"]


def test_wait_session_timeout_error_message() -> None:
    session_id = "sess-timeout"
    finished = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=chat_tool_request(),
        response_body=chat_tool_response(),
    )
    hanging = go_call(
        seq=2,
        api_family="chat.completions",
        request_body=chat_tool_request(),
        response_body=chat_tool_response(),
    )
    hanging["ended_at"] = ""
    payload = {"id": session_id, "calls": [finished, hanging]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(504, text="gateway timeout")
        if request.url.path == "/sessions":
            return httpx.Response(
                200,
                json={"sessions": [{"id": session_id, "calls": 2, "active": 1}]},
            )
        if request.url.path.endswith(f"/sessions/{session_id}"):
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(
            WaitTimeoutError, match=_timeout_message(session_id, 30, 2, 1)
        ) as caught:
            client.wait_session(session_id)

    assert caught.value.session_id == session_id
    assert caught.value.calls == 2
    assert caught.value.active == 1


def test_wait_session_timeout_does_not_invent_active_zero() -> None:
    session_id = "sess-stale"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(504, text="gateway timeout")
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": session_id, "calls": 0}]})
        if request.url.path.endswith(f"/sessions/{session_id}"):
            hanging = go_call(
                seq=1,
                api_family="chat.completions",
                request_body=chat_tool_request(),
                response_body=chat_tool_response(),
            )
            hanging["ended_at"] = ""
            return httpx.Response(200, json={"id": session_id, "calls": [hanging]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(WaitTimeoutError, match=_timeout_message(session_id, 5, 1, None)):
            client.wait_session(session_id, timeout=5, quiet=0.1)


def test_wait_session_timeout_uses_configured_timeout_in_message() -> None:
    session_id = "sess-short"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(504, text="gateway timeout")
        if request.url.path.endswith(f"/sessions/{session_id}"):
            return httpx.Response(200, json={"id": session_id, "calls": []})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(WaitTimeoutError, match=_timeout_message(session_id, 5, 0, None)):
            client.wait_session(session_id, timeout=5, quiet=0.1)


def test_wait_timeout_uses_error_active_when_present() -> None:
    payload = {
        "id": "stall",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "min_calls": 1, "active": 1},
    }
    error = wait_timeout_error("stall", 1.0, payload)
    assert error.active == 1
    assert "still active: 1" in str(error)
    assert error.calls == 0


def test_wait_timeout_explicit_zero_is_kept() -> None:
    payload = {
        "id": "quiet",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "active": 0},
    }
    error = wait_timeout_error("quiet", 1.0, payload)
    assert error.active == 0
    assert "still active: 0" in str(error)


def test_count_active_calls_ignores_missing_ended_at() -> None:
    hanging = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=chat_tool_request(),
        response_body=chat_tool_response(),
    )
    hanging["ended_at"] = ""
    payload = {"id": "s", "calls": [hanging]}
    trace = SessionTrace.from_payload("s", payload)
    assert count_active_calls(payload, trace) is None
    assert count_active_calls(payload, trace, fallback=2) == 2


def _query_string(request: httpx.Request) -> str:
    query = request.url.query
    if isinstance(query, bytes):
        return query.decode("ascii")
    return str(query)


def _timeout_message(
    session_id: str, timeout: int, calls: int, active: Optional[int]
) -> str:
    so_far = rf"{calls} calls so far"
    if active is None:
        detail = so_far
    else:
        detail = rf"{so_far}, still active: {active}"
    return (
        rf"wait timed out after {timeout}s on session {session_id} "
        rf"\({detail}\)\. "
        rf"Raise {TIMEOUT_ENV}"
    )


def test_wait_session_http_timeout_is_timeout_plus_quiet_plus_five() -> None:
    seen: list[Optional[float]] = []

    class RecordingClient:
        def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            timeout = kwargs.get("timeout")
            seen.append(float(timeout) if isinstance(timeout, (int, float)) else None)
            return httpx.Response(
                200,
                json=_wait_payload("sess-http"),
                request=httpx.Request(method, url),
            )

    client = ControlClient(
        "http://127.0.0.1:7778",
        client=cast(httpx.Client, RecordingClient()),
    )
    client.wait_session("sess-http", timeout=90, quiet=0.2)
    assert seen == [95.2]
