from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, cast

import httpx
import pytest
from test_go_session_schema import chat_tool_request, chat_tool_response, go_call

from llmentary import InFlightCall, WaitTimeoutError
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
    parse_in_flight_calls,
    wait_query_string,
    wait_timeout_error,
)

PIN = Path(__file__).resolve().parent / "testdata" / "wait_timeout_inflight.json"


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
    assert caught.value.in_flight is None


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


def test_wait_timeout_nested_session_active_skips_list_sessions() -> None:
    session_id = "sess-nested"
    listed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        listed.append(request.url.path)
        if request.url.path.endswith("/wait"):
            return httpx.Response(
                504,
                json={
                    "id": session_id,
                    "calls": [],
                    "session": {"id": session_id, "active": 1},
                    "error": {"type": "wait_timeout", "calls_seen": 0},
                },
            )
        raise AssertionError(f"unexpected request {request.url.path}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(
            WaitTimeoutError, match=_timeout_message(session_id, 1, 0, 1)
        ) as caught:
            client.wait_session(session_id, timeout=1, quiet=0.1)

    assert caught.value.active == 1
    assert listed == [f"/sessions/{session_id}/wait"]


def test_wait_timeout_uses_error_active_when_present() -> None:
    payload = {
        "id": "stall",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "min_calls": 1, "active": 1},
    }
    error = wait_timeout_error("stall", 1.0, payload)
    assert error.active == 1
    assert error.in_flight is None
    assert "still active: 1" in str(error)
    assert "awaiting_headers" not in str(error)
    assert error.calls == 0


def test_wait_timeout_explicit_zero_is_kept() -> None:
    payload = {
        "id": "quiet",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "active": 0},
    }
    error = wait_timeout_error("quiet", 1.0, payload)
    assert error.active == 0
    assert error.in_flight is None
    assert "still active: 0" in str(error)


def _pin_payloads() -> dict[str, Any]:
    parsed = json.loads(PIN.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def test_wait_timeout_old_proxy_omits_in_flight_listing() -> None:
    payload: dict[str, Any] = {
        "id": "legacy",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0},
    }
    error = wait_timeout_error("legacy", 1.0, payload)
    assert error.in_flight is None
    assert error.active is None
    assert "still active" not in str(error)
    error_mapping = payload["error"]
    assert isinstance(error_mapping, dict)
    assert parse_in_flight_calls(error_mapping, payload) is None


def test_wait_timeout_empty_in_flight() -> None:
    payload = {
        "id": "quiet",
        "calls": [],
        "error": {
            "type": "wait_timeout",
            "calls_seen": 0,
            "min_calls": 1,
            "active": 0,
            "in_flight": [],
        },
    }
    error = wait_timeout_error("quiet", 1.0, payload)
    assert error.active == 0
    assert error.in_flight == ()
    assert "still active: 0" in str(error)
    assert "#" not in str(error).split("still active: 0", 1)[1]


def test_wait_timeout_pin_awaiting_headers() -> None:
    payload = _pin_payloads()["awaiting_headers"]
    error = wait_timeout_error("stall-example", 1.0, payload)
    assert error.active == 1
    assert error.min_calls == 1
    assert error.calls == 0
    assert error.in_flight is not None
    assert len(error.in_flight) == 1
    item = error.in_flight[0]
    assert isinstance(item, InFlightCall)
    assert item.call_id == "call_1"
    assert item.seq == 1
    assert item.method == "POST"
    assert item.path == "/v1/responses"
    assert item.phase == "awaiting_headers"
    assert item.elapsed_ms == 1000
    assert item.bytes_received == 0
    assert item.http_status is None
    assert "still active: 1: #1 POST /v1/responses awaiting_headers 1000ms" in str(error)


def test_wait_timeout_pin_streaming() -> None:
    payload = _pin_payloads()["streaming"]
    error = wait_timeout_error("stall-stream-example", 1.0, payload)
    assert error.active == 1
    assert error.in_flight is not None
    item = error.in_flight[0]
    assert item.phase == "streaming"
    assert item.http_status == 200
    assert item.bytes_received == 42
    assert item.path == "/v1/chat/completions"
    assert (
        "still active: 1: #1 POST /v1/chat/completions streaming http 200 1000ms"
        in str(error)
    )


def test_wait_timeout_many_in_flight() -> None:
    payload = {
        "id": "busy",
        "calls": [],
        "error": {
            "type": "wait_timeout",
            "calls_seen": 1,
            "min_calls": 2,
            "active": 3,
            "in_flight": [
                {
                    "call_id": "a",
                    "seq": 2,
                    "method": "POST",
                    "path": "/v1/responses",
                    "phase": "awaiting_headers",
                    "elapsed_ms": 50,
                    "bytes_received": 0,
                },
                {
                    "call_id": "b",
                    "seq": 3,
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "phase": "streaming",
                    "http_status": 200,
                    "elapsed_ms": 80,
                    "bytes_received": 12,
                },
                {
                    "call_id": "c",
                    "seq": 4,
                    "method": "POST",
                    "path": "/v1/embeddings",
                    "phase": "awaiting_headers",
                    "elapsed_ms": 10,
                    "bytes_received": 0,
                },
            ],
        },
    }
    error = wait_timeout_error("busy", 2.0, payload)
    assert error.active == 3
    assert error.in_flight is not None
    assert [item.seq for item in error.in_flight] == [2, 3, 4]
    message = str(error)
    assert "still active: 3:" in message
    assert "#2 POST /v1/responses awaiting_headers 50ms" in message
    assert "#3 POST /v1/chat/completions streaming http 200 80ms" in message
    assert "#4 POST /v1/embeddings awaiting_headers 10ms" in message


def test_wait_timeout_skips_non_object_in_flight_items() -> None:
    payload = {
        "id": "mixed",
        "calls": [],
        "error": {
            "type": "wait_timeout",
            "calls_seen": 0,
            "in_flight": [
                1,
                "nope",
                None,
                {
                    "call_id": "ok",
                    "seq": 7,
                    "method": "GET",
                    "path": "/v1/models",
                    "phase": "awaiting_headers",
                    "elapsed_ms": 5,
                },
            ],
        },
    }
    error = wait_timeout_error("mixed", 1.0, payload)
    assert error.in_flight is not None
    assert len(error.in_flight) == 1
    assert error.active == 1
    assert error.in_flight[0].seq == 7
    assert "#7 GET /v1/models awaiting_headers 5ms" in str(error)


def test_wait_timeout_in_flight_len_used_when_active_omitted() -> None:
    payload = {
        "id": "no-count",
        "calls": [],
        "error": {
            "type": "wait_timeout",
            "calls_seen": 0,
            "in_flight": [
                {
                    "call_id": "only",
                    "seq": 1,
                    "method": "POST",
                    "path": "/v1/responses",
                    "phase": "streaming",
                    "http_status": 200,
                    "elapsed_ms": 3,
                }
            ],
        },
    }
    error = wait_timeout_error("no-count", 1.0, payload)
    assert error.active == 1
    assert "still active: 1:" in str(error)


def test_wait_timeout_explicit_active_wins_over_in_flight_len() -> None:
    payload = {
        "id": "mismatch",
        "calls": [],
        "error": {
            "type": "wait_timeout",
            "calls_seen": 0,
            "active": 4,
            "in_flight": [
                {
                    "call_id": "one",
                    "seq": 1,
                    "method": "POST",
                    "path": "/v1/responses",
                    "phase": "awaiting_headers",
                    "elapsed_ms": 1,
                }
            ],
        },
    }
    error = wait_timeout_error("mismatch", 1.0, payload)
    assert error.active == 4
    assert error.in_flight is not None
    assert len(error.in_flight) == 1
    assert "still active: 4: #1 POST /v1/responses awaiting_headers 1ms" in str(error)


def test_wait_timeout_invalid_in_flight_type_leaves_count_unknown() -> None:
    payload = {
        "id": "bad",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "in_flight": "nope"},
    }
    error = wait_timeout_error("bad", 1.0, payload)
    assert error.in_flight is None
    assert error.active is None
    assert "still active" not in str(error)


def test_wait_timeout_invalid_in_flight_uses_list_active_fallback() -> None:
    payload = {
        "id": "bad-fallback",
        "calls": [],
        "error": {"type": "wait_timeout", "calls_seen": 0, "in_flight": "nope"},
    }
    error = wait_timeout_error("bad-fallback", 1.0, payload, list_active=2)
    assert error.in_flight is None
    assert error.active == 2
    assert "still active: 2" in str(error)
    assert "awaiting_headers" not in str(error)


def test_in_flight_describe_falls_back_to_call_id() -> None:
    item = InFlightCall(call_id="call_x", phase="streaming")
    assert item.describe() == "call_x streaming"


def test_in_flight_call_is_hashable_and_ignores_raw() -> None:
    left = InFlightCall(call_id="call_x", seq=1, raw={"n": 1})
    right = InFlightCall(call_id="call_x", seq=1, raw={"n": 2})
    assert left == right
    assert hash(left) == hash(right)
    assert {left, right} == {left}


def test_wait_session_504_json_lists_in_flight() -> None:
    session_id = "stall-example"
    payload = _pin_payloads()["awaiting_headers"]
    listed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        listed.append(request.url.path)
        return httpx.Response(504, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(WaitTimeoutError, match="awaiting_headers") as caught:
            client.wait_session(session_id, timeout=1, quiet=0.1)

    assert caught.value.active == 1
    assert caught.value.in_flight is not None
    assert caught.value.in_flight[0].phase == "awaiting_headers"
    assert listed == [f"/sessions/{session_id}/wait"]


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
