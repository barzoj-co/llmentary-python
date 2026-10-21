from __future__ import annotations

from typing import Optional

import httpx
import pytest

from llmentary import SessionNotFound, WaitTimeoutError
from llmentary.client import (
    ControlClient,
    ProxyHandle,
    Session,
    message_lists_known_ids,
    with_known_sessions,
)
from llmentary.types import zero_calls_message


def test_list_sessions_parses_control_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions"
        return httpx.Response(
            200,
            json={
                "sessions": [
                    {
                        "id": "default",
                        "calls": 3,
                        "active": 0,
                        "sources": ["default"],
                        "first_started_at": "2026-10-20T00:00:00Z",
                        "last_ended_at": "2026-10-20T00:00:01Z",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        sessions = client.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].id == "default"
    assert sessions[0].calls == 3
    assert sessions[0].sources == ("default",)


def test_get_session_unknown_raises_session_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "known", "calls": 1}]})
        return httpx.Response(
            404,
            json={
                "error": {
                    "type": "session_not_found",
                    "message": (
                        'no calls recorded for session "nope"; '
                        "known sessions: GET /sessions"
                    ),
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(SessionNotFound, match="nope") as caught:
            client.get_session("nope")

    assert caught.value.session_id == "nope"
    assert "known" in caught.value.known_ids
    assert "known sessions: known" in str(caught.value)
    assert str(caught.value).lower().count("known sessions:") == 1
    assert "GET /sessions" not in str(caught.value)


def test_config_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/config"
        return httpx.Response(
            200,
            json={"version": "0.3.0-dev", "upstreams": {"openai": "https://api.openai.com/v1"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        assert client.version() == "0.3.0-dev"
        assert "openai" in client.config()["upstreams"]


def test_wait_404_raises_session_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "known", "calls": 1}]})
        if request.url.path.endswith("/wait"):
            return httpx.Response(
                404,
                json={
                    "error": {
                        "type": "session_not_found",
                        "message": 'no calls recorded for session "ghost"',
                    }
                },
            )
        return httpx.Response(404, json={"error": {"type": "session_not_found"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(SessionNotFound, match="ghost") as caught:
            client.wait_session("ghost", timeout=3, quiet=0.2, min_calls=0)

    assert caught.value.session_id == "ghost"
    assert "known" in caught.value.known_ids


def test_wait_504_json_uses_calls_seen() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            504,
            json={
                "error": {"type": "wait_timeout", "calls_seen": 0, "min_calls": 2},
                "id": "async",
                "calls": [],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        with pytest.raises(WaitTimeoutError) as caught:
            client.wait_session("async", timeout=1, quiet=0.1, min_calls=2)

    assert caught.value.calls == 0
    assert caught.value.min_calls == 2


def test_message_lists_known_ids_is_token_aware() -> None:
    hint = 'no calls recorded for session "nope"; known sessions: GET /sessions'
    assert not message_lists_known_ids(hint, ["known"])
    assert message_lists_known_ids(f"{hint}; known sessions: known", ["known"])


def test_zero_calls_message_includes_session_and_known_ids() -> None:
    message = zero_calls_message("sess-a", ["default", "sess-b"])
    assert "no calls recorded for session sess-a" in message
    assert "default" in message
    assert "sess-b" in message


def test_expect_wait_404_is_zero_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(
                404,
                json={"error": {"type": "session_not_found", "message": "missing"}},
            )
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "other", "calls": 2}]})
        return httpx.Response(
            404,
            json={"error": {"type": "session_not_found", "message": "missing"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        control = ControlClient("http://127.0.0.1:7778", client=http)
        session = Session(
            session_id="empty-404",
            proxy=ProxyHandle(url="http://127.0.0.1:7777", control=control),
        )
        with pytest.raises(
            AssertionError, match="no calls recorded for session empty-404"
        ) as caught:
            session.expect()

    assert "other" in str(caught.value)
    assert str(caught.value).lower().count("known sessions:") == 1


def test_expect_zero_calls_uses_known_session_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(
                504,
                json={
                    "error": {"type": "wait_timeout", "calls_seen": 0, "min_calls": 0},
                    "calls": [],
                },
            )
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "other", "calls": 2}]})
        return httpx.Response(
            404,
            json={"error": {"type": "session_not_found", "message": "missing"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        control = ControlClient("http://127.0.0.1:7778", client=http)
        session = Session(
            session_id="empty-sess",
            proxy=ProxyHandle(url="http://127.0.0.1:7777", control=control),
        )
        with pytest.raises(
            AssertionError, match="no calls recorded for session empty-sess"
        ) as caught:
            session.expect()

    assert "other" in str(caught.value)
    assert str(caught.value).lower().count("known sessions:") == 1


def _never_created_handler(session_id: str, *, calls: Optional[list[object]] = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "other", "calls": 1}]})
        if request.url.path.endswith("/wait") or request.url.path.endswith(
            f"/sessions/{session_id}"
        ):
            if calls is not None:
                return httpx.Response(200, json={"id": session_id, "calls": calls})
            return httpx.Response(
                404,
                json={
                    "error": {
                        "type": "session_not_found",
                        "message": (
                            f'no calls recorded for session "{session_id}"; '
                            "known sessions: GET /sessions"
                        ),
                    }
                },
            )
        return httpx.Response(404, json={"error": {"type": "session_not_found"}})

    return handler


def test_expect_no_calls_passes_on_404() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(_never_created_handler("ghost-zero"))
    ) as http:
        session = Session(
            session_id="ghost-zero",
            proxy=ProxyHandle(
                url="http://127.0.0.1:7777",
                control=ControlClient("http://127.0.0.1:7778", client=http),
            ),
        )
        session.expect_no_calls(quiet=0.1)


def test_expect_no_calls_fails_on_one_call() -> None:
    calls = [
        {
            "id": "call_1",
            "seq": 1,
            "session_id": "busy",
            "api_family": "chat.completions",
            "request": {"messages": [{"role": "user", "content": "ping"}]},
            "response": {"choices": [{"message": {"content": "ok"}}]},
        }
    ]
    with httpx.Client(
        transport=httpx.MockTransport(_never_created_handler("busy", calls=calls))
    ) as http:
        session = Session(
            session_id="busy",
            proxy=ProxyHandle(
                url="http://127.0.0.1:7777",
                control=ControlClient("http://127.0.0.1:7778", client=http),
            ),
        )
        with pytest.raises(AssertionError, match="expected 0 LLM calls, got 1") as caught:
            session.expect_no_calls(quiet=0.1)
    assert "llmentary session busy" in str(caught.value)
    assert "#1 (call_1)" in str(caught.value)


def test_expect_allow_empty_call_count_zero() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(_never_created_handler("empty-opt"))
    ) as http:
        session = Session(
            session_id="empty-opt",
            proxy=ProxyHandle(
                url="http://127.0.0.1:7777",
                control=ControlClient("http://127.0.0.1:7778", client=http),
            ),
        )
        session.expect(allow_empty=True).call_count(0)
        with pytest.raises(
            AssertionError, match="no calls recorded for session empty-opt"
        ):
            session.expect()


def test_with_known_sessions_dedupes_proxy_hint() -> None:
    message = with_known_sessions(
        'no calls recorded for session "ghost"; known sessions: GET /sessions',
        "ghost",
        ["iso-a", "iso-b"],
    )
    assert message.count("known sessions:") == 1
    assert "GET /sessions" not in message
    assert "iso-a" in message
    assert "iso-b" in message
