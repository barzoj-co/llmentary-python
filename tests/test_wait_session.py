from __future__ import annotations

import httpx
from test_go_session_schema import chat_tool_request, chat_tool_response, go_call

from llmentary.client import ControlClient


def test_wait_session_falls_back_to_get_when_calls_empty() -> None:
    session_id = "sess-empty-wait"
    recorded: list[str] = []
    full_payload = {
        "id": session_id,
        "calls": [
            go_call(
                seq=1,
                api_family="chat.completions",
                request_body=chat_tool_request(),
                response_body=chat_tool_response(),
            )
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/wait"):
            return httpx.Response(200, json={"id": session_id, "calls": []})
        if request.url.path.endswith(f"/sessions/{session_id}"):
            return httpx.Response(200, json=full_payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        trace = client.wait_session(session_id)

    assert recorded == [
        "GET /sessions/sess-empty-wait/wait",
        "GET /sessions/sess-empty-wait",
    ]
    assert len(trace.calls) == 1
    assert trace.calls[0].api_type == "chat_completions"
    assert trace.calls[0].tool_calls[0].name == "get_weather"


def test_wait_session_uses_wait_payload_when_calls_present() -> None:
    session_id = "sess-ready"
    recorded: list[str] = []
    wait_payload = {
        "id": session_id,
        "calls": [
            go_call(
                seq=1,
                api_family="chat.completions",
                request_body=chat_tool_request(),
                response_body=chat_tool_response(),
            )
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/wait"):
            return httpx.Response(200, json=wait_payload)
        return httpx.Response(500, json={"error": "get should not be called"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        trace = client.wait_session(session_id)

    assert recorded == ["GET /sessions/sess-ready/wait"]
    assert len(trace.calls) == 1


def test_wait_session_post_fallback_then_get_when_calls_empty() -> None:
    session_id = "sess-405"
    recorded: list[str] = []
    full_payload = {
        "id": session_id,
        "calls": [
            go_call(
                seq=1,
                api_family="chat.completions",
                request_body=chat_tool_request(),
                response_body=chat_tool_response(),
            )
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/wait") and request.method == "GET":
            return httpx.Response(405, json={"error": "method not allowed"})
        if request.url.path.endswith("/wait") and request.method == "POST":
            return httpx.Response(200, json={"id": session_id, "calls": []})
        if request.url.path.endswith(f"/sessions/{session_id}"):
            return httpx.Response(200, json=full_payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = ControlClient("http://127.0.0.1:7778", client=http)
        trace = client.wait_session(session_id)

    assert recorded == [
        "GET /sessions/sess-405/wait",
        "POST /sessions/sess-405/wait",
        "GET /sessions/sess-405",
    ]
    assert len(trace.calls) == 1
