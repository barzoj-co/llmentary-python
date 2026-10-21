from __future__ import annotations

from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

CallPayload = dict[str, Any]


def _call(
    *,
    session_id: str,
    ident: str,
    seq: int,
    user: str,
    system: Optional[str] = None,
    tools: Optional[list[str]] = None,
    source: str = "path",
    api: str = "chat",
) -> dict[str, Any]:
    if api == "responses":
        request: dict[str, Any] = {
            "instructions": system or "",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
        }
        if tools:
            request["tools"] = [{"type": "function", "name": name} for name in tools]
        return {
            "id": ident,
            "session_id": session_id,
            "session_source": source,
            "seq": seq,
            "api_family": "openai.responses",
            "request": request,
            "response": {"output_text": "ok"},
        }
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    request = {"messages": messages}
    if tools:
        request["tools"] = [
            {"type": "function", "function": {"name": name}} for name in tools
        ]
    return {
        "id": ident,
        "session_id": session_id,
        "session_source": source,
        "seq": seq,
        "api_family": "chat.completions",
        "request": request,
        "response": {"choices": [{"message": {"content": "ok"}}]},
    }


def _trace(session_id: str, *calls: CallPayload) -> SessionTrace:
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": list(calls)})


def test_isolated_sessions_pass() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="Weather in Paris?"),
    )
    barge = _trace(
        "barge",
        _call(session_id="barge", ident="call_1", seq=1, user="Tell a short story."),
    )
    Expectation(weather).isolated_from(barge)
    Expectation(barge).isolated_from(weather)


def test_system_prompt_tool_name_is_not_a_leak() -> None:
    weather = _trace(
        "weather",
        _call(
            session_id="weather",
            ident="call_4",
            seq=1,
            user="What is the weather in Paris?",
            tools=["get_current_weather"],
        ),
    )
    barge = _trace(
        "barge",
        _call(
            session_id="barge",
            ident="call_2",
            seq=1,
            user="Tell a bedtime story.",
            system="Never mention Amsterdam weather or get_current_weather.",
        ),
    )
    Expectation(barge).isolated_from(weather)


def test_shared_user_text_fails_and_names_call() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="Amsterdam canal boats"),
    )
    barge = _trace(
        "barge",
        _call(session_id="barge", ident="call_7", seq=2, user="Amsterdam canal boats"),
    )
    with pytest.raises(AssertionError, match="user text") as caught:
        Expectation(weather).isolated_from(barge)
    message = str(caught.value)
    assert "Amsterdam canal boats" in message
    assert "weather" in message
    assert "barge" in message
    assert "#1 (call_3)" in message
    assert "#2 (call_7)" in message


def test_foreign_session_id_fails() -> None:
    weather = _trace(
        "weather",
        _call(session_id="barge", ident="call_3", seq=1, user="Weather in Paris?"),
    )
    barge = _trace(
        "barge",
        _call(session_id="barge", ident="call_1", seq=1, user="Tell a story."),
    )
    with pytest.raises(AssertionError, match="session_id") as caught:
        Expectation(weather).isolated_from(barge)
    assert "expected 'weather'" in str(caught.value)
    assert "#1 (call_3)" in str(caught.value)


def test_empty_session_fails_unless_allow_empty() -> None:
    empty = _trace("ghost")
    other = _trace(
        "weather",
        _call(session_id="weather", ident="call_1", seq=1, user="Weather in Paris?"),
    )
    with pytest.raises(AssertionError, match="no calls recorded for session ghost"):
        Expectation(empty).isolated_from(other)
    with pytest.raises(AssertionError, match="no calls recorded for session ghost"):
        Expectation(other).isolated_from(empty)
    Expectation(empty).allow_empty().isolated_from(other)
    Expectation(other).allow_empty().isolated_from(empty)


def test_user_texts_ignore_system_and_read_responses_input() -> None:
    chat = _trace(
        "chat",
        _call(
            session_id="chat",
            ident="call_1",
            seq=1,
            user="user turn",
            system="system mentions get_current_weather",
        ),
    ).calls[0]
    assert chat.user_texts == ["user turn"]
    assert "get_current_weather" not in chat.user_texts

    responses = _trace(
        "resp",
        _call(
            session_id="resp",
            ident="call_2",
            seq=1,
            user="responses user",
            system="instructions",
            api="responses",
        ),
    ).calls[0]
    assert responses.user_texts == ["responses user"]
