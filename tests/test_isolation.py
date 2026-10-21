from __future__ import annotations

from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.client import SessionNotFound
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


def _replay_call(
    *,
    session_id: str,
    ident: str,
    seq: int,
    history: str,
    user: str,
) -> dict[str, Any]:
    return {
        "id": ident,
        "session_id": session_id,
        "session_source": "path",
        "seq": seq,
        "api_family": "chat.completions",
        "request": {
            "messages": [
                {"role": "user", "content": history},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": user},
            ]
        },
        "response": {"choices": [{"message": {"content": "ok"}}]},
    }


def test_same_canned_question_is_not_a_leak() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="What is my biological age?"),
    )
    barge = _trace(
        "barge",
        _call(session_id="barge", ident="call_7", seq=1, user="What is my biological age?"),
    )
    Expectation(weather).isolated_from(barge)
    Expectation(barge).isolated_from(weather)


def test_short_hi_is_not_a_substring_leak() -> None:
    hello = _trace(
        "hello",
        _call(session_id="hello", ident="call_1", seq=1, user="Hi"),
    )
    long = _trace(
        "long",
        _call(
            session_id="long",
            ident="call_2",
            seq=1,
            user="Hi, what is my biological age? This text is long.",
        ),
    )
    Expectation(hello).isolated_from(long)
    Expectation(long).isolated_from(hello)


def test_history_replay_fails_and_names_call() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="Amsterdam canal boats"),
    )
    replayed = SessionTrace.from_payload(
        "barge",
        {
            "id": "barge",
            "calls": [
                _replay_call(
                    session_id="barge",
                    ident="call_7",
                    seq=1,
                    history="Amsterdam canal boats",
                    user="And the locks?",
                )
            ],
        },
    )
    with pytest.raises(AssertionError, match="user text") as caught:
        Expectation(replayed).isolated_from(weather)
    findings, _, _ = str(caught.value).partition("\n\n")
    assert "Amsterdam canal boats" in findings
    assert "weather" in findings
    assert "barge" in findings
    assert "#1 (call_3)" in findings
    assert "#1 (call_7)" in findings


def test_ignore_texts_skips_history_replay() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="Amsterdam canal boats"),
    )
    replayed = SessionTrace.from_payload(
        "barge",
        {
            "id": "barge",
            "calls": [
                _replay_call(
                    session_id="barge",
                    ident="call_7",
                    seq=1,
                    history="Amsterdam canal boats",
                    user="And the locks?",
                )
            ],
        },
    )
    Expectation(replayed).isolated_from(weather, ignore_texts=["Amsterdam canal boats"])


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


def test_trailing_empty_user_is_not_introduced() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_1", seq=1, user="Weather in Paris?"),
    )
    barge = SessionTrace.from_payload(
        "barge",
        {
            "id": "barge",
            "calls": [
                {
                    "id": "call_2",
                    "session_id": "barge",
                    "session_source": "path",
                    "seq": 1,
                    "api_family": "chat.completions",
                    "request": {
                        "messages": [
                            {"role": "user", "content": "Weather in Paris?"},
                            {"role": "user", "content": ""},
                        ]
                    },
                    "response": {"choices": [{"message": {"content": "ok"}}]},
                }
            ],
        },
    )
    assert barge.calls[0].user_texts == ["Weather in Paris?", ""]
    with pytest.raises(AssertionError, match="Weather in Paris") as caught:
        Expectation(barge).isolated_from(weather)
    assert "#1 (call_1)" in str(caught.value)
    assert "#1 (call_2)" in str(caught.value)


def test_empty_session_fails_unless_allow_empty() -> None:
    empty = _trace("ghost")
    other = _trace(
        "weather",
        _call(session_id="weather", ident="call_1", seq=1, user="Weather in Paris?"),
    )
    with pytest.raises(AssertionError, match="no calls recorded for session ghost"):
        Expectation(empty).isolated_from(other)
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


def test_repeated_user_text_is_reported_once() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_3", seq=1, user="Amsterdam canal boats"),
        _call(session_id="weather", ident="call_5", seq=2, user="Amsterdam canal boats"),
    )
    barge = SessionTrace.from_payload(
        "barge",
        {
            "id": "barge",
            "calls": [
                _replay_call(
                    session_id="barge",
                    ident="call_7",
                    seq=1,
                    history="Amsterdam canal boats",
                    user="And the locks?",
                ),
                _replay_call(
                    session_id="barge",
                    ident="call_8",
                    seq=2,
                    history="Amsterdam canal boats",
                    user="Again the locks?",
                ),
            ],
        },
    )
    with pytest.raises(AssertionError, match="user text") as caught:
        Expectation(barge).isolated_from(weather)
    findings, sep, summary = str(caught.value).partition("\n\n")
    assert sep, str(caught.value)
    assert findings.count("Amsterdam canal boats") == 1
    assert "#1 (call_3)" in findings
    assert "#1 (call_7)" in findings
    assert "#2 (call_5)" not in findings
    assert "#2 (call_8)" not in findings
    assert "llmentary session barge" in summary


def test_missing_other_session_is_empty_trace() -> None:
    weather = _trace(
        "weather",
        _call(session_id="weather", ident="call_1", seq=1, user="Weather in Paris?"),
    )

    class _Control:
        def get_session(self, session_id: str) -> SessionTrace:
            raise SessionNotFound(session_id)

    class _Proxy:
        control = _Control()

    class _Session:
        session_id = "weather"
        proxy = _Proxy()

    # SessionNotFound is an empty Y. Empty Y is not a leak source (DOG-29).
    Expectation(weather, session=_Session()).isolated_from("typo")
    Expectation(weather, session=_Session()).allow_empty().isolated_from("typo")

    class _Other:
        session_id = "ghost-session"

        def trace(self) -> SessionTrace:
            raise SessionNotFound(self.session_id)

    Expectation(weather).isolated_from(_Other())
    Expectation(weather).allow_empty().isolated_from(_Other())


def test_responses_roleless_message_is_not_user_text() -> None:
    leak = SessionTrace.from_payload(
        "resp",
        {
            "id": "resp",
            "calls": [
                {
                    "id": "call_2",
                    "session_id": "resp",
                    "session_source": "path",
                    "seq": 1,
                    "api_family": "openai.responses",
                    "request": {
                        "input": [
                            {"type": "message", "content": "assistant leftover"},
                            {"type": "input_text", "text": "bare user"},
                            {"role": "user", "content": "explicit user"},
                        ]
                    },
                    "response": {"output_text": "ok"},
                }
            ],
        },
    ).calls[0]
    assert leak.user_texts == ["bare user", "explicit user"]
    assert "assistant leftover" not in leak.user_texts
