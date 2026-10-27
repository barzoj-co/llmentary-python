from __future__ import annotations

from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.client import SessionNotFound
from llmentary.types import SessionTrace
from llmentary.wait import WaitTimeoutError

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
        _call(session_id="weather", ident="call_3", seq=1, user="Where is my order?"),
    )
    barge = _trace(
        "barge",
        _call(session_id="barge", ident="call_7", seq=1, user="Where is my order?"),
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
            user="Hi, where is my order? This text is long.",
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
    with pytest.raises(
        AssertionError,
        match=r"no calls recorded for session ghost \(isolated_from needs calls on both sides\)",
    ):
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

    with pytest.raises(
        AssertionError,
        match=r"no calls recorded for session typo \(isolated_from needs calls on both sides\)",
    ):
        Expectation(weather, session=_Session()).isolated_from("typo")
    Expectation(weather, session=_Session()).allow_empty().isolated_from("typo")

    class _Other:
        session_id = "ghost-session"

        def trace(self) -> SessionTrace:
            raise SessionNotFound(self.session_id)

    with pytest.raises(
        AssertionError,
        match=(
            r"no calls recorded for session ghost-session "
            r"\(isolated_from needs calls on both sides\)"
        ),
    ):
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


def _users_call(
    *,
    session_id: str,
    ident: str,
    seq: int,
    users: list[str],
    api: str = "chat",
) -> dict[str, Any]:
    if api == "responses":
        return {
            "id": ident,
            "session_id": session_id,
            "session_source": "path",
            "seq": seq,
            "api_family": "openai.responses",
            "request": {
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": text}]}
                    for text in users
                ]
            },
            "response": {"output_text": "ok"},
        }
    return {
        "id": ident,
        "session_id": session_id,
        "session_source": "path",
        "seq": seq,
        "api_family": "chat.completions",
        "request": {"messages": [{"role": "user", "content": text} for text in users]},
        "response": {"choices": [{"message": {"content": "ok"}}]},
    }


def test_later_call_introduces_each_new_user_message() -> None:
    peer = _trace(
        "q",
        _call(
            session_id="q",
            ident="call_9",
            seq=1,
            user="Second message sent by P itself",
        ),
    )
    own = _trace(
        "p",
        _call(
            session_id="p",
            ident="call_7",
            seq=1,
            user="Opening question number one",
        ),
        _users_call(
            session_id="p",
            ident="call_8",
            seq=2,
            users=[
                "Opening question number one",
                "Second message sent by P itself",
                "Third message sent by P itself",
            ],
        ),
    )
    Expectation(own).isolated_from(peer)
    Expectation(peer).isolated_from(own)


def test_later_call_overlap_in_non_first_new_message_is_not_a_leak() -> None:
    peer = _trace(
        "q",
        _call(
            session_id="q",
            ident="call_q1",
            seq=1,
            user="Third message sent by P itself",
        ),
    )
    own = _trace(
        "p",
        _call(session_id="p", ident="call_p1", seq=1, user="Opening question number one"),
        _users_call(
            session_id="p",
            ident="call_2",
            seq=2,
            users=[
                "Opening question number one",
                "Second message sent by P itself",
                "Third message sent by P itself",
            ],
        ),
    )
    Expectation(own).isolated_from(peer)


def test_first_call_history_then_later_new_messages_still_leaks() -> None:
    peer = _trace(
        "q",
        _call(session_id="q", ident="call_q1", seq=1, user="Amsterdam canal boats"),
    )
    own = _trace(
        "p",
        _users_call(
            session_id="p",
            ident="call_p1",
            seq=1,
            users=["Amsterdam canal boats", "Opening question number one"],
        ),
        _users_call(
            session_id="p",
            ident="call_2",
            seq=2,
            users=[
                "Amsterdam canal boats",
                "Opening question number one",
                "Second message sent by P itself",
            ],
        ),
    )
    with pytest.raises(AssertionError, match="Amsterdam canal boats"):
        Expectation(own).isolated_from(peer)


def test_responses_later_call_two_new_user_messages() -> None:
    peer = _trace(
        "q",
        _call(
            session_id="q",
            ident="call_y1",
            seq=1,
            user="Second message sent by P itself",
            api="responses",
        ),
    )
    own = _trace(
        "p",
        _call(
            session_id="p",
            ident="call_x1",
            seq=1,
            user="Opening question number one",
            api="responses",
        ),
        _users_call(
            session_id="p",
            ident="call_x2",
            seq=2,
            users=[
                "Opening question number one",
                "Second message sent by P itself",
                "Third message sent by P itself",
            ],
            api="responses",
        ),
    )
    Expectation(own).isolated_from(peer)


def test_unknown_session_id_fails_without_allow_empty() -> None:
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

    with pytest.raises(
        AssertionError,
        match=(
            r"no calls recorded for session weather-typo "
            r"\(isolated_from needs calls on both sides\)"
        ),
    ):
        Expectation(weather, session=_Session()).isolated_from("weather-typo")
    Expectation(weather, session=_Session()).allow_empty().isolated_from("weather-typo")


def test_userless_first_call_does_not_consume_first_call_status() -> None:
    peer = _trace(
        "q",
        _call(session_id="q", ident="call_q1", seq=1, user="Amsterdam canal boats"),
    )
    own = SessionTrace.from_payload(
        "p",
        {
            "id": "p",
            "calls": [
                {
                    "id": "call_empty",
                    "session_id": "p",
                    "session_source": "path",
                    "seq": 1,
                    "api_family": "chat.completions",
                    "request": {"messages": [{"role": "assistant", "content": "hi"}]},
                    "response": {"choices": [{"message": {"content": "ok"}}]},
                },
                {
                    "id": "call_p2",
                    "session_id": "p",
                    "session_source": "path",
                    "seq": 2,
                    "api_family": "chat.completions",
                    "request": {
                        "messages": [
                            {"role": "user", "content": "Amsterdam canal boats"},
                            {"role": "user", "content": "And the locks after that?"},
                        ]
                    },
                    "response": {"choices": [{"message": {"content": "ok"}}]},
                },
            ],
        },
    )
    with pytest.raises(AssertionError, match="Amsterdam canal boats"):
        Expectation(own).isolated_from(peer)


def _wait_timeout(session_id: str, *, calls: int = 0, active: int = 0) -> WaitTimeoutError:
    return WaitTimeoutError(
        "wait timed out",
        session_id=session_id,
        timeout=1.0,
        calls=calls,
        active=active,
    )


def test_isolated_from_waits_on_other_session_in_flight_leak() -> None:
    secret = "Y secret follow-up text"
    settled = _trace(
        "y",
        _call(session_id="y", ident="call_y1", seq=1, user="Y opening question"),
    )
    after_wait = _trace(
        "y",
        _call(session_id="y", ident="call_y1", seq=1, user="Y opening question"),
        _call(session_id="y", ident="call_y2", seq=2, user=secret),
    )
    own = _trace(
        "x",
        _replay_call(
            session_id="x",
            ident="call_x1",
            seq=1,
            history=secret,
            user="X asks its own question",
        ),
    )

    class _Control:
        def __init__(self) -> None:
            self.waits = 0
            self.gets = 0

        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            self.waits += 1
            assert session_id == "y"
            return after_wait

        def get_session(self, session_id: str) -> SessionTrace:
            self.gets += 1
            return settled

    control = _Control()

    class _Proxy:
        def __init__(self) -> None:
            self.control = control

    class _Session:
        session_id = "x"
        proxy = _Proxy()
        wait_timeout = 5.0
        quiet_for = 0.1

    class _Other:
        session_id = "y"

        def wait(self, **_kwargs: object) -> SessionTrace:
            return control.wait_session("y")

        def trace(self) -> SessionTrace:
            return control.get_session("y")

    with pytest.raises(AssertionError, match="also introduced by session y"):
        Expectation(own, session=_Session()).isolated_from(_Other())
    assert control.waits == 1

    control.waits = 0
    control.gets = 0
    with pytest.raises(AssertionError, match="also introduced by session y"):
        Expectation(own, session=_Session()).isolated_from("y")
    assert control.waits == 1
    assert control.gets == 0


def test_isolated_from_prefers_wait_kwargs_from_asserting_session() -> None:
    seen: list[dict[str, float]] = []
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )
    other = _trace(
        "y",
        _call(session_id="y", ident="call_y1", seq=1, user="Tell a short story."),
    )

    class _Control:
        def wait_session(self, session_id: str, **kwargs: float) -> SessionTrace:
            seen.append(dict(kwargs))
            assert session_id == "y"
            return other

        def get_session(self, session_id: str) -> SessionTrace:
            raise AssertionError(f"must wait, not GET {session_id}")

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 7.0
        quiet_for = 0.2

    Expectation(own, session=_Session()).isolated_from("y")
    assert seen == [{"timeout": 7.0, "quiet": 0.2}]


def test_isolated_from_wait_typeerror_does_not_retry_without_kwargs() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )
    seen: list[dict[str, float]] = []

    class _Session:
        session_id = "x"
        proxy = None
        wait_timeout = 7.0
        quiet_for = 0.2

    class _Other:
        session_id = "y"

        def wait(self, timeout: float = 30.0, quiet: float = 0.5) -> SessionTrace:
            seen.append({"timeout": timeout, "quiet": quiet})
            raise TypeError("internal wait boom")

        def trace(self) -> SessionTrace:
            raise AssertionError("must not fall back to GET after wait TypeError")

    with pytest.raises(TypeError, match="internal wait boom"):
        Expectation(own, session=_Session()).isolated_from(_Other())
    assert seen == [{"timeout": 7.0, "quiet": 0.2}]


def test_isolated_from_wait_without_kwargs_still_runs() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )
    other = _trace(
        "y",
        _call(session_id="y", ident="call_y1", seq=1, user="Tell a short story."),
    )
    waits = {"n": 0}

    class _Session:
        session_id = "x"
        proxy = None
        wait_timeout = 7.0
        quiet_for = 0.2

    class _Other:
        session_id = "y"

        def wait(self) -> SessionTrace:
            waits["n"] += 1
            return other

        def trace(self) -> SessionTrace:
            raise AssertionError("must wait, not GET")

    Expectation(own, session=_Session()).isolated_from(_Other())
    assert waits["n"] == 1


def test_isolated_from_prefetched_trace_is_not_refetched() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )
    other = _trace(
        "y",
        _call(session_id="y", ident="call_y1", seq=1, user="Tell a short story."),
    )

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise AssertionError(f"must not wait on {session_id}")

        def get_session(self, session_id: str) -> SessionTrace:
            raise AssertionError(f"must not GET {session_id}")

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    Expectation(own, session=_Session()).isolated_from(other)


def test_isolated_from_other_only_inflight_names_timeout() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise _wait_timeout(session_id, calls=0, active=1)

        def get_session(self, session_id: str) -> SessionTrace:
            raise SessionNotFound(session_id)

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    class _Other:
        session_id = "z"

        def wait(self, **_kwargs: object) -> SessionTrace:
            raise _wait_timeout(self.session_id, calls=0, active=1)

        def trace(self) -> SessionTrace:
            raise SessionNotFound(self.session_id)

    with pytest.raises(AssertionError, match=r"session z has 1 call still in flight"):
        Expectation(own, session=_Session()).isolated_from(_Other())
    with pytest.raises(AssertionError, match=r"session z has 1 call still in flight"):
        Expectation(own, session=_Session()).isolated_from("z")
    Expectation(own, session=_Session()).allow_empty().isolated_from(_Other())
    Expectation(own, session=_Session()).allow_empty().isolated_from("z")


def test_isolated_from_other_timeout_unknown_active_is_not_no_calls() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise _wait_timeout(session_id, calls=0, active=0)

        def get_session(self, session_id: str) -> SessionTrace:
            return _trace(session_id)

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    with pytest.raises(
        AssertionError,
        match=r"wait timed out on session ghost \(isolated_from needs calls on both sides\)",
    ):
        Expectation(own, session=_Session()).isolated_from("ghost")
    Expectation(own, session=_Session()).allow_empty().isolated_from("ghost")


def test_isolated_from_other_timeout_with_settled_calls_runs_assertion() -> None:
    secret = "Y secret follow-up text"
    own = _trace(
        "x",
        _replay_call(
            session_id="x",
            ident="call_x1",
            seq=1,
            history=secret,
            user="X asks its own question",
        ),
    )
    settled = _trace("y", _call(session_id="y", ident="call_y2", seq=2, user=secret))

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise _wait_timeout(session_id, calls=1, active=1)

        def get_session(self, session_id: str) -> SessionTrace:
            return settled

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    with pytest.raises(AssertionError, match="also introduced by session y") as caught:
        Expectation(own, session=_Session()).isolated_from("y")
    message = str(caught.value)
    assert "wait timed out on session y" in message
    assert "session y has 1 call still in flight" in message


def test_isolated_from_other_get_after_timeout_propagates_errors() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise _wait_timeout(session_id, calls=0, active=1)

        def get_session(self, session_id: str) -> SessionTrace:
            raise RuntimeError(f"control plane down for {session_id}")

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    with pytest.raises(RuntimeError, match="control plane down"):
        Expectation(own, session=_Session()).isolated_from("y")


def test_isolated_from_other_deleted_mid_wait_is_empty() -> None:
    own = _trace(
        "x",
        _call(session_id="x", ident="call_x1", seq=1, user="Weather in Paris?"),
    )

    class _Control:
        def wait_session(self, session_id: str, **_kwargs: object) -> SessionTrace:
            raise SessionNotFound(session_id)

        def get_session(self, session_id: str) -> SessionTrace:
            raise SessionNotFound(session_id)

    class _Session:
        session_id = "x"
        proxy = type("P", (), {"control": _Control()})()
        wait_timeout = 1.0
        quiet_for = 0.1

    with pytest.raises(
        AssertionError,
        match=r"no calls recorded for session gone \(isolated_from needs calls on both sides\)",
    ):
        Expectation(own, session=_Session()).isolated_from("gone")
    Expectation(own, session=_Session()).allow_empty().isolated_from("gone")
