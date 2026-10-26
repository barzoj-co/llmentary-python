from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    INVALID_MODEL,
    WEATHER_CHAT,
    call_openai,
    openai_client,
    openai_invalid_model_errors,
)

from llmentary import OUTCOME_CANCELLED, Session, SessionNotFound, WaitTimeoutError
from llmentary.assertions import Expectation
from llmentary.types import first_present

pytestmark = pytest.mark.e2e

BEFORE_HEADERS_READ_TIMEOUT = 0.05
SLOW_MESSAGES = [{"role": "user", "content": "Count slowly from 1 to 80 in words."}]


def test_raw_passthrough_models_list(llmentary: Session) -> None:
    client = openai_client(llmentary)
    listing = call_openai(lambda: client.models.list())
    models = [item.id for item in listing.data]
    assert models
    trace = llmentary.wait()
    assert trace.calls, trace.trace_summary()
    matching = [
        call for call in trace.calls if (call.path or "").rstrip("/").endswith("/models")
    ]
    assert matching, trace.trace_summary()
    call = matching[0]
    assert call.http_status == 200, trace.trace_summary()
    body = call.raw_response_body
    blob = body if isinstance(body, str) else json.dumps(body, default=str)
    assert models[0] in blob, trace.trace_summary()


def test_assertion_helpers_fail_with_clear_messages(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Use the tool for weather in Paris."}],
            tools=[WEATHER_CHAT],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0,
            max_tokens=32,
        )
    )

    expect = llmentary.expect()
    with pytest.raises(AssertionError, match="expected tool 'missing_tool_ltl' to be called"):
        expect.tool_called("missing_tool_ltl")
    with pytest.raises(AssertionError, match="expected tool 'get_weather' not to be called"):
        expect.not_called("get_weather")
    with pytest.raises(AssertionError, match="to be called 99 time"):
        expect.called_times("get_weather", 99)
    with pytest.raises(AssertionError, match="expected model 'gpt-not-a-model'"):
        expect.model("gpt-not-a-model")
    with pytest.raises(AssertionError, match="expected prompt to contain"):
        expect.prompt_contains("this-prompt-marker-is-absent-ltl")
    with pytest.raises(AssertionError, match="expected response to contain"):
        expect.response_contains("this-response-marker-is-absent-ltl")
    with pytest.raises(AssertionError, match="expected at least one error"):
        expect.has_error()
    with pytest.raises(AssertionError, match="expected total tokens >= 1000000000"):
        expect.tokens(min_total=1_000_000_000)
    with pytest.raises(AssertionError, match="expected finish reason"):
        expect.finish_reason("this-finish-reason-is-absent")

    with pytest.raises(openai_invalid_model_errors()):
        client.chat.completions.create(
            model=INVALID_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
        )
    error_expect = Expectation(llmentary.wait())
    with pytest.raises(AssertionError, match="expected no errors"):
        error_expect.no_errors()


def _cancel_before_headers(llmentary: Session, client: Any) -> bool:
    """One SDK call cancelled before response headers; report what the proxy recorded.

    ``max_retries=0`` (set by ``openai_client``) plus a 50ms read timeout normally
    beats OpenAI's first byte, so the proxy records a ``before_headers`` client
    abort. When the upstream wins the race, ``create`` returns a stream and no
    before_headers abort is recorded.
    """

    from openai import APIConnectionError, APITimeoutError

    mark = llmentary.mark()
    try:
        stream = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=SLOW_MESSAGES,
            temperature=0,
            max_tokens=400,
            stream=True,
        )
    except (APITimeoutError, APIConnectionError, TimeoutError):
        pass
    else:
        stream.close()
    try:
        trace = llmentary.wait(min_calls=mark + 1)
    except WaitTimeoutError:
        return False
    new = trace.since(mark).calls
    return any(call.client_aborted and call.abort_phase == "before_headers" for call in new)


def _finished_calls_at_mark(llmentary: Session, mark: int) -> int:
    """Finished calls at or below ``mark``, counted over the whole session.

    ``wait(min_calls=...)`` counts finished calls session-wide, so a threshold
    for calls expected after a mark must add what the session already finished.
    """

    try:
        calls = llmentary.trace().calls
    except SessionNotFound:
        return 0
    return sum(
        1
        for call in calls
        if call.sequence_number() <= mark and first_present(call.raw, "ended_at", "endedAt")
    )


def test_live_cancelled_calls_have_cancelled_outcome(llmentary: Session) -> None:
    from httpx import Timeout

    cancel_client = openai_client(
        llmentary,
        timeout=Timeout(
            5.0,
            connect=5.0,
            write=5.0,
            read=BEFORE_HEADERS_READ_TIMEOUT,
            pool=5.0,
        ),
    )
    stream_client = openai_client(llmentary)

    mark = llmentary.mark()
    before_headers_won = _cancel_before_headers(llmentary, cancel_client)
    if not before_headers_won:
        mark = llmentary.mark()
        before_headers_won = _cancel_before_headers(llmentary, cancel_client)

    # A won before-headers cancel adds one cancelled call next to the
    # after-headers one; a lost race is dropped from ``mark`` and not counted.
    expected_cancelled = 2 if before_headers_won else 1
    if not before_headers_won:
        reason = (
            "OpenAI sent response headers before the "
            f"{BEFORE_HEADERS_READ_TIMEOUT * 1000:.0f}ms read timeout twice; "
            "skipping the before-headers part and asserting the "
            "after-headers cancel only "
            f"({expected_cancelled} cancelled call expected)"
        )
        print(reason, flush=True)
        warnings.warn(reason, UserWarning, stacklevel=2)
        mark = llmentary.mark()

    # ``min_calls`` counts finished calls over the whole session, so anchor the
    # threshold to the mark: what was already finished there plus what follows.
    finished_at_mark = _finished_calls_at_mark(llmentary, mark)

    stream = call_openai(
        lambda: stream_client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=SLOW_MESSAGES,
            temperature=0,
            max_tokens=400,
            stream=True,
        )
    )
    try:
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta is not None and delta.content:
                break
    finally:
        stream.close()

    trace = llmentary.wait(min_calls=finished_at_mark + expected_cancelled)
    # ``since`` drops a lost-race call from a retried first attempt.
    expect = llmentary.expect(since=mark)
    outcomes = sorted(call.outcome for call in expect.trace.calls)
    assert outcomes == [OUTCOME_CANCELLED] * expected_cancelled, (
        f"expected {expected_cancelled} cancelled call(s)\n\n{trace.trace_summary()}"
    )

    expect.no_errors()
    got = f"expected no cancelled calls, got {expected_cancelled}"
    with pytest.raises(AssertionError, match=got):
        expect.no_cancelled()
    llmentary.query().since(mark).cancelled().expect().call_count(
        expected_cancelled, include_cancelled=True
    )
    expect.call_count(0)
    expect.call_count(expected_cancelled, include_cancelled=True)
