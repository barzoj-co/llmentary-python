from __future__ import annotations

import json

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    INVALID_MODEL,
    WEATHER_CHAT,
    call_openai,
    openai_client,
    openai_invalid_model_errors,
)

from llmentary import Session
from llmentary.assertions import Expectation

pytestmark = pytest.mark.e2e


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
