from __future__ import annotations

import pytest
from helpers import (
    CHEAP_MODEL,
    INVALID_MODEL,
    WEATHER_CHAT,
    call_openai,
    openai_client,
    openai_client_errors,
)

from llmtestlab import LabSession
from llmtestlab.assertions import Expectation

pytestmark = pytest.mark.e2e


def test_raw_passthrough_models_list(lab: LabSession) -> None:
    client = openai_client(lab)
    listing = call_openai(lambda: client.models.list())
    models = [item.id for item in listing.data]
    assert models
    trace = lab.wait()
    assert trace.calls, trace.trace_summary()
    matching = [
        call
        for call in trace.calls
        if (call.path or "").rstrip("/").endswith("/models")
        or "models" in str(call.raw.get("path") or "")
        or "/models" in str(call.raw)
    ]
    assert matching, trace.trace_summary()
    call = matching[0]
    assert call.http_status in {None, 200} or (call.http_status or 0) < 400
    raw_blob = str(call.raw)
    assert "gpt" in raw_blob.lower() or "data" in raw_blob.lower() or models[0] in raw_blob


def test_assertion_helpers_fail_with_clear_messages(lab: LabSession) -> None:
    client = openai_client(lab)
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

    expect = lab.expect()
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

    with pytest.raises(openai_client_errors()):
        client.chat.completions.create(
            model=INVALID_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
        )
    error_expect = Expectation(lab.wait())
    with pytest.raises(AssertionError, match="expected no errors"):
        error_expect.no_errors()
