from __future__ import annotations

import os

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    JSON_SCHEMA,
    WEATHER_CHAT,
    call_openai,
    openai_client,
)

from llmentary import Session

pytestmark = pytest.mark.e2e

TOOL_RESULT = '{"temperature":20,"condition":"sunny","token":"SEED_X"}'


def test_path_attributed_openai_call(
    llmentary: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openai import OpenAI

    monkeypatch.setenv("OPENAI_BASE_URL", llmentary.path_base_url())
    monkeypatch.setenv("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
    client = OpenAI(max_retries=0)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=16,
        )
    )
    llmentary.expect().prompt_contains("ping").tokens(min_total=1).no_errors()


def test_mark_since_and_without_tools(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Return JSON with city Paris and ok true. No other text.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "route", "schema": JSON_SCHEMA, "strict": True},
            },
            temperature=0,
            max_tokens=32,
        )
    )
    llmentary.wait()
    mark = llmentary.mark()

    first = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Use the tool for weather in Paris."}],
            tools=[WEATHER_CHAT],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0,
            max_tokens=32,
        )
    )
    tool_call = first.choices[0].message.tool_calls[0]
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {"role": "user", "content": "Use the tool for weather in Paris."},
                first.choices[0].message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": '{"temperature":20,"condition":"sunny"}',
                },
            ],
            tools=[WEATHER_CHAT],
            temperature=0,
            max_tokens=32,
        )
    )

    since = llmentary.expect(since=mark)
    assert len(since.trace.calls) == 2, since.trace.trace_summary()
    since.tool_called("get_weather").tool_result_fed_back("get_weather")
    with pytest.raises(AssertionError):
        since.prompt_contains("Return JSON")
    llmentary.query().without_tools().expect().call_count(1).request_param(
        "response_format.type", "json_schema"
    )
    llmentary.expect().call_count(min=3, max=3)


def test_empty_since_mark_is_not_vacuous(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=16,
        )
    )
    llmentary.wait()
    mark = llmentary.mark()
    with pytest.raises(AssertionError, match="no calls matched query") as caught:
        llmentary.expect(since=mark).no_errors()
    message = str(caught.value)
    assert f"no calls matched query (since={mark})" in message
    with pytest.raises(AssertionError, match="no calls matched query"):
        llmentary.expect(since=mark).not_called("get_weather")
    llmentary.expect(since=mark).allow_empty().no_errors()
    llmentary.expect().call_count(1).no_errors()


@pytest.mark.parametrize("repeat", [1, 2, 3])
def test_tool_roundtrip_mentions_any(llmentary: Session, repeat: int) -> None:
    client = openai_client(llmentary)
    first = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Call get_weather for Paris, then mention the temperature "
                        "number and the condition word from the tool result."
                    ),
                }
            ],
            tools=[WEATHER_CHAT],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0,
            max_tokens=32,
        )
    )
    tool_call = first.choices[0].message.tool_calls[0]
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Call get_weather for Paris, then mention the temperature "
                        "number and the condition word from the tool result."
                    ),
                },
                first.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": TOOL_RESULT},
            ],
            tools=[WEATHER_CHAT],
            temperature=0,
            max_tokens=64,
        )
    )
    llmentary.expect().tool_result_in_prompt("get_weather").prompt_contains(
        "SEED_X"
    ).final_response_mentions_any("20", "sunny")
