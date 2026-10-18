from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import pytest

from llmtestlab import LabSession
from llmtestlab.assertions import Expectation
from llmtestlab.types import CapturedCall, SessionTrace

CHEAP_MODEL = "gpt-4.1-nano"
PARALLEL_MODEL = "gpt-4.1-mini"
REASONING_MODEL = "o4-mini"
INVALID_MODEL = "ltl-e2e-invalid-model-does-not-exist"
INVALID_API_KEY = "sk-invalid-ltl-e2e-override"
INVALID_KEY_HEADERS = {"Authorization": f"Bearer {INVALID_API_KEY}"}
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "ok": {"type": "boolean"},
    },
    "required": ["city", "ok"],
    "additionalProperties": False,
}

T = TypeVar("T")


def require_e2e() -> None:
    if os.environ.get("LLMTESTLAB_E2E") != "1":
        pytest.skip("set LLMTESTLAB_E2E=1 to run real OpenAI E2E tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required when LLMTESTLAB_E2E=1")


def openai_client(lab: LabSession, *, api_key: str | None = None) -> Any:
    from openai import OpenAI

    key = api_key if api_key is not None else os.environ["OPENAI_API_KEY"]
    return OpenAI(
        api_key=key,
        base_url=lab.openai_base_url,
        default_headers=lab.headers,
        max_retries=0,
    )


def openai_client_errors() -> tuple[type[BaseException], ...]:
    from openai import APIStatusError, AuthenticationError, BadRequestError, NotFoundError

    return (APIStatusError, AuthenticationError, BadRequestError, NotFoundError)


def call_openai(operation: Callable[[], T]) -> T:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    last: BaseException | None = None
    for attempt in range(3):
        try:
            return operation()
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as exc:
            last = exc
        except APIStatusError as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                last = exc
            else:
                raise
        if attempt < 2:
            time.sleep(0.4 * (2**attempt))
    assert last is not None
    raise last


def chat_function_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def responses_function_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }


WEATHER_CHAT = chat_function_tool("get_weather", "Get weather for a city.")
TIME_CHAT = chat_function_tool("get_time", "Get the current time for a city.")
WEATHER_RESPONSES = responses_function_tool("get_weather", "Get weather for a city.")
TIME_RESPONSES = responses_function_tool("get_time", "Get the current time for a city.")


def assert_plain_text_recorded(expect: Expectation, *, model: str = CHEAP_MODEL) -> None:
    expect.model(model).tokens(min_prompt=1, min_completion=1, min_total=1).no_errors()
    reasons = [call.finish_reason for call in expect.trace.calls if call.finish_reason]
    assert reasons, expect.trace.trace_summary()
    texts = [call.response_text for call in expect.trace.calls if call.response_text.strip()]
    assert texts, expect.trace.trace_summary()


def assert_streamed(trace: SessionTrace) -> None:
    streamed = [call for call in trace.calls if call.streamed]
    assert streamed, f"expected a streamed call\n\n{trace.trace_summary()}"


def assert_assembled_matches_recorded(assembled: str, recorded: str) -> None:
    left, right = assembled.strip(), recorded.strip()
    assert left, "expected concatenated stream deltas"
    assert right, "expected recorded output text"
    assert left == right or left in right or right in left, (left, right)


def error_fields(call: CapturedCall) -> tuple[int | None, str | None, str | None]:
    return call.http_status, call.error_type, call.error_code


def assert_recorded_error(trace: SessionTrace, *, pattern: str) -> CapturedCall:
    Expectation(trace).has_error(pattern)
    errored = [call for call in trace.calls if call.error or (call.http_status or 0) >= 400]
    assert errored, trace.trace_summary()
    call = errored[0]
    status, err_type, err_code = error_fields(call)
    assert status is not None and status >= 400, trace.trace_summary()
    assert err_type or err_code or call.error, trace.trace_summary()
    return call


def assemble_chat_text(stream: Any) -> str:
    parts: list[str] = []
    for event in stream:
        for choice in getattr(event, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str) and content:
                parts.append(content)
    return "".join(parts)


def assemble_chat_tool_arguments(stream: Any) -> str:
    parts: list[str] = []
    for event in stream:
        for choice in getattr(event, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            tool_calls = getattr(delta, "tool_calls", None) if delta is not None else None
            if not tool_calls:
                continue
            for tool in tool_calls:
                function = getattr(tool, "function", None)
                arguments = getattr(function, "arguments", None) if function is not None else None
                if isinstance(arguments, str) and arguments:
                    parts.append(arguments)
    return "".join(parts)


def assemble_responses_text(stream: Any) -> str:
    parts: list[str] = []
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if isinstance(delta, str) and delta:
                parts.append(delta)
    return "".join(parts)


def assemble_responses_tool_arguments(stream: Any) -> str:
    parts: list[str] = []
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type in {
            "response.function_call_arguments.delta",
            "response.output_item.delta",
        }:
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                parts.append(delta)
                continue
            if isinstance(delta, Mapping):
                arguments = delta.get("arguments")
                if isinstance(arguments, str) and arguments:
                    parts.append(arguments)
    return "".join(parts)


def responses_function_call(response: Any) -> Any:
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            return item
    raise AssertionError(f"no function_call in Responses output: {response.output!r}")


def first_call(trace: SessionTrace) -> CapturedCall:
    assert trace.calls, trace.trace_summary()
    return trace.calls[0]


def tool_argument_json(trace: SessionTrace, name: str) -> Mapping[str, Any]:
    for call in trace.calls:
        for tool in call.tool_calls:
            if tool.name == name:
                if tool.arguments:
                    return tool.arguments
                if tool.raw_arguments:
                    parsed = json.loads(tool.raw_arguments)
                    assert isinstance(parsed, Mapping)
                    return parsed
    raise AssertionError(f"no arguments for {name!r}\n\n{trace.trace_summary()}")


def recorded_incomplete_or_errored(calls: Sequence[CapturedCall]) -> bool:
    for call in calls:
        if call.error or call.mid_stream_error:
            return True
        if call.http_status is not None and call.http_status >= 400:
            return True
        ended = call.raw.get("ended_at")
        if ended in (None, ""):
            return True
        if call.streamed and call.finish_reason in {None, "length"}:
            return True
    return False
