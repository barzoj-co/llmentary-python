from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import pytest

from llmentary import Session
from llmentary.assertions import Expectation
from llmentary.types import CapturedCall, SessionTrace

CHEAP_MODEL = "gpt-4o-mini"
PARALLEL_MODEL = "gpt-4.1-mini"
REASONING_MODEL = "gpt-6-luna"
INVALID_MODEL = "ltl-e2e-invalid-model-does-not-exist"
INVALID_API_KEY = "sk-invalid-ltl-e2e-override"
INVALID_KEY_HEADERS = {"Authorization": f"Bearer {INVALID_API_KEY}"}
# Observed OpenAI error envelope through the live proxy (CI 39127536406):
# unknown model -> 404, type=invalid_request_error, code=model_not_found
# (Chat Completions and Responses).
# Invalid API key -> 401, type=invalid_request_error, code=invalid_api_key.
INVALID_MODEL_STATUS = 404
INVALID_MODEL_ERROR_TYPE = "invalid_request_error"
INVALID_MODEL_ERROR_CODE = "model_not_found"
INVALID_KEY_STATUS = 401
INVALID_KEY_ERROR_TYPE = "invalid_request_error"
INVALID_KEY_ERROR_CODE = "invalid_api_key"
REASONING_PROMPT = (
    "Compute 3847 * 5921 by expanding (3800+47)*(5900+21). "
    "Show each partial product, then reply with the final integer only."
)
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
    if os.environ.get("LLMENTARY_E2E") != "1":
        pytest.skip("set LLMENTARY_E2E=1 to run real OpenAI E2E tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required when LLMENTARY_E2E=1")


def openai_client(
    lm: Session, *, api_key: str | None = None, timeout: float | Any | None = None
) -> Any:
    from openai import OpenAI

    key = api_key if api_key is not None else os.environ["OPENAI_API_KEY"]
    kwargs: dict[str, Any] = {
        "api_key": key,
        "base_url": lm.openai_base_url,
        "default_headers": lm.headers,
        "max_retries": 0,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def openai_invalid_model_errors() -> tuple[type[BaseException], ...]:
    from openai import BadRequestError, NotFoundError

    return (BadRequestError, NotFoundError)


def openai_invalid_key_errors() -> tuple[type[BaseException], ...]:
    from openai import AuthenticationError, PermissionDeniedError

    return (AuthenticationError, PermissionDeniedError)


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
    assert " ".join(left.split()) == " ".join(right.split()), (left, right)


def assert_recorded_error(
    trace: SessionTrace,
    *,
    pattern: str,
    http_status: int,
    error_type: str,
    error_code: str,
) -> CapturedCall:
    Expectation(trace).has_error(pattern)
    errored = [call for call in trace.calls if call.error or (call.http_status or 0) >= 400]
    assert errored, trace.trace_summary()
    call = errored[0]
    summary = (
        f"http_status={call.http_status} error_type={call.error_type!r} "
        f"error_code={call.error_code!r}\n{trace.trace_summary()}"
    )
    assert call.http_status == http_status, summary
    assert call.error_type == error_type, summary
    assert call.error_code == error_code, summary
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


def recorded_client_aborted(calls: Sequence[CapturedCall]) -> bool:
    """True when the Go proxy recorded a client disconnect.

    The concrete signal is ``normalized.client_aborted`` (CI dump also set
    ``stream_truncated=True`` and still populated ``ended_at``). Missing
    ``ended_at`` is not evidence of cancellation.
    """

    return any(call.client_aborted for call in calls)
