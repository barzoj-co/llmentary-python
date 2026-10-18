from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, dict[str, "JSONValue"], list["JSONValue"]]
JSONObject = dict[str, JSONValue]
APIType = Literal["chat_completions", "responses", "chat", "response"]


@dataclass(frozen=True)
class ToolCall:
    """Normalized view of a tool call captured by the proxy."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    raw_arguments: Optional[str] = None
    call_id: Optional[str] = None
    index: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapturedCall:
    """A single upstream LLM call as returned by the control API.

    The raw object is preserved. Properties derive the common OpenAI Chat
    Completions and Responses fields used by the assertion layer.
    """

    raw: Mapping[str, Any]
    index: int

    @property
    def api_type(self) -> Optional[str]:
        value = first_present(
            self.raw,
            "api_type",
            "apiType",
            "api",
            "type",
            "kind",
            "endpoint_type",
            "endpointType",
        )
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_")
            if normalized in {"chat", "chat_completion", "chat_completions"}:
                return "chat_completions"
            if normalized in {"response", "responses"}:
                return "responses"

        path = self.path
        if path:
            if "/chat/completions" in path:
                return "chat_completions"
            if "/responses" in path:
                return "responses"
        return None

    @property
    def path(self) -> Optional[str]:
        value = first_present(self.raw, "path", "url", "endpoint")
        if isinstance(value, str):
            return value
        request = mapping_at(self.raw, "request")
        value = first_present(request, "path", "url", "endpoint") if request else None
        return value if isinstance(value, str) else None

    @property
    def request(self) -> Mapping[str, Any]:
        request = mapping_at(self.raw, "request")
        if not request:
            return {}
        body = mapping_at(request, "body")
        if body:
            return body
        json_body = mapping_at(request, "json")
        if json_body:
            return json_body
        return request

    @property
    def response(self) -> Mapping[str, Any]:
        response = mapping_at(self.raw, "response")
        if not response:
            return {}
        body = mapping_at(response, "body")
        if body:
            return body
        json_body = mapping_at(response, "json")
        if json_body:
            return json_body
        return response

    @property
    def model(self) -> Optional[str]:
        for source in (self.raw, self.request, self.response):
            value = first_present(source, "model")
            if isinstance(value, str):
                return value
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        value = first_present(self.raw, "finish_reason", "finishReason")
        if isinstance(value, str):
            return value

        choices = sequence_at(self.response, "choices")
        if choices:
            choice0 = as_mapping(choices[0])
            value = first_present(choice0, "finish_reason", "finishReason")
            if isinstance(value, str):
                return value

        output = sequence_at(self.response, "output")
        if output:
            reasons = [
                str(reason)
                for item in output
                for reason in [first_present(as_mapping(item), "finish_reason", "finishReason")]
                if reason
            ]
            if reasons:
                return ",".join(reasons)
        return None

    @property
    def usage(self) -> Mapping[str, Any]:
        usage = mapping_at(self.raw, "usage") or mapping_at(self.response, "usage")
        return usage or {}

    @property
    def latency_ms(self) -> Optional[float]:
        return numeric_at_any(
            self.raw,
            ("latency_ms", "latencyMs", "duration_ms", "durationMs"),
            nested=("timings", "metrics"),
        )

    @property
    def ttft_ms(self) -> Optional[float]:
        return numeric_at_any(
            self.raw,
            ("ttft_ms", "ttftMs", "time_to_first_token_ms", "timeToFirstTokenMs"),
            nested=("timings", "metrics"),
        )

    @property
    def retries(self) -> int:
        value = first_present(self.raw, "retries", "retry_count", "retryCount")
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    @property
    def error(self) -> Optional[Any]:
        return first_present(self.raw, "error", "errors")

    @property
    def mid_stream_error(self) -> Optional[Any]:
        return first_present(
            self.raw,
            "mid_stream_error",
            "midStreamError",
            "stream_error",
            "streamError",
        )

    @property
    def prompt_text(self) -> str:
        return stringify_messages(self.request)

    @property
    def response_text(self) -> str:
        return stringify_response(self.response)

    @property
    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        calls.extend(extract_tool_calls(self.raw, len(calls)))
        calls.extend(extract_tool_calls(self.response, len(calls)))
        return dedupe_tool_calls(calls)

    @property
    def tool_results(self) -> list[Mapping[str, Any]]:
        return extract_tool_results(self.request)


@dataclass(frozen=True)
class SessionTrace:
    """Control API session payload plus normalized captured calls."""

    session_id: str
    raw: Mapping[str, Any]
    calls: Sequence[CapturedCall]

    @classmethod
    def from_payload(cls, session_id: str, payload: Mapping[str, Any]) -> SessionTrace:
        raw_calls = find_call_list(payload)
        calls = [CapturedCall(raw=call, index=index) for index, call in enumerate(raw_calls)]
        return cls(session_id=session_id, raw=payload, calls=calls)

    def trace_summary(self) -> str:
        if not self.calls:
            return f"llm-test-lab session {self.session_id}: no captured calls"

        lines = [f"llm-test-lab session {self.session_id}: {len(self.calls)} captured call(s)"]
        for call in self.calls:
            tool_names = ", ".join(tool.name for tool in call.tool_calls) or "-"
            status = "error" if call.error or call.mid_stream_error else "ok"
            latency = f", latency={call.latency_ms:.1f}ms" if call.latency_ms is not None else ""
            ttft = f", ttft={call.ttft_ms:.1f}ms" if call.ttft_ms is not None else ""
            finish = f", finish={call.finish_reason}" if call.finish_reason else ""
            lines.append(
                f"  [{call.index}] api={call.api_type or '?'} model={call.model or '?'} "
                f"status={status}{finish}{latency}{ttft} tools=[{tool_names}]"
            )
            if call.error:
                lines.append(f"      error: {call.error}")
            if call.mid_stream_error:
                lines.append(f"      mid-stream error: {call.mid_stream_error}")
        return "\n".join(lines)

    def filtered(
        self,
        *,
        api_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SessionTrace:
        calls = list(self.calls)
        if api_type:
            normalized = normalize_api_type(api_type)
            calls = [call for call in calls if call.api_type == normalized]
        if model:
            calls = [call for call in calls if call.model == model]
        return SessionTrace(session_id=self.session_id, raw=self.raw, calls=calls)


def normalize_api_type(api_type: str) -> str:
    normalized = api_type.lower().replace("-", "_")
    if normalized in {"chat", "chat_completion", "chat_completions"}:
        return "chat_completions"
    if normalized in {"response", "responses"}:
        return "responses"
    return normalized


def find_call_list(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("calls", "requests", "records", "events", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]

    session = payload.get("session")
    if isinstance(session, Mapping):
        return find_call_list(session)
    return []


def first_present(mapping: Mapping[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def mapping_at(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def sequence_at(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = mapping.get(key)
    return value if isinstance(value, list) else []


def numeric_at_any(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
    *,
    nested: Iterable[str] = (),
) -> Optional[float]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for nested_key in nested:
        nested_mapping = mapping_at(mapping, nested_key)
        if nested_mapping:
            value = numeric_at_any(nested_mapping, keys)
            if value is not None:
                return value
    return None


def extract_tool_calls(mapping: Mapping[str, Any], start_index: int) -> list[ToolCall]:
    calls: list[ToolCall] = []

    direct = sequence_at(mapping, "tool_calls") or sequence_at(mapping, "toolCalls")
    calls.extend(tool_call_from_raw(item, start_index + len(calls)) for item in direct)

    choices = sequence_at(mapping, "choices")
    for choice in choices:
        choice_map = as_mapping(choice)
        message = mapping_at(choice_map, "message") or mapping_at(choice_map, "delta")
        nested = sequence_at(message, "tool_calls") or sequence_at(message, "toolCalls")
        calls.extend(tool_call_from_raw(item, start_index + len(calls)) for item in nested)

    output = sequence_at(mapping, "output")
    for item in output:
        item_map = as_mapping(item)
        if item_map.get("type") in {"function_call", "custom_tool_call", "tool_call"}:
            calls.append(tool_call_from_raw(item_map, start_index + len(calls)))

    return [call for call in calls if call.name]


def tool_call_from_raw(value: Any, index: int) -> ToolCall:
    raw = as_mapping(value)
    function = mapping_at(raw, "function")
    name = first_present(raw, "name") or first_present(function, "name")
    args = first_present(raw, "arguments") or first_present(function, "arguments")
    parsed_args = parse_args(args)
    return ToolCall(
        name=str(name or ""),
        arguments=parsed_args,
        raw_arguments=args if isinstance(args, str) else None,
        call_id=string_or_none(first_present(raw, "id", "call_id", "callId")),
        index=index,
        raw=raw,
    )


def dedupe_tool_calls(calls: Sequence[ToolCall]) -> list[ToolCall]:
    seen: set[tuple[Optional[str], str, str]] = set()
    deduped: list[ToolCall] = []
    for call in calls:
        key = (call.call_id, call.name, repr(sorted(call.arguments.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def parse_args(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value:
        return {}
    import json

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def extract_tool_results(mapping: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    results: list[Mapping[str, Any]] = []

    messages = sequence_at(mapping, "messages")
    for message in messages:
        message_map = as_mapping(message)
        if message_map.get("role") == "tool":
            results.append(message_map)

    input_value = mapping.get("input")
    if isinstance(input_value, list):
        for item in input_value:
            item_map = as_mapping(item)
            if item_map.get("type") in {"function_call_output", "tool_result"}:
                results.append(item_map)
    return results


def stringify_messages(mapping: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("messages", "input", "prompt"):
        value = mapping.get(key)
        if value is None:
            continue
        parts.append(stringify_value(value))
    return "\n".join(parts)


def stringify_response(mapping: Mapping[str, Any]) -> str:
    parts: list[str] = []
    output_text = mapping.get("output_text")
    if isinstance(output_text, str):
        parts.append(output_text)

    choices = sequence_at(mapping, "choices")
    for choice in choices:
        message = mapping_at(as_mapping(choice), "message")
        content = message.get("content")
        if content is not None:
            parts.append(stringify_value(content))

    output = sequence_at(mapping, "output")
    for item in output:
        item_map = as_mapping(item)
        if item_map.get("type") == "message":
            parts.append(stringify_value(item_map.get("content")))
        elif "content" in item_map:
            parts.append(stringify_value(item_map.get("content")))

    if not parts:
        parts.append(stringify_value(mapping))
    return "\n".join(part for part in parts if part)


def stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(stringify_value(item) for item in value)
    if isinstance(value, Mapping):
        content = value.get("text") or value.get("content")
        if isinstance(content, str):
            return content
        import json

        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None
