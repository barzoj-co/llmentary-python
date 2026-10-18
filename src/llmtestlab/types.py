from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, dict[str, "JSONValue"], list["JSONValue"]]
JSONObject = dict[str, JSONValue]
APIType = Literal["chat_completions", "responses", "chat", "response"]

REQUEST_KEYS = (
    "request",
    "req",
    "http_request",
    "httpRequest",
    "raw_request",
    "rawRequest",
)
RESPONSE_KEYS = (
    "response",
    "resp",
    "http_response",
    "httpResponse",
    "raw_response",
    "rawResponse",
)
BODY_KEYS = ("body", "json", "payload", "Body")
LLM_PAYLOAD_KEYS = (
    "messages",
    "input",
    "output",
    "choices",
    "model",
    "output_text",
    "tool_calls",
    "tools",
)
PATH_KEYS = ("path", "url", "endpoint")


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
    def normalized_payload(self) -> Mapping[str, Any]:
        return coerce_mapping(self.raw.get("normalized"))

    @property
    def api_type(self) -> Optional[str]:
        for source in (self.raw, self.normalized_payload):
            value = first_present(
                source,
                "api_type",
                "apiType",
                "api_family",
                "apiFamily",
                "api",
                "type",
                "kind",
                "endpoint_type",
                "endpointType",
            )
            mapped = normalize_api_type(value) if isinstance(value, str) else None
            if mapped in {"chat_completions", "responses"}:
                return mapped

        path = self.path or ""
        if "/chat/completions" in path:
            return "chat_completions"
        if "/responses" in path:
            return "responses"

        request = self.request
        response = self.response
        if "messages" in request:
            return "chat_completions"
        if any(key in request or key in response for key in ("input", "output", "output_text")):
            return "responses"
        return None

    @property
    def path(self) -> Optional[str]:
        sources: list[Mapping[str, Any]] = [self.raw, self.normalized_payload]
        for key in REQUEST_KEYS:
            if key in self.raw and self.raw[key] is not None:
                decoded = coerce_mapping(self.raw[key])
                if decoded:
                    sources.append(decoded)
        for source in sources:
            value = first_present(source, *PATH_KEYS)
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def request(self) -> Mapping[str, Any]:
        return extract_llm_side(self.raw, REQUEST_KEYS, side="request")

    @property
    def response(self) -> Mapping[str, Any]:
        return extract_llm_side(self.raw, RESPONSE_KEYS, side="response")

    @property
    def model(self) -> Optional[str]:
        # Prefer the request model: OpenAI responses often echo a dated snapshot ID.
        for source in (self.request, self.raw, self.normalized_payload, self.response):
            value = first_present(source, "model")
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        for source in (self.raw, self.normalized_payload):
            value = first_present(source, "finish_reason", "finishReason")
            if isinstance(value, str) and value:
                return value

        choices = sequence_at(self.response, "choices")
        if choices:
            choice0 = as_mapping(choices[0])
            value = first_present(choice0, "finish_reason", "finishReason")
            if isinstance(value, str) and value:
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
        for source in (self.raw, self.normalized_payload, self.response):
            usage = mapping_at(source, "usage")
            if usage:
                return usage
        return {}

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
            nested=("timings", "metrics", "normalized"),
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
        # Only the model output counts as an invocation. Do not walk request
        # messages: those echo prior assistant tool_calls on later turns.
        calls: list[ToolCall] = []
        normalized = self.normalized_payload
        if normalized:
            calls.extend(extract_tool_calls(normalized, len(calls)))
            for key in ("response", "body", "result"):
                nested = decode_value(normalized.get(key))
                if isinstance(nested, Mapping):
                    calls.extend(extract_tool_calls(nested, len(calls)))
        calls.extend(extract_tool_calls(self.response, len(calls)))
        return dedupe_tool_calls(calls)

    @property
    def tool_results(self) -> list[Mapping[str, Any]]:
        results = extract_tool_results(self.request)
        normalized = self.normalized_payload
        if normalized:
            results.extend(extract_tool_results(coerce_mapping(normalized.get("request"))))
            results.extend(extract_tool_results(normalized))
        return results


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
            return (
                f"llm-test-lab session {self.session_id}: no captured calls "
                f"(payload keys={list(self.raw.keys())})"
            )

        lines = [f"llm-test-lab session {self.session_id}: {len(self.calls)} captured call(s)"]
        for call in self.calls:
            tool_names = ", ".join(tool.name for tool in call.tool_calls) or "-"
            status = "error" if call.error or call.mid_stream_error else "ok"
            latency = f", latency={call.latency_ms:.1f}ms" if call.latency_ms is not None else ""
            ttft = f", ttft={call.ttft_ms:.1f}ms" if call.ttft_ms is not None else ""
            finish = f", finish={call.finish_reason}" if call.finish_reason else ""
            api_label = call.api_type or first_present(call.raw, "api_family", "apiFamily") or "?"
            lines.append(
                f"  [{call.index}] api={api_label} model={call.model or '?'} "
                f"status={status}{finish}{latency}{ttft} tools=[{tool_names}]"
            )
            if not call.tool_calls:
                lines.append(f"      call keys={list(call.raw.keys())}")
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
            calls = [call for call in calls if model_matches(call.model, model)]
        return SessionTrace(session_id=self.session_id, raw=self.raw, calls=calls)


def model_matches(actual: Optional[str], expected: str) -> bool:
    if not actual:
        return False
    if actual == expected:
        return True
    prefix = expected + "-"
    if not actual.startswith(prefix):
        return False
    # OpenAI snapshot ids are `{model}-{YYYY}-{MM}-{DD}`, not sibling names like mini/nano.
    parts = actual[len(prefix) :].split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and len(parts[2]) == 2
        and all(part.isdigit() for part in parts)
    )


def normalize_api_type(api_type: str) -> str:
    normalized = api_type.lower().replace("-", "_").replace(".", "_").replace("/", "_")
    collapsed = "_".join(part for part in normalized.split("_") if part)
    if collapsed in {"chat", "chat_completion", "chat_completions"}:
        return "chat_completions"
    if "chat" in collapsed and "completion" in collapsed:
        return "chat_completions"
    if collapsed in {"response", "responses"} or "response" in collapsed:
        return "responses"
    return collapsed


def find_call_list(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in (
        "calls",
        "requests",
        "records",
        "events",
        "entries",
        "spans",
        "llm_calls",
        "captured",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            mapped = [coerce_mapping(item) for item in value]
            return [item for item in mapped if item]
        if isinstance(value, Mapping):
            mapped = [coerce_mapping(item) for item in value.values()]
            found = [item for item in mapped if item]
            if found:
                return found

    session = payload.get("session")
    if isinstance(session, Mapping):
        return find_call_list(session)
    return []


def first_present(mapping: Mapping[str, Any], *keys: str) -> Optional[Any]:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None or value == "":
            continue
        return value
    return None


def extract_llm_side(
    raw: Mapping[str, Any],
    keys: Sequence[str],
    *,
    side: Literal["request", "response"],
) -> Mapping[str, Any]:
    fallback: Mapping[str, Any] = {}
    for key in keys:
        if key not in raw or raw[key] is None:
            continue
        for candidate in candidate_mappings(raw[key]):
            if looks_like_llm_payload(candidate, side=side):
                return candidate
            if not fallback:
                fallback = candidate

    normalized = coerce_mapping(raw.get("normalized"))
    if normalized:
        nested = first_present(normalized, side, "body")
        if nested is not None:
            for candidate in candidate_mappings(nested):
                if looks_like_llm_payload(candidate, side=side):
                    return candidate
                if not fallback:
                    fallback = candidate
        if looks_like_llm_payload(normalized, side=side):
            return normalized
    return fallback


def candidate_mappings(value: Any) -> list[Mapping[str, Any]]:
    decoded = decode_value(value)
    candidates: list[Mapping[str, Any]] = []
    if isinstance(decoded, Mapping):
        candidates.append(decoded)
        for key in BODY_KEYS:
            nested = first_present(decoded, key)
            if nested is None:
                continue
            nested_decoded = decode_value(nested)
            if isinstance(nested_decoded, Mapping):
                candidates.append(nested_decoded)
    elif isinstance(decoded, str):
        extracted = extract_json_object(decoded)
        if extracted is not None:
            candidates.append(extracted)
    return candidates


def looks_like_llm_payload(
    mapping: Mapping[str, Any],
    *,
    side: Optional[Literal["request", "response"]] = None,
) -> bool:
    if side == "request":
        return any(key in mapping for key in ("messages", "input", "tools", "model", "prompt"))
    if side == "response":
        return any(
            key in mapping
            for key in ("choices", "output", "output_text", "tool_calls", "model", "usage")
        )
    return any(key in mapping for key in LLM_PAYLOAD_KEYS)


def coerce_mapping(value: Any) -> Mapping[str, Any]:
    decoded = decode_value(value)
    if isinstance(decoded, Mapping):
        return decoded
    if isinstance(decoded, str):
        extracted = extract_json_object(decoded)
        if extracted is not None:
            return extracted
    return {}


def decode_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return value
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        stripped = data.lstrip()
        if stripped.startswith((b"{", b"[")):
            try:
                return decode_value(json.loads(data), depth=depth + 1)
            except json.JSONDecodeError:
                pass
        try:
            return decode_value(data.decode("utf-8"), depth=depth + 1)
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return decode_value(parsed, depth=depth + 1)

    decoded_b64 = decode_base64_json(text)
    if decoded_b64 is not None:
        return decode_value(decoded_b64, depth=depth + 1)

    extracted = extract_json_object(text)
    return extracted if extracted is not None else value


def decode_base64_json(text: str) -> Any:
    compact = "".join(text.split())
    if len(compact) < 8:
        return None
    try:
        padding = (-len(compact)) % 4
        decoded = base64.b64decode(compact + ("=" * padding), validate=False)
    except (ValueError, binascii.Error):
        return None
    stripped = decoded.lstrip()
    if stripped.startswith((b"{", b"[")):
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            return None
    try:
        as_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return extract_json_object(as_text)


def extract_json_object(text: str) -> Optional[Mapping[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def as_mapping(value: Any) -> Mapping[str, Any]:
    return coerce_mapping(value)


def mapping_at(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return coerce_mapping(mapping.get(key))


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
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None
