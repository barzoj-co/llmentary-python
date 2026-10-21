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
class Validation:
    """Per-call spec-validation metadata (was it checked, and how)."""

    mode: Optional[str] = None
    profile: Optional[str] = None
    spec_sha: Optional[str] = None
    schema_subset_retrieved: Optional[str] = None
    request_checked: Optional[bool] = None
    response_checked: Optional[bool] = None
    skipped_reason: Optional[str] = None
    duration_us: Optional[int] = None
    truncated: Optional[bool] = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecViolation:
    """One spec-validation finding from ``normalized.spec_violations``."""

    side: Optional[str] = None
    source: Optional[str] = None
    path: Optional[str] = None
    pointer: Optional[str] = None
    keyword: Optional[str] = None
    message: Optional[str] = None
    profile: Optional[str] = None
    schema_ref: Optional[str] = None
    offset: Optional[int] = None
    event_index: Optional[int] = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IgnoredField:
    """One ignored-field warning from ``normalized.ignored_fields``."""

    side: Optional[str] = None
    path: Optional[str] = None
    pointer: Optional[str] = None
    kind: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    hint: Optional[str] = None
    key: Optional[str] = None
    action: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionInfo:
    """Cheap session list row from ``GET /sessions`` (no call bodies)."""

    id: str
    calls: int = 0
    active: int = 0
    sources: tuple[str, ...] = ()
    first_started_at: Optional[str] = None
    last_ended_at: Optional[str] = None
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
        upstream = coerce_mapping(self.raw.get("upstream"))
        if upstream:
            sources.append(upstream)
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

        for source in (self.response, self.normalized_payload, self.raw):
            value = first_present(source, "status")
            if isinstance(value, str) and value and not value.isdigit():
                return value
        return None

    @property
    def usage(self) -> Mapping[str, Any]:
        sources: list[Mapping[str, Any]] = [self.raw, self.normalized_payload, self.response]
        nested = coerce_mapping(self.normalized_payload.get("response"))
        if nested:
            sources.append(nested)
        raw_body = self.raw_response_body
        if isinstance(raw_body, Mapping):
            sources.append(raw_body)
        for source in sources:
            usage = mapping_at(source, "usage")
            if usage:
                return usage
        return {}

    @property
    def raw_response_body(self) -> Any:
        return decode_value(first_present(self.raw, "raw_response", "rawResponse"))

    @property
    def streamed(self) -> bool:
        value = first_present(self.raw, "stream", "streaming", "is_stream")
        if isinstance(value, bool):
            return value
        request_stream = self.request.get("stream")
        return request_stream is True

    @property
    def http_status(self) -> Optional[int]:
        value = first_present(self.raw, "http_status", "httpStatus", "status_code", "statusCode")
        if isinstance(value, bool):
            value = None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        wrapped = coerce_mapping(self.raw.get("response"))
        nested = first_present(wrapped, "status", "status_code", "http_status", "httpStatus")
        if isinstance(nested, int) and not isinstance(nested, bool):
            return nested
        return None

    @property
    def cached_tokens(self) -> Optional[int]:
        return usage_detail(
            self.usage,
            "cached_tokens",
            "cachedTokens",
            "cached_input_tokens",
            "cachedInputTokens",
        )

    @property
    def reasoning_tokens(self) -> Optional[int]:
        return usage_detail(self.usage, "reasoning_tokens", "reasoningTokens")

    @property
    def client_aborted(self) -> bool:
        """True when the client disconnected before the proxy finished the call.

        Prefers the top-level Call field (DOG-27). Older proxies only set
        ``normalized.client_aborted``; that remains the fallback.
        """

        top = first_present(self.raw, "client_aborted", "clientAborted")
        if isinstance(top, bool):
            return top
        return bool_flag(self.normalized_payload, "client_aborted", "clientAborted")

    @property
    def abort_phase(self) -> Optional[str]:
        """``before_headers`` or ``after_headers`` when aborted; else ``None``.

        Prefers the top-level Call field, then normalized, then a ``phase``
        alias when it is one of those two values.
        """

        for mapping in (self.raw, self.normalized_payload):
            value = first_present(mapping, "abort_phase", "abortPhase")
            if isinstance(value, str) and value:
                return value
            phase = first_present(mapping, "phase")
            if isinstance(phase, str) and phase in {"before_headers", "after_headers"}:
                return phase
        return None

    @property
    def stream_truncated(self) -> bool:
        return self._flag("stream_truncated", "streamTruncated")

    def _flag(self, *keys: str) -> bool:
        return bool_flag(self.normalized_payload, *keys) or bool_flag(self.raw, *keys)

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
        value = first_present(self.raw, "error", "errors")
        if value:
            return value
        mapping = self.error_mapping
        return mapping or None

    @property
    def error_mapping(self) -> Mapping[str, Any]:
        return extract_error_mapping(self.raw)

    @property
    def error_type(self) -> Optional[str]:
        value = self.error_mapping.get("type")
        return value if isinstance(value, str) else None

    @property
    def error_code(self) -> Optional[str]:
        value = self.error_mapping.get("code")
        if value is None or isinstance(value, Mapping):
            return None
        return str(value)

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

    @property
    def id(self) -> Optional[str]:
        """Process-global call id (``call_N``). Not session order; use ``seq``."""

        value = first_present(self.raw, "id")
        return value if isinstance(value, str) and value else None

    @property
    def seq(self) -> int:
        """1-based per-session sequence (``seq``, else ``index + 1``)."""

        seq = first_present(self.raw, "seq", "sequence", "sequence_number", "sequenceNumber")
        if isinstance(seq, int) and not isinstance(seq, bool):
            return seq
        return self.index + 1

    def sequence_number(self) -> int:
        """Alias of ``seq``. Kept for 0.x compatibility."""

        return self.seq

    def offered_tool_names(self) -> list[str]:
        """Tool names offered on the request (not tools the model called)."""

        tools = self.request.get("tools")
        if not isinstance(tools, list):
            return []
        names: list[str] = []
        seen: set[str] = set()
        for tool in tools:
            mapping = as_mapping(tool)
            name = first_present(mapping, "name")
            if not isinstance(name, str) or not name:
                name = first_present(mapping_at(mapping, "function"), "name")
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @property
    def validation(self) -> Optional[Validation]:
        """Metadata: whether spec validation ran. Not a finding list."""

        for source in (self.normalized_payload, self.raw):
            if not isinstance(source, Mapping) or "validation" not in source:
                continue
            value = source.get("validation")
            if isinstance(value, Mapping):
                return validation_from_mapping(value)
        return None

    @property
    def spec_violations(self) -> Optional[list[SpecViolation]]:
        """Findings, or ``None`` when validation was not run (key absent)."""

        raw_list = optional_list_field(
            self.normalized_payload, "spec_violations", "specViolations"
        )
        if raw_list is None:
            raw_list = optional_list_field(self.raw, "spec_violations", "specViolations")
        if raw_list is None:
            return None
        return [spec_violation_from_mapping(item) for item in raw_list]

    @property
    def ignored_fields(self) -> Optional[list[IgnoredField]]:
        """Ignored-field warnings, or ``None`` when the key is absent."""

        raw_list = optional_list_field(
            self.normalized_payload, "ignored_fields", "ignoredFields"
        )
        if raw_list is None:
            raw_list = optional_list_field(self.raw, "ignored_fields", "ignoredFields")
        if raw_list is None:
            return None
        return [ignored_field_from_mapping(item) for item in raw_list]

    def spec_status_label(self) -> str:
        """``clean``, ``N violations``, or ``not checked`` for ``trace_summary``."""

        items = self.spec_violations
        if items is None:
            return "not checked"
        if not items:
            return "clean"
        count = len(items)
        return f"{count} violation" if count == 1 else f"{count} violations"

    def call_number(self) -> str:
        return str(self.seq)

    @property
    def session_id(self) -> Optional[str]:
        """Session id recorded on the call, when the proxy sent one."""

        value = first_present(self.raw, "session_id", "sessionId")
        return value if isinstance(value, str) and value else None

    @property
    def user_texts(self) -> list[str]:
        """User-role turn texts only (Chat ``messages``, Responses ``input``)."""

        return extract_user_texts(self.request)

    @property
    def session_source(self) -> Optional[str]:
        """How the proxy attributed this call: ``header``, ``baggage``, ``path``, or ``default``."""

        value = first_present(self.raw, "session_source", "sessionSource")
        return value if isinstance(value, str) else None

    @property
    def upstream_url(self) -> Optional[str]:
        """Upstream URL the proxy forwarded to. ``None`` when the field is absent."""

        upstream = self.raw.get("upstream")
        if isinstance(upstream, str) and upstream:
            return upstream
        value = first_present(coerce_mapping(upstream), "url", "URL")
        return value if isinstance(value, str) else None

    def usage_token_pair(self) -> Optional[tuple[str, str]]:
        """``(input, output)`` token counts for ``trace_summary``, or ``None``."""

        usage = self.usage
        if not usage:
            return None
        inbound = int_or_none(
            first_present(
                usage,
                "prompt_tokens",
                "promptTokens",
                "input_tokens",
                "inputTokens",
            )
        )
        outbound = int_or_none(
            first_present(
                usage,
                "completion_tokens",
                "completionTokens",
                "output_tokens",
                "outputTokens",
            )
        )
        if inbound is None and outbound is None:
            return None
        return (
            str(inbound) if inbound is not None else "?",
            str(outbound) if outbound is not None else "?",
        )


@dataclass(frozen=True)
class SessionTrace:
    """Control API session payload plus normalized captured calls."""

    session_id: str
    raw: Mapping[str, Any]
    calls: Sequence[CapturedCall]
    filter_labels: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, session_id: str, payload: Mapping[str, Any]) -> SessionTrace:
        raw_calls = find_call_list(payload)
        calls = [CapturedCall(raw=call, index=index) for index, call in enumerate(raw_calls)]
        return cls(session_id=session_id, raw=payload, calls=calls)

    def trace_summary(self) -> str:
        if not self.calls:
            return (
                f"llmentary session {self.session_id}: no captured calls "
                f"(payload keys={list(self.raw.keys())})"
            )

        lines = [f"llmentary session {self.session_id}: {len(self.calls)} captured call(s)"]
        for call in self.calls:
            tool_names = ", ".join(tool.name for tool in call.tool_calls) or "-"
            status = "error" if call.error or call.mid_stream_error else "ok"
            latency = f", latency={call.latency_ms:.1f}ms" if call.latency_ms is not None else ""
            ttft = f", ttft={call.ttft_ms:.1f}ms" if call.ttft_ms is not None else ""
            finish_reason = call.finish_reason
            finish = (
                f", finish={finish_reason}"
                if finish_reason and finish_reason != "unknown"
                else ""
            )
            http_status = f", http={call.http_status}" if call.http_status is not None else ""
            streamed = ", streamed" if call.streamed else ""
            aborted = ", client_aborted" if call.client_aborted else ""
            if call.client_aborted and call.abort_phase:
                aborted = f"{aborted}, phase={call.abort_phase}"
            truncated = ", stream_truncated" if call.stream_truncated else ""
            source = f", src={call.session_source}" if call.session_source else ""
            token_pair = call.usage_token_pair()
            tokens = f", tokens={token_pair[0]}/{token_pair[1]}" if token_pair else ""
            api_label = call.api_type or first_present(call.raw, "api_family", "apiFamily") or "?"
            spec = f", spec: {call.spec_status_label()}"
            call_id = call.id or "?"
            lines.append(
                f"  #{call.seq} ({call_id}) api={api_label} model={call.model or '?'} "
                f"status={status}{http_status}{streamed}{aborted}{truncated}{finish}"
                f"{latency}{ttft}{source}{tokens} tools=[{tool_names}]{spec}"
            )
            if not call.request and not call.response:
                lines.append(f"      unparsed call, keys={list(call.raw.keys())}")
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
        offers_tool: Optional[str] = None,
        with_tools: Optional[bool] = None,
        since: Optional[int] = None,
    ) -> SessionTrace:
        calls = list(self.calls)
        if api_type:
            normalized = normalize_api_type(api_type)
            calls = [call for call in calls if call.api_type == normalized]
        if model:
            calls = [call for call in calls if model_matches(call.model, model)]
        if since is not None:
            calls = [call for call in calls if call.sequence_number() > since]
        if offers_tool:
            calls = [call for call in calls if offers_tool in call.offered_tool_names()]
        if with_tools is True:
            calls = [call for call in calls if call.offered_tool_names()]
        elif with_tools is False:
            calls = [call for call in calls if not call.offered_tool_names()]
        labels = list(self.filter_labels)
        if offers_tool:
            labels.append(f'offers_tool("{offers_tool}")')
        if with_tools is True:
            labels.append("with_tools")
        elif with_tools is False:
            labels.append("without_tools")
        if since is not None:
            labels.append(f"since={since}")
        if api_type:
            labels.append(f"api_type={api_type}")
        if model:
            labels.append(f"model={model}")
        return SessionTrace(
            session_id=self.session_id,
            raw=self.raw,
            calls=calls,
            filter_labels=tuple(labels),
        )

    def empty_match_message(self) -> str:
        """Go-compatible empty-filter message: ``no calls matched query (<filters>)``."""

        if self.filter_labels:
            return f"no calls matched query ({', '.join(self.filter_labels)})"
        return "no calls matched query"

    def spec_violations(self) -> list[SpecViolation]:
        found: list[SpecViolation] = []
        for call in self.calls:
            items = call.spec_violations
            if items:
                found.extend(items)
        return found

    def ignored_fields(self) -> list[IgnoredField]:
        found: list[IgnoredField] = []
        for call in self.calls:
            items = call.ignored_fields
            if items:
                found.extend(items)
        return found

    def offers_tool(self, name: str) -> SessionTrace:
        return self.filtered(offers_tool=name)

    def with_tools(self) -> SessionTrace:
        return self.filtered(with_tools=True)

    def without_tools(self) -> SessionTrace:
        return self.filtered(with_tools=False)

    def since(self, mark: int) -> SessionTrace:
        return self.filtered(since=mark)


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


def looks_like_error_object(mapping: Mapping[str, Any]) -> bool:
    return any(mapping.get(key) not in (None, "") for key in ("type", "code", "message"))


def unwrap_error(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        nested = value.get("error")
        if isinstance(nested, Mapping):
            return nested
        if looks_like_error_object(value):
            return value
        return {}
    if isinstance(value, str) and value.strip():
        extracted = extract_json_object(value)
        if extracted:
            nested = extracted.get("error")
            if isinstance(nested, Mapping):
                return nested
            if looks_like_error_object(extracted):
                return extracted
    return {}


def extract_error_mapping(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in (raw, coerce_mapping(raw.get("normalized"))):
        mapped = unwrap_error(first_present(source, "error", "errors"))
        if mapped:
            return mapped

    for key in RESPONSE_KEYS:
        if key not in raw or raw[key] is None:
            continue
        for candidate in candidate_mappings(raw[key]):
            if "error" in candidate:
                mapped = unwrap_error(candidate.get("error"))
                if mapped:
                    return mapped
            if looks_like_error_object(candidate):
                return candidate
    return {}


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
        return has_payload_field(mapping, ("messages", "input", "tools", "prompt"))
    if side == "response":
        return has_payload_field(
            mapping, ("choices", "output", "output_text", "tool_calls", "usage")
        )
    return has_payload_field(mapping, LLM_PAYLOAD_KEYS)


def has_payload_field(mapping: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return True
    return False


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
    if isinstance(value, list) and value and all(
        isinstance(item, int) and 0 <= item <= 255 for item in value
    ):
        return decode_value(bytes(value), depth=depth + 1)
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


def usage_detail(usage: Mapping[str, Any], *keys: str) -> Optional[int]:
    """Read a usage counter, preferring OpenAI nested details over flattened aliases.

    Go traces flatten ``completion_tokens_details.reasoning_tokens`` to a top-level
    ``reasoning_tokens`` that is often ``0`` even when the nested details are
    non-zero. Nested detail objects are therefore searched across every container
    before falling back to top-level keys such as ``cached_input_tokens``.
    """

    containers = [usage, mapping_at(usage, "raw"), mapping_at(usage, "body")]
    detail_keys = (
        "prompt_tokens_details",
        "promptTokensDetails",
        "input_tokens_details",
        "inputTokensDetails",
        "completion_tokens_details",
        "completionTokensDetails",
        "output_tokens_details",
        "outputTokensDetails",
    )
    for container in containers:
        if not container:
            continue
        for container_key in detail_keys:
            details = mapping_at(container, container_key)
            if not details:
                continue
            parsed = int_or_none(first_present(details, *keys))
            if parsed is not None:
                return parsed
    for container in containers:
        if not container:
            continue
        parsed = int_or_none(first_present(container, *keys))
        if parsed is not None:
            return parsed
    return None


def bool_flag(mapping: Mapping[str, Any], *keys: str) -> bool:
    value = first_present(mapping, *keys)
    return value is True


def int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
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

    for message in sequence_at(mapping, "messages"):
        message_map = as_mapping(message)
        if message_map.get("role") != "assistant":
            continue
        nested = sequence_at(message_map, "tool_calls") or sequence_at(message_map, "toolCalls")
        calls.extend(tool_call_from_raw(item, start_index + len(calls)) for item in nested)

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
        call_id=string_or_none(first_present(raw, "call_id", "callId", "id")),
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


def extract_user_texts(request: Mapping[str, Any]) -> list[str]:
    """User-role texts from Chat ``messages`` and Responses ``input``.

    System, developer, assistant, and tool turns are ignored.
    """

    texts: list[str] = []
    for message in sequence_at(request, "messages"):
        mapping = as_mapping(message)
        if mapping.get("role") != "user":
            continue
        text = _user_content_text(mapping.get("content"))
        if text:
            texts.append(text)

    raw_input = request.get("input")
    if isinstance(raw_input, str) and raw_input.strip():
        texts.append(raw_input.strip())
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
                continue
            mapping = as_mapping(item)
            role = mapping.get("role")
            typ = mapping.get("type")
            if typ in {"function_call", "function_call_output", "tool_result"}:
                continue
            if role == "user" or (role is None and typ == "input_text"):
                text = _user_content_text(
                    first_present(mapping, "content", "text")
                )
                if text:
                    texts.append(text)
    return texts


def _user_content_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_user_content_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        text = first_present(value, "text", "content")
        if text is value:
            return ""
        return _user_content_text(text)
    return ""


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


def optional_list_field(mapping: Mapping[str, Any], *keys: str) -> Optional[list[Any]]:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            return None
        if isinstance(value, list):
            return list(value)
        return None
    return None


def optional_str(mapping: Mapping[str, Any], *keys: str) -> Optional[str]:
    value = first_present(mapping, *keys)
    return value if isinstance(value, str) else None


def optional_bool(mapping: Mapping[str, Any], *keys: str) -> Optional[bool]:
    value = first_present(mapping, *keys)
    return value if isinstance(value, bool) else None


def optional_int(mapping: Mapping[str, Any], *keys: str) -> Optional[int]:
    value = first_present(mapping, *keys)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def validation_from_mapping(raw: Mapping[str, Any]) -> Validation:
    mapping = coerce_mapping(raw)
    return Validation(
        mode=optional_str(mapping, "mode"),
        profile=optional_str(mapping, "profile"),
        spec_sha=optional_str(mapping, "spec_sha", "specSHA", "specSha"),
        schema_subset_retrieved=optional_str(
            mapping, "schema_subset_retrieved", "schemaSubsetRetrieved"
        ),
        request_checked=optional_bool(mapping, "request_checked", "requestChecked"),
        response_checked=optional_bool(mapping, "response_checked", "responseChecked"),
        skipped_reason=optional_str(mapping, "skipped_reason", "skippedReason"),
        duration_us=optional_int(mapping, "duration_us", "durationUs"),
        truncated=optional_bool(mapping, "truncated"),
        raw=mapping,
    )


def spec_violation_from_mapping(raw: Any) -> SpecViolation:
    mapping = coerce_mapping(raw)
    return SpecViolation(
        side=optional_str(mapping, "side"),
        source=optional_str(mapping, "source"),
        path=optional_str(mapping, "path"),
        pointer=optional_str(mapping, "pointer"),
        keyword=optional_str(mapping, "keyword"),
        message=optional_str(mapping, "message"),
        profile=optional_str(mapping, "profile"),
        schema_ref=optional_str(mapping, "schema_ref", "schemaRef"),
        offset=optional_int(mapping, "offset"),
        event_index=optional_int(mapping, "event_index", "eventIndex"),
        raw=mapping,
    )


def ignored_field_from_mapping(raw: Any) -> IgnoredField:
    mapping = coerce_mapping(raw)
    return IgnoredField(
        side=optional_str(mapping, "side"),
        path=optional_str(mapping, "path"),
        pointer=optional_str(mapping, "pointer"),
        kind=optional_str(mapping, "kind", "keyword"),
        severity=optional_str(mapping, "severity"),
        message=optional_str(mapping, "message"),
        hint=optional_str(mapping, "hint"),
        key=optional_str(mapping, "key"),
        action=optional_str(mapping, "action"),
        raw=mapping,
    )


def session_info_from_mapping(raw: Mapping[str, Any]) -> SessionInfo:
    mapping = coerce_mapping(raw)
    session_id = optional_str(mapping, "id", "session_id", "sessionId") or ""
    calls_value = mapping.get("calls")
    if isinstance(calls_value, list):
        call_count = len(calls_value)
    else:
        call_count = optional_int(mapping, "calls") or 0
    sources_value = mapping.get("sources")
    sources = (
        tuple(str(item) for item in sources_value if item is not None)
        if isinstance(sources_value, list)
        else ()
    )
    return SessionInfo(
        id=session_id,
        calls=call_count,
        active=optional_int(mapping, "active") or 0,
        sources=sources,
        first_started_at=optional_str(
            mapping, "first_started_at", "firstStartedAt"
        ),
        last_ended_at=optional_str(mapping, "last_ended_at", "lastEndedAt"),
        raw=mapping,
    )


def zero_calls_message(session_id: str, known_ids: Sequence[str] = ()) -> str:
    listed = ", ".join(known_ids) if known_ids else "(none)"
    return (
        f"no calls recorded for session {session_id}; known sessions: {listed}"
    )
