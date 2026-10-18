from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from re import Pattern
from typing import Any, Callable, Optional, Union

from jsonschema import ValidationError
from jsonschema.validators import Draft202012Validator

from .types import CapturedCall, SessionTrace, ToolCall, as_mapping, sequence_at, stringify_value

ArgsMatcher = Union[Mapping[str, Any], Callable[[Mapping[str, Any]], bool]]
Regex = Union[str, Pattern[str]]


class Expectation:
    """Fluent deterministic assertions over a llm-test-lab session trace."""

    def __init__(self, trace: SessionTrace) -> None:
        self.trace = trace

    def tool_called(self, name: str) -> Expectation:
        self._assert(
            any(call.name == name for call in self._tool_calls()),
            f"expected tool {name!r} to be called",
        )
        return self

    def not_called(self, name: str) -> Expectation:
        self._assert(
            all(call.name != name for call in self._tool_calls()),
            f"expected tool {name!r} not to be called",
        )
        return self

    def called_times(self, name: str, count: int) -> Expectation:
        actual = sum(1 for call in self._tool_calls() if call.name == name)
        self._assert(
            actual == count,
            f"expected tool {name!r} to be called {count} time(s), got {actual}",
        )
        return self

    def args_subset(self, name: str, subset: Mapping[str, Any]) -> Expectation:
        matching = [tool for tool in self._tool_calls() if tool.name == name]
        self._assert(bool(matching), f"expected tool {name!r} to be called")
        self._assert(
            any(mapping_contains(tool.arguments, subset) for tool in matching),
            f"expected a {name!r} call with argument subset {subset!r}",
        )
        return self

    def args_matcher(self, name: str, matcher: ArgsMatcher) -> Expectation:
        matching = [tool for tool in self._tool_calls() if tool.name == name]
        self._assert(bool(matching), f"expected tool {name!r} to be called")
        if isinstance(matcher, Mapping):
            return self.args_subset(name, matcher)
        self._assert(
            any(matcher(tool.arguments) for tool in matching),
            f"expected a {name!r} call to satisfy matcher",
        )
        return self

    def order(self, *names: str, strict: bool = False) -> Expectation:
        actual = [tool.name for tool in self._tool_calls()]
        if strict:
            self._assert(
                actual == list(names),
                f"expected strict tool order {list(names)!r}, got {actual!r}",
            )
            return self

        position = 0
        for expected in names:
            try:
                index = actual.index(expected, position)
            except ValueError:
                self._assert(False, f"expected tool order {list(names)!r}, got {actual!r}")
                return self
            position = index + 1
        return self

    def parallel_tool_calls(self, expected: bool = True) -> Expectation:
        actual = any(len(call.tool_calls) > 1 for call in self.trace.calls)
        request_param = any(
            call.request.get("parallel_tool_calls") is True
            or call.request.get("parallelToolCalls") is True
            for call in self.trace.calls
        )
        actual = actual or request_param
        self._assert(
            actual is expected,
            f"expected parallel tool calls to be {expected}, got {actual}",
        )
        return self

    def tool_result_fed_back(self, name: str) -> Expectation:
        tool_ids = {
            tool.call_id
            for call in self.trace.calls
            for tool in call.tool_calls
            if tool.name == name and tool.call_id
        }
        any_named_tool = any(tool.name == name for tool in self._tool_calls())
        result_matches_id = any(
            result.get("tool_call_id") in tool_ids
            or result.get("call_id") in tool_ids
            or result.get("callId") in tool_ids
            for call in self.trace.calls
            for result in call.tool_results
        )
        result_mentions_name = any(
            name in json.dumps(result, sort_keys=True)
            for call in self.trace.calls
            for result in call.tool_results
        )
        self._assert(
            any_named_tool and (result_matches_id or result_mentions_name),
            f"expected tool result for {name!r} to be fed back in a later request",
        )
        return self

    def model(self, expected: str) -> Expectation:
        models = {call.model for call in self.trace.calls if call.model}
        self._assert(expected in models, f"expected model {expected!r}, got {sorted(models)!r}")
        return self

    def request_param(self, name: str, expected: Any) -> Expectation:
        matches = [call for call in self.trace.calls if nested_get(call.request, name) == expected]
        self._assert(
            bool(matches),
            f"expected request parameter {name!r} to equal {expected!r}",
        )
        return self

    def finish_reason(self, expected: str) -> Expectation:
        reasons = [call.finish_reason for call in self.trace.calls if call.finish_reason]
        self._assert(
            expected in reasons,
            f"expected finish reason {expected!r}, got {reasons!r}",
        )
        return self

    def tokens(
        self,
        *,
        min_total: Optional[int] = None,
        max_total: Optional[int] = None,
        min_prompt: Optional[int] = None,
        max_prompt: Optional[int] = None,
        min_completion: Optional[int] = None,
        max_completion: Optional[int] = None,
    ) -> Expectation:
        usage = aggregate_usage(self.trace.calls)
        checks = [
            ("total", min_total, max_total),
            ("prompt", min_prompt, max_prompt),
            ("completion", min_completion, max_completion),
        ]
        for key, minimum, maximum in checks:
            if minimum is None and maximum is None:
                continue
            value = usage.get(key)
            if value is None:
                self._assert(False, f"expected usage token field {key!r} to be present")
                continue
            if minimum is not None:
                self._assert(value >= minimum, f"expected {key} tokens >= {minimum}, got {value}")
            if maximum is not None:
                self._assert(value <= maximum, f"expected {key} tokens <= {maximum}, got {value}")
        return self

    def latency(
        self,
        *,
        min_ms: Optional[float] = None,
        max_ms: Optional[float] = None,
    ) -> Expectation:
        return self._bounds(
            "latency",
            [call.latency_ms for call in self.trace.calls],
            min_ms,
            max_ms,
        )

    def ttft(
        self,
        *,
        min_ms: Optional[float] = None,
        max_ms: Optional[float] = None,
    ) -> Expectation:
        return self._bounds(
            "ttft",
            [call.ttft_ms for call in self.trace.calls],
            min_ms,
            max_ms,
        )

    def retries(
        self,
        *,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ) -> Expectation:
        total = sum(call.retries for call in self.trace.calls)
        if min_count is not None:
            self._assert(total >= min_count, f"expected retries >= {min_count}, got {total}")
        if max_count is not None:
            self._assert(total <= max_count, f"expected retries <= {max_count}, got {total}")
        return self

    def no_retries(self) -> Expectation:
        return self.retries(max_count=0)

    def no_errors(self) -> Expectation:
        errors = [call for call in self.trace.calls if call.error or call.mid_stream_error]
        self._assert(not errors, f"expected no errors, got {len(errors)} errored call(s)")
        return self

    def has_error(self, pattern: Optional[Regex] = None) -> Expectation:
        errors = [call.error for call in self.trace.calls if call.error]
        self._assert(bool(errors), "expected at least one error")
        if pattern is not None:
            self._assert(
                any(regex_search(pattern, str(error)) for error in errors),
                f"expected an error matching {pattern!r}",
            )
        return self

    def mid_stream_error(self, pattern: Optional[Regex] = None) -> Expectation:
        errors = [call.mid_stream_error for call in self.trace.calls if call.mid_stream_error]
        self._assert(bool(errors), "expected at least one mid-stream error")
        if pattern is not None:
            self._assert(
                any(regex_search(pattern, str(error)) for error in errors),
                f"expected a mid-stream error matching {pattern!r}",
            )
        return self

    def prompt_contains(self, text: str) -> Expectation:
        prompt = "\n".join(call.prompt_text for call in self.trace.calls)
        self._assert(text in prompt, f"expected prompt to contain {text!r}")
        return self

    def prompt_regex(self, pattern: Regex) -> Expectation:
        prompt = "\n".join(call.prompt_text for call in self.trace.calls)
        self._assert(regex_search(pattern, prompt), f"expected prompt to match {pattern!r}")
        return self

    def response_contains(self, text: str) -> Expectation:
        response = "\n".join(call.response_text for call in self.trace.calls)
        self._assert(text in response, f"expected response to contain {text!r}")
        return self

    def response_regex(self, pattern: Regex) -> Expectation:
        response = "\n".join(call.response_text for call in self.trace.calls)
        self._assert(regex_search(pattern, response), f"expected response to match {pattern!r}")
        return self

    def response_json_valid(self) -> Expectation:
        for text in self._response_json_candidates():
            try:
                json.loads(text)
                return self
            except json.JSONDecodeError:
                continue
        self._assert(False, "expected at least one response body/content to be valid JSON")
        return self

    def response_json_schema(self, schema: Mapping[str, Any]) -> Expectation:
        validator = Draft202012Validator(schema)
        for text in self._response_json_candidates():
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            try:
                validator.validate(payload)
                return self
            except ValidationError:
                continue
        self._assert(False, "expected at least one JSON response to satisfy schema")
        return self

    def _tool_calls(self) -> Sequence[ToolCall]:
        return [tool for call in self.trace.calls for tool in call.tool_calls]

    def _response_json_candidates(self) -> Sequence[str]:
        candidates: list[str] = []
        for call in self.trace.calls:
            candidates.extend(model_output_text_candidates(call))
        return candidates

    def _bounds(
        self,
        name: str,
        values: Sequence[Optional[float]],
        minimum: Optional[float],
        maximum: Optional[float],
    ) -> Expectation:
        present = [value for value in values if value is not None]
        self._assert(bool(present), f"expected {name} measurements to be present")
        for value in present:
            if minimum is not None:
                self._assert(value >= minimum, f"expected {name} >= {minimum}ms, got {value}ms")
            if maximum is not None:
                self._assert(value <= maximum, f"expected {name} <= {maximum}ms, got {value}ms")
        return self

    def _assert(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(f"{message}\n\n{self.trace.trace_summary()}")


class SoftExpectation(Expectation):
    """Non-fatal variant that accumulates failures until ``assert_all``."""

    def __init__(self, trace: SessionTrace) -> None:
        super().__init__(trace)
        self.failures: list[str] = []

    def assert_all(self) -> None:
        if self.failures:
            joined = "\n\n".join(f"- {failure}" for failure in self.failures)
            raise AssertionError(f"{len(self.failures)} soft assertion(s) failed:\n{joined}")

    def _assert(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(f"{message}\n\n{self.trace.trace_summary()}")


def mapping_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                return False
            if not mapping_contains(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def regex_search(pattern: Regex, value: str) -> bool:
    if isinstance(pattern, str):
        return re.search(pattern, value, flags=re.DOTALL) is not None
    return pattern.search(value) is not None


def nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def model_output_text_candidates(call: CapturedCall) -> list[str]:
    response = call.response
    candidates: list[str] = []
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        candidates.append(output_text)

    for choice in sequence_at(response, "choices"):
        message = as_mapping(as_mapping(choice).get("message"))
        content = message.get("content")
        if content is not None:
            candidates.append(stringify_value(content))

    for item in sequence_at(response, "output"):
        item_map = as_mapping(item)
        if item_map.get("type") == "message" or "content" in item_map:
            candidates.append(stringify_value(item_map.get("content")))

    return [candidate for candidate in candidates if candidate]


def aggregate_usage(calls: Sequence[CapturedCall]) -> dict[str, int]:
    totals = {"total": 0, "prompt": 0, "completion": 0}
    seen_any = {"total": False, "prompt": False, "completion": False}
    aliases = {
        "total": ("total_tokens", "totalTokens"),
        "prompt": ("prompt_tokens", "promptTokens", "input_tokens", "inputTokens"),
        "completion": ("completion_tokens", "completionTokens", "output_tokens", "outputTokens"),
    }
    for call in calls:
        usage = call.usage
        for canonical, keys in aliases.items():
            for key in keys:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[canonical] += value
                    seen_any[canonical] = True
                    break
    return {key: value for key, value in totals.items() if seen_any[key]}
