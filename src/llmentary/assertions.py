from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from re import Pattern
from typing import Any, Callable, Optional, Union

from jsonschema import ValidationError
from jsonschema.validators import Draft202012Validator

from .types import (
    CapturedCall,
    IgnoredField,
    SessionTrace,
    SpecViolation,
    ToolCall,
    as_mapping,
    first_present,
    model_matches,
    sequence_at,
    stringify_value,
)

ArgsMatcher = Union[Mapping[str, Any], Callable[[Mapping[str, Any]], bool]]
Regex = Union[str, Pattern[str]]


class _ResolvedOther:
    """Trace loaded for ``isolated_from``, plus wait-timeout metadata."""

    def __init__(
        self,
        trace: SessionTrace,
        *,
        timed_out: bool = False,
        active: Optional[int] = None,
    ) -> None:
        self.trace = trace
        self.timed_out = timed_out
        self.active = active


def require_non_negative_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"call_count() {name} must be a non-negative integer, got {value!r}")
    return value


class Expectation:
    """Fluent deterministic assertions over an llmentary session trace."""

    def __init__(self, trace: SessionTrace, *, session: Optional[object] = None) -> None:
        self.trace = trace
        self._session = session
        self._allow_empty = False

    def allow_empty(self) -> Expectation:
        """Opt out of the empty-trace guard on assertions that need calls.

        Same semantics as Go ``AllowEmpty()``: a filtered, ``since``, or
        never-created trace with zero matching calls then passes
        ``no_errors`` / ``not_called``. ``isolated_from`` also uses this
        to allow an empty self or an empty/unknown other session.
        """

        self._allow_empty = True
        return self

    def call_count(
        self,
        expected: Optional[int] = None,
        *,
        min: Optional[int] = None,
        max: Optional[int] = None,
    ) -> Expectation:
        """Assert how many captured calls the (possibly filtered) trace has.

        ``call_count(2)`` requires an exact count. ``call_count(min=1, max=4)``
        is the range form. The failure message includes the actual count and
        the trace summary.
        """

        if expected is not None and (min is not None or max is not None):
            raise TypeError("call_count() takes either an exact count or min/max, not both")
        if expected is None and min is None and max is None:
            raise TypeError("call_count() requires an exact count or min/max")
        if expected is not None:
            expected = require_non_negative_count("expected", expected)
        if min is not None:
            min = require_non_negative_count("min", min)
        if max is not None:
            max = require_non_negative_count("max", max)
        if min is not None and max is not None and min > max:
            raise TypeError(f"call_count() min={min} is greater than max={max}")

        actual = len(self.trace.calls)
        if expected is not None:
            self._assert(
                actual == expected,
                f"expected {expected} captured call(s), got {actual}",
            )
            return self
        if min is not None:
            self._assert(
                actual >= min,
                f"expected at least {min} captured call(s), got {actual}",
            )
        if max is not None:
            self._assert(
                actual <= max,
                f"expected at most {max} captured call(s), got {actual}",
            )
        return self

    def isolated_from(
        self,
        other: object,
        *,
        require_path: bool = False,
        min_chars: int = 8,
        ignore_texts: Optional[Sequence[str]] = None,
    ) -> Expectation:
        """Assert this session's calls are isolated from ``other``.

        ``other`` is a ``Session``, a session id, or a ``SessionTrace``.
        Checks, with one failure line per violated rule (DOG-29):

        1. every call has ``session_id`` equal to this session
        2. no call id appears in both sessions
        3. no **foreign** user turn: a user-role message in this session's
           prompt that this session never introduced but ``other`` did.
           The first call introduces only its last user-role message;
           later calls introduce **each** new user-role message whose
           normalized form was not already present in an earlier call.
           Comparison is exact after normalize (trim, collapse whitespace,
           casefold), never substring. Texts both sessions introduced
           themselves are not a leak.

        ``min_chars`` (default 8) ignores shorter texts. ``ignore_texts``
        skips those normalized messages. System/developer/tools are ignored.

        Empty self fails with ``no calls recorded for session X`` unless
        ``allow_empty()`` was called. Empty or unknown other (404 / no
        calls) fails with ``no calls recorded for session Y (isolated_from
        needs calls on both sides)`` unless ``allow_empty()`` was called.
        A ``Session`` or session id is loaded with the same ``wait``
        semantics as ``expect()`` so an in-flight call on the other side
        is included once it settles. A pre-fetched ``SessionTrace`` is
        used as given. If that wait times out with zero settled calls,
        the failure names the in-flight / timed-out state instead of a
        bare empty other. ``require_path=True`` also requires
        ``session_source == "path"``.
        """

        resolved = self._resolve_isolated_other(other)
        other_trace = resolved.trace
        findings: list[str] = []
        if not self.trace.calls:
            if not self._allow_empty:
                self._assert(
                    False, f"no calls recorded for session {self.trace.session_id}"
                )
            return self

        expected_session = self.trace.session_id
        for call in _calls_in_seq(self.trace.calls):
            label = _call_label(call)
            if call.session_id != expected_session:
                findings.append(
                    f"{label} session_id={call.session_id!r}, expected {expected_session!r}"
                )
            if require_path and call.session_source != "path":
                findings.append(
                    f"{label} session_source={call.session_source!r}, expected 'path'"
                )

        if not other_trace.calls:
            if not self._allow_empty:
                findings.append(_other_empty_message(resolved))
            if findings:
                self._assert(False, "\n".join(findings))
            return self

        self_ids = {ident for ident in (_call_id(call) for call in self.trace.calls) if ident}
        other_ids = {ident for ident in (_call_id(call) for call in other_trace.calls) if ident}
        shared_ids = sorted(self_ids & other_ids)
        for ident in shared_ids:
            findings.append(
                f"call id {ident} appears in sessions {self.trace.session_id} "
                f"and {other_trace.session_id}"
            )

        ignored = {
            normalize_user_text(text)
            for text in (ignore_texts or ())
            if normalize_user_text(text)
        }
        intro_self = _introduced_user_texts(self.trace.calls, min_chars, ignored)
        intro_other = _introduced_user_texts(other_trace.calls, min_chars, ignored)
        seen_leaks: set[str] = set()
        for call in _calls_in_seq(self.trace.calls):
            for text in call.user_texts:
                key = normalize_user_text(text)
                if not key or len(key) < min_chars or key in ignored or key in intro_self:
                    continue
                if key not in intro_other or key in seen_leaks:
                    continue
                seen_leaks.add(key)
                _other_text, other_call = intro_other[key]
                findings.append(
                    f"user text {text!r} also introduced by session "
                    f"{other_trace.session_id} {_call_label(other_call)} "
                    f"appears in session {self.trace.session_id} "
                    f"{_call_label(call)}"
                )

        if resolved.timed_out:
            findings.append(f"wait timed out on session {other_trace.session_id}")
            if resolved.active and resolved.active > 0:
                findings.append(_inflight_message(other_trace.session_id, resolved.active))

        if findings:
            self._assert(False, "\n".join(findings))
        return self

    def _resolve_isolated_other(self, other: object) -> _ResolvedOther:
        if isinstance(other, SessionTrace):
            return _ResolvedOther(other)
        if isinstance(other, str):
            session = self._session
            control = getattr(getattr(session, "proxy", None), "control", None)
            wait_session = getattr(control, "wait_session", None)
            get_session = getattr(control, "get_session", None)
            if not callable(wait_session) and not callable(get_session):
                raise TypeError("isolated_from(session_id) requires Session.expect()")
            kwargs = self._other_wait_kwargs()
            if callable(wait_session):
                getter: Callable[[], object] = (
                    (lambda: get_session(other)) if callable(get_session) else (lambda: None)
                )
                return self._wait_other_trace(
                    lambda: wait_session(other, **kwargs),
                    getter,
                    other,
                )
            if callable(get_session):
                return self._trace_or_empty(lambda: get_session(other), other)
            raise TypeError("isolated_from(session_id) requires Session.expect()")
        session_id = getattr(other, "session_id", None)
        wait_fn = getattr(other, "wait", None)
        trace_fn = getattr(other, "trace", None)
        if callable(wait_fn) and isinstance(session_id, str):
            kwargs = self._other_wait_kwargs()
            return self._wait_other_trace(
                lambda: _call_with_wait_kwargs(wait_fn, kwargs),
                lambda: trace_fn() if callable(trace_fn) else None,
                session_id,
            )
        if callable(trace_fn) and isinstance(session_id, str):
            return self._trace_or_empty(trace_fn, session_id)
        raise TypeError(
            "isolated_from() expects a Session, session id, or SessionTrace"
        )

    def _other_wait_kwargs(self) -> dict[str, float]:
        session = self._session
        kwargs: dict[str, float] = {}
        timeout = getattr(session, "wait_timeout", None)
        quiet = getattr(session, "quiet_for", None)
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            kwargs["timeout"] = float(timeout)
        if isinstance(quiet, (int, float)) and not isinstance(quiet, bool):
            kwargs["quiet"] = float(quiet)
        return kwargs

    def _wait_other_trace(
        self,
        waiter: Callable[[], object],
        getter: Callable[[], object],
        session_id: str,
    ) -> _ResolvedOther:
        from .client import SessionNotFound
        from .wait import WaitTimeoutError

        try:
            loaded = waiter()
        except SessionNotFound:
            return _ResolvedOther(_empty_other_trace(session_id))
        except WaitTimeoutError as exc:
            trace = self._other_trace_after_timeout(getter, session_id)
            raw_active = exc.active
            active = (
                raw_active
                if isinstance(raw_active, int) and not isinstance(raw_active, bool)
                else None
            )
            return _ResolvedOther(trace, timed_out=True, active=active)
        if isinstance(loaded, SessionTrace):
            return _ResolvedOther(loaded)
        return _ResolvedOther(_empty_other_trace(session_id))

    def _other_trace_after_timeout(
        self, getter: Callable[[], object], session_id: str
    ) -> SessionTrace:
        from .client import SessionNotFound

        try:
            loaded = getter()
        except SessionNotFound:
            return _empty_other_trace(session_id)
        except Exception:
            return _empty_other_trace(session_id)
        if isinstance(loaded, SessionTrace):
            return loaded
        return _empty_other_trace(session_id)

    def _trace_or_empty(self, loader: Callable[[], object], session_id: str) -> _ResolvedOther:
        from .client import SessionNotFound

        try:
            loaded = loader()
        except SessionNotFound:
            return _ResolvedOther(_empty_other_trace(session_id))
        if isinstance(loaded, SessionTrace):
            return _ResolvedOther(loaded)
        return _ResolvedOther(_empty_other_trace(session_id))

    def tool_called(self, name: str) -> Expectation:
        self._assert(
            any(call.name == name for call in self._tool_calls()),
            f"expected tool {name!r} to be called",
        )
        return self

    def not_called(self, name: str) -> Expectation:
        if not self._require_matching_calls():
            return self
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
        result_mentions_name = not tool_ids and any(
            result.get("name") == name
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
        self._assert(
            any(model_matches(model, expected) for model in models),
            f"expected model {expected!r}, got {sorted(models)!r}",
        )
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
        if not self._require_matching_calls():
            return self
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

    def aborted(self, *, phase: Optional[str] = None) -> Expectation:
        """Assert at least one captured call has ``client_aborted``.

        ``phase`` is ``before_headers`` or ``after_headers`` when the proxy
        recorded ``abort_phase`` (DOG-27 / DOG-28).
        """

        calls = [call for call in self.trace.calls if call.client_aborted]
        self._assert(bool(calls), "expected a client-aborted call")
        if phase is not None:
            actual = [call.abort_phase for call in calls]
            self._assert(
                any(value == phase for value in actual),
                f"expected abort_phase={phase!r}, got {actual}",
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

    def final_response_mentions_any(
        self,
        *candidates: str,
        normalize_numbers: bool = True,
        casefold: bool = True,
    ) -> Expectation:
        if not self.trace.calls:
            self._assert(False, "expected at least one captured call")
            return self
        text = "\n".join(model_output_text_candidates(self.trace.calls[-1]))
        if not text.strip():
            self._assert(False, "expected the last response to include model output text")
            return self
        try:
            answer_mentions_any(
                text,
                *candidates,
                normalize_numbers=normalize_numbers,
                casefold=casefold,
            )
        except AssertionError as exc:
            self._assert(False, str(exc))
        return self

    def tool_result_in_prompt(self, name: str) -> Expectation:
        """Fail unless a later request has a non-empty tool result for ``name``.

        The result is matched by tool-call id or tool name. This is stronger
        than ``tool_result_fed_back`` (content must be present) but does not
        compare against an independently supplied expected string.
        """

        calls = list(self.trace.calls)
        for index, call in enumerate(calls):
            invoked = [tool for tool in call.tool_calls if tool.name == name]
            if not invoked:
                continue
            later = calls[index + 1 :]
            if not later:
                self._assert(
                    False,
                    f"expected a later request after tool {name!r} that carries its result",
                )
                return self
            tool_ids = {tool.call_id for tool in invoked if tool.call_id}
            for later_call in later:
                if tool_result_contents(later_call, name, tool_ids):
                    return self
            self._assert(
                False,
                f"expected the next request after tool {name!r} to contain its result content",
            )
            return self
        self._assert(False, f"expected tool {name!r} to be called")
        return self

    def no_spec_violations(
        self,
        side: Optional[str] = None,
        calls: Optional[Sequence[int]] = None,
        ignore_paths: Sequence[str] = (),
        ignore_keywords: Sequence[str] = (),
        allow_unvalidated: bool = False,
    ) -> Expectation:
        """Fail when checked calls have spec findings. Never vacuous.

        Schema findings live on ``normalized.spec_violations``. The LTL-API-2
        unknown nested message field is recorded as
        ``ignored_fields`` with ``kind=unknown_request_field`` (OpenAI accepts
        it and ignores it). Those are findings for this assertion too; JS
        ``noSpecViolations`` fails the same live L2 case.
        """

        selected = self._select_calls(calls)
        findings: list[str] = []
        checked = 0
        for call in selected:
            items = call.spec_violations
            unknown = unknown_request_fields(call)
            if items is None and not unknown:
                continue
            if not selected_side_was_checked(call, side):
                continue
            checked += 1
            for item in items or ():
                if side and (item.side or "").lower() != side.lower():
                    continue
                if path_is_ignored(item.path, item.pointer, ignore_paths):
                    continue
                if item.keyword and item.keyword in ignore_keywords:
                    continue
                findings.append(format_spec_finding(call, item))
            for ignored in unknown:
                item_side = ignored.side or "request"
                if side and item_side.lower() != side.lower():
                    continue
                if path_is_ignored(ignored.path, ignored.pointer, ignore_paths):
                    continue
                kind = ignored.kind or ""
                if kind and kind in ignore_keywords:
                    continue
                findings.append(format_ignored_finding(call, ignored))
        if findings:
            self._assert(False, "\n".join(findings))
            return self
        if checked == 0 and not allow_unvalidated:
            self._assert(False, "validation was not run")
        return self

    def no_ignored_fields(
        self,
        side: Optional[str] = None,
        calls: Optional[Sequence[int]] = None,
        ignore_paths: Sequence[str] = (),
        ignore_keywords: Sequence[str] = (),
        allow_unvalidated: bool = False,
    ) -> Expectation:
        """Fail when checked calls recorded ignored fields. Never vacuous."""

        selected = self._select_calls(calls)
        findings: list[str] = []
        checked = 0
        for call in selected:
            items = call.ignored_fields
            if items is None:
                if not call_was_validated(call):
                    continue
                items = []
            checked += 1
            for item in items:
                if side and (item.side or "").lower() != side.lower():
                    continue
                if path_is_ignored(item.path, item.pointer, ignore_paths):
                    continue
                kind = item.kind or ""
                if kind and kind in ignore_keywords:
                    continue
                findings.append(format_ignored_finding(call, item))
        if findings:
            self._assert(False, "\n".join(findings))
            return self
        if checked == 0 and not allow_unvalidated:
            self._assert(False, "validation was not run")
        return self

    def _select_calls(self, calls: Optional[Sequence[int]]) -> Sequence[CapturedCall]:
        if calls is None:
            return self.trace.calls
        wanted = {int(value) for value in calls}
        return [call for call in self.trace.calls if call.sequence_number() in wanted]

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

    def _require_matching_calls(self) -> bool:
        if self.trace.calls or self._allow_empty:
            return True
        self._assert(False, self.trace.empty_match_message())
        return False

    def _assert(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(f"{message}\n\n{self.trace.trace_summary()}")


class SoftExpectation(Expectation):
    """Non-fatal variant that accumulates failures until ``assert_all``."""

    def __init__(self, trace: SessionTrace, *, session: Optional[object] = None) -> None:
        super().__init__(trace, session=session)
        self.failures: list[str] = []

    def assert_all(self) -> None:
        if self.failures:
            joined = "\n\n".join(f"- {failure}" for failure in self.failures)
            raise AssertionError(
                f"{len(self.failures)} soft assertion(s) failed:\n{joined}"
                f"\n\n{self.trace.trace_summary()}"
            )

    def _assert(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)


def normalize_user_text(text: str) -> str:
    """Trim, collapse whitespace, casefold. Exact-match key for isolation."""

    return re.sub(r"\s+", " ", text.strip()).casefold()


def _empty_other_trace(session_id: str) -> SessionTrace:
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": []})


def _inflight_message(session_id: str, active: int) -> str:
    noun = "call" if active == 1 else "calls"
    return f"session {session_id} has {active} {noun} still in flight"


def _other_empty_message(resolved: _ResolvedOther) -> str:
    session_id = resolved.trace.session_id
    if resolved.timed_out:
        if resolved.active and resolved.active > 0:
            return _inflight_message(session_id, resolved.active)
        return (
            f"wait timed out on session {session_id} "
            "(isolated_from needs calls on both sides)"
        )
    return (
        f"no calls recorded for session {session_id} "
        "(isolated_from needs calls on both sides)"
    )


def _call_with_wait_kwargs(wait_fn: Callable[..., object], kwargs: Mapping[str, float]) -> object:
    if not kwargs:
        return wait_fn()
    try:
        return wait_fn(**dict(kwargs))
    except TypeError:
        return wait_fn()


def _calls_in_seq(calls: Sequence[CapturedCall]) -> list[CapturedCall]:
    return sorted(calls, key=lambda call: (call.sequence_number(), call.index))


def _introduced_user_texts(
    calls: Sequence[CapturedCall],
    min_chars: int,
    ignored: set[str],
) -> dict[str, tuple[str, CapturedCall]]:
    """User-role messages this session introduced, walked in ``seq`` order.

    The first call introduces only its last user-role message; earlier
    user turns on that call are inbound history (seen, not introduced).
    Later calls introduce each user-role message whose normalized form
    was not already present in an earlier call of the same session.
    """

    introduced: dict[str, tuple[str, CapturedCall]] = {}
    seen: set[str] = set()
    first = True
    for call in _calls_in_seq(calls):
        texts = list(call.user_texts)
        if not texts:
            continue
        is_first = first
        first = False
        if is_first:
            for hist in texts[:-1]:
                key = normalize_user_text(hist)
                if key:
                    seen.add(key)
            candidates = texts[-1:]
        else:
            candidates = texts
        for raw in candidates:
            key = normalize_user_text(raw)
            if not key:
                continue
            if len(key) >= min_chars and key not in ignored and key not in seen:
                introduced.setdefault(key, (raw, call))
            seen.add(key)
    return introduced


def _call_id(call: CapturedCall) -> Optional[str]:
    value = first_present(call.raw, "id")
    return value if isinstance(value, str) and value else None


def _call_label(call: CapturedCall) -> str:
    ident = _call_id(call) or "?"
    return f"#{call.sequence_number()} ({ident})"


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


_DEGREE = "°"
_UNIT_WORDS = re.compile(r"(?i)\b(?:degrees?|celsius|fahrenheit)\b")
_TRAILING_TEMP_UNIT = re.compile(r"(?<=\d)\s*[cCfF]\b")
_SPACES = re.compile(r"\s+")
_EXCERPT_LIMIT = 200


def answer_mentions_any(
    text: str,
    *candidates: str,
    normalize_numbers: bool = True,
    casefold: bool = True,
) -> None:
    """Assert that free-text ``text`` mentions at least one candidate.

    With ``normalize_numbers``, ``20°C``, ``20 °C``, and ``20 degrees`` match
    each other. With ``casefold``, ``Moderate`` matches ``moderate``. Failure
    lists the candidates and a 200-character excerpt of ``text``.
    """

    if not candidates:
        raise TypeError("answer_mentions_any() requires at least one candidate")
    if _mentions_any(
        text,
        candidates,
        normalize_numbers=normalize_numbers,
        casefold=casefold,
    ):
        return
    excerpt = text if len(text) <= _EXCERPT_LIMIT else text[:_EXCERPT_LIMIT]
    listed = ", ".join(repr(candidate) for candidate in candidates)
    raise AssertionError(
        f"expected answer to mention any of [{listed}]; excerpt={excerpt!r}"
    )


def _mentions_any(
    text: str,
    candidates: Sequence[str],
    *,
    normalize_numbers: bool,
    casefold: bool,
) -> bool:
    haystacks = {_normalize_mention(text, normalize_numbers=False, casefold=casefold)}
    if normalize_numbers:
        haystacks.add(_normalize_mention(text, normalize_numbers=True, casefold=casefold))
    for candidate in candidates:
        needles = {_normalize_mention(candidate, normalize_numbers=False, casefold=casefold)}
        if normalize_numbers:
            needles.add(_normalize_mention(candidate, normalize_numbers=True, casefold=casefold))
        for needle in needles:
            if not needle:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
            if any(pattern.search(haystack) for haystack in haystacks):
                return True
    return False


def _normalize_mention(text: str, *, normalize_numbers: bool, casefold: bool) -> str:
    out = text.casefold() if casefold else text
    if not normalize_numbers:
        return out
    out = out.replace(_DEGREE, " ")
    out = _UNIT_WORDS.sub(" ", out)
    out = _TRAILING_TEMP_UNIT.sub(" ", out)
    return _SPACES.sub(" ", out).strip()


def tool_result_contents(
    call: CapturedCall,
    name: str,
    tool_ids: Collection[Optional[str]],
) -> list[str]:
    wanted_ids = {tool_id for tool_id in tool_ids if tool_id}
    contents: list[str] = []
    for result in call.tool_results:
        result_id = first_present(result, "tool_call_id", "call_id", "callId")
        result_name = result.get("name")
        matched = (wanted_ids and result_id in wanted_ids) or result_name == name
        if not matched and not wanted_ids and result_name is None:
            matched = True
        if not matched:
            continue
        content = first_present(result, "content", "output")
        if content is None:
            continue
        rendered = stringify_value(content)
        if rendered:
            contents.append(rendered)
    return contents


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


def call_was_validated(call: CapturedCall) -> bool:
    validation = call.validation
    if validation is None:
        return False
    if validation.request_checked is True or validation.response_checked is True:
        return True
    return bool(validation.mode and validation.mode != "off")


def selected_side_was_checked(call: CapturedCall, side: Optional[str]) -> bool:
    """Whether validation ran for ``side`` (or any side when ``side`` is None)."""

    if side is None:
        return True
    validation = call.validation
    if validation is None:
        return True
    wanted = side.lower()
    if wanted == "request":
        return validation.request_checked is not False
    if wanted == "response":
        return validation.response_checked is not False
    return True


def path_is_ignored(
    path: Optional[str],
    pointer: Optional[str],
    ignore_paths: Sequence[str],
) -> bool:
    if not ignore_paths:
        return False
    candidates = [value for value in (path, pointer) if value]
    for ignore in ignore_paths:
        if not ignore:
            continue
        ignore_slash = ignore if ignore.startswith("/") else f"/{ignore}"
        ignore_bare = ignore.lstrip("/")
        for candidate in candidates:
            if candidate in {ignore, ignore_slash, ignore_bare}:
                return True
            candidate_slash = candidate if candidate.startswith("/") else f"/{candidate}"
            if candidate_slash == ignore_slash:
                return True
    return False


def format_spec_finding(call: CapturedCall, item: SpecViolation) -> str:
    api = first_present(call.raw, "api_family", "apiFamily") or call.api_type or "?"
    path = item.pointer or item.path or "?"
    message = item.message or item.keyword or "spec violation"
    hint = item.keyword or item.source or "spec"
    return f"{call.call_number()} {api} {item.side or '?'} {path}: {message} — {hint}"


def unknown_request_fields(call: CapturedCall) -> list[IgnoredField]:
    """Ignored fields the relaxed profile records instead of schema violations."""

    items = call.ignored_fields
    if not items:
        return []
    return [
        item
        for item in items
        if (item.kind or "").lower() == "unknown_request_field"
    ]


def format_ignored_finding(call: CapturedCall, item: IgnoredField) -> str:
    api = first_present(call.raw, "api_family", "apiFamily") or call.api_type or "?"
    path = item.pointer or item.path or item.key or "?"
    message = item.message or item.hint or item.kind or "ignored field"
    hint = item.kind or item.severity or "ignored"
    side = item.side or ("request" if (item.kind or "") == "unknown_request_field" else "?")
    return f"{call.call_number()} {api} {side} {path}: {message} — {hint}"


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
