from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

# Chatservice live run 1 (2026-10-20, head 76c8ace): the hand-written scraper
# treated ``normalized.validation`` as a finding. That payload is clean.
CHATSERVICE_RUN1_NORMALIZED: dict[str, Any] = {
    "validation": {
        "mode": "warn",
        "profile": "relaxed",
        "spec_sha": "f88d2f78deadbeefcafebabe01234567",
        "schema_subset_retrieved": "2026-10-18",
        "request_checked": True,
        "response_checked": True,
        "skipped_reason": None,
        "duration_us": 400,
    },
    "spec_violations": [],
}

FOO_VIOLATION: dict[str, Any] = {
    "side": "request",
    "source": "schema",
    "path": "foo",
    "pointer": "/foo",
    "keyword": "unevaluatedProperties",
    "message": "additional property not allowed",
    "profile": "relaxed",
}

ENUM_IGNORED: dict[str, Any] = {
    "side": "request",
    "path": "tools/0/function/parameters/properties/city/enumDescriptions",
    "pointer": "/tools/0/function/parameters/properties/city/enumDescriptions",
    "kind": "enumDescriptions",
    "severity": "warning",
    "message": "vendor extension is ignored",
}

# Live OpenAI records this on ignored_fields, not spec_violations (LTL-API-2 L2).
L2_UNKNOWN_NESTED: dict[str, Any] = {
    "path": "messages[0].content[0].ltl_unknown_nested",
    "pointer": "/messages/0/content/0/ltl_unknown_nested",
    "key": "ltl_unknown_nested",
    "kind": "unknown_request_field",
    "context": "chat.message",
    "effective_strict": False,
    "upstream_behavior": "silently_ignored",
    "severity": "warning",
    "action": "reject_in_enforce",
    "hint": (
        "The live OpenAI API accepts this field and silently ignores it. "
        "Remove it, or it will never reach the model."
    ),
}


def _call(
    *,
    seq: int = 1,
    api_family: str = "chat.completions",
    normalized: Optional[Mapping[str, Any]] = None,
    extra_normalized: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seq": seq,
        "api_family": api_family,
        "request": {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        "normalized": dict(normalized or {}),
    }
    if extra_normalized:
        merged = dict(payload["normalized"])
        merged.update(extra_normalized)
        payload["normalized"] = merged
    return payload


def _trace(*calls: Mapping[str, Any], session_id: str = "sess") -> SessionTrace:
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": list(calls)})


def test_chatservice_run1_payload_is_clean_not_a_violation() -> None:
    trace = _trace(_call(normalized=CHATSERVICE_RUN1_NORMALIZED), session_id="chatservice-run1")
    call = trace.calls[0]
    assert call.validation is not None
    assert call.validation.mode == "warn"
    assert call.validation.profile == "relaxed"
    assert call.validation.request_checked is True
    assert call.validation.response_checked is True
    assert call.validation.spec_sha is not None
    assert call.validation.schema_subset_retrieved == "2026-10-18"
    assert call.spec_violations == []
    assert call.spec_status_label() == "clean"
    Expectation(trace).no_spec_violations()
    Expectation(trace).no_ignored_fields()
    assert trace.spec_violations() == []


def test_spec_violations_absent_means_not_checked() -> None:
    trace = _trace(_call(normalized={"model": "fake-model"}))
    call = trace.calls[0]
    assert call.validation is None
    assert call.spec_violations is None
    assert call.spec_status_label() == "not checked"
    with pytest.raises(AssertionError, match="validation was not run"):
        Expectation(trace).no_spec_violations()
    Expectation(trace).no_spec_violations(allow_unvalidated=True)


def test_validation_present_with_empty_findings_is_clean() -> None:
    trace = _trace(
        _call(
            normalized={
                "validation": {"mode": "warn", "request_checked": True, "response_checked": True},
                "spec_violations": [],
            }
        )
    )
    assert trace.calls[0].spec_violations == []
    Expectation(trace).no_spec_violations()


def test_spec_violations_present_fail_with_contract_message() -> None:
    trace = _trace(
        _call(
            normalized={
                "validation": {"mode": "warn", "request_checked": True},
                "spec_violations": [FOO_VIOLATION],
            }
        )
    )
    call = trace.calls[0]
    assert call.spec_violations is not None
    assert len(call.spec_violations) == 1
    assert call.spec_violations[0].pointer == "/foo"
    assert call.spec_violations[0].keyword == "unevaluatedProperties"
    assert call.spec_status_label() == "1 violation"
    with pytest.raises(
        AssertionError,
        match=(
            r"1 chat\.completions request /foo: additional property not allowed "
            r"— unevaluatedProperties"
        ),
    ) as caught:
        Expectation(trace).no_spec_violations()
    assert "spec: 1 violation" in str(caught.value)


def test_no_spec_violations_options_table() -> None:
    dirty = _call(
        seq=1,
        normalized={
            "validation": {"mode": "warn"},
            "spec_violations": [FOO_VIOLATION],
        },
    )
    clean = _call(
        seq=2,
        normalized={"validation": {"mode": "warn"}, "spec_violations": []},
    )
    dirty_trace = _trace(dirty, clean)

    Expectation(dirty_trace).no_spec_violations(side="response")
    with pytest.raises(AssertionError, match="/foo"):
        Expectation(dirty_trace).no_spec_violations(side="request")
    Expectation(dirty_trace).no_spec_violations(ignore_paths=("/foo",))
    Expectation(dirty_trace).no_spec_violations(ignore_paths=("foo",))
    Expectation(dirty_trace).no_spec_violations(ignore_keywords=("unevaluatedProperties",))
    Expectation(dirty_trace).no_spec_violations(calls=[2])
    with pytest.raises(AssertionError, match="/foo"):
        Expectation(dirty_trace).no_spec_violations(calls=[1])


def test_ignored_fields_accessors_and_assertion() -> None:
    clean = _trace(
        _call(normalized={"validation": {"mode": "warn"}, "ignored_fields": []}),
        session_id="ignored-clean",
    )
    assert clean.calls[0].ignored_fields == []
    Expectation(clean).no_ignored_fields()

    off = _trace(_call(normalized={"model": "x"}), session_id="ignored-off")
    assert off.calls[0].ignored_fields is None
    with pytest.raises(AssertionError, match="validation was not run"):
        Expectation(off).no_ignored_fields()
    Expectation(off).no_ignored_fields(allow_unvalidated=True)

    dirty = _trace(
        _call(
            normalized={
                "validation": {"mode": "warn"},
                "ignored_fields": [ENUM_IGNORED],
            }
        ),
        session_id="ignored-dirty",
    )
    fields = dirty.calls[0].ignored_fields
    assert fields is not None
    assert fields[0].kind == "enumDescriptions"
    with pytest.raises(AssertionError, match="enumDescriptions"):
        Expectation(dirty).no_ignored_fields()
    Expectation(dirty).no_ignored_fields(ignore_paths=(ENUM_IGNORED["pointer"],))
    Expectation(dirty).no_ignored_fields(ignore_keywords=("enumDescriptions",))
    Expectation(dirty).no_ignored_fields(side="response")


def test_select_calls_uses_seq_not_conflicting_index() -> None:
    dirty = _call(
        seq=1,
        normalized={
            "validation": {"mode": "warn"},
            "spec_violations": [FOO_VIOLATION],
        },
    )
    clean = _call(
        seq=2,
        normalized={"validation": {"mode": "warn"}, "spec_violations": []},
    )
    # indexes are 0 and 1; seq values are 1 and 2. calls=[1] must mean seq 1.
    Expectation(_trace(dirty, clean)).no_spec_violations(calls=[2])
    with pytest.raises(AssertionError, match="/foo"):
        Expectation(_trace(dirty, clean)).no_spec_violations(calls=[1])


def test_unknown_request_field_fails_no_spec_violations() -> None:
    trace = _trace(
        _call(
            normalized={
                "validation": {
                    "mode": "warn",
                    "profile": "relaxed",
                    "request_checked": True,
                    "response_checked": True,
                },
                "spec_violations": [],
                "ignored_fields": [L2_UNKNOWN_NESTED],
            }
        )
    )
    call = trace.calls[0]
    assert call.spec_violations == []
    fields = call.ignored_fields
    assert fields is not None
    assert fields[0].kind == "unknown_request_field"
    assert fields[0].key == "ltl_unknown_nested"
    assert fields[0].hint is not None
    with pytest.raises(AssertionError) as caught:
        Expectation(trace).no_spec_violations()
    message = str(caught.value)
    assert "validation was not run" not in message
    assert " — " in message
    assert "ltl_unknown_nested" in message or "/ltl_unknown" in message
    Expectation(trace).no_spec_violations(ignore_keywords=("unknown_request_field",))
    Expectation(trace).no_spec_violations(ignore_paths=("/messages/0/content/0/ltl_unknown_nested",))
    Expectation(trace).no_spec_violations(side="response")
    with pytest.raises(AssertionError, match="ltl_unknown"):
        Expectation(trace).no_spec_violations(side="request")


def test_no_spec_violations_side_requires_that_side_checked() -> None:
    trace = _trace(
        _call(
            normalized={
                "validation": {
                    "mode": "warn",
                    "request_checked": True,
                    "response_checked": False,
                },
                "spec_violations": [],
            }
        )
    )
    Expectation(trace).no_spec_violations()
    Expectation(trace).no_spec_violations(side="request")
    with pytest.raises(AssertionError, match="validation was not run"):
        Expectation(trace).no_spec_violations(side="response")
    Expectation(trace).no_spec_violations(side="response", allow_unvalidated=True)


def test_enum_ignored_fields_do_not_fail_no_spec_violations() -> None:
    trace = _trace(
        _call(
            normalized={
                "validation": {"mode": "warn"},
                "spec_violations": [],
                "ignored_fields": [ENUM_IGNORED],
            }
        )
    )
    Expectation(trace).no_spec_violations()
    with pytest.raises(AssertionError, match="enumDescriptions"):
        Expectation(trace).no_ignored_fields()


def test_trace_summary_includes_spec_status() -> None:
    clean = _trace(
        _call(normalized={"validation": {"mode": "warn"}, "spec_violations": []}),
        session_id="summary",
    )
    assert "spec: clean" in clean.trace_summary()
    missing = _trace(_call(normalized={"model": "x"}), session_id="summary-off")
    assert "spec: not checked" in missing.trace_summary()
