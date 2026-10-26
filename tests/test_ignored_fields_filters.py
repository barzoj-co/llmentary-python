from __future__ import annotations

from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace


def _trace(*calls: dict[str, Any], session_id: str = "sess") -> SessionTrace:
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": list(calls)})


def _validated_call(*, ignored: list[dict[str, Any]], seq: int = 1) -> dict[str, Any]:
    return {
        "seq": seq,
        "api_family": "chat.completions",
        "request": {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        "normalized": {
            "validation": {
                "mode": "warn",
                "profile": "relaxed",
                "target": "openai-compatible",
                "request_checked": True,
                "response_checked": True,
            },
            "spec_violations": [],
            "ignored_fields": ignored,
        },
    }


def _entry(
    *,
    kind: str,
    context: str,
    severity: Optional[str] = None,
    key: Optional[str] = None,
) -> dict[str, Any]:
    name = key or kind
    item: dict[str, Any] = {
        "path": name,
        "pointer": f"/{name}",
        "key": name,
        "kind": kind,
        "context": context,
        "message": f"{kind} is ignored",
        "action": "report_only",
    }
    if severity is not None:
        item["severity"] = severity
    return item


VENDOR_FIELD = _entry(kind="vendor_field", context="request", severity="info", key="provider")
COMPATIBLE_OPTIONAL = _entry(
    kind="compatible_optional", context="response", severity="info", key="usage"
)
WARNING_ENTRY = _entry(kind="unknown_request_field", context="request", severity="warning")
UNLABELED_ENTRY = _entry(kind="response_shape", context="response")

ALL_ENTRIES = [VENDOR_FIELD, WARNING_ENTRY, COMPATIBLE_OPTIONAL, UNLABELED_ENTRY]


def _expect(*ignored: dict[str, Any]) -> Expectation:
    return Expectation(_trace(_validated_call(ignored=list(ignored))))


def test_ignored_field_severities_are_parsed() -> None:
    trace = _trace(_validated_call(ignored=ALL_ENTRIES))
    fields = trace.calls[0].ignored_fields
    assert fields is not None
    assert [item.kind for item in fields] == [
        "vendor_field",
        "unknown_request_field",
        "compatible_optional",
        "response_shape",
    ]
    assert [item.side for item in fields] == ["request", "request", "response", "response"]
    assert [item.severity for item in fields] == ["info", "warning", "info", None]


def test_default_behaviour_unchanged() -> None:
    expect = _expect(*ALL_ENTRIES)
    with pytest.raises(AssertionError) as caught:
        expect.no_ignored_fields()
    message = str(caught.value)
    for kind in ("vendor_field", "compatible_optional", "unknown_request_field", "response_shape"):
        assert kind in message
    with pytest.raises(AssertionError, match="provider"):
        _expect(VENDOR_FIELD).no_ignored_fields()
    with pytest.raises(AssertionError, match="enumDescriptions"):
        Expectation(
            _trace(
                _validated_call(
                    ignored=[
                        _entry(
                            kind="enumDescriptions",
                            context="request",
                            severity="warning",
                            key="enumDescriptions",
                        )
                    ]
                )
            )
        ).no_ignored_fields()


def test_min_severity_warning_keeps_warning_and_unlabeled() -> None:
    expect = _expect(*ALL_ENTRIES)
    with pytest.raises(AssertionError) as caught:
        expect.no_ignored_fields(min_severity="warning")
    message = str(caught.value)
    assert "unknown_request_field" in message
    assert "response_shape" in message
    assert "vendor_field" not in message
    assert "compatible_optional" not in message


def test_min_severity_info_and_error_rungs() -> None:
    expect = _expect(*ALL_ENTRIES)
    with pytest.raises(AssertionError) as caught:
        expect.no_ignored_fields(min_severity="info")
    message = str(caught.value)
    for kind in ("vendor_field", "compatible_optional", "unknown_request_field", "response_shape"):
        assert kind in message
    with pytest.raises(AssertionError) as caught:
        expect.no_ignored_fields(min_severity="error")
    message = str(caught.value)
    assert "response_shape" in message
    for kind in ("vendor_field", "compatible_optional", "unknown_request_field"):
        assert kind not in message


def test_min_severity_keeps_a_checked_and_clean_trace() -> None:
    _expect(VENDOR_FIELD, COMPATIBLE_OPTIONAL).no_ignored_fields(min_severity="warning")
    _expect(VENDOR_FIELD).no_ignored_fields(min_severity="warning", ignore_kinds=["vendor_field"])


def test_ignore_kinds_matches_kind_like_ignore_keywords() -> None:
    expected = _expect(*ALL_ENTRIES)
    for kinds in (
        ["vendor_field"],
        ["vendor_field", "compatible_optional"],
        ["unknown_request_field", "response_shape"],
    ):
        with pytest.raises(AssertionError) as by_kinds:
            _expect(*ALL_ENTRIES).no_ignored_fields(ignore_kinds=kinds)
        with pytest.raises(AssertionError) as by_keywords:
            expected.no_ignored_fields(ignore_keywords=tuple(kinds))
        assert str(by_kinds.value) == str(by_keywords.value)
    _expect(*ALL_ENTRIES).no_ignored_fields(
        ignore_kinds=[
            "vendor_field",
            "compatible_optional",
            "unknown_request_field",
            "response_shape",
        ]
    )


def test_ignore_kinds_and_ignore_keywords_union() -> None:
    _expect(*ALL_ENTRIES).no_ignored_fields(
        ignore_keywords=("vendor_field", "compatible_optional"),
        ignore_kinds=("unknown_request_field", "response_shape"),
    )
    with pytest.raises(AssertionError, match="response_shape") as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(
            ignore_keywords=("vendor_field",),
            ignore_kinds=("compatible_optional", "unknown_request_field"),
        )
    message = str(caught.value)
    assert "vendor_field" not in message
    assert "compatible_optional" not in message


def test_single_string_ignore_filters_are_one_entry() -> None:
    _expect(VENDOR_FIELD).no_ignored_fields(ignore_keywords="vendor_field")
    _expect(VENDOR_FIELD).no_ignored_fields(ignore_kinds="vendor_field")
    _expect(VENDOR_FIELD, COMPATIBLE_OPTIONAL).no_ignored_fields(
        ignore_keywords="vendor_field",
        ignore_kinds="compatible_optional",
    )
    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(ignore_keywords="vendor_field")
    message = str(caught.value)
    assert "vendor_field" not in message
    for kind in ("unknown_request_field", "compatible_optional", "response_shape"):
        assert kind in message
    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(ignore_kinds="vendor_field")
    message = str(caught.value)
    assert "vendor_field" not in message
    for kind in ("unknown_request_field", "compatible_optional", "response_shape"):
        assert kind in message
    near_miss = _entry(kind="vendor", context="request", severity="info", key="vendor")
    with pytest.raises(AssertionError, match="vendor"):
        _expect(near_miss).no_ignored_fields(ignore_kinds="vendor_field")


def test_filters_combine_with_side() -> None:
    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(ignore_kinds=["vendor_field"])
    both_sides = str(caught.value)
    assert "vendor_field" not in both_sides
    for kind in ("unknown_request_field", "compatible_optional", "response_shape"):
        assert kind in both_sides

    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(side="request", ignore_kinds=["vendor_field"])
    message = str(caught.value)
    assert "unknown_request_field" in message
    for kind in ("vendor_field", "compatible_optional", "response_shape"):
        assert kind not in message
    _expect(*ALL_ENTRIES).no_ignored_fields(
        side="request", ignore_kinds=["vendor_field", "unknown_request_field"]
    )

    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(side="response", ignore_kinds=["response_shape"])
    message = str(caught.value)
    assert "compatible_optional" in message
    for kind in ("vendor_field", "unknown_request_field", "response_shape"):
        assert kind not in message

    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(side="response", min_severity="warning")
    message = str(caught.value)
    assert "response_shape" in message
    assert "compatible_optional" not in message

    _expect(*ALL_ENTRIES).no_ignored_fields(
        side="request", ignore_kinds=["vendor_field"], min_severity="error"
    )


def test_filters_combine_with_ignore_paths_and_calls() -> None:
    with pytest.raises(AssertionError) as caught:
        _expect(*ALL_ENTRIES).no_ignored_fields(
            ignore_paths=(VENDOR_FIELD["pointer"], WARNING_ENTRY["pointer"]),
            min_severity="warning",
        )
    message = str(caught.value)
    assert "response_shape" in message
    for kind in ("vendor_field", "compatible_optional", "unknown_request_field"):
        assert kind not in message
    _expect(*ALL_ENTRIES).no_ignored_fields(
        ignore_paths=(VENDOR_FIELD["pointer"], WARNING_ENTRY["pointer"]),
        ignore_kinds=["response_shape"],
        min_severity="warning",
    )
    dirty = _validated_call(ignored=[WARNING_ENTRY], seq=1)
    clean = _validated_call(ignored=[], seq=2)
    trace = _trace(dirty, clean, session_id="calls")
    Expectation(trace).no_ignored_fields(calls=[2], min_severity="info")
    with pytest.raises(AssertionError, match="unknown_request_field"):
        Expectation(trace).no_ignored_fields(calls=[1], min_severity="info")


def test_invalid_min_severity_raises_value_error() -> None:
    expect = _expect(*ALL_ENTRIES)
    for value in ("critical", "high", "warn", "errors", ""):
        with pytest.raises(ValueError, match="min_severity"):
            expect.no_ignored_fields(min_severity=value)
    with pytest.raises(ValueError, match="min_severity"):
        Expectation(_trace({"seq": 1}, session_id="no-calls")).no_ignored_fields(
            min_severity="critical"
        )
