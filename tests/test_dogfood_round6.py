from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from llmentary.assertions import Expectation
from llmentary.client import ControlClient, ProxyHandle, Session
from llmentary.types import (
    CapturedCall,
    SessionInfo,
    SessionTrace,
    ignored_field_from_mapping,
    session_info_from_mapping,
    stringify_response,
)

ROOT = Path(__file__).resolve().parent
CONTEXT_ONLY = json.loads(
    (ROOT / "testdata" / "ignored_fields_context_only.json").read_text()
)
RESPONSE_CASES = json.loads((ROOT / "testdata" / "response_text_cases.json").read_text())


def _trace(*calls: dict[str, Any], session_id: str = "sess") -> SessionTrace:
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": list(calls)})


def _validated_call(*, ignored: list[dict[str, Any]], seq: int = 1) -> dict[str, Any]:
    return {
        "seq": seq,
        "api_family": "chat.completions",
        "request": {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "ok"}}]},
        "normalized": {
            "validation": {
                "mode": "warn",
                "request_checked": True,
                "response_checked": True,
            },
            "spec_violations": [],
            "ignored_fields": ignored,
        },
    }


def _vendor(*, side: Optional[str] = None, context: Optional[str] = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": "provider",
        "pointer": "/provider",
        "key": "provider",
        "kind": "vendor_field",
        "message": "vendor extra",
        "hint": "ignored",
    }
    if side is not None:
        item["side"] = side
    if context is not None:
        item["context"] = context
    return item


def _unknown(*, side: Optional[str] = None, context: Optional[str] = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": "messages[0].extra",
        "pointer": "/messages/0/extra",
        "key": "extra",
        "kind": "unknown_request_field",
        "message": "unknown request field",
    }
    if side is not None:
        item["side"] = side
    if context is not None:
        item["context"] = context
    return item


@pytest.mark.parametrize(
    "label",
    [
        {"context": "request"},
        {"side": "request"},
        {"context": "request", "side": "request"},
    ],
    ids=["context-only", "side-only", "both"],
)
def test_no_ignored_fields_side_request_sees_vendor_fields(label: dict[str, str]) -> None:
    trace = _trace(_validated_call(ignored=[_vendor(**label)]))
    assert trace.calls[0].ignored_fields is not None
    assert trace.calls[0].ignored_fields[0].side == "request"
    with pytest.raises(AssertionError, match=r"request /provider") as caught:
        Expectation(trace).no_ignored_fields(side="request")
    assert "?" not in str(caught.value).split("\n", 1)[0]
    Expectation(trace).no_ignored_fields(side="response")


def test_shared_context_only_vector_fails_side_request() -> None:
    call = CONTEXT_ONLY["call"]
    trace = _trace(call, session_id="context-only")
    fields = trace.calls[0].ignored_fields
    assert fields is not None
    assert [item.side for item in fields] == ["request", "request"]
    assert [item.raw.get("context") for item in fields] == ["request", "request"]
    with pytest.raises(AssertionError, match="vendor_field"):
        Expectation(trace).no_ignored_fields()
    with pytest.raises(AssertionError, match=r"request /provider"):
        Expectation(trace).no_ignored_fields(side="request")
    Expectation(trace).no_ignored_fields(side="response")


def test_ignored_field_neither_label_fail_closes() -> None:
    item = _vendor()
    parsed = ignored_field_from_mapping(item)
    assert parsed.side is None
    trace = _trace(_validated_call(ignored=[item]))
    with pytest.raises(AssertionError, match="vendor extra"):
        Expectation(trace).no_ignored_fields(side="request")
    with pytest.raises(AssertionError, match="vendor extra"):
        Expectation(trace).no_ignored_fields(side="response")


def test_context_wins_over_conflicting_side() -> None:
    item = ignored_field_from_mapping(_vendor(side="response", context="request"))
    assert item.side == "request"


def test_non_side_context_is_not_a_filter_label() -> None:
    item = ignored_field_from_mapping(_vendor(context="chat.message"))
    assert item.side is None


def test_no_spec_violations_side_request_sees_unknown_request_field() -> None:
    for label in (
        {"context": "request"},
        {"side": "request"},
        {"context": "request", "side": "request"},
    ):
        trace = _trace(_validated_call(ignored=[_unknown(**label)]))
        with pytest.raises(AssertionError, match="unknown_request_field"):
            Expectation(trace).no_spec_violations(side="request")
        Expectation(trace).no_spec_violations(side="response")
    neither = _trace(_validated_call(ignored=[_unknown()]))
    with pytest.raises(AssertionError, match="unknown_request_field"):
        Expectation(neither).no_spec_violations(side="request")
    with pytest.raises(AssertionError, match="unknown_request_field"):
        Expectation(neither).no_spec_violations(side="response")


@pytest.mark.parametrize("case", RESPONSE_CASES["cases"], ids=lambda row: row["name"])
def test_response_text_prefers_normalized_output_text(case: dict[str, Any]) -> None:
    call = CapturedCall(raw=case["call"], index=0)
    assert call.response_text == case["response_text"]
    assert call.response_text != "{}"
    assert not call.response_text.lstrip().startswith('{"choices"')


def test_truncated_stream_response_text_is_the_partial_text() -> None:
    case = next(
        row
        for row in RESPONSE_CASES["cases"]
        if row["name"] == "truncated_stream_idle_timeout"
    )
    call = CapturedCall(raw=case["call"], index=0)
    assert call.normalized_payload["output_text"] == "Hello"
    assert call.response_text == "Hello"
    assert "choices" not in call.response_text
    trace = _trace(case["call"])
    Expectation(trace).response_contains("Hello")
    with pytest.raises(AssertionError, match="choices"):
        Expectation(trace).response_contains("choices")
    with pytest.raises(AssertionError):
        Expectation(trace).response_regex(r'"object": "chat.completion.chunk"')
    with pytest.raises(AssertionError, match="choices"):
        Expectation(trace).final_response_mentions_any("choices")


def test_header_timeout_504_response_text_is_empty() -> None:
    case = next(row for row in RESPONSE_CASES["cases"] if row["name"] == "header_timeout_504")
    call = CapturedCall(raw=case["call"], index=0)
    assert call.http_status == 504
    assert call.response_text == ""
    assert stringify_response({}) == ""
    assert stringify_response({"choices": [{"delta": {"content": "Hi"}}]}) == "Hi"


def test_call_count_exclude_aborted_and_query_committed() -> None:
    committed_a = {
        "seq": 1,
        "id": "call_1",
        "client_aborted": False,
        "http_status": 200,
        "request": {"model": "m", "messages": [{"role": "user", "content": "one"}]},
        "response": {"choices": [{"message": {"content": "ok"}}]},
        "normalized": {"output_text": "ok"},
    }
    committed_b = {
        "seq": 2,
        "id": "call_2",
        "client_aborted": False,
        "http_status": 200,
        "request": {"model": "m", "messages": [{"role": "user", "content": "two"}]},
        "response": {"choices": [{"message": {"content": "ok"}}]},
        "normalized": {"output_text": "ok"},
    }
    aborted = {
        "seq": 3,
        "id": "call_3",
        "client_aborted": True,
        "abort_phase": "after_headers",
        "http_status": 200,
        "request": {"model": "m", "messages": [{"role": "user", "content": "barge"}]},
        "response": {"choices": [{"delta": {"content": "He"}}]},
        "normalized": {"output_text": "He", "client_aborted": True},
    }
    trace = _trace(committed_a, committed_b, aborted, session_id="counts")
    expect = Expectation(trace)
    with pytest.raises(AssertionError, match="got 3"):
        expect.call_count(2, include_cancelled=True)
    expect.call_count(2)
    expect.call_count(2, exclude_aborted=True)
    expect.call_count(min=2, max=2, exclude_aborted=True)
    assert [call.id for call in trace.committed().calls] == ["call_1", "call_2"]
    assert [call.id for call in trace.aborted().calls] == ["call_3"]
    Expectation(trace.committed()).call_count(2)
    Expectation(trace.aborted()).call_count(1, include_cancelled=True)
    empty_committed = _trace(aborted, session_id="only-abort").committed()
    assert "committed" in empty_committed.empty_match_message()
    empty_aborted = _trace(committed_a, session_id="only-ok").aborted()
    assert "aborted" in empty_aborted.empty_match_message()
    with pytest.raises(AssertionError, match="committed call"):
        Expectation(trace).call_count(3, exclude_aborted=True)


def test_session_info_exposes_spec_summary() -> None:
    row = session_info_from_mapping(
        {
            "id": "s1",
            "calls": 3,
            "active": 0,
            "sources": ["path"],
            "spec": {
                "checked": 2,
                "clean": 1,
                "findings": 1,
                "request_only": 1,
                "not_checked": 0,
            },
        }
    )
    positional = SessionInfo(
        "legacy",
        1,
        0,
        (),
        None,
        None,
        {"id": "legacy"},
    )
    assert positional.raw == {"id": "legacy"}
    assert positional.spec is None
    assert isinstance(row, SessionInfo)
    assert row.spec is not None
    assert row.spec.checked == 2
    assert row.spec.clean == 1
    assert row.spec.findings == 1
    assert row.spec.request_only == 1
    assert row.spec.not_checked == 0

    aliases = session_info_from_mapping(
        {
            "id": "s2",
            "calls": 1,
            "spec": {
                "Checked": 1,
                "Clean": 1,
                "Findings": 0,
                "RequestOnly": 0,
                "NotChecked": 0,
            },
        }
    )
    assert aliases.spec is not None
    assert aliases.spec.checked == 1
    assert aliases.spec.clean == 1

    missing = session_info_from_mapping({"id": "old", "calls": 1})
    assert missing.spec is None
    assert session_info_from_mapping({"id": "bad", "spec": "nope"}).spec is None


def test_query_committed_and_aborted_filters() -> None:
    payload = {
        "id": "s",
        "calls": [
            {
                "seq": 1,
                "id": "call_1",
                "client_aborted": False,
                "request": {"model": "m", "messages": [{"role": "user", "content": "a"}]},
                "response": {"choices": [{"message": {"content": "ok"}}]},
                "normalized": {"output_text": "ok"},
            },
            {
                "seq": 2,
                "id": "call_2",
                "client_aborted": True,
                "abort_phase": "after_headers",
                "http_status": 200,
                "request": {"model": "m", "messages": [{"role": "user", "content": "b"}]},
                "response": {"choices": [{"delta": {"content": "H"}}]},
                "normalized": {"output_text": "H", "client_aborted": True},
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait") or request.url.path.endswith("/sessions/s"):
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        session = Session(
            session_id="s",
            proxy=ProxyHandle(
                url="http://127.0.0.1:7777",
                control=ControlClient("http://127.0.0.1:7778", client=http),
            ),
        )
        session.query().committed().expect().call_count(1)
        session.query().aborted().expect().call_count(1, include_cancelled=True)
        session.expect().call_count(1, exclude_aborted=True)
        session.expect().call_count(2, include_cancelled=True)
