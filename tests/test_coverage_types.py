from __future__ import annotations

from typing import Any

import pytest
from test_go_session_schema import go_call

from llmentary.assertions import Expectation
from llmentary.types import (
    SessionTrace,
    coerce_mapping,
    decode_base64_json,
    decode_value,
    extract_json_object,
    extract_llm_side,
    extract_user_texts,
    find_call_list,
    first_present,
    int_or_none,
    json_pointer,
    model_matches,
    normalize_api_type,
    optional_int,
    optional_list_field,
    parse_args,
    session_info_from_mapping,
    stringify_response,
    stringify_value,
    unwrap_error,
)


def _ok_call() -> dict[str, Any]:
    return go_call(
        seq=1,
        api_family="chat.completions",
        request_body={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        response_body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )


def test_api_type_from_path_and_payload_corners() -> None:
    chat = _ok_call()
    chat.pop("api_family", None)
    chat["normalized"] = {}
    chat["path"] = "/v1/chat/completions"
    captured = SessionTrace.from_payload("p", {"id": "p", "calls": [chat]}).calls[0]
    assert captured.api_type == "chat_completions"

    responses = _ok_call()
    responses.pop("api_family", None)
    responses["normalized"] = {}
    responses["path"] = "/v1/responses"
    responses["request"] = {"input": "hi"}
    responses["response"] = {"output": []}
    captured = SessionTrace.from_payload("r", {"id": "r", "calls": [responses]}).calls[0]
    assert captured.api_type == "responses"


def test_http_status_and_finish_reason_corners() -> None:
    call = _ok_call()
    call["http_status"] = True
    call["response"] = {"status": 204}
    captured = SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0]
    assert captured.http_status == 204

    call = _ok_call()
    call["http_status"] = "200"
    assert SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0].http_status == 200

    call = _ok_call()
    call["http_status"] = "0"
    assert SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0].http_status is None

    call = _ok_call()
    call.pop("normalized", None)
    call["response"] = {"status": "completed"}
    call["raw_response"] = None
    assert (
        SessionTrace.from_payload("st", {"id": "st", "calls": [call]}).calls[0].finish_reason
        == "completed"
    )


def test_offered_tools_validation_filters_and_helpers() -> None:
    call = _ok_call()
    call["request"] = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "direct"},
            {"function": {"name": "nested"}},
            {"function": {}},
            "skip",
        ],
    }
    call["validation"] = {"mode": "live", "response_checked": False, "skipped_reason": ""}
    call["spec_violations"] = [{"message": "one"}]
    call["ignored_fields"] = [{"path": "/temperature"}]
    call["normalized"] = {
        "response": {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_nested",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    }
    trace = SessionTrace.from_payload("f", {"id": "f", "calls": [call]})
    captured = trace.calls[0]
    assert captured.offered_tool_names() == ["direct", "nested"]
    assert captured.validation is not None
    assert captured.spec_status_label() == "1 violation"
    assert [tool.name for tool in captured.tool_calls] == ["lookup"]
    assert len(trace.spec_violations()) == 1
    assert len(trace.ignored_fields()) == 1
    assert trace.empty_match_message() == "no calls matched query"
    filtered = trace.without_tools()
    assert "without_tools" in filtered.empty_match_message()
    assert filtered.spec_violations() == []
    assert filtered.ignored_fields() == []

    skipped = _ok_call()
    skipped["spec"] = "skipped:no_schema"
    assert (
        SessionTrace.from_payload("s", {"id": "s", "calls": [skipped]}).calls[0].spec_status_label()
        == "skipped:no_schema"
    )


def test_model_matches_normalize_and_find_call_list() -> None:
    assert model_matches(None, "gpt-4.1") is False
    assert model_matches("gpt-4.1", "gpt-4.1") is True
    assert model_matches("gpt-4.1-mini", "gpt-4.1") is False
    assert model_matches("gpt-4.1-2025-01-01", "gpt-4.1") is True
    assert normalize_api_type("chat") == "chat_completions"
    assert normalize_api_type("something-else") == "something_else"
    assert len(find_call_list({"calls": {"a": _ok_call(), "b": _ok_call()}})) == 2
    assert len(find_call_list({"session": {"calls": [_ok_call()]}})) == 1
    assert find_call_list({}) == []


def test_parse_and_decode_helpers() -> None:
    not_a_map: Any = "not-a-map"
    assert first_present(not_a_map, "x") is None
    assert unwrap_error({"foo": 1}) == {}
    assert unwrap_error({"type": "invalid"}) == {"type": "invalid"}
    assert unwrap_error("not-json") == {}
    assert unwrap_error('{"error": {"message": "boom"}}') == {"message": "boom"}
    assert coerce_mapping('{"ok": true}') == {"ok": True}
    assert extract_json_object("no object") is None
    assert extract_json_object("{not json}") is None
    assert decode_value(b"{", depth=6) == b"{"
    assert decode_value(b"{not-json}") == "{not-json}"
    assert decode_value(bytes([0xFF, 0xFE])) == bytes([0xFF, 0xFE])
    assert decode_value("   ") == "   "
    assert decode_base64_json("abcd") is None
    assert int_or_none(3.0) == 3
    assert int_or_none(3.5) is None
    assert optional_int({"n": 2.0}, "n") == 2
    assert optional_int({"n": 2.5}, "n") is None
    assert optional_list_field(not_a_map, "items") is None
    assert optional_list_field({"items": None}, "items") is None
    assert optional_list_field({"items": "nope"}, "items") is None
    assert parse_args({"a": 1}) == {"a": 1}
    assert parse_args("") == {}
    assert parse_args("{") == {}
    assert parse_args("[1]") == {}
    assert json_pointer(["a", "b/c"]) == "/a/b~1c"
    assert json_pointer([]) == ""
    assert stringify_value({"text": "hi"}) == "hi"
    assert stringify_value(None) == ""
    assert stringify_response({"output_text": "done"}) == "done"
    assert stringify_response({"output": [{"type": "message", "content": "x"}]}) == "x"
    assert stringify_response({"output": [{"content": "y"}]}) == "y"
    assert extract_user_texts(
        {
            "input": [
                "plain",
                {"type": "input_text", "text": "typed"},
                {"type": "function_call"},
            ]
        }
    ) == ["plain", "typed"]
    assert extract_llm_side(
        {"normalized": {"request": {"messages": [{"role": "user", "content": "hi"}]}}},
        ("request",),
        side="request",
    )["messages"][0]["content"] == "hi"
    info = session_info_from_mapping({"id": "s", "calls": [_ok_call()]})
    assert info.calls == 1


def test_expectation_call_count_and_order_corners() -> None:
    trace = SessionTrace.from_payload("t", {"id": "t", "calls": [_ok_call()]})
    expect = Expectation(trace)
    with pytest.raises(TypeError, match="exact count"):
        expect.call_count()
    with pytest.raises(TypeError, match="greater than max"):
        expect.call_count(min=2, max=1)
    with pytest.raises(TypeError, match="Session, session id"):
        expect.isolated_from(123)
    expect.not_called("missing")
    with pytest.raises(AssertionError, match="tool order"):
        expect.order("lookup")
    expect.retries(min_count=0, max_count=3)
    expect.no_retries()
    expect.no_errors()
    with pytest.raises(AssertionError, match="at least one error"):
        expect.has_error()
