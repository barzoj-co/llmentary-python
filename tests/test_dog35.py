from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_go_session_schema import go_call

from llmentary.binary import BinaryResolutionError
from llmentary.doctor import (
    binary_version_output,
    doctor,
    parse_proxy_semver,
    proxy_binary_path,
    proxy_version,
    proxy_version_line,
)
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


def _abort_call(
    *,
    phase: str | None,
    http_status: int | None,
    stream_truncated: bool = False,
    upstream_called: bool | None = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body={"model": "fake-model", "messages": [{"role": "user", "content": "count"}]},
        response_body={"choices": [{"message": {"content": "One,"}}]}
        if phase == "after_headers"
        else {"choices": []},
    )
    call["client_aborted"] = True
    call["stream"] = True
    if phase is not None:
        call["abort_phase"] = phase
    if http_status is None:
        call["http_status"] = 0
    else:
        call["http_status"] = http_status
    if upstream_called is not None:
        call["upstream_called"] = upstream_called
    call["normalized"] = {
        "client_aborted": True,
        "finish_reason": "cancelled",
        "stream_truncated": stream_truncated,
    }
    if extra:
        call.update(extra)
    return call


def _ok_call() -> dict[str, Any]:
    return go_call(
        seq=1,
        api_family="chat.completions",
        request_body={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        response_body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("llmentary 0.4.0 (f970bafe, 2026-10-21T18:04:54Z, go1.26.10, linux/amd64)", "0.4.0"),
        ("llmentary v0.4.0 (sha)", "0.4.0"),
        ("0.4.0", "0.4.0"),
        ("llmentary 0.4.0-dev+dirty", "0.4.0-dev+dirty"),
        ("llmentary", None),
        ("llmentary unknown", None),
        ("unknown", None),
        ("not found", None),
        ("", None),
        ("   ", None),
        (None, None),
        ("llmentary 0.4.0\nsecond line", "0.4.0"),
        ("other-bin 1.2.3", None),
    ],
)
def test_parse_proxy_semver_corners(line: str | None, expected: str | None) -> None:
    assert parse_proxy_semver(line) == expected


def test_proxy_version_returns_semver_from_version_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text(
        "#!/bin/sh\necho 'llmentary 0.4.0 (f970bafe3fc2f9374a916b6f06b71471fd764bab, "
        "2026-10-21T18:04:54Z, go1.26.10, linux/amd64)'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    assert proxy_version() == "0.4.0"
    line = proxy_version_line()
    assert line is not None
    assert line.startswith("llmentary 0.4.0 (")


def test_proxy_version_none_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMENTARY_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    env = {"PATH": ""}
    assert proxy_version(env=env) is None
    assert proxy_version_line(env=env) is None


def test_doctor_in_process_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.0 (deadbeef)'\n", encoding="utf-8")
    binary.chmod(0o755)
    out = io.StringIO()
    assert doctor(env={"LLMENTARY_BIN": str(binary), "PATH": ""}, out=out) == 0
    assert "binary --version: llmentary 0.4.0" in out.getvalue()

    monkeypatch.setattr(
        "llmentary.doctor.find_binary",
        lambda env=None: (_ for _ in ()).throw(BinaryResolutionError("no binary")),
    )
    failed = io.StringIO()
    assert doctor(env={"PATH": ""}, out=failed) == 1
    assert "no binary" in failed.getvalue()


def test_binary_version_output_corners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert binary_version_output(None) == "unknown"

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("nope")

    monkeypatch.setattr("llmentary.doctor.subprocess.run", boom)
    assert binary_version_output(tmp_path / "missing") == "unknown"

    def timed_out(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="llmentary", timeout=5)

    monkeypatch.setattr("llmentary.doctor.subprocess.run", timed_out)
    assert binary_version_output(tmp_path / "slow") == "unknown"

    empty = tmp_path / "empty"
    empty.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    empty.chmod(0o755)
    monkeypatch.undo()
    assert binary_version_output(empty) == "unknown"


def test_proxy_binary_path_none_on_resolution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llmentary.doctor.find_binary",
        lambda env=None: (_ for _ in ()).throw(BinaryResolutionError("missing")),
    )
    assert proxy_binary_path(env={"PATH": ""}) is None


def test_proxy_version_none_when_version_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    assert proxy_version_line() is None
    assert proxy_version() is None


def test_upstream_called_top_level_true_false_and_absent() -> None:
    present = _abort_call(phase="after_headers", http_status=200, upstream_called=True)
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [present]}).calls[0]
    assert captured.upstream_called is True

    denied = _abort_call(phase="before_headers", http_status=None, upstream_called=False)
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [denied]}).calls[0]
    assert captured.upstream_called is False

    missing = _abort_call(phase="before_headers", http_status=None, upstream_called=None)
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [missing]}).calls[0]
    assert captured.upstream_called is None


def test_upstream_called_camel_case_and_non_bool_corners() -> None:
    call = _abort_call(phase="after_headers", http_status=200, upstream_called=None)
    call["upstreamCalled"] = True
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [call]}).calls[0]
    assert captured.upstream_called is True

    call = _abort_call(phase="after_headers", http_status=200, upstream_called=None)
    call["upstream_called"] = "true"
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [call]}).calls[0]
    assert captured.upstream_called is None

    call = _abort_call(phase="after_headers", http_status=200, upstream_called=None)
    call["upstream_called"] = 1
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [call]}).calls[0]
    assert captured.upstream_called is None

    call = _abort_call(phase="after_headers", http_status=200, upstream_called=None)
    call["normalized"] = {**call["normalized"], "upstream_called": True}
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [call]}).calls[0]
    assert captured.upstream_called is None


def test_abort_summary_status_is_aborted_for_both_phases() -> None:
    before = _abort_call(phase="before_headers", http_status=None, stream_truncated=False)
    after = _abort_call(phase="after_headers", http_status=200, stream_truncated=True)
    before_summary = SessionTrace.from_payload(
        "before", {"id": "before", "calls": [before]}
    ).trace_summary()
    after_summary = SessionTrace.from_payload(
        "after", {"id": "after", "calls": [after]}
    ).trace_summary()
    assert "status=aborted" in before_summary
    assert "status=error" not in before_summary
    assert "status=ok" not in before_summary
    assert "phase=before_headers" in before_summary
    assert "http=" not in before_summary
    assert "stream_truncated" not in before_summary
    assert "status=aborted" in after_summary
    assert "status=ok" not in after_summary
    assert "phase=after_headers" in after_summary
    assert "http=200" in after_summary
    assert "stream_truncated" in after_summary


def test_non_aborted_summary_status_unchanged() -> None:
    ok = _ok_call()
    ok["http_status"] = 200
    ok_summary = SessionTrace.from_payload("ok", {"id": "ok", "calls": [ok]}).trace_summary()
    assert "status=ok" in ok_summary
    assert "status=aborted" not in ok_summary
    assert "phase=" not in ok_summary

    errored = _ok_call()
    errored["http_status"] = 401
    errored["error"] = {"type": "invalid_request_error", "code": "invalid_api_key"}
    error_summary = SessionTrace.from_payload(
        "err", {"id": "err", "calls": [errored]}
    ).trace_summary()
    assert "status=error" in error_summary
    assert "status=aborted" not in error_summary


def test_aborted_with_error_payload_still_status_aborted() -> None:
    call = _abort_call(phase="before_headers", http_status=None)
    call["error"] = {"message": "context canceled"}
    summary = SessionTrace.from_payload("e", {"id": "e", "calls": [call]}).trace_summary()
    assert "status=aborted" in summary
    assert "status=error" not in summary


def test_aborted_without_phase_omits_phase_token() -> None:
    call = _abort_call(phase=None, http_status=None)
    summary = SessionTrace.from_payload("p", {"id": "p", "calls": [call]}).trace_summary()
    assert "status=aborted" in summary
    assert "phase=" not in summary


def test_find_call_list_mapping_and_nested_session() -> None:
    mapped = find_call_list({"calls": {"a": _ok_call(), "b": _ok_call()}})
    assert len(mapped) == 2
    nested = find_call_list({"session": {"calls": [_ok_call()]}})
    assert len(nested) == 1
    assert find_call_list({}) == []


def test_finish_reason_from_responses_output() -> None:
    call = _ok_call()
    call["response"] = {
        "output": [{"type": "message", "finish_reason": "stop"}],
        "status": "completed",
    }
    call["raw_response"] = None
    captured = SessionTrace.from_payload("r", {"id": "r", "calls": [call]}).calls[0]
    assert captured.finish_reason in {"stop", "completed"}


def test_mid_stream_error_without_abort_is_status_error() -> None:
    call = _ok_call()
    call["http_status"] = 200
    call["mid_stream_error"] = "upstream reset"
    summary = SessionTrace.from_payload("m", {"id": "m", "calls": [call]}).trace_summary()
    assert "status=error" in summary
    assert "status=aborted" not in summary


def test_doctor_writes_to_stdout_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    assert doctor() == 0
    assert "binary --version: llmentary 0.4.0" in capsys.readouterr().out


def test_api_type_from_path_and_payload_corners() -> None:
    chat = _ok_call()
    chat.pop("api_family", None)
    chat.pop("api_type", None)
    chat["normalized"] = {}
    chat["path"] = "/v1/chat/completions"
    captured = SessionTrace.from_payload("p", {"id": "p", "calls": [chat]}).calls[0]
    assert captured.api_type == "chat_completions"

    responses = _ok_call()
    responses.pop("api_family", None)
    responses.pop("api_type", None)
    responses["normalized"] = {}
    responses["path"] = "/v1/responses"
    responses["request"] = {"input": "hi"}
    responses["response"] = {"output": []}
    captured = SessionTrace.from_payload("r", {"id": "r", "calls": [responses]}).calls[0]
    assert captured.api_type == "responses"


def test_http_status_bool_string_nested_and_zero() -> None:
    call = _ok_call()
    call["http_status"] = True
    call["response"] = {"status": 204}
    captured = SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0]
    assert captured.http_status == 204

    call = _ok_call()
    call["http_status"] = "200"
    captured = SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0]
    assert captured.http_status == 200

    call = _ok_call()
    call["http_status"] = "0"
    captured = SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0]
    assert captured.http_status is None

    call = _ok_call()
    call.pop("http_status", None)
    call["response"] = {"status": 0}
    captured = SessionTrace.from_payload("h", {"id": "h", "calls": [call]}).calls[0]
    assert captured.http_status is None


def test_retries_error_code_and_nested_tool_calls() -> None:
    call = _ok_call()
    call["retries"] = "nope"
    call["error"] = {"code": {"nested": True}}
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
    captured = SessionTrace.from_payload("t", {"id": "t", "calls": [call]}).calls[0]
    assert captured.retries == 0
    assert captured.error_code is None
    assert [tool.name for tool in captured.tool_calls] == ["lookup"]


def test_offered_tools_validation_and_filter_helpers() -> None:
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
    call["validation"] = {
        "mode": "live",
        "response_checked": False,
        "skipped_reason": "",
    }
    call["spec_violations"] = [{"message": "one"}]
    call["ignored_fields"] = [{"path": "/temperature"}]
    trace = SessionTrace.from_payload("f", {"id": "f", "calls": [call]})
    captured = trace.calls[0]
    assert captured.offered_tool_names() == ["direct", "nested"]
    assert captured.validation is not None
    assert captured.spec_status_label() == "1 violation"
    assert len(trace.spec_violations()) == 1
    assert len(trace.ignored_fields()) == 1
    assert trace.empty_match_message() == "no calls matched query"
    filtered = trace.without_tools()
    assert filtered.empty_match_message() == "no calls matched query (without_tools)"
    assert filtered.spec_violations() == []
    assert filtered.ignored_fields() == []

    skipped = _ok_call()
    skipped["spec"] = "skipped:no_schema"
    skipped_call = SessionTrace.from_payload("s", {"id": "s", "calls": [skipped]}).calls[0]
    assert skipped_call.spec_status_label() == "skipped:no_schema"


def test_model_matches_and_normalize_api_type_corners() -> None:
    assert model_matches(None, "gpt-4.1") is False
    assert model_matches("", "gpt-4.1") is False
    assert model_matches("gpt-4.1", "gpt-4.1") is True
    assert model_matches("gpt-4.1-mini", "gpt-4.1") is False
    assert model_matches("gpt-4.1-2025-01-01", "gpt-4.1") is True
    assert normalize_api_type("chat") == "chat_completions"
    assert normalize_api_type("something-else") == "something_else"


def test_finish_reason_falls_back_to_non_digit_status() -> None:
    call = _ok_call()
    call.pop("normalized", None)
    call["response"] = {"status": "completed"}
    call["raw_response"] = None
    captured = SessionTrace.from_payload("st", {"id": "st", "calls": [call]}).calls[0]
    assert captured.finish_reason == "completed"


def test_usage_from_normalized_response_and_path_from_request() -> None:
    call = _ok_call()
    call["usage"] = None
    call["normalized"] = {"response": {"usage": {"prompt_tokens": 3, "completion_tokens": 1}}}
    captured = SessionTrace.from_payload("u", {"id": "u", "calls": [call]}).calls[0]
    assert captured.usage["prompt_tokens"] == 3

    call = _ok_call()
    call.pop("path", None)
    call["request"] = {
        "path": "/v1/chat/completions",
        "messages": [{"role": "user", "content": "x"}],
    }
    captured = SessionTrace.from_payload("p", {"id": "p", "calls": [call]}).calls[0]
    assert captured.path == "/v1/chat/completions"


def test_parse_and_decode_helpers_cover_error_paths() -> None:
    not_a_map: Any = "not-a-map"
    assert first_present(not_a_map, "x") is None
    assert unwrap_error({"foo": 1}) == {}
    assert unwrap_error({"type": "invalid"}) == {"type": "invalid"}
    assert unwrap_error("not-json") == {}
    assert unwrap_error('{"error": {"message": "boom"}}') == {"message": "boom"}
    assert unwrap_error('{"type": "invalid_request_error"}') == {
        "type": "invalid_request_error"
    }
    assert coerce_mapping('{"ok": true}') == {"ok": True}
    assert extract_json_object("no object") is None
    assert extract_json_object("{not json}") is None
    assert decode_value(b"{", depth=6) == b"{"
    assert decode_value(b"{not-json}") == "{not-json}"
    assert decode_value(bytes([0xFF, 0xFE])) == bytes([0xFF, 0xFE])
    assert decode_value("   ") == "   "
    assert decode_base64_json("abcd") is None
    assert decode_base64_json("!!!!????") is None
    assert int_or_none(3.0) == 3
    assert int_or_none(3.5) is None
    assert optional_int({}, "n") is None
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
        {"input": ["plain", {"type": "input_text", "text": "typed"}, {"type": "function_call"}]}
    ) == ["plain", "typed"]
    assert extract_llm_side(
        {"normalized": {"request": {"messages": [{"role": "user", "content": "hi"}]}}},
        ("request",),
        side="request",
    )["messages"][0]["content"] == "hi"
    assert extract_llm_side(
        {"normalized": {"foo": 1}},
        ("request",),
        side="request",
    ) == {}
    info = session_info_from_mapping({"id": "s", "calls": [_ok_call()]})
    assert info.calls == 1
    assert model_matches("other", "gpt") is False
