from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from llmtestlab.client import (
    LabSession,
    apply_upstream_api_key,
    build_start_command,
    default_control_url,
    encode_baggage_value,
    redact_command,
)


def test_build_start_command_matches_go_cli(tmp_path: Path) -> None:
    binary = tmp_path / "llmtestlab"
    command = build_start_command(
        bin_path=binary,
        host="127.0.0.1",
        port=7777,
        control_port=7778,
        upstream_url="http://127.0.0.1:9/v1",
        upstream_api_key="unused-by-flags",
        token="secret",
        env={},
    )
    assert command == [
        str(binary),
        "-listen",
        "127.0.0.1:7777",
        "-control",
        "127.0.0.1:7778",
        "-upstream",
        "openai=http://127.0.0.1:9",
        "-control-token",
        "secret",
    ]


def test_build_start_command_template(tmp_path: Path) -> None:
    binary = tmp_path / "llmtestlab"
    command = build_start_command(
        bin_path=binary,
        host="127.0.0.1",
        port=9,
        control_port=10,
        upstream_url="http://upstream/v1",
        upstream_api_key="",
        token=None,
        env={
            "LLMTESTLAB_START_COMMAND": (
                "{bin} -listen {listen} -control {control} -upstream {upstream}"
            )
        },
    )
    assert command == [
        str(binary),
        "-listen",
        "127.0.0.1:9",
        "-control",
        "127.0.0.1:10",
        "-upstream",
        "openai=http://upstream",
    ]


def test_redact_command_hides_control_token() -> None:
    assert redact_command(["llmtestlab", "-control-token", "super-secret"]) == [
        "llmtestlab",
        "-control-token",
        "***",
    ]
    assert redact_command(["llmtestlab", "-control-token=super-secret"]) == [
        "llmtestlab",
        "-control-token=***",
    ]


def test_apply_upstream_api_key_overwrites_existing_openai_key() -> None:
    env = {"OPENAI_API_KEY": "parent-key", "OTHER": "keep"}
    apply_upstream_api_key(env, "fixture-key")
    assert env["OPENAI_API_KEY"] == "fixture-key"
    assert env["LLMTESTLAB_UPSTREAM_API_KEY"] == "fixture-key"
    assert env["OTHER"] == "keep"


def test_apply_upstream_api_key_skips_missing() -> None:
    env = {"OPENAI_API_KEY": "parent-key"}
    apply_upstream_api_key(env, None)
    assert env == {"OPENAI_API_KEY": "parent-key"}


def test_baggage_percent_encodes_reserved_characters() -> None:
    session_id = "test_foo, bar=baz"
    session = LabSession(session_id=session_id, proxy=Mock())
    assert session.headers["X-Test-Session"] == session_id
    assert session.baggage == f"llmtestlab.session.id={encode_baggage_value(session_id)}"
    assert "," not in session.baggage.split("=", 1)[1]
    assert " " not in session.baggage
    assert session.baggage.endswith("test_foo%2C%20bar%3Dbaz")


def test_default_control_url_uses_go_control_port() -> None:
    assert (
        default_control_url("http://127.0.0.1:7777", {}) == "http://127.0.0.1:7778"
    )
    assert (
        default_control_url("http://127.0.0.1:7777", {"LLMTESTLAB_CONTROL_URL": "http://127.0.0.1:9"})
        == "http://127.0.0.1:9"
    )
