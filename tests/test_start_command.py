from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from llmentary.client import (
    Session,
    allocate_listen_and_control_ports,
    apply_upstream_api_key,
    build_start_command,
    default_control_url,
    encode_baggage_value,
    free_ports,
    normalize_upstream_url,
    redact_command,
)


def test_build_start_command_matches_go_cli(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
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
        "openai=http://127.0.0.1:9/v1",
        "-control-token",
        "secret",
    ]


def test_build_start_command_omits_upstream_when_unset(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
    command = build_start_command(
        bin_path=binary,
        host="127.0.0.1",
        port=7777,
        control_port=7778,
        upstream_url=None,
        upstream_api_key=None,
        token=None,
        env={},
    )
    assert command == [
        str(binary),
        "-listen",
        "127.0.0.1:7777",
        "-control",
        "127.0.0.1:7778",
    ]
    assert "-upstream" not in command


def test_build_start_command_template(tmp_path: Path) -> None:
    binary = tmp_path / "llmentary"
    command = build_start_command(
        bin_path=binary,
        host="127.0.0.1",
        port=9,
        control_port=10,
        upstream_url="http://upstream/v1",
        upstream_api_key="",
        token=None,
        env={
            "LLMENTARY_START_COMMAND": (
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
        "openai=http://upstream/v1",
    ]


def test_normalize_upstream_url_keeps_v1_suffix() -> None:
    assert normalize_upstream_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert normalize_upstream_url("https://api.openai.com/v1/") == "https://api.openai.com/v1"
    assert normalize_upstream_url("https://api.openai.com") == "https://api.openai.com"
    assert (
        normalize_upstream_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"
    )


def test_redact_command_hides_control_token() -> None:
    assert redact_command(["llmentary", "-control-token", "super-secret"]) == [
        "llmentary",
        "-control-token",
        "***",
    ]
    assert redact_command(["llmentary", "-control-token=super-secret"]) == [
        "llmentary",
        "-control-token=***",
    ]


def test_apply_upstream_api_key_overwrites_existing_openai_key() -> None:
    env = {"OPENAI_API_KEY": "parent-key", "OTHER": "keep"}
    apply_upstream_api_key(env, "fixture-key")
    assert env["OPENAI_API_KEY"] == "fixture-key"
    assert env["LLMENTARY_UPSTREAM_API_KEY"] == "fixture-key"
    assert env["OTHER"] == "keep"


def test_apply_upstream_api_key_skips_missing() -> None:
    env = {"OPENAI_API_KEY": "parent-key"}
    apply_upstream_api_key(env, None)
    assert env == {"OPENAI_API_KEY": "parent-key"}


def test_baggage_percent_encodes_reserved_characters() -> None:
    session_id = "test_foo, bar=baz"
    session = Session(session_id=session_id, proxy=Mock())
    assert session.headers["X-Test-Session"] == session_id
    assert session.baggage == f"llmentary.session.id={encode_baggage_value(session_id)}"
    assert "," not in session.baggage.split("=", 1)[1]
    assert " " not in session.baggage
    assert session.baggage.endswith("test_foo%2C%20bar%3Dbaz")


def test_free_ports_are_distinct() -> None:
    first, second = free_ports("127.0.0.1", 2)
    assert first != second


def test_allocate_listen_and_control_ports_keeps_explicit_listen_off_control() -> None:
    listen, control = allocate_listen_and_control_ports(
        "127.0.0.1",
        {"LLMENTARY_PORT": "59999"},
    )
    assert listen == 59999
    assert control != listen


def test_default_control_url_uses_go_control_port() -> None:
    assert (
        default_control_url("http://127.0.0.1:7777", {}) == "http://127.0.0.1:7778"
    )
    assert (
        default_control_url("http://127.0.0.1:7777", {"LLMENTARY_CONTROL_URL": "http://127.0.0.1:9"})
        == "http://127.0.0.1:9"
    )
