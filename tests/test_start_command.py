from __future__ import annotations

from pathlib import Path

from llmtestlab.client import build_start_command, default_control_url, redact_command


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


def test_default_control_url_uses_go_control_port() -> None:
    assert (
        default_control_url("http://127.0.0.1:7777", {}) == "http://127.0.0.1:7778"
    )
    assert (
        default_control_url("http://127.0.0.1:7777", {"LLMTESTLAB_CONTROL_URL": "http://127.0.0.1:9"})
        == "http://127.0.0.1:9"
    )
