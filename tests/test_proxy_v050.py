from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

import pytest
from test_go_session_schema import go_call

from llmentary.types import (
    SessionTrace,
    proxy_config_from_mapping,
    validation_from_mapping,
)


def hostname_of(url: str) -> Optional[str]:
    """Return the URL hostname, or ``None`` when the value is not a URL."""

    if not isinstance(url, str) or not url.strip():
        return None
    return urlparse(url).hostname


def _ok_call() -> dict[str, Any]:
    return go_call(
        seq=1,
        api_family="chat.completions",
        request_body={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
        response_body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )


def _captured(call: dict[str, Any]) -> Any:
    return SessionTrace.from_payload("s", {"id": "s", "calls": [call]}).calls[0]


def test_validation_target_openai_and_compatible() -> None:
    openai = validation_from_mapping(
        {"mode": "warn", "profile": "relaxed", "target": "openai"}
    )
    assert openai.target == "openai"
    compatible = validation_from_mapping(
        {"mode": "warn", "target": "openai-compatible"}
    )
    assert compatible.target == "openai-compatible"


def test_validation_target_omitted_empty_and_invalid() -> None:
    assert validation_from_mapping({"mode": "warn"}).target is None
    assert validation_from_mapping({"target": ""}).target is None
    assert validation_from_mapping({"target": None}).target is None
    assert validation_from_mapping({"target": 1}).target is None
    assert validation_from_mapping({"target": True}).target is None
    assert validation_from_mapping({"target": ["openai"]}).target is None
    assert validation_from_mapping({"Target": "openai"}).target is None


def test_validation_target_on_captured_call_prefers_normalized() -> None:
    call = _ok_call()
    call["normalized"] = {
        "validation": {"mode": "warn", "target": "openai-compatible"}
    }
    captured = _captured(call)
    assert captured.validation is not None
    assert captured.validation.target == "openai-compatible"

    top_only = _ok_call()
    top_only["validation"] = {"mode": "warn", "target": "openai"}
    top_only["normalized"] = {}
    assert _captured(top_only).validation is not None
    assert _captured(top_only).validation.target == "openai"


def test_proxy_config_parses_v050_fields() -> None:
    cfg = proxy_config_from_mapping(
        {
            "version": "0.5.0",
            "upstreams": {"openai": "https://api.openai.com/v1"},
            "spec_validation": "warn",
            "spec_profile": "relaxed",
            "spec_sha": "abc",
            "schema_subset_retrieved": "2026-10-18",
            "spec_targets": {"openai": "openai", "groq": "openai-compatible"},
            "upstream_header_timeout": "0s",
            "upstream_idle_timeout": "2s",
        }
    )
    assert cfg.version == "0.5.0"
    assert cfg.upstreams == {"openai": "https://api.openai.com/v1"}
    assert cfg.spec_validation == "warn"
    assert cfg.spec_profile == "relaxed"
    assert cfg.spec_sha == "abc"
    assert cfg.schema_subset_retrieved == "2026-10-18"
    assert cfg.spec_targets == {"openai": "openai", "groq": "openai-compatible"}
    assert cfg.upstream_header_timeout == "0s"
    assert cfg.upstream_idle_timeout == "2s"


def test_proxy_config_omitted_keys_are_none_or_empty() -> None:
    cfg = proxy_config_from_mapping({"version": "0.4.1"})
    assert cfg.version == "0.4.1"
    assert cfg.upstreams == {}
    assert cfg.spec_targets is None
    assert cfg.upstream_header_timeout is None
    assert cfg.upstream_idle_timeout is None
    assert cfg.spec_validation is None


def test_proxy_config_empty_and_zero_timeouts() -> None:
    cfg = proxy_config_from_mapping(
        {
            "upstream_header_timeout": "",
            "upstream_idle_timeout": "0s",
            "spec_targets": {},
        }
    )
    assert cfg.upstream_header_timeout is None
    assert cfg.upstream_idle_timeout == "0s"
    assert cfg.spec_targets == {}


def test_proxy_config_invalid_types_are_dropped() -> None:
    cfg = proxy_config_from_mapping(
        {
            "version": 5,
            "upstreams": "openai",
            "spec_targets": ["openai"],
            "upstream_header_timeout": 2,
            "upstream_idle_timeout": False,
            "spec_validation": {"mode": "warn"},
        }
    )
    assert cfg.version is None
    assert cfg.upstreams == {}
    assert cfg.spec_targets == {}
    assert cfg.upstream_header_timeout is None
    assert cfg.upstream_idle_timeout is None
    assert cfg.spec_validation is None


def test_proxy_config_upstreams_object_and_list_shapes() -> None:
    from_objects = proxy_config_from_mapping(
        {
            "upstreams": {
                "openai": {"url": "https://api.openai.com/v1"},
                "groq": {"base_url": "https://api.groq.com/openai/v1"},
                "empty": {},
                "bad": 1,
            }
        }
    )
    assert from_objects.upstreams == {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
    }

    from_list = proxy_config_from_mapping(
        {
            "upstreams": [
                {"name": "openai", "url": "https://api.openai.com/v1"},
                {"name": "skip", "url": ""},
                {"url": "https://example.test"},
                "nope",
            ]
        }
    )
    assert from_list.upstreams == {"openai": "https://api.openai.com/v1"}


def test_proxy_config_camel_case_aliases() -> None:
    cfg = proxy_config_from_mapping(
        {
            "specTargets": {"local": "openai-compatible"},
            "upstreamHeaderTimeout": "1s",
            "upstreamIdleTimeout": "3s",
            "specValidation": "enforce",
            "specProfile": "strict",
            "schemaSubsetRetrieved": "2026-10-22",
        }
    )
    assert cfg.spec_targets == {"local": "openai-compatible"}
    assert cfg.upstream_header_timeout == "1s"
    assert cfg.upstream_idle_timeout == "3s"
    assert cfg.spec_validation == "enforce"
    assert cfg.spec_profile == "strict"
    assert cfg.schema_subset_retrieved == "2026-10-22"


def test_proxy_config_spec_targets_skip_non_string_values() -> None:
    cfg = proxy_config_from_mapping(
        {"spec_targets": {"openai": "openai", "bad": 1, "": "openai", 2: "openai"}}
    )
    assert cfg.spec_targets == {"openai": "openai"}


def test_error_normalized_timeout_categories() -> None:
    header = _ok_call()
    header["http_status"] = 504
    header["upstream_called"] = False
    header["error"] = {
        "type": "upstream_timeout",
        "code": "upstream_timeout",
        "message": "upstream sent no headers",
        "normalized": "upstream_timeout",
        "source": "proxy",
        "mid_stream": False,
    }
    captured = _captured(header)
    assert captured.http_status == 504
    assert captured.upstream_called is False
    assert captured.error_type == "upstream_timeout"
    assert captured.error_code == "upstream_timeout"
    assert captured.error_normalized == "upstream_timeout"
    assert captured.error_source == "proxy"
    assert captured.error_mapping["source"] == "proxy"

    idle = _ok_call()
    idle["http_status"] = 200
    idle["upstream_called"] = True
    idle["error"] = {
        "type": "upstream_idle_timeout",
        "code": "upstream_idle_timeout",
        "message": "upstream went idle",
        "normalized": "upstream_idle_timeout",
        "source": "proxy",
        "mid_stream": False,
    }
    idle["normalized"] = {"stream_truncated": True}
    captured = _captured(idle)
    assert captured.http_status == 200
    assert captured.upstream_called is True
    assert captured.error_normalized == "upstream_idle_timeout"
    assert captured.error_source == "proxy"
    assert captured.mid_stream_error is None


def test_error_source_omitted_empty_and_invalid() -> None:
    source_only = _ok_call()
    source_only["error"] = {"source": "proxy"}
    assert _captured(source_only).error_source == "proxy"

    missing = _ok_call()
    missing["error"] = {"normalized": "upstream_timeout"}
    assert _captured(missing).error_source is None

    empty = _ok_call()
    empty["error"] = {"normalized": "upstream_timeout", "source": ""}
    assert _captured(empty).error_source is None

    none = _ok_call()
    none["error"] = {"normalized": "upstream_timeout", "source": None}
    assert _captured(none).error_source is None

    bad = _ok_call()
    bad["error"] = {"normalized": "upstream_timeout", "source": ["proxy"]}
    assert _captured(bad).error_source is None

    camel = _ok_call()
    camel["error"] = {"normalized": "upstream_timeout", "Source": "proxy"}
    assert _captured(camel).error_source is None

    nested = _ok_call()
    nested["error"] = None
    nested["normalized"] = {
        "error": {
            "normalized": "upstream_timeout",
            "source": "proxy",
        }
    }
    assert _captured(nested).error_source == "proxy"
    assert _captured(nested).error_normalized == "upstream_timeout"

    response_source = _ok_call()
    response_source["error"] = None
    response_source["response"] = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "source": "body",
    }
    captured = _captured(response_source)
    assert captured.error_mapping == {}
    assert captured.error_source is None


def test_error_normalized_only_field_and_invalid_corners() -> None:
    only = _ok_call()
    only["error"] = {"normalized": "upstream_timeout"}
    assert _captured(only).error_normalized == "upstream_timeout"

    missing = _ok_call()
    assert _captured(missing).error_normalized is None

    empty = _ok_call()
    empty["error"] = {"normalized": ""}
    assert _captured(empty).error_normalized is None

    bad = _ok_call()
    bad["error"] = {"normalized": ["upstream_timeout"]}
    assert _captured(bad).error_normalized is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.openai.com/v1", "api.openai.com"),
        ("https://api.openai.com/v1/chat/completions", "api.openai.com"),
        ("http://api.openai.com", "api.openai.com"),
        ("https://API.OPENAI.COM/v1", "api.openai.com"),
        ("https://user:pass@api.openai.com/v1", "api.openai.com"),
        ("https://api.openai.com:443/v1", "api.openai.com"),
        ("https://evil.example/?q=api.openai.com", "evil.example"),
        ("https://evil.example/api.openai.com/v1", "evil.example"),
        ("https://api.openai.com.evil.example/v1", "api.openai.com.evil.example"),
        ("https://not-api.openai.com/v1", "not-api.openai.com"),
        ("https://prefix-api.openai.com", "prefix-api.openai.com"),
        ("api.openai.com/v1", None),
        ("not-a-url", None),
        ("", None),
        ("   ", None),
        ("/v1", None),
    ],
)
def test_hostname_of_uses_exact_host_not_substring(url: str, expected: Optional[str]) -> None:
    assert hostname_of(url) == expected
    assert (hostname_of(url) == "api.openai.com") is (expected == "api.openai.com")


def test_hostname_of_rejects_non_strings() -> None:
    assert hostname_of(None) is None  # type: ignore[arg-type]
    assert hostname_of(123) is None  # type: ignore[arg-type]
    assert hostname_of(["https://api.openai.com"]) is None  # type: ignore[arg-type]


@pytest.mark.integration
def test_proxy_config_typed_defaults_against_real_proxy() -> None:
    from test_proxy_dogfood import isolated_proxy

    with isolated_proxy() as handle:
        raw = handle.control.config()
        cfg = handle.control.proxy_config()
        assert cfg.version
        assert "openai" in cfg.upstreams
        assert hostname_of(cfg.upstreams["openai"]) == "api.openai.com"
        assert cfg.upstream_header_timeout == "0s", raw
        assert cfg.upstream_idle_timeout == "0s", raw
        assert cfg.spec_targets is not None, raw
        assert cfg.spec_targets.get("openai") == "openai", raw
        assert cfg.raw == raw


@pytest.mark.integration
def test_proxy_config_timeouts_and_spec_target_override() -> None:
    from test_proxy_dogfood import isolated_proxy

    with isolated_proxy(
        extra_env={
            "LLMENTARY_UPSTREAM_HEADER_TIMEOUT": "2s",
            "LLMENTARY_UPSTREAM_IDLE_TIMEOUT": "3s",
            "LLMENTARY_SPEC_TARGETS": "openai=openai-compatible",
        }
    ) as handle:
        cfg = handle.control.proxy_config()
        assert cfg.upstream_header_timeout == "2s", cfg.raw
        assert cfg.upstream_idle_timeout == "3s", cfg.raw
        assert cfg.spec_targets is not None, cfg.raw
        assert cfg.spec_targets.get("openai") == "openai-compatible", cfg.raw


@pytest.mark.integration
def test_named_upstream_skips_path_allow_list() -> None:
    from fake_upstream import FakeOpenAIServer
    from test_proxy_dogfood import _post_chat, isolated_proxy

    server = FakeOpenAIServer().start()
    try:
        with isolated_proxy(upstream_url=server.url, upstream_name="groq") as handle:
            extra = _post_chat(handle.url, "/u/groq/openai/v1/chat/completions")
            extra.raise_for_status()
            assert server.received_paths, "named upstream dropped the extra prefix"
            assert server.received_paths[-1].endswith("/chat/completions")
    finally:
        server.close()
