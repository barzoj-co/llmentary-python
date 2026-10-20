# llmentary

Python (pytest) client for llmentary: deterministic trace-based integration tests for LLM apps.

[![CI](https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/barzoj-co/llmentary-python/graph/badge.svg)](https://codecov.io/gh/barzoj-co/llmentary-python)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/barzoj-co/llmentary-python/badge)](https://scorecard.dev/viewer/?uri=github.com/barzoj-co/llmentary-python)
[![License: MIT](https://img.shields.io/github/license/barzoj-co/llmentary-python)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llmentary)](https://pypi.org/project/llmentary/)
[![Python versions](https://img.shields.io/pypi/pyversions/llmentary)](https://pypi.org/project/llmentary/)

`llmentary` is the Python/pytest client for the
[`barzoj-co/llmentary`](https://github.com/barzoj-co/llmentary) recording proxy.
It gives tests deterministic, trace-based assertions for OpenAI-compatible Chat
Completions and Responses calls. There is no LLM judge: assertions inspect the
requests, responses, tool calls, usage, timings, retries, and errors the proxy
recorded.

Published platform wheels bundle a version-pinned proxy binary; see
[docs/binaries.md](docs/binaries.md) for how it is resolved and how to override it.

## Install

```bash
pip install llmentary
```

Platform wheels bundle the version-pinned proxy binary. The sdist is client-only
and does not include a binary. Install a wheel with
`pip install --only-binary=:all: llmentary`, or set `LLMENTARY_BIN`, or put
`llmentary` on `PATH`. Run `python -m llmentary doctor` to see what this
install resolved.

Supported wheel tags: `manylinux2014_x86_64`, `manylinux2014_aarch64`,
`macosx_11_0_x86_64`, `macosx_11_0_arm64`, `win_amd64`.

For pytest fixture support:

```bash
pip install "llmentary[pytest]"
```

For the quickstart using `openai-python`:

```bash
pip install "llmentary[pytest,openai]"
```

## Quickstart

Point pytest at an existing proxy, or let the fixture start one:

```bash
export LLMENTARY_PROXY_URL=http://127.0.0.1:7777
export LLMENTARY_CONTROL_URL=http://127.0.0.1:7778
pytest
```

```bash
export LLMENTARY_BIN=/path/to/llmentary
export LLMENTARY_UPSTREAM_BASE_URL=https://api.openai.com/v1
export LLMENTARY_UPSTREAM_API_KEY="$OPENAI_API_KEY"
pytest
```

In-process tests that can set headers:

```python
from openai import OpenAI


def test_tool_flow(llmentary):
    client = OpenAI(
        api_key="test-key",
        base_url=llmentary.openai_base_url,
        default_headers=llmentary.headers,
    )

    first = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )
    tool_call = first.choices[0].message.tool_calls[0]

    client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            first.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": '{"weather":"sunny"}'},
        ],
    )

    llmentary.expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").no_errors()
```

Black-box / import-time clients that cannot send headers: set the service base
URL to `llmentary.path_base_url()` **before** the app is imported.

```python
import os

from openai import OpenAI


def test_import_time_client(llmentary):
    os.environ["OPENAI_BASE_URL"] = llmentary.path_base_url()
    client = OpenAI()
    client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "ping"}],
    )
    llmentary.expect().prompt_contains("ping").no_errors()
```

See the proxy
[black-box guide](https://github.com/barzoj-co/llmentary/blob/main/docs/black-box.md)
and
[assertions guide](https://github.com/barzoj-co/llmentary/blob/main/docs/assertions-guide.md).

The function-scoped `llmentary` fixture returns a `Session`: it creates a unique
session id, exposes headers with `X-Test-Session`, waits for proxy quiescence
before assertions, and deletes the session during teardown.

## Waiting for calls

`wait` asks the proxy to return the session trace once the session has been
idle for `quiet`, or to fail when `timeout` elapses. The proxy only reads these
query parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `timeout` | `30s` | Maximum time to wait for the session to become quiet |
| `quiet` | `500ms` | How long the session must stay idle before wait returns |
| `min_calls` | `0` | Wait until this many finished calls exist, then for `quiet` |

Override the defaults with environment variables. Each accepts a Go duration
(`45s`, `200ms`) or a bare number of seconds (`45`, `0.2`):

| Environment | Default |
| --- | --- |
| `LLMENTARY_WAIT_TIMEOUT` | `30s` |
| `LLMENTARY_WAIT_QUIET` | `500ms` |

The client HTTP deadline is `timeout + quiet + 5s`, so the client cannot abort
before the proxy answers. A timeout is an error: `wait timed out after 30s on
session X (k calls so far, still active: n). Raise LLMENTARY_WAIT_TIMEOUT …`.

## Assertions

```python
llmentary.expect().tool_called("search").called_times("search", 1)
llmentary.query(api_type="responses", model="gpt-4.1-mini").expect().no_errors()
llmentary.query().without_tools().expect().request_param("response_format.type", "json_schema")
mark = llmentary.mark()
llmentary.expect(since=mark).final_response_mentions_any("20", "sunny")

with llmentary.soft_expect() as expect:
    expect.tool_called("search")
    expect.response_contains("done")
```

Supported deterministic assertions include tool calls (`tool_called`,
`not_called`, `called_times`, `args_subset`, `args_matcher`, `order`,
`parallel_tool_calls`, `tool_result_fed_back`), request and finish metadata
(`model`, `request_param`, `finish_reason`), usage and timing (`tokens`,
`latency`, `ttft`, `retries`, `no_retries`), errors (`no_errors`, `has_error`,
`mid_stream_error`), prompt or response text (`prompt_contains`,
`prompt_regex`, `response_contains`, `response_regex`, `response_json_valid`,
`response_json_schema`), and spec observability (`no_spec_violations`,
`no_ignored_fields`). Failures raise `AssertionError` with the session trace
summary. Use `llmentary.list_sessions()` when a wait or expect fails with
zero calls, so the message can name the known session ids.

See [`docs/api.md`](docs/api.md), [`docs/pytest.md`](docs/pytest.md), and the
proxy [assertions guide](https://github.com/barzoj-co/llmentary/blob/main/docs/assertions-guide.md).

<!-- ltl-repos:start -->
## llmentary repositories

| Repository | What |
|---|---|
| [llmentary](https://github.com/barzoj-co/llmentary) | Recording proxy binary (Go) |
| [llmentary-go](https://github.com/barzoj-co/llmentary-go) | Go client for `go test` |
| [llmentary-python](https://github.com/barzoj-co/llmentary-python) | Python client and pytest plugin |
| [llmentary-js](https://github.com/barzoj-co/llmentary-js) | TypeScript/JavaScript client for Vitest and Jest |
<!-- ltl-repos:end -->

## License

Licensed under MIT, see LICENSE
