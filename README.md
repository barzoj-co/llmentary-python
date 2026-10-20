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

The function-scoped `llmentary` fixture returns a `Session`: it creates a unique
session id, exposes headers with `X-Test-Session`, waits for proxy quiescence
before assertions, and deletes the session during teardown.

## Assertions

```python
llmentary.expect().tool_called("search").called_times("search", 1)
llmentary.query(api_type="responses", model="gpt-4.1-mini").expect().no_errors()

with llmentary.soft_expect() as expect:
    expect.tool_called("search")
    expect.response_contains("done")
```

Supported deterministic assertions include tool calls (`tool_called`,
`not_called`, `called_times`, `args_subset`, `args_matcher`, `order`,
`parallel_tool_calls`, `tool_result_fed_back`), request and finish metadata
(`model`, `request_param`, `finish_reason`), usage and timing (`tokens`,
`latency`, `ttft`, `retries`, `no_retries`), errors (`no_errors`, `has_error`,
`mid_stream_error`), and prompt or response text (`prompt_contains`,
`prompt_regex`, `response_contains`, `response_regex`, `response_json_valid`,
`response_json_schema`). Failures raise `AssertionError` with the session trace
summary.

See [`docs/api.md`](docs/api.md) and [`docs/pytest.md`](docs/pytest.md).

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
