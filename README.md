# llm-test-lab Python

`llm-test-lab` is the Python/pytest client for the
[`barzoj-co/llm-test-lab`](https://github.com/barzoj-co/llm-test-lab) proxy.
It gives Python tests deterministic, trace-based assertions for OpenAI-compatible
Chat Completions and Responses calls. There is no LLM judge: assertions inspect
the requests, responses, tool calls, usage, timings, retries, and errors recorded
by the proxy.

## Install

```bash
pip install llm-test-lab
```

For pytest fixture support:

```bash
pip install "llm-test-lab[pytest]"
```

For the quickstart using `openai-python`:

```bash
pip install "llm-test-lab[pytest,openai]"
```

## Quickstart

Point the client at an existing proxy. The Go control API listens on a
separate address (default `127.0.0.1:7778`):

```bash
export LLMTESTLAB_PROXY_URL=http://127.0.0.1:7777
export LLMTESTLAB_CONTROL_URL=http://127.0.0.1:7778
pytest
```

or let the pytest fixture start the proxy:

```bash
export LLMTESTLAB_BIN=/path/to/llmtestlab
export LLMTESTLAB_UPSTREAM_BASE_URL=https://api.openai.com/v1
export LLMTESTLAB_UPSTREAM_API_KEY="$OPENAI_API_KEY"
pytest
```

```python
from openai import OpenAI


def test_tool_flow(lab):
    client = OpenAI(
        api_key="test-key",
        base_url=lab.openai_base_url,
        default_headers=lab.headers,
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

    lab.expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").no_errors()
```

The function-scoped `lab` fixture creates a unique session id, exposes headers
with `X-Test-Session`, waits for proxy quiescence before assertions, and deletes
the session during teardown.

## Binary management

There is no download-on-first-run. Platform wheels bundle a version-pinned Go
proxy binary taken from the matching `barzoj-co/llm-test-lab` GoReleaser
release. The Python package version and the bundled binary version are the same.
Release packaging verifies each archive against `checksums.txt` (sha256) before
it is copied into the wheel.

Lookup order:

1. `LLMTESTLAB_BIN` — explicit override, including CI builds from source
2. Binary bundled in the installed platform wheel
3. `llmtestlab` on `PATH`

Published wheels:

- `manylinux2014_x86_64`
- `manylinux2014_aarch64`
- `macosx_11_0_x86_64`
- `macosx_11_0_arm64`
- `win_amd64`

The bundled proxy is a separate process. This Python client is MIT; the proxy
binary remains AGPL-3.0.

The client starts the proxy with the Go binary's flags: `-listen` for the
OpenAI-compatible proxy, `-control` for the control API, and
`-upstream name=https://host/base` (default name `openai`). Override the whole
command with `LLMTESTLAB_START_COMMAND` if needed:

```bash
export LLMTESTLAB_START_COMMAND='{bin} -listen {listen} -control {control} -upstream {upstream}'
```

Available template variables are `{bin}`, `{host}`, `{port}`, `{listen}`,
`{control}`, `{control_listen}`, `{upstream}`, `{upstream_url}`,
`{upstream_api_key}`, and `{token}`.

## Assertion API

```python
lab.expect().tool_called("search").called_times("search", 1)
lab.query(api_type="responses", model="gpt-4.1-mini").expect().no_errors()

with lab.soft_expect() as expect:
    expect.tool_called("search")
    expect.response_contains("done")
```

Supported deterministic assertions include:

- tool calls: `tool_called`, `not_called`, `called_times`, `args_subset`,
  `args_matcher`, `order`, `parallel_tool_calls`, `tool_result_fed_back`
- request/model/finish: `model`, `request_param`, `finish_reason`
- usage/timing/retries: `tokens`, `latency`, `ttft`, `retries`, `no_retries`
- errors: `no_errors`, `has_error`, `mid_stream_error`
- prompt/response: `prompt_contains`, `prompt_regex`, `response_contains`,
  `response_regex`, `response_json_valid`, `response_json_schema`

Failures raise `AssertionError` with the session trace summary from the captured
calls.

See [`docs/api.md`](docs/api.md) and [`docs/pytest.md`](docs/pytest.md).

## CI configuration required

- `secrets.LLMTESTLAB_REPO_TOKEN`: read-only Contents token for private
  repository `barzoj-co/llm-test-lab`. Used only to check out and build the real
  Go proxy in CI, and to fetch GoReleaser assets when building release wheels.
  TODO: remove once llm-test-lab is public.
- `secrets.OPENAI_API_KEY`: used by OpenAI E2E tests on push and non-fork PRs.
  Those tests actually run (they do not skip) when this secret is present.
- PyPI Trusted Publishing: configure PyPI project `llm-test-lab` to trust this
  repository's `release.yml` workflow and `pypi` environment for tag `v*`
  releases. No PyPI API token is used. The workflow is prepared for Trusted
  Publishing but does not publish yet.

CI builds the real Go proxy from source (`go build ./cmd/llmtestlab`) and points
tests at it with `LLMTESTLAB_BIN`. Integration tests talk to a fake OpenAI
upstream through that real binary. E2E tests talk to the real OpenAI API
through the same real binary.

## License

This Python client library is MIT licensed. See [LICENSE](LICENSE).

The `llm-test-lab` proxy binary bundled in platform wheels, started by this
client, or already running as a separate process, comes from
`barzoj-co/llm-test-lab` and is licensed AGPL-3.0. The client and the proxy are
separate processes; distributing or modifying the proxy is subject to AGPL.
