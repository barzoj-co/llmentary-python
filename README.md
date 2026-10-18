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

Point the client at an existing proxy:

```bash
export LLMTESTLAB_PROXY_URL=http://127.0.0.1:8080
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

Binary lookup order:

1. `LLMTESTLAB_BIN`
2. `llmtestlab` on `PATH`
3. cache directory (`LLMTESTLAB_CACHE_DIR` or the platform cache dir)
4. optional GitHub Releases download

Downloads are disabled by default because the Go proxy repository is private.
Enable them with `LLMTESTLAB_DOWNLOAD=1` or `--llmtestlab-download` and provide a
`GITHUB_TOKEN` that can read `barzoj-co/llm-test-lab`. Release assets are verified
against the GoReleaser checksum file before installation.

If the Go binary uses a different CLI flag shape, set
`LLMTESTLAB_START_COMMAND`, for example:

```bash
export LLMTESTLAB_START_COMMAND='{bin} --listen {listen} --upstream-url {upstream_url}'
```

Available template variables are `{bin}`, `{host}`, `{port}`, `{listen}`,
`{upstream_url}`, `{upstream_api_key}`, and `{token}`.

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

This repository's CI intentionally fails loudly until these are configured:

- `secrets.LLMTESTLAB_REPO_TOKEN`: token with read access to private repository
  `barzoj-co/llm-test-lab`, used to check out and build the real Go proxy.
- `secrets.OPENAI_API_KEY`: used by non-fork OpenAI E2E tests.
- PyPI Trusted Publishing: configure PyPI project `llm-test-lab` to trust this
  repository's `release.yml` workflow and `pypi` environment for tag `v*`
  releases. No PyPI API token is used.

## License

This Python client library is MIT licensed.

The `llm-test-lab` proxy binary that this client starts or talks to is a
separate process from `barzoj-co/llm-test-lab` and is licensed AGPL-3.0.
