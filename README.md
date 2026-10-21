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
`llmentary.__version__` is the Python package; `llmentary.proxy_version()` is
the semver of the resolved proxy binary (for example `0.4.0`, or `None` when
no binary resolves); `proxy_version_line()` is the full `llmentary --version`
line.
It gives tests deterministic, trace-based assertions for OpenAI-compatible Chat
Completions and Responses calls. There is no LLM judge: assertions inspect the
requests, responses, tool calls, usage, timings, retries, and errors the proxy
recorded. The control API contract (`GET {control}/openapi.yaml`) describes `/healthz`,
`/sessions`, and `/sessions/{id}/wait`. Platform wheels bundle the same
`openapi.yaml` next to the proxy binary. `ControlClient.openapi()` returns the
served document; `contract_path()` is the bundled file.

Published platform wheels bundle proxy v0.4.0; see
[docs/binaries.md](docs/binaries.md) for how it is resolved and how to override it.

## Install

```bash
pip install llmentary
```

Platform wheels bundle the version-pinned proxy binary (`PROXY_VERSION`,
currently 0.4.0). The sdist is client-only
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

    llmentary.expect().call_count(2).tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").no_errors()
```

Black-box / import-time clients that cannot send headers: set the service base
URL to `llmentary.path_base_url()` **inside the test**, then import the app.
The session id exists only after the fixture runs, so a module-level import is
too early.

```python
import importlib


def test_import_time_client(llmentary, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", llmentary.path_base_url())
    app = importlib.import_module("myapp")
    app.client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "ping"}],
    )
    llmentary.expect().prompt_contains("ping").no_errors()
```

A path prefix isolates **client processes**, not conversations. One process
with one env-configured SDK client is one session. For N parallel
conversations use one of: (a) one worker process per session; (b) construct
the SDK client per conversation with `base_url=lm.path_base_url()`; (c)
headers/baggage when the app can set them; (d) run conversations sequentially
with `DELETE` between them. A second bus worker needs its own topic or
partition; two consumer groups on one topic both get every event.

Prove isolation with `isolated_from`. The rule is the isolation spec: same
canned user question in two sessions is not a leak; a foreign user turn
replayed as history is. Later calls introduce each new user-role message.
An empty or unknown other session fails unless you pass `allow_empty`.
Comparison is exact (normalized), never substring.

```python
other.expect().isolated_from(llmentary)
llmentary.expect().isolated_from(other)
```

See the proxy
[isolation spec](https://github.com/barzoj-co/llmentary/blob/main/docs/isolation-spec.md)
(DOG-29 / DOG-32 / DOG-33),
[black-box guide](https://github.com/barzoj-co/llmentary/blob/main/docs/black-box.md),
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

`wait` on a session that was never created (or was `DELETE`d):

| `min_calls` | Control status | Client exception |
| --- | --- | --- |
| `0` | `404 session_not_found` | `SessionNotFound` |
| `>=1` | `504 wait_timeout` after `timeout` | `WaitTimeoutError` |

```python
from llmentary import SessionNotFound, WaitTimeoutError

try:
    llmentary.wait(timeout=3, quiet=0.2, min_calls=0)
except SessionNotFound:
    pass

try:
    llmentary.wait(timeout=3, quiet=0.2, min_calls=1)
except WaitTimeoutError:
    pass
```

`wait()` still raises `SessionNotFound` on that 404. `expect()` without
`allow_empty=True` turns it into `AssertionError('no calls recorded for
session …; known sessions: …')` — it does not return an empty expectation
you can chain. Use `expect_no_calls()` or `expect(allow_empty=True)` for
the negative test. A never-created session that will be filled by an
async worker should wait with `min_calls=1` (or higher), not `0`.

## Assertions

```python
llmentary.expect().call_count(1).tool_called("search").called_times("search", 1)
llmentary.expect_no_calls()
llmentary.expect(allow_empty=True).call_count(0)
llmentary.query(api_type="responses", model="gpt-4.1-mini").expect().no_errors()
llmentary.query().without_tools().expect().call_count(1).request_param(
    "response_format.type", "json_schema"
)
mark = llmentary.mark()
llmentary.expect(since=mark).final_response_mentions_any("20", "sunny")

with llmentary.soft_expect() as expect:
    expect.tool_called("search")
    expect.response_contains("done")
```

Supported deterministic assertions include call count (`call_count`), tool
calls (`tool_called`, `not_called`, `called_times`, `args_subset`,
`args_matcher`, `order`, `parallel_tool_calls`, `tool_result_fed_back`),
request and finish metadata (`model`, `request_param`, `finish_reason`), usage
and timing (`tokens`, `latency`, `ttft`, `retries`, `no_retries`), errors
(`no_errors`, `has_error`, `mid_stream_error`), prompt or response text
(`prompt_contains`, `prompt_regex`, `response_contains`, `response_regex`,
`response_json_valid`, `response_json_schema`), and spec observability
(`no_spec_violations`, `no_ignored_fields`). Failures raise `AssertionError`
with the session trace summary. A filter or `since=` that matches zero calls
fails `no_errors()` / `not_called()` unless you call `allow_empty()` first.
`expect()` itself still fails on a never-created session unless you pass
`allow_empty=True`. Use `llmentary.list_sessions()` when a wait or expect
fails with zero calls, so the message can name the known session ids (once).

Negative tests — asserting zero LLM calls:

```python
# reading history / a GET that must not hit the model
unused.expect_no_calls(quiet=0.2)
unused.expect(allow_empty=True).call_count(0)
active.expect().allow_empty().isolated_from(unused)
```

Testing interruptions / barge-in: cancel a streaming call (~50 ms after
send, or after the first chunk) and read the typed abort fields:

```python
stream.close()
expectation = llmentary.expect().aborted()
call = next(call for call in expectation.trace.calls if call.client_aborted)
assert call.client_aborted
assert call.upstream_called is True
assert call.abort_phase in (None, "before_headers", "after_headers")
assert call.finish_reason in {"cancelled", "unknown"}
assert call.http_status != 502
assert "status=aborted" in expectation.trace.trace_summary()
```

`client_aborted` prefers the top-level Call field and falls back to
`normalized.client_aborted` on older proxies. `abort_phase` is
`before_headers` or `after_headers` when the call was aborted, and
omitted (`None`) otherwise. `upstream_called` is the top-level Call
field (`True`/`False`), or `None` on older proxies. Closing after
`[DONE]` or a terminal Responses event is not an abort. Proxy 0.3.0 still
records `finish_reason=unknown` on abort; `cancelled` is the documented
value.

Trace-summary `status=` is `ok`, `error`, or `aborted`. Both abort
phases print `status=aborted` (a scan for `status=error` must not miss
an after-headers cancel that already has `http=200`). `http=` is
included only when an upstream status was recorded (after-headers,
typically 200; omitted for before-headers). `stream_truncated` is
printed only when the proxy recorded a truncated stream (after-headers).
`phase=` is printed when `abort_phase` is known.

Call `id` values (`call_3`) are unique within one proxy process and can
interleave across sessions. `CapturedCall.seq` is 1-based per session
(`sequence_number()` is an alias). The trace summary prints `#seq (id)`.

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
