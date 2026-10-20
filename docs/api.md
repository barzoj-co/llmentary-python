# API reference

## Core objects

### `ControlClient`

```python
from llmentary import ControlClient

control = ControlClient("http://127.0.0.1:7778", token=None)
control.health()
sessions = control.list_sessions()
trace = control.wait_session("session-id", timeout=30, quiet=0.5, min_calls=2)
control.delete_session("session-id")
otlp_bytes = control.otlp("session-id")
version = control.version()
```

The Go proxy serves OpenAI-compatible traffic on `-listen` and the control API
on a separate `-control` address (default `127.0.0.1:7778`). The control client
uses:

- `GET /healthz`
- `GET /config` (includes `version` and the effective upstreams)
- `GET /sessions` and `GET /sessions?source=…`
- `GET /sessions/{id}`
- `GET /sessions/{id}/wait?timeout=30s&quiet=500ms&min_calls=2`
- fallback `POST /sessions/{id}/wait`
- `DELETE /sessions/{id}`
- `GET /sessions/{id}/otlp`

`GET /sessions/{id}` and `GET /sessions/{id}/otlp` raise `SessionNotFound` on
404. Expect helpers on a session with zero calls fail with
`no calls recorded for session <id>; known sessions: …`.

For non-loopback proxies, pass the bearer token as `token` or set
`LLMENTARY_CONTROL_TOKEN`.

### `Session`

The pytest `llmentary` fixture returns `llmentary.Session`.

Useful properties:

- `llmentary.session_id`
- `llmentary.base_url`: proxy URL
- `llmentary.openai_base_url`: proxy URL with `/v1`, suitable for `openai-python`
- `llmentary.headers`: `X-Test-Session` plus W3C `baggage` with `llmentary.session.id`
- `llmentary.baggage`: `llmentary.session.id=<session-id>`

Methods:

- `llmentary.wait(timeout=None, quiet=None, min_calls=None)`: wait for
  quiescence and return a trace. `quiet_for` remains a deprecated alias of
  `quiet`. Defaults are `timeout=30s` and `quiet=500ms`
  (`LLMENTARY_WAIT_TIMEOUT` / `LLMENTARY_WAIT_QUIET`). `min_calls` waits until
  that many finished calls exist. A 504 raises `WaitTimeoutError`.
- `llmentary.trace()`: fetch the current trace without waiting
- `llmentary.list_sessions()`: cheap session list from `GET /sessions`
- `llmentary.otlp()`: fetch OTLP bytes for the session
- `llmentary.cleanup()`: delete the session
- `llmentary.expect(soft=False)`: wait, then return an expectation object
- `llmentary.query(api_type=None, model=None)`: scope assertions by API type/model
- `llmentary.soft_expect()`: context manager that accumulates assertion failures

### `Query`

```python
llmentary.query(api_type="chat_completions", model="gpt-4.1-mini").expect().no_errors()
llmentary.query(api_type="responses").expect().tool_called("lookup")
```

Accepted API type spellings include `chat`, `chat_completions`, `chat.completions`,
`response`, `responses`, and the Go proxy's `api_family` values.

## Assertions

All assertion methods are chainable. Hard expectations raise `AssertionError` on
the first failure. Soft expectations collect failures and raise from
`assert_all()` or when leaving `llmentary.soft_expect()`.

### Tool calls

```python
expect.tool_called("get_weather")
expect.not_called("delete_user")
expect.called_times("search", 2)
expect.args_subset("get_weather", {"city": "Paris"})
expect.args_matcher("get_weather", lambda args: args["city"].startswith("Par"))
expect.order("search", "summarize")
expect.order("search", "summarize", strict=True)
expect.parallel_tool_calls()
expect.tool_result_fed_back("get_weather")
```

`args_subset` recursively checks only the expected keys. `args_matcher` accepts
either a mapping (same behavior as `args_subset`) or a callable returning bool.

### Request and response metadata

```python
expect.model("gpt-4.1-mini")
expect.request_param("temperature", 0)
expect.request_param("response_format.type", "json_schema")
expect.finish_reason("stop")
```

`request_param` supports dot notation for nested request JSON fields.

### Usage, latency, TTFT, retries

```python
expect.tokens(min_total=1, max_total=500)
expect.tokens(min_prompt=1, max_completion=100)
expect.latency(max_ms=2_000)
expect.ttft(max_ms=500)
expect.retries(max_count=1)
expect.no_retries()
```

The client accepts both Chat Completions token names
(`prompt_tokens`, `completion_tokens`, `total_tokens`) and Responses names
(`input_tokens`, `output_tokens`, `total_tokens`).

### Errors

```python
expect.no_errors()
expect.has_error("429")
expect.mid_stream_error("connection reset")
```

### Prompt and response content

```python
expect.prompt_contains("weather")
expect.prompt_regex(r"weather.+Paris")
expect.response_contains("sunny")
expect.response_regex(r"sunny|clear")
expect.response_json_valid()
expect.response_json_schema({"type": "object", "required": ["answer"]})
```

JSON assertions inspect model output text only (`output_text`, Chat
`choices[].message.content`, Responses `output[].content`), not the HTTP
response envelope.

### Spec validation

`validation` is metadata (was the call checked, and how). `spec_violations`
and `ignored_fields` are findings. `[]` means checked and clean; a missing key
means validation was not run.

| Field | Meaning |
| --- | --- |
| `call.validation` | Frozen `Validation` metadata, or `None` |
| `call.spec_violations` | `list[SpecViolation]` or `None` if not checked |
| `call.ignored_fields` | `list[IgnoredField]` or `None` if not checked |
| `trace.spec_violations()` | Flattened findings from checked calls |

```python
expect.no_spec_violations()
expect.no_spec_violations(side="request", ignore_paths=("/foo",), allow_unvalidated=False)
expect.no_ignored_fields()
```

`no_spec_violations` and `no_ignored_fields` are never vacuous: they fail with
`validation was not run` when nothing was checked, unless
`allow_unvalidated=True`. Finding failures use
`<call#> <api_family> <side> <path>: <message> — <hint>`.
`trace_summary()` adds `spec: clean | N violations | not checked` per call.
`no_spec_violations` also fails on `ignored_fields` with
`kind=unknown_request_field` (the live unknown nested message field): the
proxy records that on `ignored_fields`, not `spec_violations`.

## Trace payload handling

The client preserves the exact JSON returned by the Go proxy and normalizes the
common fields used for assertions:

- call list keys: `calls`, `requests`, `records`, `events`, `entries`
- API type keys and endpoint paths
- request/response `body` or `json`
- Chat Completions `choices[].message.tool_calls`
- Responses `output[]` function calls
- Chat `role=tool` results and Responses `function_call_output` results
- usage, latency, TTFT, retry, and error fields in the names used by the Go
  proxy and OpenAI APIs

Every assertion failure includes `SessionTrace.trace_summary()`, which lists each
captured call with API type, model, status, finish reason, timings, and tools.
