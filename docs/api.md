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
- `llmentary.path_base_url(upstream=None)`: `{proxy}/s/{id}/v1`, or
  `{proxy}/s/{id}/u/{name}` for a named upstream. Use this when the service
  cannot send session headers.
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
- `llmentary.mark()`: current max call sequence number (`0` if none)
- `llmentary.expect(soft=False, since=None)`: wait, then return an expectation
  object scoped to calls after `since` when that mark is set. Call
  ``allow_empty()`` before ``no_errors`` / ``not_called`` to opt out of the
  empty-filter guard (Go ``AllowEmpty()``)
- `llmentary.query(api_type=None, model=None)`: scope assertions by API type/model
- `llmentary.soft_expect()`: context manager that accumulates assertion failures

### `Query`

```python
llmentary.query(api_type="chat_completions", model="gpt-4.1-mini").expect().no_errors()
llmentary.query(api_type="responses").expect().tool_called("lookup")
llmentary.query().offers_tool("get_weather").expect().tool_called("get_weather")
llmentary.query().without_tools().expect().request_param("response_format.type", "json_schema")
mark = llmentary.mark()
llmentary.query().since(mark).expect().prompt_contains("turn 2")
llmentary.expect().call_count(1)
router = llmentary.query().without_tools().expect()
router.call_count(1).no_errors()
```

Accepted API type spellings include `chat`, `chat_completions`, `chat.completions`,
`response`, `responses`, and the Go proxy's `api_family` values.

`offers_tool(name)` keeps calls that offered that tool. `without_tools()` keeps
calls that offered none (router-style structured output). `since(mark)` keeps
calls whose sequence number is greater than `mark`. The same filters exist on
`SessionTrace.filtered(...)` and as chainable `SessionTrace` methods.
`trace_summary()` prints those sequence numbers.

### Filters and empty traces

`Session.expect()` still fails when the **whole session** has zero calls. After a
filter or `since=` cuts the trace to zero matching calls, `no_errors()` and
`not_called()` also fail, with `no calls matched query (without_tools, since=3)`
plus the trace summary. That stops a missed turn or a bypassed proxy from
looking green.

Opt out only when the test really expects zero matching calls:

```python
llmentary.expect(since=mark).allow_empty().no_errors()
```

Use `call_count(n)` or `call_count(min=1, max=4)` to state how many calls the
(filtered) trace should have.

## Assertions

All assertion methods are chainable. Hard expectations raise `AssertionError` on
the first failure. Soft expectations collect failures and raise from
`assert_all()` or when leaving `llmentary.soft_expect()`.

### Tool calls

```python
expect.call_count(2)
expect.call_count(min=1, max=4)
expect.tool_called("get_weather")
expect.not_called("delete_user")
expect.called_times("search", 2)
expect.args_subset("get_weather", {"city": "Paris"})
expect.args_matcher("get_weather", lambda args: args["city"].startswith("Par"))
expect.order("search", "summarize")
expect.order("search", "summarize", strict=True)
expect.parallel_tool_calls()
expect.tool_result_fed_back("get_weather")
expect.tool_result_in_prompt("get_weather")
```

`tool_result_in_prompt` requires a later request to include a non-empty tool
result that matches `name` by tool-call id or tool name. `tool_result_fed_back`
only checks that a later request references the tool call id.

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
expect.final_response_mentions_any("20", "sunny")
```

`final_response_mentions_any` runs `answer_mentions_any` on the last call's
model output text only (same fields as the JSON assertions), not the response
envelope. Matching is case-insensitive; `20°C` / `20 °C` / `20 degrees` match
when `normalize_numbers` is on. Use the helper on app-level answers that are
not on the trace:

```python
from llmentary import answer_mentions_any

answer_mentions_any(app_text, "20", "sunny")
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

`CapturedCall.session_source` is `header`, `baggage`, `path`, or `default` when
the proxy recorded attribution. `CapturedCall.upstream_url` is the forwarded
upstream URL (string or `upstream.url`); it should not contain `/s/{session}`.

```python
call = llmentary.expect().trace.calls[0]
assert call.session_source == "path"
assert call.upstream_url is not None and "/s/" not in call.upstream_url
```

Every hard assertion failure includes `SessionTrace.trace_summary()` once.
`SoftExpectation.assert_all()` prints each failure message, then the summary
once at the end. The summary lists each captured call with its sequence number
(the same numbers `mark` / `since` use), API type, model, status, source,
tokens, finish reason (omitted when absent or `unknown`), timings, and tools.
Unparsed records (no request and no response) print `unparsed call, keys=[…]`.
Passing pytest tests that used the `llmentary` fixture also print that summary.

```python
from llmentary import proxy_version

print(proxy_version())  # first line of `llmentary --version`, or "not found"
```
