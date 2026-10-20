# Proxy binary

There is no download-on-first-run. Platform wheels bundle a version-pinned Go
proxy binary taken from the `barzoj-co/llmentary` GoReleaser release named in
`PROXY_VERSION`. That pin is independent of this Python package version. The
bundled version is also written to the wheel `VERSION` file. Release packaging
verifies each archive against `checksums.txt` (sha256) before it is copied into
the wheel.

Lookup order:

1. `LLMENTARY_BIN` — explicit override, including CI builds from source
2. Binary bundled in the installed platform wheel
3. `llmentary` on `PATH`

Published wheels:

- `manylinux2014_x86_64`
- `manylinux2014_aarch64`
- `macosx_11_0_x86_64`
- `macosx_11_0_arm64`
- `win_amd64`

The client starts the proxy with the Go binary's flags: `-listen` for the
OpenAI-compatible proxy, `-control` for the control API, and
`-upstream name=https://host/base` (default name `openai`). Override the whole
command with `LLMENTARY_START_COMMAND` if needed:

```bash
export LLMENTARY_START_COMMAND='{bin} -listen {listen} -control {control} -upstream {upstream}'
```

Available template variables are `{bin}`, `{host}`, `{port}`, `{listen}`,
`{control}`, `{control_listen}`, `{upstream}`, `{upstream_url}`,
`{upstream_api_key}`, and `{token}`.
