# Security policy

## Supported versions

Only the latest published `0.x` minor of `llm-test-lab` on PyPI is supported
with security updates.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub Security Advisories:

https://github.com/barzoj-co/llm-test-lab-python/security/advisories/new

Do **not** open a public issue, pull request, or discussion for a security
report. Private vulnerability reporting becomes available once this repository
is public.

Please include:

- The `llm-test-lab` package version
- The bundled proxy version from the wheel `VERSION` file, or the
  `LLMTESTLAB_BIN` path you used
- A minimal reproduction without API keys or private prompts

This is a best-effort process. There is no SLA.

### Proxy issues

Issues in the recording proxy binary belong in
[`barzoj-co/llm-test-lab`](https://github.com/barzoj-co/llm-test-lab), not here.
Use that repository's private reporting channel when the issue is a
vulnerability.

## In scope

- Bundled-binary selection and execution (`LLMTESTLAB_BIN`, wheel `VERSION`,
  PATH lookup)
- Wheel contents (platform binaries, license files, metadata)
- Pytest plugin handling of environment variables and tokens
- The control-API client

## Out of scope

- The Go proxy implementation itself (report it upstream)
- Issues that require leaking `OPENAI_API_KEY`, `LLMTESTLAB_REPO_TOKEN`, or
  other secrets in public artifacts

## Verifying PyPI attestations

Published wheels and sdists are intended to carry PEP 740 attestations from
GitHub Actions trusted publishing. After the first release you can verify them
with:

```bash
pip install pypi-attestations
pypi-attestations verify pypi \
  --repository https://github.com/barzoj-co/llm-test-lab-python \
  llm_test_lab-<version>-py3-none-any.whl
```

or in the PyPI UI under **Verified details** on the project or file page.
