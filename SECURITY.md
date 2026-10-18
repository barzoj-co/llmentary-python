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

## CI secrets and the e2e gate

Live OpenAI tests run in the separate `openai-e2e` job (`OpenAI E2E`). That job
is the only one that references `environment: openai-e2e` and the only one that
receives `OPENAI_API_KEY`. Only the `Real OpenAI E2E tests` step gets the key.
Unit and fake-upstream jobs do not use the environment, so they run without
approval. `LLMTESTLAB_REPO_TOKEN` stays the private-proxy checkout crutch it is
today: a boolean presence check plus the `actions/checkout` `token:` input (and
the release wheel-build step). Install and tooling steps run with no secrets.

`openai-e2e` is skipped for fork PRs. GitHub treats a skipped required check as
passing, so `e2e-gate` is the required check: it is green only when live e2e
succeeded. For a fork PR a maintainer reviews the diff, pushes it to a
same-repo branch, and live e2e runs there. Never use `pull_request_target`.

The `openai-e2e` environment requires maintainer approval on each run that
needs `OPENAI_API_KEY`. If that approval is rejected or times out, `e2e-gate`
goes red. That is intended.

On `pull_request`, GitHub runs the workflow and scripts from the PR's own merge
commit, so a PR that edits `ci.yml` or `scripts/e2e-gate.sh` can make the gate
pass. The gate catches the normal fork case (e2e skipped, nobody tampered with
CI). Tampering is caught by review of every `.github/**` and `scripts/**` diff.
Never "fix" this with `pull_request_target`.
