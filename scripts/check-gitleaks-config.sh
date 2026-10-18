#!/usr/bin/env bash
# Prove the committed gitleaks config allowlists only the synthetic strings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/.gitleaks.toml"

require_gitleaks() {
  if ! command -v gitleaks >/dev/null 2>&1; then
    echo "gitleaks is required on PATH" >&2
    exit 2
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing ${CONFIG}" >&2
    exit 1
  fi
}

random_alnum() {
  local n="$1"
  python3 -c "import secrets,string; print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(${n})))"
}

self_test() {
  require_gitleaks
  local tmp output rc
  tmp="$(mktemp -d)"
  trap "rm -rf '$(printf '%s' "${tmp}")'" EXIT

  mkdir -p "${tmp}/clean"
  printf 'placeholder sk-...\n' >"${tmp}/clean/docs.txt"
  printf 'sk-invalid-key-for-401-test\n' >"${tmp}/clean/invalid.txt"

  set +e
  output="$(gitleaks dir --config "${CONFIG}" --redact --no-banner --exit-code 1 "${tmp}/clean" 2>&1)"
  rc=$?
  set -e
  echo "case clean-placeholders: rc=${rc}"
  echo "${output}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "self-test: placeholders were reported as leaks" >&2
    exit 1
  fi

  mkdir -p "${tmp}/leaky/tests/e2e" "${tmp}/leaky/src/llmtestlab"
  local openai_key github_pat openai_marker
  openai_marker="$(python3 -c 'import base64; print(base64.b64encode(b"OpenAI").decode())')"
  openai_key="sk-$(random_alnum 20)${openai_marker}$(random_alnum 20)"
  github_pat="ghp_$(random_alnum 36)"
  printf '%s\n' "${openai_key}" >"${tmp}/leaky/tests/e2e/not-really-allowlisted.py"
  printf '%s\n' "${github_pat}" >"${tmp}/leaky/src/llmtestlab/not-really-allowlisted.py"
  unset openai_key github_pat openai_marker

  set +e
  output="$(gitleaks dir --config "${CONFIG}" --redact --no-banner --exit-code 1 \
    --report-format json --report-path "${tmp}/leaky-report.json" "${tmp}/leaky" 2>&1)"
  rc=$?
  set -e
  echo "case realistic-keys-under-allowlisted-paths: rc=${rc}"
  echo "${output}"
  if [[ "${rc}" -eq 0 ]]; then
    echo "self-test: realistic keys under tests/e2e/ and src/llmtestlab/ were not flagged" >&2
    exit 1
  fi
  if ! grep -F -q "tests/e2e/not-really-allowlisted.py" "${tmp}/leaky-report.json"; then
    echo "self-test: OpenAI-shaped key under tests/e2e/ was not flagged" >&2
    exit 1
  fi
  if ! grep -F -q "src/llmtestlab/not-really-allowlisted.py" "${tmp}/leaky-report.json"; then
    echo "self-test: GitHub PAT under src/llmtestlab/ was not flagged" >&2
    exit 1
  fi
  echo "self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

echo "usage: $0 --self-test" >&2
exit 2
