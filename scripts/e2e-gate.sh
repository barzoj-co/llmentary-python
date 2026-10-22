#!/usr/bin/env bash
# Fail closed unless the live e2e job succeeded. Fork PRs that skip live e2e
# (no secrets) pass with a clear skipped result. Same script in every client repo.
set -euo pipefail

FORK_SKIP_MSG="skipped (fork, no secrets)"

run_gate() {
  local result="${E2E_RESULT:-}"
  local event="${EVENT_NAME:-}"
  local is_fork="${IS_FORK:-false}"

  if [[ "${result}" == "success" ]]; then
    echo "live e2e succeeded"
    return 0
  fi

  if [[ "${is_fork}" == "true" && "${result}" == "skipped" ]]; then
    echo "${FORK_SKIP_MSG}"
    return 0
  fi

  if [[ "${result}" == "skipped" ]]; then
    echo "::error::live e2e did not run (event=${event}, result=skipped)"
    return 1
  fi

  echo "::error::live e2e did not succeed (result=${result})"
  return 1
}

self_test() {
  local failed=0
  local name result event fork want_rc want_msg output rc

  run_case() {
    name="$1"
    result="$2"
    event="$3"
    fork="$4"
    want_rc="$5"
    want_msg="$6"
    set +e
    output="$(E2E_RESULT="${result}" EVENT_NAME="${event}" IS_FORK="${fork}" bash "${BASH_SOURCE[0]}" 2>&1)"
    rc=$?
    set -e
    echo "case ${name}: result=${result} event=${event} fork=${fork} rc=${rc} want_rc=${want_rc}"
    echo "${output}"
    if [[ "${rc}" -ne "${want_rc}" ]]; then
      echo "self-test: ${name}: expected exit ${want_rc}, got ${rc}" >&2
      failed=1
    fi
    if [[ -n "${want_msg}" ]] && ! grep -F -q "${want_msg}" <<<"${output}"; then
      echo "self-test: ${name}: expected message: ${want_msg}" >&2
      failed=1
    fi
  }

  run_case "success" "success" "pull_request" "false" 0 "live e2e succeeded"
  run_case "fork-success" "success" "pull_request" "true" 0 "live e2e succeeded"
  run_case "fork-skipped" "skipped" "pull_request" "true" 0 "${FORK_SKIP_MSG}"
  run_case "same-repo-skipped" "skipped" "pull_request" "false" 1 "live e2e did not run (event=pull_request, result=skipped)"
  run_case "dispatch-skipped" "skipped" "workflow_dispatch" "false" 1 "live e2e did not run (event=workflow_dispatch, result=skipped)"
  run_case "fork-failure" "failure" "pull_request" "true" 1 "live e2e did not succeed (result=failure)"
  run_case "failure" "failure" "pull_request" "false" 1 "live e2e did not succeed (result=failure)"
  run_case "cancelled" "cancelled" "pull_request" "false" 1 "live e2e did not succeed (result=cancelled)"
  run_case "empty" "" "pull_request" "false" 1 "live e2e did not succeed (result=)"
  run_case "unknown" "unknown" "pull_request" "false" 1 "live e2e did not succeed (result=unknown)"

  if [[ "${failed}" -ne 0 ]]; then
    echo "self-test: failed" >&2
    exit 1
  fi
  echo "self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

run_gate
