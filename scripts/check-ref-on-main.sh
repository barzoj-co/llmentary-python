#!/usr/bin/env bash
# Refuse to release unless the tagged commit is an ancestor of origin/main.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check_ref() {
  local ref="${1:-HEAD}"
  local commit short

  if ! commit="$(git rev-parse --verify "${ref}^{commit}" 2>/dev/null)"; then
    echo "::error::cannot resolve ${ref} to a commit"
    return 1
  fi
  if ! git rev-parse --verify refs/remotes/origin/main >/dev/null 2>&1; then
    echo "::error::origin/main not available; checkout needs fetch-depth: 0"
    return 1
  fi
  short="$(git rev-parse --short "${commit}")"
  if ! git merge-base --is-ancestor "${commit}" origin/main; then
    echo "::error::${ref} (${short}) is not on main; refusing to release"
    return 1
  fi
  echo "${commit}"
  return 0
}

self_test() {
  local tmp failed=0 output rc commit doomed
  tmp="$(mktemp -d)"
  trap "rm -rf '$(printf '%s' "${tmp}")'" EXIT

  git init --bare "${tmp}/origin.git" >/dev/null
  git clone "${tmp}/origin.git" "${tmp}/repo" >/dev/null 2>&1
  (
    cd "${tmp}/repo"
    git config user.email "self-test@example.com"
    git config user.name "self-test"
    git checkout -b main
    echo one >file
    git add file
    git commit -m "on main" >/dev/null
    git push -u origin main >/dev/null 2>&1
  )

  run_in_repo() {
    local want_rc="$1"
    local want_msg="$2"
    local label="$3"
    shift 3
    set +e
    output="$(cd "${tmp}/repo" && bash "${ROOT}/scripts/check-ref-on-main.sh" "$@" 2>&1)"
    rc=$?
    set -e
    echo "case ${label}: rc=${rc} want_rc=${want_rc}"
    echo "${output}"
    if [[ "${rc}" -ne "${want_rc}" ]]; then
      echo "self-test: ${label}: expected exit ${want_rc}, got ${rc}" >&2
      failed=1
    fi
    if [[ -n "${want_msg}" ]] && ! grep -F -q "${want_msg}" <<<"${output}"; then
      echo "self-test: ${label}: expected message: ${want_msg}" >&2
      failed=1
    fi
  }

  commit="$(cd "${tmp}/repo" && git rev-parse HEAD)"
  run_in_repo 0 "" "commit-on-main" "${commit}"
  run_in_repo 0 "" "main-tip" "HEAD"

  (
    cd "${tmp}/repo"
    git tag -a annotated-main -m "annotated tag on main"
  )
  run_in_repo 0 "" "annotated-tag-on-main" "annotated-main"

  (
    cd "${tmp}/repo"
    git checkout -b side
    echo side >>file
    git commit -am "unmerged side branch" >/dev/null
    git tag light-side
    git checkout main >/dev/null 2>&1
  )
  run_in_repo 1 "is not on main; refusing to release" "lightweight-tag-unmerged" "light-side"

  (
    cd "${tmp}/repo"
    git checkout -b doomed
    echo doomed >>file
    git commit -am "doomed branch" >/dev/null
    git rev-parse HEAD >"${tmp}/doomed.sha"
    git checkout main >/dev/null 2>&1
    git branch -D doomed >/dev/null
  )
  doomed="$(cat "${tmp}/doomed.sha")"
  run_in_repo 1 "" "deleted-branch-commit" "${doomed}"

  (
    cd "${tmp}/repo"
    git update-ref -d refs/remotes/origin/main
  )
  run_in_repo 1 "origin/main not available; checkout needs fetch-depth: 0" "missing-origin-main" "HEAD"

  (
    cd "${tmp}/repo"
    git fetch origin >/dev/null 2>&1
  )
  run_in_repo 1 "" "nonexistent-ref" "this-ref-does-not-exist"

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

check_ref "${1:-HEAD}"
