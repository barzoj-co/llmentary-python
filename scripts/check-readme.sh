#!/usr/bin/env bash
# Fail if the README is missing the required badge URLs or the canonical
# llm-test-lab repository table.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${ROOT}/README.md"

if [[ ! -f "$README" ]]; then
  echo "missing README.md" >&2
  exit 1
fi

badges=(
  'https://github.com/barzoj-co/llm-test-lab-python/actions/workflows/ci.yml/badge.svg?branch=main'
  'https://github.com/barzoj-co/llm-test-lab-python/actions/workflows/ci.yml'
  'https://codecov.io/gh/barzoj-co/llm-test-lab-python/graph/badge.svg'
  'https://codecov.io/gh/barzoj-co/llm-test-lab-python'
  'https://api.scorecard.dev/projects/github.com/barzoj-co/llm-test-lab-python/badge'
  'https://scorecard.dev/viewer/?uri=github.com/barzoj-co/llm-test-lab-python'
  'https://img.shields.io/github/license/barzoj-co/llm-test-lab-python'
  'https://img.shields.io/pypi/v/llm-test-lab'
  'https://pypi.org/project/llm-test-lab/'
  'https://img.shields.io/pypi/pyversions/llm-test-lab'
)

missing=0
for url in "${badges[@]}"; do
  if ! grep -F -q -- "$url" "$README"; then
    echo "README.md is missing badge URL: $url" >&2
    missing=1
  fi
done

canonical_block='<!-- ltl-repos:start -->
## llm-test-lab repositories

| Repository | What | License |
|---|---|---|
| [llm-test-lab](https://github.com/barzoj-co/llm-test-lab) | Recording proxy binary (Go) | AGPL-3.0 |
| [llm-test-lab-go](https://github.com/barzoj-co/llm-test-lab-go) | Go client for `go test` | MIT |
| [llm-test-lab-python](https://github.com/barzoj-co/llm-test-lab-python) | Python client and pytest plugin | MIT |
| [llm-test-lab-js](https://github.com/barzoj-co/llm-test-lab-js) | TypeScript/JavaScript client for Vitest and Jest | MIT |
<!-- ltl-repos:end -->'

if ! grep -F -q -- "$canonical_block" "$README"; then
  echo "README.md is missing or altered the canonical llm-test-lab repositories block" >&2
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
echo "README.md badges and repository block are present"
