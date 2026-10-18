#!/usr/bin/env bash
# Fail if the README is missing the required badge URLs, the canonical
# llm-test-lab repository table, or the MIT license one-liner.
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

| Repository | What |
|---|---|
| [llm-test-lab](https://github.com/barzoj-co/llm-test-lab) | Recording proxy binary (Go) |
| [llm-test-lab-go](https://github.com/barzoj-co/llm-test-lab-go) | Go client for `go test` |
| [llm-test-lab-python](https://github.com/barzoj-co/llm-test-lab-python) | Python client and pytest plugin |
| [llm-test-lab-js](https://github.com/barzoj-co/llm-test-lab-js) | TypeScript/JavaScript client for Vitest and Jest |
<!-- ltl-repos:end -->'

if ! grep -F -q -- "$canonical_block" "$README"; then
  echo "README.md is missing or altered the canonical llm-test-lab repositories block" >&2
  missing=1
fi

license_section="$(awk '/^## License$/{p=1} p' "$README" | sed '/^$/d')"
expected_license="$(printf '%s\n%s' '## License' 'Licensed under MIT, see LICENSE')"
if [ "$license_section" != "$expected_license" ]; then
  echo "README.md License section must be a single line: Licensed under MIT, see LICENSE" >&2
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi
echo "README.md badges, repository block, and license one-liner are present"
