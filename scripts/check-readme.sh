#!/usr/bin/env bash
# Fail if the README is missing the required badge URLs, the canonical
# llmentary repository table, or the MIT license one-liner.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="${ROOT}/README.md"

if [[ ! -f "$README" ]]; then
  echo "missing README.md" >&2
  exit 1
fi

badges=(
  'https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml/badge.svg?branch=main'
  'https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml'
  'https://codecov.io/gh/barzoj-co/llmentary-python/graph/badge.svg'
  'https://codecov.io/gh/barzoj-co/llmentary-python'
  'https://api.scorecard.dev/projects/github.com/barzoj-co/llmentary-python/badge'
  'https://scorecard.dev/viewer/?uri=github.com/barzoj-co/llmentary-python'
  'https://img.shields.io/github/license/barzoj-co/llmentary-python'
  'https://img.shields.io/pypi/v/llmentary'
  'https://pypi.org/project/llmentary/'
  'https://img.shields.io/pypi/pyversions/llmentary'
)

missing=0
for url in "${badges[@]}"; do
  if ! grep -F -q -- "$url" "$README"; then
    echo "README.md is missing badge URL: $url" >&2
    missing=1
  fi
done

canonical_block='<!-- ltl-repos:start -->
## llmentary repositories

| Repository | What |
|---|---|
| [llmentary](https://github.com/barzoj-co/llmentary) | Recording proxy binary (Go) |
| [llmentary-go](https://github.com/barzoj-co/llmentary-go) | Go client for `go test` |
| [llmentary-python](https://github.com/barzoj-co/llmentary-python) | Python client and pytest plugin |
| [llmentary-js](https://github.com/barzoj-co/llmentary-js) | TypeScript/JavaScript client for Vitest and Jest |
<!-- ltl-repos:end -->'

if ! grep -F -q -- "$canonical_block" "$README"; then
  echo "README.md is missing or altered the canonical llmentary repositories block" >&2
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
