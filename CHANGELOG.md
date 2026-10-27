# Changelog

## [0.7.3](https://github.com/barzoj-co/llmentary-python/compare/v0.7.2...v0.7.3) (2026-10-26)


### Upgrade notes

0.7.3 changes test outcomes for some existing suites:

- `call_count(n)` no longer counts cancelled calls (calls the client aborted, e.g. on barge-in). The failure message says how many were not counted, e.g. `expected 3 captured call(s), got 0 (3 cancelled not counted; pass include_cancelled=True)`. To keep the old total, pass `call_count(3, include_cancelled=True)`.
- `no_errors()` now checks failed calls only; cancelled calls no longer make it fail. Use `no_cancelled()` for the old strictness. The failure message changed from `... N errored call(s)` to `expected no errors, got N failed call(s): #1 ...`, so `pytest.raises(AssertionError, match="errored call")` needs `match="failed call"`.
- `llmentary --version` now prints one format: `llmentary X; proxy Y (path)`. Update scripts that parse the old output.

### Bug Fixes

* add min_severity and ignore_kinds filters to no_ignored_fields (DOG-55) ([#65](https://github.com/barzoj-co/llmentary-python/issues/65)) ([82111bc](https://github.com/barzoj-co/llmentary-python/commit/82111bce881c065f0824d04ef34891fad91feedd))
* add wait(exclude_cancelled=True), pin proxy v0.5.3, live cancel e2e (DOG-58 Y5-Y6) ([#66](https://github.com/barzoj-co/llmentary-python/issues/66)) ([f684976](https://github.com/barzoj-co/llmentary-python/commit/f684976d4db008a64fccacde6151f7dc0bc6fc83))
* classify call outcomes; no_errors checks failed calls only; call_count excludes cancelled (DOG-58 Y1-Y4) ([#62](https://github.com/barzoj-co/llmentary-python/issues/62)) ([3975396](https://github.com/barzoj-co/llmentary-python/commit/397539675079add3ad17c3a251ce5c9f70c02b55))
* wait-timeout hint names the right timeout; one version string format (DOG-49b, DOG-49d) ([#64](https://github.com/barzoj-co/llmentary-python/issues/64)) ([961a06c](https://github.com/barzoj-co/llmentary-python/commit/961a06ce2ec3e45bb76e1d7d1dac7d1fe1fd64d4))

## [0.7.2](https://github.com/barzoj-co/llmentary-python/compare/v0.7.1...v0.7.2) (2026-10-23)


### Bug Fixes

* **deps:** bundle proxy v0.5.1 ([#59](https://github.com/barzoj-co/llmentary-python/issues/59)) ([6b48664](https://github.com/barzoj-co/llmentary-python/commit/6b4866453c1457520708618921f30d8abe837673))

## [0.7.1](https://github.com/barzoj-co/llmentary-python/compare/v0.7.0...v0.7.1) (2026-10-22)


### Bug Fixes

* ignored field side, response_text, committed counts ([#57](https://github.com/barzoj-co/llmentary-python/issues/57)) ([f807606](https://github.com/barzoj-co/llmentary-python/commit/f807606007a4b9973c21fb32ad548e5925d58377))

## [0.7.0](https://github.com/barzoj-co/llmentary-python/compare/v0.6.1...v0.7.0) (2026-10-22)


### Features

* bundle proxy v0.5.0 ([#53](https://github.com/barzoj-co/llmentary-python/issues/53)) ([f577145](https://github.com/barzoj-co/llmentary-python/commit/f57714520b0c8dbb985ffd319ee302b130b4fd62))


### Bug Fixes

* expose error.source ([#55](https://github.com/barzoj-co/llmentary-python/issues/55)) ([89c10a6](https://github.com/barzoj-co/llmentary-python/commit/89c10a6cbf525ebc495b859b3fd7be2c073b849f))

## [0.6.1](https://github.com/barzoj-co/llmentary-python/compare/v0.6.0...v0.6.1) (2026-10-22)


### Bug Fixes

* list in-flight calls on wait timeout ([#51](https://github.com/barzoj-co/llmentary-python/issues/51)) ([115767a](https://github.com/barzoj-co/llmentary-python/commit/115767afb9c08298565d07f9d9afddb6821070d3))

## [0.6.0](https://github.com/barzoj-co/llmentary-python/compare/v0.5.2...v0.6.0) (2026-10-22)


### Features

* add python -m llmentary --version ([#47](https://github.com/barzoj-co/llmentary-python/issues/47)) ([319d465](https://github.com/barzoj-co/llmentary-python/commit/319d465f06cca71ec79b6e7102fc6bea06be86bd))


### Bug Fixes

* stop inventing still active 0 on wait timeout ([#48](https://github.com/barzoj-co/llmentary-python/issues/48)) ([3f8624f](https://github.com/barzoj-co/llmentary-python/commit/3f8624ffe5e2f9e16977c9fc5ca7d63779f5fcc8))
* wait on the other isolated_from session ([#46](https://github.com/barzoj-co/llmentary-python/issues/46)) ([b746196](https://github.com/barzoj-co/llmentary-python/commit/b746196d215e708de8d373f7a1833dfc610cd6c7))

### 0.5.1

`proxy_version()` now returns a bare semver (`"0.4.1"`) instead of the full
`llmentary --version` line. Use `proxy_version_line()` for the full line.
`proxy_version().split()[1]` raises `IndexError` after this change.

## [0.5.2](https://github.com/barzoj-co/llmentary-python/compare/v0.5.1...v0.5.2) (2026-10-22)


### Bug Fixes

* **deps:** bundle proxy v0.4.1 ([#44](https://github.com/barzoj-co/llmentary-python/issues/44)) ([6b5d9c4](https://github.com/barzoj-co/llmentary-python/commit/6b5d9c46d79bdca0ef529015abeb4c7863e0d08b))

## [0.5.1](https://github.com/barzoj-co/llmentary-python/compare/v0.5.0...v0.5.1) (2026-10-21)


### Bug Fixes

* fail isolated_from on empty other ([#43](https://github.com/barzoj-co/llmentary-python/issues/43)) ([7c6986a](https://github.com/barzoj-co/llmentary-python/commit/7c6986a0410b53fde361a9fa4dbc94cf1298a3fb))
* return proxy semver and aborted summary status ([#39](https://github.com/barzoj-co/llmentary-python/issues/39)) ([19e7218](https://github.com/barzoj-co/llmentary-python/commit/19e7218605268586313aefb40436a9507df63468))

## [0.5.0](https://github.com/barzoj-co/llmentary-python/compare/v0.4.0...v0.5.0) (2026-10-21)


### Features

* add typed client_aborted and abort_phase accessors ([#36](https://github.com/barzoj-co/llmentary-python/issues/36)) ([bb48a5a](https://github.com/barzoj-co/llmentary-python/commit/bb48a5a040ed97efcaaecb2e0eb21babdbab168e))
* ship openapi.yaml and ControlClient.openapi() ([#35](https://github.com/barzoj-co/llmentary-python/issues/35)) ([d460089](https://github.com/barzoj-co/llmentary-python/commit/d4600899e1c8a5957030719fc95df71f42d5edce))


### Bug Fixes

* add expect_no_calls and allow_empty on never-created sessions ([#32](https://github.com/barzoj-co/llmentary-python/issues/32)) ([875f784](https://github.com/barzoj-co/llmentary-python/commit/875f784dc8543f52d3d30092ef9414529413ad5c))
* bundle proxy v0.4.0 ([#38](https://github.com/barzoj-co/llmentary-python/issues/38)) ([ca5644a](https://github.com/barzoj-co/llmentary-python/commit/ca5644a54763ba55e0c51b2d3af3e2285bed4a0e))
* label unchecked responses request_only in the summary ([#33](https://github.com/barzoj-co/llmentary-python/issues/33)) ([877ab4f](https://github.com/barzoj-co/llmentary-python/commit/877ab4ff3f6e3ca4d2f7183373f3b7863b5a5ad5))
* match isolated_from on foreign user turns only ([#34](https://github.com/barzoj-co/llmentary-python/issues/34)) ([98ce37b](https://github.com/barzoj-co/llmentary-python/commit/98ce37b1ed614192adc5357b84350eb7684d24e5))

## [0.4.0](https://github.com/barzoj-co/llmentary-python/compare/v0.3.1...v0.4.0) (2026-10-21)


### Features

* add isolated_from and user_texts ([#29](https://github.com/barzoj-co/llmentary-python/issues/29)) ([3ccb46a](https://github.com/barzoj-co/llmentary-python/commit/3ccb46a2db883cf76e5ac00f64dba87e1ff62c48))
* expose CapturedCall.seq and print #seq (id) ([#27](https://github.com/barzoj-co/llmentary-python/issues/27)) ([a2102b4](https://github.com/barzoj-co/llmentary-python/commit/a2102b4ad56af20aeb52ba33c9c52ec13210c6e0))


### Bug Fixes

* bundle proxy v0.2.2 ([#30](https://github.com/barzoj-co/llmentary-python/issues/30)) ([c7ff944](https://github.com/barzoj-co/llmentary-python/commit/c7ff9448b88ba8566611a7e1cec3692d90fb8922))
* bundle proxy v0.3.0 ([#31](https://github.com/barzoj-co/llmentary-python/issues/31)) ([49a03f8](https://github.com/barzoj-co/llmentary-python/commit/49a03f880330642abdfc90a34382152382e516bd))
* fail empty filtered expectations and add call_count ([#21](https://github.com/barzoj-co/llmentary-python/issues/21)) ([850b9e1](https://github.com/barzoj-co/llmentary-python/commit/850b9e1dee71adafb7b3110407a4d3de01a4d1db))
* map wait 404 to SessionNotFound ([#26](https://github.com/barzoj-co/llmentary-python/issues/26)) ([2f49a26](https://github.com/barzoj-co/llmentary-python/commit/2f49a26d409a5f7ed220e4e721a3c6ce578efdbc))
* quiet trace summaries and expose session attribution ([#22](https://github.com/barzoj-co/llmentary-python/issues/22)) ([191e9fd](https://github.com/barzoj-co/llmentary-python/commit/191e9fde9021c7b4874d56cffcd76b6487e5c987))


### Documentation

* link the control API contract in the README ([#28](https://github.com/barzoj-co/llmentary-python/issues/28)) ([e18e05c](https://github.com/barzoj-co/llmentary-python/commit/e18e05cf4b2988ce6c4cf132b38a4347873815b7))

## [0.3.1](https://github.com/barzoj-co/llmentary-python/compare/v0.3.0...v0.3.1) (2026-10-21)


### Bug Fixes

* bundle proxy v0.2.1 ([#23](https://github.com/barzoj-co/llmentary-python/issues/23)) ([6abf622](https://github.com/barzoj-co/llmentary-python/commit/6abf622ee61a778b2958ce7d3d3649f7795ec1af))

## [0.3.0](https://github.com/barzoj-co/llmentary-python/compare/v0.2.0...v0.3.0) (2026-10-20)


### Features

* add control API helpers, spec accessors, and default upstream ([#18](https://github.com/barzoj-co/llmentary-python/issues/18)) ([1d7b97e](https://github.com/barzoj-co/llmentary-python/commit/1d7b97ed345a1a95f60aab8851522485e966a915))
* add path session URL, call filters, and answer helpers ([#19](https://github.com/barzoj-co/llmentary-python/issues/19)) ([15155c5](https://github.com/barzoj-co/llmentary-python/commit/15155c5c3902aaf7560e1244628e7d3e03710ba8))


### Bug Fixes

* bundle proxy v0.2.0 ([#20](https://github.com/barzoj-co/llmentary-python/issues/20)) ([3c6f9cc](https://github.com/barzoj-co/llmentary-python/commit/3c6f9cc627c8f953ddaf4e65205ff9356809e4de))
* fail clearly when the proxy binary is missing ([#16](https://github.com/barzoj-co/llmentary-python/issues/16)) ([3cda18e](https://github.com/barzoj-co/llmentary-python/commit/3cda18e900926e36c75614e41f30f7d8c2adc3ec))
* send timeout and quiet wait params the proxy reads ([#15](https://github.com/barzoj-co/llmentary-python/issues/15)) ([c34f9d2](https://github.com/barzoj-co/llmentary-python/commit/c34f9d2ba53166ecf300605d3b3db94582c14aad))

## [0.2.0](https://github.com/barzoj-co/llmentary-python/compare/v0.1.1...v0.2.0) (2026-10-20)


### Features

* rename lab API to llmentary and slim README ([#10](https://github.com/barzoj-co/llmentary-python/issues/10)) ([e14b350](https://github.com/barzoj-co/llmentary-python/commit/e14b350e0b793d7e6e240bc26893082c41302aa1))


### Bug Fixes

* place bundled AGPL at the PEP 639 License-File path ([#13](https://github.com/barzoj-co/llmentary-python/issues/13)) ([45210e1](https://github.com/barzoj-co/llmentary-python/commit/45210e1715707424554bd7764bb28a786dfe1060))

## [0.1.1](https://github.com/barzoj-co/llmentary-python/compare/v0.1.0...v0.1.1) (2026-10-20)


### Bug Fixes

* build release wheels with current wheel tags CLI ([#11](https://github.com/barzoj-co/llmentary-python/issues/11)) ([8cfae98](https://github.com/barzoj-co/llmentary-python/commit/8cfae986af00c6dec26a9fca526be6cbb39bab67))

## 0.1.0 (2026-10-19)


### ⚠ BREAKING CHANGES

* rename Python client to llmentary (LTL-RENAME-3) ([#7](https://github.com/barzoj-co/llmentary-python/issues/7))

### Features

* rename Python client to llmentary (LTL-RENAME-3) ([#7](https://github.com/barzoj-co/llmentary-python/issues/7)) ([1c3b09e](https://github.com/barzoj-co/llmentary-python/commit/1c3b09e6f2cc3922512be0c3ce499fea82ea184d))


### Documentation

* state MIT plainly and drop AGPL commentary ([#4](https://github.com/barzoj-co/llmentary-python/issues/4)) ([2bdf155](https://github.com/barzoj-co/llmentary-python/commit/2bdf155dbf84d4407f33928d19f19d661421bdb2))
