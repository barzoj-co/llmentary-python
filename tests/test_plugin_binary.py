from __future__ import annotations

import pytest

from llmentary.binary import BinaryResolutionError


def test_plugin_surfaces_binary_resolution_error(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMENTARY_BIN", raising=False)
    monkeypatch.delenv("LLMENTARY_PROXY_URL", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: None)
    monkeypatch.setattr("llmentary.binary.package_bin_dir", lambda: pytester.path / "missing-bin")
    pytester.makepyfile(
        """
        def test_needs_session(llmentary):
            assert llmentary.session_id
        """
    )
    result = pytester.runpytest("-q")
    result.assert_outcomes(errors=1)
    text = result.stdout.str() + result.stderr.str()
    assert "BinaryResolutionError" in text
    assert "no bundled proxy" in text or "no wheel for platform" in text
    assert "RuntimeError: " not in text
    assert BinaryResolutionError.__name__ in text
