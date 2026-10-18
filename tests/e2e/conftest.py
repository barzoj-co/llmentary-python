from __future__ import annotations

import pytest
from helpers import require_e2e


@pytest.fixture(autouse=True)
def _require_real_openai_e2e() -> None:
    require_e2e()
