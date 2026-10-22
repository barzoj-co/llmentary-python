from __future__ import annotations

import ast
import re
from pathlib import Path

import llmentary
from llmentary import (
    InFlightCall,
    Query,
    Session,
    SessionNotFound,
    proxy_version,
    proxy_version_line,
    pytest_plugin,
)
from llmentary.assertions import Expectation, SoftExpectation

ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    ROOT / "README.md",
    ROOT / "docs" / "api.md",
    ROOT / "docs" / "pytest.md",
    ROOT / "docs" / "binaries.md",
)
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
FORBIDDEN_README = (
    "Renamed from llm-test-lab",
    "LLMTESTLAB",
    "CI configuration required",
    "trusted publishing",
    "private repo",
    "private repository",
)


def _python_fences(path: Path) -> list[str]:
    return PYTHON_FENCE.findall(path.read_text())


def test_package_exports_session_not_lab_aliases() -> None:
    assert "Session" in llmentary.__all__
    assert "answer_mentions_any" in llmentary.__all__
    assert "proxy_version" in llmentary.__all__
    assert "proxy_version_line" in llmentary.__all__
    assert llmentary.Session is Session
    assert llmentary.proxy_version is proxy_version
    assert llmentary.proxy_version_line is proxy_version_line
    assert "SessionNotFound" in llmentary.__all__
    assert llmentary.SessionNotFound is SessionNotFound
    assert "InFlightCall" in llmentary.__all__
    assert llmentary.InFlightCall is InFlightCall
    assert "contract_path" in llmentary.__all__
    for removed in ("LabSession", "LlmentarySession"):
        assert removed not in llmentary.__all__
        assert not hasattr(llmentary, removed)


def test_plugin_registers_llmentary_fixture_only() -> None:
    marker = getattr(pytest_plugin.llmentary, "_fixture_function_marker", None)
    assert marker is not None
    assert getattr(marker, "scope", None) == "function"
    assert not hasattr(pytest_plugin, "lab")


def test_query_binds_session_not_lab() -> None:
    assert "session" in Query.__dataclass_fields__
    assert "lab" not in Query.__dataclass_fields__


def test_readme_is_user_facing() -> None:
    readme = (ROOT / "README.md").read_text()
    for needle in FORBIDDEN_README:
        assert needle not in readme, f"README must not mention {needle!r}"
    assert not re.search(r"\blab\b", readme)
    assert not re.search(r"\bLabSession\b", readme)
    assert "LlmentarySession" not in readme
    headings = re.findall(r"^## .+$", readme, flags=re.MULTILINE)
    assert headings == [
        "## Install",
        "## Versions",
        "## Quickstart",
        "## Waiting for calls",
        "## Assertions",
        "## llmentary repositories",
        "## License",
    ]


def test_doc_python_snippets_compile_and_match_api() -> None:
    session_attrs = {
        "session_id",
        "base_url",
        "openai_base_url",
        "path_base_url",
        "headers",
        "baggage",
        "wait",
        "trace",
        "list_sessions",
        "otlp",
        "cleanup",
        "expect",
        "expect_no_calls",
        "query",
        "mark",
        "soft_expect",
    }
    assertion_methods = {
        name
        for cls in (Expectation, SoftExpectation)
        for name, value in vars(cls).items()
        if callable(value) and not name.startswith("_")
    }

    for path in DOC_PATHS:
        for index, source in enumerate(_python_fences(path), start=1):
            compile(source, f"{path}#{index}", "exec")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    arg_names = [arg.arg for arg in node.args.args]
                    assert "lab" not in arg_names, f"{path} snippet uses fixture lab"
                    if "llmentary" in arg_names:
                        assert arg_names.count("llmentary") == 1
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "llmentary":
                        assert node.attr in session_attrs, (
                            f"{path} uses unknown llmentary.{node.attr}"
                        )
                    if isinstance(node.value, ast.Name) and node.value.id == "expect":
                        assert node.attr in assertion_methods, (
                            f"{path} uses unknown expect.{node.attr}"
                        )
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        if isinstance(func, ast.Attribute) and func.attr in {"expect", "query"}:
                            if node.attr not in assertion_methods | {
                                "expect",
                                "trace",
                                "offers_tool",
                                "with_tools",
                                "without_tools",
                                "since",
                            }:
                                raise AssertionError(
                                    f"{path} uses unknown chained attribute {node.attr}"
                                )
