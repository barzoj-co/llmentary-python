"""Python client for llm-test-lab deterministic LLM integration tests."""

from importlib.metadata import PackageNotFoundError, version

from .assertions import Expectation, SoftExpectation
from .binary import BinaryResolutionError, find_binary
from .client import ControlClient, LabSession, ProxyHandle, Query, start_proxy
from .types import APIType, SessionTrace

try:
    __version__ = version("llm-test-lab")
except PackageNotFoundError:
    __version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "APIType",
    "BinaryResolutionError",
    "ControlClient",
    "Expectation",
    "LabSession",
    "ProxyHandle",
    "Query",
    "SessionTrace",
    "SoftExpectation",
    "__version__",
    "find_binary",
    "start_proxy",
]
