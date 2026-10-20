"""Python client for llmentary deterministic LLM integration tests."""

from importlib.metadata import PackageNotFoundError, version

from .assertions import Expectation, SoftExpectation
from .binary import BinaryResolutionError, find_binary
from .client import ControlClient, ProxyHandle, Query, Session, start_proxy
from .types import APIType, SessionTrace

try:
    __version__ = version("llmentary")
except PackageNotFoundError:
    __version__ = "0.1.1"  # x-release-please-version

__all__ = [
    "APIType",
    "BinaryResolutionError",
    "ControlClient",
    "Expectation",
    "ProxyHandle",
    "Query",
    "Session",
    "SessionTrace",
    "SoftExpectation",
    "__version__",
    "find_binary",
    "start_proxy",
]
