"""Python client for llmentary deterministic LLM integration tests."""

from importlib.metadata import PackageNotFoundError, version

from .assertions import Expectation, SoftExpectation
from .binary import BinaryResolutionError, find_binary
from .client import ControlClient, ProxyHandle, Query, Session, start_proxy
from .types import APIType, SessionTrace
from .wait import WaitTimeoutError

try:
    __version__ = version("llmentary")
except PackageNotFoundError:
    __version__ = "0.2.0"  # x-release-please-version

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
    "WaitTimeoutError",
    "__version__",
    "find_binary",
    "start_proxy",
]
