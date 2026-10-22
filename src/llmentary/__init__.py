"""Python client for llmentary deterministic LLM integration tests."""

from importlib.metadata import PackageNotFoundError, version

from .assertions import Expectation, SoftExpectation, answer_mentions_any
from .binary import BinaryResolutionError, contract_path, find_binary
from .client import (
    ControlClient,
    ProxyHandle,
    Query,
    Session,
    SessionNotFound,
    start_proxy,
)
from .doctor import (
    proxy_binary_path,
    proxy_report_header,
    proxy_version,
    proxy_version_line,
)
from .types import (
    APIType,
    IgnoredField,
    SessionInfo,
    SessionTrace,
    SpecViolation,
    Validation,
)
from .wait import WaitTimeoutError

try:
    __version__ = version("llmentary")
except PackageNotFoundError:
    __version__ = "0.5.2"  # x-release-please-version

__all__ = [
    "APIType",
    "BinaryResolutionError",
    "ControlClient",
    "Expectation",
    "IgnoredField",
    "ProxyHandle",
    "Query",
    "Session",
    "SessionInfo",
    "SessionNotFound",
    "SessionTrace",
    "SoftExpectation",
    "SpecViolation",
    "Validation",
    "WaitTimeoutError",
    "__version__",
    "answer_mentions_any",
    "contract_path",
    "find_binary",
    "proxy_binary_path",
    "proxy_report_header",
    "proxy_version",
    "proxy_version_line",
    "start_proxy",
]
