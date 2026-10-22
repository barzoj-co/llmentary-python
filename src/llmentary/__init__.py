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
    ProxyConfig,
    SessionInfo,
    SessionSpecSummary,
    SessionTrace,
    SpecViolation,
    Validation,
)
from .wait import InFlightCall, WaitTimeoutError

try:
    __version__ = version("llmentary")
except PackageNotFoundError:
    __version__ = "0.7.0"  # x-release-please-version

__all__ = [
    "APIType",
    "BinaryResolutionError",
    "ControlClient",
    "Expectation",
    "IgnoredField",
    "InFlightCall",
    "ProxyConfig",
    "ProxyHandle",
    "Query",
    "Session",
    "SessionInfo",
    "SessionNotFound",
    "SessionSpecSummary",
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
