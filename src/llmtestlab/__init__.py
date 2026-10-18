"""Python client for llm-test-lab deterministic LLM integration tests."""

from .assertions import Expectation, SoftExpectation
from .binary import BinaryResolutionError, find_binary
from .client import ControlClient, LabSession, ProxyHandle, Query, start_proxy
from .types import APIType, SessionTrace

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
    "find_binary",
    "start_proxy",
]
