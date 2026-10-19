from __future__ import annotations

import httpx
import pytest

from llmentary import LabSession


@pytest.mark.integration
def test_chat_completions_tool_flow_through_real_proxy(lab: LabSession) -> None:
    with httpx.Client(base_url=lab.openai_base_url, headers=lab.headers) as client:
        first = client.post(
            "/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
        )
        first.raise_for_status()
        tool_call = first.json()["choices"][0]["message"]["tool_calls"][0]
        second = client.post(
            "/chat/completions",
            json={
                "model": "fake-model",
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": '{"weather":"sunny"}',
                    },
                ],
            },
        )
        second.raise_for_status()

    lab.expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").response_json_valid().response_json_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    ).no_errors()


@pytest.mark.integration
def test_responses_tool_flow_through_real_proxy(lab: LabSession) -> None:
    with httpx.Client(base_url=lab.openai_base_url, headers=lab.headers) as client:
        first = client.post(
            "/responses",
            json={
                "model": "fake-model",
                "input": "What is the weather in Paris?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )
        first.raise_for_status()
        tool_call = first.json()["output"][0]
        second = client.post(
            "/responses",
            json={
                "model": "fake-model",
                "input": [
                    tool_call,
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": '{"weather":"sunny"}',
                    },
                ],
            },
        )
        second.raise_for_status()

    lab.query(api_type="responses").expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").response_json_valid().no_errors()
