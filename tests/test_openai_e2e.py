from __future__ import annotations

import os

import pytest

from llmtestlab import LabSession

pytestmark = pytest.mark.e2e


def require_e2e() -> None:
    if os.environ.get("LLMTESTLAB_E2E") != "1":
        pytest.skip("set LLMTESTLAB_E2E=1 to run real OpenAI E2E tests")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required when LLMTESTLAB_E2E=1")


def test_openai_chat_tool_flow(lab: LabSession) -> None:
    require_e2e()
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=lab.openai_base_url,
        default_headers=lab.headers,
    )
    first = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Use the tool for weather in Paris."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=64,
    )
    tool_call = first.choices[0].message.tool_calls[0]
    client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Use the tool for weather in Paris."},
            first.choices[0].message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": '{"weather":"sunny"}'},
        ],
        max_tokens=32,
    )

    lab.query(api_type="chat_completions", model="gpt-4.1-mini").expect().tool_called(
        "get_weather"
    ).tool_result_fed_back("get_weather").no_errors()


def test_openai_responses_tool_flow(lab: LabSession) -> None:
    require_e2e()
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=lab.openai_base_url,
        default_headers=lab.headers,
    )
    first = client.responses.create(
        model="gpt-4.1-mini",
        input="Use the tool for weather in Paris.",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "function", "name": "get_weather"},
        max_output_tokens=64,
    )
    function_call = next(item for item in first.output if item.type == "function_call")
    client.responses.create(
        model="gpt-4.1-mini",
        input=[
            function_call,
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": '{"weather":"sunny"}',
            },
        ],
        max_output_tokens=32,
    )

    lab.query(api_type="responses", model="gpt-4.1-mini").expect().tool_called(
        "get_weather"
    ).tool_result_fed_back("get_weather").no_errors()
