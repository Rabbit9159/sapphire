"""Gemini 3 tool calls must round-trip extra_content.google.thought_signature.

Google returns 400 on the tool-result request when the signature that came
back on the function call is missing. Other providers must not receive it.
"""
from types import SimpleNamespace

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from core.chat.llm_providers.base import ToolCall, LLMResponse
from core.chat.llm_providers.gemini import GeminiProvider
from core.chat.llm_providers.openai_compat import OpenAICompatProvider, gemini_thought_extra


SIG = "sig-a"
SIG_B = "sig-b"


def _gemini():
    return GeminiProvider({
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": "k",
        "model": "gemini-3.5-flash",
        "thinking_enabled": False,
    })


def _local():
    return OpenAICompatProvider({
        "provider": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "api_key": "test",
        "model": "qwen3",
    })


def _turn(calls):
    return [{
        "role": "assistant",
        "content": "",
        "tool_calls": calls,
    }]


def test_signature_is_sent_back_on_gemini_only():
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_recent_memories", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": SIG}},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "web_search", "arguments": "{}"},
        },
    ]
    cleaned = _gemini()._sanitize_messages(_turn(calls))
    out = cleaned[0]["tool_calls"]
    assert out[0]["extra_content"] == {"google": {"thought_signature": SIG}}
    assert "extra_content" not in out[1]

    local = _local()._sanitize_messages(_turn(calls))
    assert "extra_content" not in local[0]["tool_calls"][0]


def test_sdk_extra_field_round_trips_through_the_stream():
    delta_call = ChoiceDeltaToolCall.model_validate({
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_recent_memories", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": SIG}},
    })
    assert gemini_thought_extra(delta_call) == {"google": {"thought_signature": SIG}}

    chunk = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=[delta_call]),
        )],
    )
    provider = _gemini()
    provider._create = lambda request_kwargs: [chunk]

    events = list(provider.chat_completion_stream(
        [{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "function": {"name": "get_recent_memories", "parameters": {}}}],
    ))
    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert tool_events[-1]["extra_content"]["google"]["thought_signature"] == SIG
    done = next(e for e in events if e["type"] == "done")
    assert done["response"].tool_calls[0].extra_content == {"google": {"thought_signature": SIG}}
    assert done["response"].get_tool_calls_as_dicts()[0]["extra_content"]["google"]["thought_signature"] == SIG


def test_tool_call_dict_omits_extra_when_absent():
    plain = ToolCall(id="c", name="n", arguments="{}").to_dict()
    assert "extra_content" not in plain
    signed = LLMResponse(tool_calls=[
        ToolCall(id="c", name="n", arguments="{}",
                 extra_content={"google": {"thought_signature": SIG_B}})
    ]).get_tool_calls_as_dicts()
    assert signed[0]["extra_content"]["google"]["thought_signature"] == SIG_B
