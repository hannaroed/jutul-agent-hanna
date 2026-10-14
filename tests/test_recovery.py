"""Tests for invalid-tool-call recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from jutul_agent.agent.builder import build_agent
from jutul_agent.agent.recovery import InvalidToolCallRecoveryMiddleware, ModelOutputError, _recover
from jutul_agent.agent.turns import TurnRunner
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from jutul_agent.session import Session

_INVALID = {
    "name": "read_file",
    "args": '{"path": "a"}{"path": "b"}',
    "id": "call_bad_1",
    "error": "Failed to parse tool call arguments as JSON",
    "type": "invalid_tool_call",
}


def _invalid_message() -> AIMessage:
    return AIMessage(content="", invalid_tool_calls=[_INVALID])


def test_recover_feeds_errors_back_and_jumps() -> None:
    state = {"messages": [HumanMessage(content="go"), _invalid_message()]}
    update = _recover(state)
    assert update is not None
    assert update["jump_to"] == "model"
    [error] = update["messages"]
    assert isinstance(error, ToolMessage)
    assert error.tool_call_id == "call_bad_1"
    assert error.status == "error"
    assert "could not be parsed" in str(error.content)


def test_recover_ignores_valid_calls_and_plain_replies() -> None:
    valid = AIMessage(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "1"}],
        invalid_tool_calls=[_INVALID],
    )
    assert _recover({"messages": [valid]}) is None  # the tool node handles these
    assert _recover({"messages": [AIMessage(content="done")]}) is None
    assert _recover({"messages": []}) is None


def test_recover_gives_up_after_consecutive_failures() -> None:
    recovery_result = ToolMessage(
        content="x",
        tool_call_id="call_bad_1",
        status="error",
        additional_kwargs={"invalid_tool_call_recovery": True},
    )
    messages = [
        HumanMessage(content="go"),
        _invalid_message(),
        recovery_result,
        _invalid_message(),
        recovery_result,
        _invalid_message(),
    ]
    assert _recover({"messages": messages}) is None
    # A fresh user message resets the budget.
    assert _recover({"messages": [*messages, HumanMessage(content="again"), _invalid_message()]})


def test_recover_nudges_empty_and_reasoning_only_turns() -> None:
    # A blank reply with no tool call: the model fell silent after thinking.
    update = _recover({"messages": [HumanMessage(content="go"), AIMessage(content="")]})
    assert update is not None
    assert update["jump_to"] == "model"
    [nudge] = update["messages"]
    assert isinstance(nudge, HumanMessage)
    assert nudge.additional_kwargs.get("empty_turn_recovery") is True

    # Reasoning-only content (no text block) is also empty as far as an answer goes.
    reasoning_only = AIMessage(content=[{"type": "reasoning", "reasoning": "thinking…"}])
    assert _recover({"messages": [reasoning_only]}) is not None

    # A real answer (string or text block) ends the turn — no nudge.
    assert _recover({"messages": [AIMessage(content="here is the result")]}) is None
    answered = AIMessage(content=[{"type": "text", "text": "the answer"}])
    assert _recover({"messages": [answered]}) is None


def test_recover_gives_up_after_consecutive_empty_turns() -> None:
    nudge = HumanMessage(content="x", additional_kwargs={"empty_turn_recovery": True})
    empty = AIMessage(content="")
    messages = [HumanMessage(content="go"), empty, nudge, empty, nudge, empty]
    assert _recover({"messages": messages}) is None
    # A fresh user message resets the budget.
    assert _recover({"messages": [*messages, HumanMessage(content="again"), empty]})


async def test_agent_recovers_from_empty_turn(tmp_path: Path) -> None:
    """End to end: an empty reasoning-only turn is nudged into an answer."""
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    model = make_scripted_model([AIMessage(content=""), scripted_final("now answered")])

    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    result = await runner.run_prompt("do the thing")

    contents = [str(getattr(m, "content", "")) for m in result.messages]
    assert any("now answered" in c for c in contents)
    session.finalize()


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"stop_reason": "max_tokens"},
        {"finish_reason": "MAX_TOKENS"},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
@pytest.mark.parametrize("kind", ["empty", "answer", "tool", "invalid-tool"])
def test_truncated_response_stops_before_recovery_or_execution(metadata, kind):
    message = AIMessage(content="" if kind == "empty" else "partial", response_metadata=metadata)
    if kind == "tool":
        message.tool_calls = [{"name": "run_julia", "args": {"code": "1+1"}, "id": "partial"}]
    elif kind == "invalid-tool":
        message.invalid_tool_calls = [_INVALID]
    with pytest.raises(ModelOutputError, match="truncated at the output token limit"):
        _recover({"messages": [HumanMessage(content="go"), message]})


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name": "run_julia"',
        '<tool_call>{"name":"run_julia","arguments":{"code":"1+1"}}</tool_call>',
        [
            {"type": "reasoning", "reasoning": "think"},
            {"type": "text", "text": '\n <tool_call>{"name": "run_julia"'},
        ],
    ],
)
def test_unparsed_tool_text_stops_even_without_finish_metadata(content):
    with pytest.raises(ModelOutputError, match="without structured tool calls"):
        _recover({"messages": [AIMessage(content=content)]})


@pytest.mark.parametrize("metadata", [{}, {"finish_reason": "stop"}, {"stop_reason": "end_turn"}])
def test_complete_answers_and_tool_examples_remain_usable(metadata):
    for text in (
        '{"run_id":"run-cedar"}',
        "Example: `<tool_call>{...}</tool_call>`",
        "```xml\n<tool_call>example</tool_call>\n```",
    ):
        assert _recover({"messages": [AIMessage(content=text, response_metadata=metadata)]}) is None


async def test_sync_and_async_hooks_stop_on_the_same_error():
    middleware = InvalidToolCallRecoveryMiddleware()
    state = {
        "messages": [AIMessage(content="partial", response_metadata={"finish_reason": "length"})]
    }
    with pytest.raises(ModelOutputError):
        middleware.after_model(state, None)
    with pytest.raises(ModelOutputError):
        await middleware.aafter_model(state, None)


def _recorded_truncated_message(*, streaming=False):
    import httpx
    from langchain_openai import ChatOpenAI

    raw = json.loads((Path(__file__).parent / "fixtures/truncated_tool_response.json").read_text())

    def respond(request):
        if not streaming:
            return httpx.Response(200, json=raw)
        chunk = {
            "id": raw["id"],
            "object": "chat.completion.chunk",
            "created": raw["created"],
            "model": raw["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "delta": {
                        "role": "assistant",
                        "content": raw["choices"][0]["message"]["content"],
                    },
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="offline",
            api_key="offline",
            base_url="http://offline/v1",
            http_client=client,
            streaming=streaming,
            max_retries=0,
        )
        message = model.invoke("go")
    assert message.response_metadata["finish_reason"] == "length"
    assert message.tool_calls == []
    assert message.content == raw["choices"][0]["message"]["content"]
    return message


@pytest.mark.parametrize("streaming", [False, True])
async def test_real_agent_stops_on_recorded_qwen_response_without_retry(tmp_path, streaming):
    adapter = make_fake_adapter(tmp_path)
    julia = FakeJulia()
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)
    message = _recorded_truncated_message(streaming=streaming)
    model = make_scripted_model([message, scripted_final("must not be requested")])
    try:
        agent, _ = build_agent(session, model=model)
        runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
        with pytest.raises(ModelOutputError, match="truncated"):
            await runner.run_prompt("run the campaign")
        assert model.i == 1
        assert julia.calls == []
        events = session.trace.iter_events()
        assert not any(e.kind in ("tool_call", "tool_result") for e in events)
        assert any(
            e.kind == "message_assistant" and "<tool_call>" in e.payload["content"] for e in events
        )
    finally:
        session.finalize()


async def test_truncated_structured_call_cannot_run_a_tool(tmp_path):
    adapter = make_fake_adapter(tmp_path)
    julia = FakeJulia()
    session = Session.create(julia=julia, state_root=tmp_path, simulator=adapter)
    partial = AIMessage(
        content="",
        response_metadata={"finish_reason": "length"},
        tool_calls=[{"name": "run_julia", "args": {"code": "1+1"}, "id": "cut"}],
    )
    try:
        agent, _ = build_agent(session, model=make_scripted_model([partial]))
        with pytest.raises(ModelOutputError):
            await TurnRunner(agent, thread_id=session.session_id).run_prompt("go")
        assert julia.calls == []
    finally:
        session.finalize()


async def test_agent_recovers_from_invalid_tool_call(tmp_path: Path) -> None:
    """End to end: a malformed tool call costs one round-trip, not the turn."""
    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    model = make_scripted_model([_invalid_message(), scripted_final("recovered and done")])

    agent, _ = build_agent(session, model=model)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    result = await runner.run_prompt("do the thing")

    contents = [str(getattr(m, "content", "")) for m in result.messages]
    assert any("recovered and done" in c for c in contents)
    assert any("could not be parsed" in c for c in contents)
    session.finalize()
