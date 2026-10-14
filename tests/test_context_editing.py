"""Tests for tool-result clearing (ContextEditingMiddleware)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from jutul_agent.agent.context_editing import (
    CLEARED_RESULT_PREFIX_MAX_CHARS,
    SMALL_TOOL_RESULT_MAX_CHARS,
    build_context_editing_middleware,
    clear_tool_uses_trigger_tokens,
    keep_recent_tool_results,
)


def test_clear_tool_uses_trigger_tokens() -> None:
    assert clear_tool_uses_trigger_tokens(65_536) == int(65_536 * 0.6)
    # Below the summarization trigger (0.8) so clearing is the first response.
    assert clear_tool_uses_trigger_tokens(100_000) < int(100_000 * 0.8)
    assert clear_tool_uses_trigger_tokens(None) == 60_000


def test_keep_recent_tool_results_scales_with_window() -> None:
    # A small (local) window keeps fewer, so the working set can't fill it.
    assert keep_recent_tool_results(65_536) == 3
    assert keep_recent_tool_results(200_000) == 6
    assert keep_recent_tool_results(None) == 6
    assert keep_recent_tool_results(16_384) == 1


def test_build_context_editing_middleware_config(monkeypatch) -> None:
    from jutul_agent import models

    monkeypatch.setattr(models, "context_window", lambda model_id: 50_000)
    middleware = build_context_editing_middleware(model_id="ollama:qwen3.6:27b")
    (edit,) = middleware.edits
    assert isinstance(edit, ClearToolUsesEdit)
    assert edit.trigger == int(50_000 * 0.6)
    assert edit.keep == 3  # small window → fewer recent results kept
    # The attempt log is referred to by value, so it is never cleared.
    assert "record_attempt" in edit.exclude_tools


def test_build_context_editing_uses_prebuilt_model_profile() -> None:
    model = MagicMock(profile={"max_input_tokens": 16_384})
    middleware = build_context_editing_middleware(model=model)
    (edit,) = middleware.edits
    assert edit.trigger == int(16_384 * 0.6)
    assert edit.keep == 1


def _tool_round(call_id: str, name: str, result: str) -> list:
    """An AIMessage requesting one tool call plus its ToolMessage result."""
    return [
        AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}]),
        ToolMessage(content=result, tool_call_id=call_id, name=name),
    ]


async def test_clears_old_tool_results_but_keeps_recent_and_excluded() -> None:
    """Old tool results become a placeholder; recent and excluded ones stay,
    and state["messages"] is left intact (the edit rides the request)."""
    edit = ClearToolUsesEdit(trigger=10, keep=2, exclude_tools=("record_attempt",))
    middleware = ContextEditingMiddleware(edits=[edit])

    messages = [
        HumanMessage(content="start"),
        *_tool_round("a", "sim_read", "OLD SOURCE DUMP " * 50),  # oldest → cleared
        *_tool_round("b", "record_attempt", "attempt #1 id=keep-me"),  # old but excluded
        *_tool_round("c", "run_julia", "recent output one"),  # recent → kept
        *_tool_round("d", "sim_read", "recent output two"),  # recent → kept
    ]
    captured: dict[str, list] = {}

    async def handler(req: ModelRequest) -> AIMessage:
        captured["messages"] = req.messages
        return AIMessage(content="ok")

    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 100_000}),
        messages=messages,
        system_message=None,
        runtime=MagicMock(),
        state={"messages": messages},
    )
    await middleware.awrap_model_call(request, handler)

    seen = {m.tool_call_id: m for m in captured["messages"] if isinstance(m, ToolMessage)}
    assert seen["a"].content == "[cleared]"  # old sim_read cleared
    assert "keep-me" in str(seen["b"].content)  # record_attempt excluded → intact
    assert seen["c"].content == "recent output one"  # within keep window
    assert seen["d"].content == "recent output two"

    # Non-mutating: the original messages still carry the full result.
    original = next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "a")
    assert original.content.startswith("OLD SOURCE DUMP")


@pytest.mark.parametrize("overhead", ["system", "tools"])
async def test_clearing_counts_system_and_schema_overhead_without_mutating_history(overhead):
    old = "source columns\n" + "old " * 1000
    messages = [
        HumanMessage(content="start"),
        *_tool_round("a", "run_julia", old),
        *_tool_round("b", "run_julia", "latest"),
    ]
    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 16_384}),
        messages=messages,
        system_message=SystemMessage(content="system rules " * 4000)
        if overhead == "system"
        else None,
        tools=[{"type": "function", "function": {"name": "probe", "description": "schema " * 6000}}]
        if overhead == "tools"
        else [],
        runtime=MagicMock(),
        state={"messages": messages},
    )
    middleware = build_context_editing_middleware(model=request.model)
    observed = []

    async def handler(req):
        observed.extend(req.messages)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(request, handler)
    results = {m.tool_call_id: m.content for m in observed if isinstance(m, ToolMessage)}
    assert results["a"].startswith("[cleared; first lines only]\nsource columns\n")
    assert len(results["a"]) < len(old)
    assert results["b"] == "latest"
    assert messages[2].content == old


def test_clearing_can_use_the_guard_counter_when_estimates_disagree():
    old = "old " * 1000
    messages = [
        HumanMessage(content="start"),
        *_tool_round("a", "run_julia", old),
        *_tool_round("b", "run_julia", "latest"),
    ]
    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 16384}),
        messages=messages,
        system_message=None,
        tools=[],
        runtime=MagicMock(),
        state={"messages": messages},
    )
    middleware = build_context_editing_middleware(model=request.model)
    reduced = middleware.edit_request(request, token_counter=lambda messages, system, tools: 20000)
    assert reduced.messages[2].content.startswith("[cleared; first lines only]")
    assert reduced.messages[4].content == "latest"
    assert messages[2].content == old


@pytest.mark.parametrize("size", [0, SMALL_TOOL_RESULT_MAX_CHARS, SMALL_TOOL_RESULT_MAX_CHARS + 1])
def test_small_result_boundary_and_cleared_prefix_are_bounded(size):
    original = "x" * size
    messages = [
        HumanMessage(content="goal"),
        *_tool_round("old", "read_file", original),
        *_tool_round("new", "grep", "latest"),
    ]
    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 16384}),
        messages=messages,
        tools=[],
        system_message=None,
        runtime=MagicMock(),
        state={"messages": messages},
    )
    middleware = build_context_editing_middleware(model=request.model)
    reduced = middleware.edit_request(request, token_counter=lambda *_: 20000)
    result = reduced.messages[2]
    if size <= SMALL_TOOL_RESULT_MAX_CHARS:
        assert result.content == original
        assert "context_editing" not in result.response_metadata
    else:
        assert (
            result.content
            == "[cleared; first lines only]\n" + original[:CLEARED_RESULT_PREFIX_MAX_CHARS]
        )
        assert result.response_metadata["context_editing"]["cleared"] is True
    assert messages[2].content == original
    assert reduced.messages[4].content == "latest"


@pytest.mark.parametrize("async_call", [False, True])
async def test_small_ledger_schema_and_values_survive_later_grep(async_call):
    ledger = (
        "@@ lines 1-3 of 3 @@\n"
        "run_id,porosity,recovery_factor,max_pressure_bar,artifact_path\n"
        "run-a,0.21,0.4198097956633321,290.50194377165514,results/a.csv\n"
        "run-b,0.24,0.34422282769482515,280.07102680388783,results/b.csv"
    )
    messages = [
        HumanMessage(content="select the winner"),
        *_tool_round("ledger", "read_file", ledger),
        *_tool_round("cell", "grep", "run-a,73,0.9867380629912129"),
    ]
    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 16384}),
        messages=messages,
        tools=[],
        system_message=SystemMessage(content="rules " * 9000),
        runtime=MagicMock(),
        state={"messages": messages},
    )
    middleware = build_context_editing_middleware(model=request.model)
    captured = []

    def handler(reduced):
        captured.extend(reduced.messages)
        return AIMessage(content="ok")

    if async_call:

        async def async_handler(reduced):
            return handler(reduced)

        await middleware.awrap_model_call(request, async_handler)
    else:
        middleware.wrap_model_call(request, handler)
    assert captured[2].content == ledger
    assert captured[4].content == "run-a,73,0.9867380629912129"
    assert messages[2].content == ledger


def test_large_csv_keeps_header_with_file_wrapper_but_not_bulk_rows():
    header = "run_id,porosity,recovery_factor,max_pressure_bar,cell_id,water_saturation"
    table = (
        "@@ lines 1-193 of 193 @@\n"
        + header
        + "\n"
        + "run-a,0.21,0.4198098,290.501944,1,0.7\n" * 192
    )
    messages = [
        HumanMessage(content="goal"),
        *_tool_round("table", "read_file", table),
        *_tool_round("new", "grep", "latest"),
    ]
    request = ModelRequest(
        model=MagicMock(profile={"max_input_tokens": 16384}),
        messages=messages,
        tools=[],
        system_message=None,
        runtime=MagicMock(),
        state={"messages": messages},
    )
    reduced = build_context_editing_middleware(model=request.model).edit_request(
        request, token_counter=lambda *_: 20000
    )
    assert header in reduced.messages[2].content
    assert reduced.messages[2].content.count("run-a,") == 1
    assert reduced.messages[2].tool_call_id == "table"
    assert messages[2].content == table
