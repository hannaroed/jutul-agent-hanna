"""Tests for the /context trigger figure and the manual /compact path.

Auto-compaction is deepagents' stock SummarizationMiddleware (covered by the
builder and recorder tests); this module covers the manual ``compact_thread``,
which drives the same engine non-mutatingly and recoverably.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from jutul_agent.agent.builder import build_agent
from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.agent.summarization import (
    MESSAGE_HISTORY_SANITIZATION_POLICY,
    SCIENTIFIC_SUMMARY_PROMPT,
    SUMMARY_SANITIZATION_POLICY,
    auto_compact_trigger_tokens,
    build_scientific_summarization_middleware,
    compact_thread,
    sanitize_message_history,
    sanitize_summary,
)
from jutul_agent.agent.turns import TurnRunner
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter, make_scripted_model, scripted_final
from jutul_agent.session import Session


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["empty", "thinking-only", "truncated", "tool-call"])
async def test_incomplete_summary_is_rejected_before_context_replacement(
    tmp_path, asynchronous, failure
):
    from deepagents.backends import FilesystemBackend

    from jutul_agent.agent.recovery import ModelOutputError

    response = AIMessage(content="")
    if failure == "thinking-only":
        response = AIMessage(content="<think>Still considering the task.</think>")
    elif failure == "truncated":
        response = AIMessage(
            content="Partial summary", response_metadata={"finish_reason": "length"}
        )
    elif failure == "tool-call":
        response = AIMessage(content="", tool_calls=[{"id": "one", "name": "grep", "args": {}}])
    model = make_scripted_model([response])
    middleware = build_scientific_summarization_middleware(
        model, FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    history = [HumanMessage(content="Keep the updated run status.")]
    with pytest.raises(ModelOutputError):
        if asynchronous:
            await middleware._acreate_summary(history)
        else:
            middleware._create_summary(history)
    assert history == [HumanMessage(content="Keep the updated run status.")]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_complete_summary_is_accepted(tmp_path, asynchronous):
    from deepagents.backends import FilesystemBackend

    model = make_scripted_model([AIMessage(content="Goal: select a valid run. Completed: None.")])
    middleware = build_scientific_summarization_middleware(
        model, FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    history = [HumanMessage(content="Select a valid run.")]
    result = (
        await middleware._acreate_summary(history)
        if asynchronous
        else middleware._create_summary(history)
    )
    assert result == "Goal: select a valid run. Completed: None."


def test_summary_prompt_has_no_media_examples_and_keeps_literal_constraints():
    from jutul_agent.agent.summarization import GENERIC_SUMMARY_PROMPT

    assert len(GENERIC_SUMMARY_PROMPT) < 1000
    assert "{messages}" in GENERIC_SUMMARY_PROMPT
    assert "160 words" in GENERIC_SUMMARY_PROMPT
    assert "search" in GENERIC_SUMMARY_PROMPT and "patterns" in GENERIC_SUMMARY_PROMPT
    assert "/conversation_history/media" not in GENERIC_SUMMARY_PROMPT
    assert "<image" not in SCIENTIFIC_SUMMARY_PROMPT


def test_pinned_request_ignores_recovery_nudges_and_previous_summaries():
    from jutul_agent.agent.summarization import _pin_user_request

    task = HumanMessage(content="Use literal pattern ,73,; pressure limit is 310 bar.")
    history = [
        HumanMessage(content="Earlier task"),
        task,
        HumanMessage(
            content="incorrect generated summary", additional_kwargs={"lc_source": "summarization"}
        ),
        HumanMessage(content="try again", additional_kwargs={"empty_turn_recovery": True}),
    ]
    summary = HumanMessage(
        content="unverified notes", additional_kwargs={"lc_source": "summarization"}
    )
    pinned = _pin_user_request(summary, history)
    assert task.content in pinned.content
    assert "Earlier task" not in pinned.content and "try again" not in pinned.content
    assert pinned.additional_kwargs == summary.additional_kwargs
    assert summary.content == "unverified notes"
    assert _pin_user_request(summary, []) is summary


def test_recorded_qwen_overflow_is_detected_before_model_invocation(tmp_path):
    import json
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.exceptions import ContextOverflowError
    from langchain_openai.chat_models.base import _convert_dict_to_message

    raw = json.loads((Path(__file__).parent / "fixtures/context_overflow_request.json").read_text())
    messages = [_convert_dict_to_message(item) for item in raw["messages"]]
    model = make_scripted_model([])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model, FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    request = ModelRequest(
        model=model,
        messages=messages[1:],
        system_message=messages[0],
        tools=raw["tools"],
        model_settings={"max_tokens": raw["max_tokens"]},
        runtime=MagicMock(),
        state={},
    )
    old_estimate = (
        middleware._count_tokens(request.messages, request.system_message, request.tools) - 1024
    )
    assert old_estimate < middleware._input_budget(request)
    assert middleware._over_budget(request)
    invoked = []
    with pytest.raises(ContextOverflowError):
        middleware._call_with_budget(request, lambda current: invoked.append(current))
    assert invoked == []


@pytest.mark.parametrize("clear_results", [False, True])
async def test_verbatim_user_request_survives_repeated_summaries(
    tmp_path, monkeypatch, clear_results
):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import ToolMessage

    task = HumanMessage(
        content="Pressure <= 310 bar. Use grep pattern `,73,` on only the selected CSV."
    )
    bad_summary = "Use cell 73. The only feasible case is run-cedar. Read fake-image.png."
    model = make_scripted_model([scripted_final(bad_summary), scripted_final(bad_summary)])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        strategies=HarnessStrategies(clear_tool_results=clear_results),
    )
    monkeypatch.setattr(middleware, "_should_summarize", lambda *_: True)
    messages = [task]
    state = {"messages": messages}
    received = []

    async def handler(request):
        summary = request.messages[0]
        assert ",73," in summary.content
        assert "Pressure &lt;= 310 bar" in summary.content
        assert "generated summary is unverified" in summary.content
        assert summary.content.count("User request (verbatim") == 1
        received.append(summary.content)
        return AIMessage(content="ok")

    for identifier in ("one", "two", "three"):
        messages.extend(
            [
                AIMessage(
                    content="", tool_calls=[{"id": identifier, "name": "run_julia", "args": {}}]
                ),
                ToolMessage(content="recorded " + identifier, tool_call_id=identifier),
            ]
        )
        if identifier == "one":
            continue
        request = ModelRequest(
            model=model,
            messages=messages,
            tools=[],
            system_message=None,
            runtime=MagicMock(),
            state=state,
        )
        response = await middleware.awrap_model_call(request, handler)
        state.update(response.command.update)
    assert len(received) == 2
    assert task.content == "Pressure <= 310 bar. Use grep pattern `,73,` on only the selected CSV."


def test_auto_compact_trigger_tokens() -> None:
    # Mirrors deepagents' stock summarizer default (0.85 of the window).
    assert auto_compact_trigger_tokens(65_536) == int(65_536 * 0.85)
    assert auto_compact_trigger_tokens(200_000) == 170_000
    # No discoverable window → deepagents' fixed fallback.
    assert auto_compact_trigger_tokens(None) == 170_000


def test_scientific_summary_policy_preserves_state_and_retrieval_paths() -> None:
    model = make_scripted_model([scripted_final("summary")])
    middleware = build_scientific_summarization_middleware(model, backend=object())
    assert middleware is not None
    assert middleware.name == "ScientificSummarizationMiddleware"


def test_scientific_summary_policy_defers_unresolved_model_specs() -> None:
    assert build_scientific_summarization_middleware("openai:test", backend=object()) is None


async def test_budget_sees_the_same_system_prompt_as_the_model(tmp_path, monkeypatch) -> None:
    from jutul_agent.agent.summarization import ScientificSummarizationMiddleware
    from jutul_agent.lab.fakes import ScriptedChatModel

    monkeypatch.chdir(tmp_path)
    counted = []
    received = []
    original_count = ScientificSummarizationMiddleware._count_tokens
    original_generate = ScriptedChatModel._generate

    def count(self, messages, system_message, tools):
        counted.append(system_message.content)
        return original_count(self, messages, system_message, tools)

    def generate(self, messages, *args, **kwargs):
        received.append(messages[0].content)
        return original_generate(self, messages, *args, **kwargs)

    monkeypatch.setattr(ScientificSummarizationMiddleware, "_count_tokens", count)
    monkeypatch.setattr(ScriptedChatModel, "_generate", generate)
    model = make_scripted_model([scripted_final("done")])
    model.profile = {"max_input_tokens": 65_536}
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    try:
        agent, _ = build_agent(session, model=model)
        inputs = {"messages": [HumanMessage(content="hello")]}
        await agent.ainvoke(inputs)
        assert received and counted
        assert counted[-1] == received[-1]
        assert "<memory_guidelines>" in str(counted[-1])
        assert "write_todos" in str(counted[-1])
    finally:
        session.trace.close()


@pytest.mark.parametrize("enabled", [True, False])
def test_strategy_switches_control_installed_middleware(tmp_path, monkeypatch, enabled) -> None:
    from langchain.agents.middleware import ContextEditingMiddleware

    from jutul_agent.agent import builder
    from jutul_agent.agent.strategies import HarnessStrategies
    from jutul_agent.agent.summarization import (
        GENERIC_SUMMARY_PROMPT,
        ScientificSummarizationMiddleware,
    )
    from jutul_agent.agent.tool_scheduling import SequentialToolCallsMiddleware

    captured = {}

    def capture_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(builder, "create_deep_agent", capture_agent)
    strategies = HarnessStrategies(
        summarization="scientific" if enabled else "generic",
        clear_tool_results=enabled,
        sequential_tools=enabled,
        sanitize_history=enabled,
    )
    session = Session.create(
        julia=FakeJulia(), state_root=tmp_path, simulator=make_fake_adapter(tmp_path)
    )
    try:
        build_agent(
            session,
            model=make_scripted_model([scripted_final("summary")]),
            harness_strategies=strategies,
        )
        middleware = captured["middleware"]
        assert any(isinstance(m, SequentialToolCallsMiddleware) for m in middleware) is enabled
        [summary] = [m for m in middleware if isinstance(m, ScientificSummarizationMiddleware)]
        assert isinstance(summary.context_editor, ContextEditingMiddleware) is enabled
        prompt = SCIENTIFIC_SUMMARY_PROMPT if enabled else GENERIC_SUMMARY_PROMPT
        assert summary._lc_helper.summary_prompt == prompt
        raw = "<think>private</think>visible"
        [message] = summary._build_new_messages_with_path(raw, "/history/session.md")
        assert ("private" not in message.content) is enabled
        recorded = session.trace.first_payload("eval_harness_config")
        assert recorded["summary_prompt"] == prompt
        assert recorded["system_prompt"] == captured["system_prompt"]
    finally:
        session.trace.close()


def test_summary_reinjection_strips_internal_reasoning_capsules() -> None:
    raw = (
        '<think signature="opaque">private reasoning</think>\n'
        "durable scientific state\n"
        "<content-internal>bridge metadata</content-internal>"
    )
    assert sanitize_summary(raw) == "durable scientific state"
    assert SUMMARY_SANITIZATION_POLICY == "strip-internal-reasoning-capsules-v1"

    model = make_scripted_model([scripted_final("summary")])
    middleware = build_scientific_summarization_middleware(model, backend=object())
    [message] = middleware._build_new_messages_with_path(raw, "/history/session.md")
    assert "durable scientific state" in message.content
    assert "private reasoning" not in message.content
    assert "bridge metadata" not in message.content


def test_model_history_strips_capsules_without_mutating_tool_calls_or_state() -> None:
    original = AIMessage(
        content=(
            '<think signature="opaque">long private reasoning</think>'
            "visible decision"
            "<content-internal>bridge metadata</content-internal>"
        ),
        tool_calls=[{"id": "call-1", "name": "run_julia", "args": {"code": "f()"}}],
    )
    human = HumanMessage(content="objective")

    cleaned = sanitize_message_history([human, original])

    assert cleaned[0] is human
    assert cleaned[1] is not original
    assert cleaned[1].content == "visible decision"
    assert cleaned[1].tool_calls == original.tool_calls
    assert "long private reasoning" in original.content
    assert MESSAGE_HISTORY_SANITIZATION_POLICY == (
        "strip-internal-reasoning-capsules-before-model-v1"
    )


async def test_compact_thread_round_trip(tmp_path: Path) -> None:
    """/compact records a summarization event without rewriting the raw log."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    adapter = make_fake_adapter(tmp_path)
    session = Session.create(julia=FakeJulia(), state_root=tmp_path, simulator=adapter)
    # Substantial replies so the summary is meaningfully smaller than the turns
    # it replaces (a few words would summarize to something larger).
    finals = [scripted_final(f"answer {i}: " + "detail " * 80) for i in range(8)]
    ckpt = session.state_dir / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        agent, backend = build_agent(session, model=make_scripted_model(finals), checkpointer=saver)
        runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
        for i in range(6):
            await runner.run_prompt(f"question {i}")

        result = await compact_thread(
            agent,
            thread_id=session.session_id,
            model=make_scripted_model([scripted_final("SUMMARY of earlier work")]),
            backend=backend,
            trace=session.trace,
        )
        assert result is not None
        # 12 messages (6 turns), keep 8 → summarize the oldest 4.
        assert result.messages_summarized == 4
        assert result.messages_kept == 8
        assert result.freed_tokens > 0
        assert result.offloaded is True
        current = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        pinned = current.values["_summarization_event"]["summary_message"]
        assert "question 5" in pinned.content
        assert "generated summary is unverified" in pinned.content

        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        # Non-mutating: the raw conversation log is preserved in full...
        assert len(state.values["messages"]) == 12
        # ...and the compaction is recorded as an event the next turn applies.
        assert state.values["_summarization_event"]["cutoff_index"] == 4
        history_id = state.values["_summarization_session_id"]
        assert history_id

        for i in range(6, 8):
            await runner.run_prompt(f"question {i}")
        again = await compact_thread(
            agent,
            thread_id=session.session_id,
            model=make_scripted_model([scripted_final("SUMMARY of more work")]),
            backend=backend,
            trace=session.trace,
        )
        assert again is not None and again.offloaded
        state = await agent.aget_state({"configurable": {"thread_id": session.session_id}})
        assert state.values["_summarization_session_id"] == history_id
        assert state.values["_summarization_event"]["cutoff_index"] == 8

    # The offloaded turns were written somewhere recoverable under the session.
    offloaded = list((session.state_dir / "conversation_history").glob("*.md"))
    assert len(offloaded) == 1
    history = offloaded[0].read_text(encoding="utf-8")
    assert "question 0" in history and "question 2" in history

    events = [e for e in session.trace.iter_events() if e.kind == "context_compaction"]
    assert sum(e.payload.get("manual") is True for e in events) == 2
    session.finalize()


async def test_compact_thread_skips_short_threads() -> None:
    class _NoState:
        pass

    assert (
        await compact_thread(_NoState(), thread_id="t", model=None, backend=None, trace=None)
    ) is None


async def test_small_window_compacts_history_that_fits_old_retention_but_exceeds_request_budget(
    tmp_path,
):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage, ToolMessage

    model = make_scripted_model(
        [scripted_final("Earlier run completed. Continue with the latest result.")]
    )
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        strategies=HarnessStrategies(clear_tool_results=False),
    )
    messages = [
        HumanMessage(content="goal " * 200),
        AIMessage(content="", tool_calls=[{"id": "old", "name": "run_julia", "args": {}}]),
        ToolMessage(content="old output " * 500, tool_call_id="old"),
        AIMessage(content="", tool_calls=[{"id": "new", "name": "run_julia", "args": {}}]),
        ToolMessage(content="new output " * 160, tool_call_id="new"),
    ]
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="rules " * 6500),
        model_settings={"max_tokens": 3072},
        tools=[],
        runtime=MagicMock(),
        state={"messages": messages},
    )
    count = middleware._count_tokens(messages, request.system_message, request.tools)
    assert count > middleware._input_budget(request)
    captured = []

    async def handler(reduced):
        captured.extend(reduced.messages)
        assert not middleware._over_budget(reduced)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(request, handler)
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "new" for m in captured)
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "old" for m in captured)
    assert "old output" in messages[2].content


async def test_clearing_runs_before_summary_budget_guard(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage, ToolMessage

    model = make_scripted_model([])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        strategies=HarnessStrategies(clear_tool_results=True),
    )
    messages = [HumanMessage(content="goal")]
    for identifier, result in [("a", "older " * 600), ("b", "old " * 900), ("c", "latest")]:
        messages.extend(
            [
                AIMessage(
                    content="", tool_calls=[{"id": identifier, "name": "run_julia", "args": {}}]
                ),
                ToolMessage(content=result, tool_call_id=identifier, name="run_julia"),
            ]
        )
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="rules " * 7200),
        tools=[],
        model_settings={"max_tokens": 3072},
        runtime=MagicMock(),
        state={"messages": messages},
    )
    assert middleware._over_budget(request)

    async def unexpected_summary(*args):
        raise AssertionError("Clearing should leave room without a summary call")

    monkeypatch.setattr(middleware, "_acreate_summary", unexpected_summary)
    observed = []

    async def handler(reduced):
        assert not middleware._over_budget(reduced)
        observed.extend(reduced.messages)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(request, handler)
    results = {m.tool_call_id: m.content for m in observed if isinstance(m, ToolMessage)}
    assert results["a"].startswith("[cleared; first lines only]")
    assert results["b"].startswith("[cleared; first lines only]")
    assert results["c"] == "latest"
    assert messages[2].content.startswith("older")


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("prior_summary", [False, True])
async def test_summary_and_saved_history_keep_results_removed_from_model_request(
    tmp_path, monkeypatch, async_call, prior_summary
):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage, ToolMessage

    from jutul_agent.agent.summarization import _SUMMARY_REQUEST
    from jutul_agent.lab.fakes import ScriptedChatModel

    model = make_scripted_model([scripted_final("Both completed runs are recorded.")])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        strategies=HarnessStrategies(summarization="generic", clear_tool_results=True),
    )
    messages = [HumanMessage(content="Complete the runs, then retrieve the saved result.")]
    for identifier, result in [
        ("failed", "ERROR: UndefVarError: supplied function not defined"),
        ("completed-a", "recorded run_id=completed-a artifact_path=results/a.csv"),
        ("completed-b", "recorded run_id=completed-b artifact_path=results/b.csv"),
        ("latest", "latest output"),
    ]:
        if identifier in ("completed-a", "completed-b"):
            result = "progress\n" + "details " * 400 + "\n" + result
        messages.extend(
            [
                AIMessage(
                    content="", tool_calls=[{"id": identifier, "name": "run_julia", "args": {}}]
                ),
                ToolMessage(content=result, tool_call_id=identifier, name="run_julia"),
            ]
        )
    state = {"messages": messages}
    if prior_summary:
        stale = [HumanMessage(content="obsolete raw history"), AIMessage(content="old reply")]
        messages = [*stale, *messages]
        state = {
            "messages": messages,
            "_summarization_event": {
                "cutoff_index": 2,
                "summary_message": HumanMessage(content="Earlier session summary"),
                "file_path": None,
            },
        }
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="rules " * 7600),
        tools=[],
        model_settings={"max_tokens": 2048},
        runtime=MagicMock(),
        state=state,
    )
    prompts = []
    generate = ScriptedChatModel._generate

    def capture(self, summary_messages, *args, **kwargs):
        prompts.extend(message.content for message in summary_messages)
        return generate(self, summary_messages, *args, **kwargs)

    monkeypatch.setattr(ScriptedChatModel, "_generate", capture)
    monkeypatch.setattr(middleware, "_should_summarize", lambda *_: True)
    effective = middleware._get_effective_messages(request)
    outputs = [m.content for m in effective if isinstance(m, ToolMessage)]
    assert outputs[0] == "ERROR: UndefVarError: supplied function not defined"
    assert outputs[1].startswith("[cleared; first lines only]")
    assert outputs[2].startswith("[cleared; first lines only]")
    assert "recorded run_id=completed" not in outputs[1] + outputs[2]
    assert outputs[3] == "latest output"
    assert _SUMMARY_REQUEST.get() is None
    delivered = []

    def handler(reduced):
        assert not middleware._over_budget(reduced)
        delivered.extend(reduced.messages)
        return AIMessage(content="ok")

    if async_call:

        async def async_handler(reduced):
            return handler(reduced)

        response = await middleware.awrap_model_call(request, async_handler)
    else:
        response = middleware.wrap_model_call(request, handler)

    prompt = "\n".join(prompts)
    [history_path] = (tmp_path / "conversation_history").glob("*.md")
    history = history_path.read_text()
    for text in (prompt, history):
        assert "ERROR: UndefVarError" in text
        assert "recorded run_id=completed-a artifact_path=results/a.csv" in text
        assert "recorded run_id=completed-b artifact_path=results/b.csv" in text
        assert "[cleared]" not in text
        assert "obsolete raw history" not in text
        assert "latest output" not in text
    assert [m.content for m in delivered if isinstance(m, ToolMessage)] == ["latest output"]
    assert response.command.update["_summarization_event"]["cutoff_index"] == (
        9 if prior_summary else 7
    )
    assert _SUMMARY_REQUEST.get() is None
    assert any("recorded run_id=completed-a" in str(m.content) for m in messages)


@pytest.mark.parametrize("async_call", [False, True])
async def test_restored_summary_stops_before_model_call_if_history_exceeds_budget(
    tmp_path, monkeypatch, async_call
):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.exceptions import ContextOverflowError
    from langchain_core.messages import SystemMessage, ToolMessage

    from jutul_agent.agent.summarization import _SUMMARY_REQUEST
    from jutul_agent.lab.fakes import ScriptedChatModel

    model = make_scripted_model([])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model, FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    messages = [HumanMessage(content="goal")]
    for identifier, result in [("old", "output " * 12000), ("latest", "latest result")]:
        messages.extend(
            [
                AIMessage(
                    content="", tool_calls=[{"id": identifier, "name": "run_julia", "args": {}}]
                ),
                ToolMessage(content=result, tool_call_id=identifier, name="run_julia"),
            ]
        )
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="rules " * 7600),
        tools=[],
        model_settings={"max_tokens": 2048},
        runtime=MagicMock(),
        state={"messages": messages},
    )
    monkeypatch.setattr(middleware, "_should_summarize", lambda *_: True)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("An oversized summary must not reach the model")

    monkeypatch.setattr(ScriptedChatModel, "_generate", unexpected_call)
    with pytest.raises(ContextOverflowError, match="Summary input exceeds"):
        if async_call:
            await middleware.awrap_model_call(request, unexpected_call)
        else:
            middleware.wrap_model_call(request, unexpected_call)
    assert _SUMMARY_REQUEST.get() is None
    assert messages[2].content == "output " * 12000


async def test_many_retained_small_results_still_compact_before_exceeding_budget(tmp_path):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage, ToolMessage

    model = make_scripted_model([scripted_final("Earlier results are saved in the history file.")])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model, FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    messages = [HumanMessage(content="Review the recorded results.")]
    for index in range(20):
        identifier = f"result-{index}"
        messages.extend(
            [
                AIMessage(
                    content="", tool_calls=[{"id": identifier, "name": "read_file", "args": {}}]
                ),
                ToolMessage(
                    content="run_id,recovery_factor\n" + "values " * 200,
                    name="read_file",
                    tool_call_id=identifier,
                ),
            ]
        )
    request = ModelRequest(
        model=model,
        messages=messages,
        tools=[],
        system_message=SystemMessage(content="rules " * 7600),
        model_settings={"max_tokens": 2048},
        runtime=MagicMock(),
        state={"messages": messages},
    )
    effective = middleware._get_effective_messages(request)
    assert [m.content for m in effective] == [m.content for m in messages]
    assert middleware._over_budget(request)
    received = []

    async def handler(reduced):
        assert not middleware._over_budget(reduced)
        received.extend(reduced.messages)
        return AIMessage(content="ok")

    response = await middleware.awrap_model_call(request, handler)
    assert response.command.update["_summarization_event"]["cutoff_index"] > 0
    assert len(received) < len(messages)
    assert not any("[cleared" in str(m.content) for m in messages)


@pytest.mark.parametrize("clear_results", [False, True])
async def test_compacted_request_ignores_previous_usage_scaling(
    tmp_path, monkeypatch, clear_results
):
    from unittest.mock import MagicMock

    from deepagents.backends import FilesystemBackend
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_core.messages.utils import count_tokens_approximately

    model = make_scripted_model([])
    model.profile = {"max_input_tokens": 16384}
    middleware = build_scientific_summarization_middleware(
        model,
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        strategies=HarnessStrategies(clear_tool_results=clear_results),
    )
    messages = [
        HumanMessage(content="Earlier simulations completed. Read the current result."),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "latest", "name": "read_file", "args": {"file_path": "results/run.csv"}}
            ],
            usage_metadata={"input_tokens": 14000, "output_tokens": 2048, "total_tokens": 16048},
            response_metadata={"model_provider": "openai"},
        ),
        ToolMessage(content="result " * 200, tool_call_id="latest", name="read_file"),
    ]
    request = ModelRequest(
        model=model,
        messages=messages,
        system_message=SystemMessage(content="rules " * 7600),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a saved result.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                },
            }
        ],
        model_settings={"max_tokens": 2048},
        runtime=MagicMock(),
        state={"messages": messages},
    )
    full_request = [request.system_message, *messages]
    budget = middleware._input_budget(request)
    assert count_tokens_approximately(full_request, tools=request.tools) < budget
    assert (
        count_tokens_approximately(
            full_request, tools=request.tools, use_usage_metadata_scaling=True
        )
        > budget
    )

    async def unexpected_summary(*args):
        raise AssertionError("Previous input/output usage must not inflate the current request")

    monkeypatch.setattr(middleware, "_acreate_summary", unexpected_summary)
    received = []

    async def handler(current):
        assert not middleware._over_budget(current)
        received.extend(current.messages)
        return AIMessage(content="ok")

    await middleware.awrap_model_call(request, handler)
    assert [message.content for message in received] == [message.content for message in messages]
    assert messages[1].usage_metadata["total_tokens"] == 16048
