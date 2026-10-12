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


def test_auto_compact_trigger_tokens() -> None:
    # Mirrors deepagents' stock summarizer default (0.85 of the window).
    assert auto_compact_trigger_tokens(65_536) == int(65_536 * 0.85)
    assert auto_compact_trigger_tokens(200_000) == 170_000
    # No discoverable window → deepagents' fixed fallback.
    assert auto_compact_trigger_tokens(None) == 170_000


def test_scientific_summary_policy_preserves_state_and_retrieval_paths() -> None:
    assert "exact run, case, and attempt identifiers" in SCIENTIFIC_SUMMARY_PROMPT
    assert "every reported comparison metric" in SCIENTIFIC_SUMMARY_PROMPT
    assert "filesystem search/read tools" in SCIENTIFIC_SUMMARY_PROMPT
    assert "exact workspace-relative" in SCIENTIFIC_SUMMARY_PROMPT
    assert "change a filename or extension" in SCIENTIFIC_SUMMARY_PROMPT
    assert "verification gate or pending action" in SCIENTIFIC_SUMMARY_PROMPT
    assert "not complete while a verification-gate action is still" in SCIENTIFIC_SUMMARY_PROMPT

    model = make_scripted_model([scripted_final("summary")])
    middleware = build_scientific_summarization_middleware(model, backend=object())
    assert middleware is not None
    assert middleware.name == "SummarizationMiddleware"


def test_scientific_summary_policy_defers_unresolved_model_specs() -> None:
    assert build_scientific_summarization_middleware("openai:test", backend=object()) is None


@pytest.mark.parametrize("enabled", [True, False])
def test_strategy_switches_control_installed_middleware(tmp_path, monkeypatch, enabled) -> None:
    from deepagents.middleware.summarization import DEEPAGENTS_DEFAULT_SUMMARY_PROMPT
    from langchain.agents.middleware import ContextEditingMiddleware

    from jutul_agent.agent import builder
    from jutul_agent.agent.strategies import HarnessStrategies
    from jutul_agent.agent.summarization import ScientificSummarizationMiddleware
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
        assert any(isinstance(m, ContextEditingMiddleware) for m in middleware) is enabled
        assert any(isinstance(m, SequentialToolCallsMiddleware) for m in middleware) is enabled
        [summary] = [m for m in middleware if isinstance(m, ScientificSummarizationMiddleware)]
        prompt = SCIENTIFIC_SUMMARY_PROMPT if enabled else DEEPAGENTS_DEFAULT_SUMMARY_PROMPT
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
