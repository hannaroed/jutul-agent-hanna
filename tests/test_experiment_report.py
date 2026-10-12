"""Tests for the long-horizon intervention trajectory report."""

from jutul_agent.eval.experiment_report import (
    ExperimentRecord,
    _condition,
    _trace_metrics,
    render_markdown,
)


def _record(**overrides) -> ExperimentRecord:
    values = dict(
        run="2026-10-11T00-34-45-00-00",
        trial=1,
        condition="Targeted retrieval + path-contract fix",
        task="experiment_targeted",
        model="openai/Qwen/Qwen3-8B",
        completed=True,
        protocol_correct=True,
        winner_correct=True,
        numeric_correct=True,
        grounded_evidence=True,
        tool_calls=8,
        file_ops=2,
        compactions=2,
        filesystem_result_chars=1198,
        total_tokens=203_763,
        wall_s=183.0,
        context_window=16_384,
        summary_prompt_sha256="d30558fafdd690929105883cc5e3b61c",
        summary_sanitization=None,
        memory_strategy="structured-ledger-targeted-retrieval",
        path_policy="agent-visible-to-real-path-resolution-v1",
        tool_scheduling=None,
    )
    values.update(overrides)
    return ExperimentRecord(**values)


def test_condition_labels_recorded_interventions() -> None:
    assert _condition("jutul_agent/experiment_state", {}, {}) == "Unaware 16k baseline"
    assert (
        _condition(
            "jutul_agent/experiment_state",
            {"context_window": 16_384},
            {"context_window": 16_384},
        )
        == "Window-aware generic compaction"
    )
    assert (
        _condition(
            "jutul_agent/experiment_targeted",
            {"memory_strategy": "structured-ledger-targeted-retrieval"},
            {"path_policy": "agent-visible-to-real-path-resolution-v1"},
        )
        == "Targeted retrieval + path-contract fix"
    )
    assert (
        _condition(
            "jutul_agent/experiment_targeted",
            {"memory_strategy": "structured-ledger-targeted-retrieval-stable-locators"},
            {"path_policy": "agent-visible-to-real-path-resolution-v1"},
        )
        == "Targeted retrieval + verified artifact locators"
    )
    assert (
        _condition(
            "jutul_agent/experiment_autonomous",
            {"verification_strategy": "tool-returned-completion-gate-v1"},
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous campaign + completion marker"
    )
    assert (
        _condition(
            "jutul_agent/experiment_autonomous",
            {
                "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
                "answer_precision_policy": "six-significant-digits-value-specific-v1",
            },
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous campaign + action-aware compaction + 6-digit check"
    )
    assert (
        _condition(
            "jutul_agent/experiment_autonomous",
            {"verification_strategy": "compaction-safe-ordered-verification-gate-v2"},
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous campaign + action-aware compaction"
    )
    assert (
        _condition(
            "jutul_agent/experiment_autonomous",
            {"interaction_strategy": "single-goal-autonomous-loop"},
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous campaign + sequential scheduling"
    )
    assert (
        _condition(
            "jutul_agent/experiment_autonomous",
            {"interaction_strategy": "single-goal-autonomous-loop"},
            {},
        )
        == "Autonomous campaign (parallel scheduling)"
    )
    assert (
        _condition(
            "jutul_agent/experiment_recovery",
            {
                "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
                "recovery_strategy": "checkpoint-read-retry-gate-v2",
            },
            {
                "tool_scheduling": "sequential-provider-enforced-v1",
                "summary_sanitization": "strip-internal-reasoning-capsules-v1",
            },
        )
        == "Autonomous fault recovery + sanitized compaction"
    )
    assert (
        _condition(
            "jutul_agent/experiment_recovery",
            {
                "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
                "recovery_strategy": "checkpoint-read-retry-gate-v2",
            },
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous fault recovery + compaction-safe retry gate"
    )
    assert (
        _condition(
            "jutul_agent/experiment_recovery",
            {"verification_strategy": "compaction-safe-ordered-verification-gate-v2"},
            {"tool_scheduling": "sequential-provider-enforced-v1"},
        )
        == "Autonomous fault recovery + action-aware verification"
    )


def test_render_keeps_intervention_runs_separate() -> None:
    page = render_markdown(
        [
            _record(
                condition="Unaware 16k baseline",
                winner_correct=False,
                numeric_correct=False,
                grounded_evidence=None,
                total_tokens=233_733,
            ),
            _record(),
        ]
    )
    assert page.count("| Unaware 16k baseline |") == 1
    assert page.count("| Targeted retrieval + path-contract fix |") == 1
    assert "203,763" in page
    assert "successive harness interventions or independent trials" in page


def test_render_summarizes_repeated_conditions_without_hiding_trials() -> None:
    page = render_markdown(
        [
            _record(trial=1),
            _record(trial=2, numeric_correct=False, grounded_evidence=False),
            _record(trial=3, numeric_correct=False, grounded_evidence=False),
        ]
    )
    assert "## Replicated-condition summary" in page
    assert (
        "| Targeted retrieval + path-contract fix | 3 | 3/3 (100%) | 3/3 (100%) | "
        "3/3 (100%) | 1/3 (33%) | 1/3 (33%) |" in page
    )
    assert page.count("| Targeted retrieval + path-contract fix |") == 4


def test_returned_execution_does_not_claim_campaign_completion() -> None:
    page = render_markdown(
        [
            _record(
                completed=True,
                protocol_correct=False,
                numeric_correct=False,
                grounded_evidence=False,
            )
        ]
    )
    assert "| 1 | returned | fail |" in page
    assert "campaign unfinished" in page
    assert "adaptive campaigns allow three" in page
    assert "| complete |" not in page


def test_trace_metrics_recover_pre_scorer_efficiency(tmp_path) -> None:
    from types import SimpleNamespace

    from jutul_agent.trace import TraceLog

    trace_path = tmp_path / "trace.sqlite"
    trace = TraceLog(trace_path)
    trace.append("tool_call", {"name": "run_julia"})
    trace.append("tool_call", {"name": "read_file"})
    trace.append("tool_result", {"name": "read_file", "content": "12345"})
    trace.append("context_compaction", {"manual": False})
    trace.close()

    metrics = _trace_metrics(SimpleNamespace(store={"jutul/trace_db": str(trace_path)}))
    assert metrics == {
        "tool_calls": 2,
        "file_ops": 1,
        "compactions": 1,
        "filesystem_chars": 5,
    }
