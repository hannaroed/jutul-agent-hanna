"""Autonomous campaign recovery from a deterministic transient failure."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_case_call_coverage,
    julia_case_recovered_from_ledger,
    tool_call_matches,
    used_tools,
    workspace_file_count,
    workspace_file_exists,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.eval.tasks._campaign import (
    CAMPAIGN_CASES,
    CAMPAIGN_FIXTURE,
    CAMPAIGN_TARGET,
    campaign_answer_scorers,
    campaign_efficiency_scorers,
    campaign_sample,
)

load_eval_credentials()

_FAULT_CASE = "run-maple"
_RECOVERY_FUNCTION = "run_campaign_case_with_recovery"
_RECOVERY_FIXTURE = (
    CAMPAIGN_FIXTURE
    + """

const campaign_fault_marker = ".campaign-run-maple-failed-once"

function run_campaign_case_with_recovery(name::AbstractString)
    if name == "run-maple" && !isfile(campaign_fault_marker)
        write(campaign_fault_marker, "injected transient failure observed\\n")
        error(
            "Injected transient campaign failure for run-maple. " *
            "recovery_gate_id=campaign-maple-v2 " *
            "recovery_step_1=pending tool=read_file path=campaign-index.csv " *
            "recovery_step_2=pending_after_step_1 tool=run_julia " *
            "function=run_campaign_case_with_recovery case=run-maple " *
            "completion_rule=do_not_answer_or_summarize_until_retry_succeeds_" *
            "and_remaining_cases_complete",
        )
    end
    return run_autonomous_campaign_case(name)
end
"""
)


@task
def experiment_recovery() -> Task:
    """Recover one failed experiment without duplicating completed work."""
    cases = ", ".join(f"`{name}`" for name in CAMPAIGN_CASES)
    counts = tuple(2 if name == _FAULT_CASE else 1 for name in CAMPAIGN_CASES)
    sample = Sample(
        id="experiment-recovery-jutuldarcy",
        input=(
            "Run this scientific campaign end to end without waiting for another "
            "message. The workspace file `campaign.jl` defines a JutulDarcy "
            "campaign with a durable ledger. Include it once. Run these cases in "
            f"the listed order: {cases}. Call "
            f'`{_RECOVERY_FUNCTION}("<run-id>")` with one case per `run_julia` '
            "response, and wait for each result before acting again. The campaign "
            "may contain one recoverable transient failure. If a call fails, do "
            "not restart the campaign or repeat a successful case: read "
            "`campaign-index.csv` to identify the durable completed set, then "
            "retry only the failed case and continue the remaining order. After "
            "six successful result files exist, use `read_file` on the ledger to "
            "select the highest-recovery run whose final maximum reservoir "
            "pressure is at most 320 bar. Use `grep` with literal pattern `,150,` "
            "on only the selected artifact. Report its run ID, exact recovery "
            "factor, maximum pressure in bar, and cell 150 water saturation to at "
            "least six significant digits. Never guess a value absent from tool "
            "output."
        ),
        target=CAMPAIGN_TARGET,
        metadata={
            "fixtures": {"campaign.jl": _RECOVERY_FIXTURE},
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "context_window": 16_384,
            "interaction_strategy": "single-goal-fault-recovery-loop",
            "human_follow_up_count": 0,
            "fault_model": "deterministic-transient-before-run-maple-v1",
            "recovery_strategy": "checkpoint-read-retry-gate-v2",
            "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
            "memory_strategy": "durable-ledger-targeted-retrieval-stable-locators",
            "artifact_locator_strategy": "tool-returned-workspace-relative-paths-v1",
            "path_policy": "agent-visible-to-real-path-resolution-v1",
        },
    )
    return Task(
        dataset=[campaign_sample(sample)],
        solver=jutul_agent_solver(),
        scorer=[
            *campaign_answer_scorers(evidence_tools=("grep",)),
            workspace_file_count("results/*.csv", len(CAMPAIGN_CASES)),
            workspace_file_exists("campaign-index.csv"),
            julia_case_call_coverage(_RECOVERY_FUNCTION, CAMPAIGN_CASES, counts),
            julia_case_recovered_from_ledger(_RECOVERY_FUNCTION, _FAULT_CASE),
            used_tools(["run_julia"]),
            tool_call_matches("read_file", r"campaign-index\.csv"),
            tool_call_matches("grep", r"(?=.*run-falcon\.csv)(?=.*,150,)"),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=120,
    )
