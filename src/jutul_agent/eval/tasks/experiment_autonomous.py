"""Single-prompt autonomous execution of a real JutulDarcy campaign."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_case_call_coverage,
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
    PRECISE_TOLERANCES,
    campaign_answer_scorers,
    campaign_efficiency_scorers,
    campaign_sample,
)

load_eval_credentials()


@task
def experiment_autonomous() -> Task:
    """Give the campaign objective once and let the agent run its own loop."""
    cases = ", ".join(f"`{name}`" for name in CAMPAIGN_CASES)
    sample = Sample(
        id="experiment-autonomous-jutuldarcy",
        input=(
            "Run this scientific campaign end to end without waiting for any "
            "follow-up message. The workspace file `campaign.jl` defines a fixed "
            "JutulDarcy waterflood campaign and a structured experiment ledger. "
            f"The six required cases are {cases}. Include `campaign.jl` once, then "
            "run every required case exactly once with "
            "`run_autonomous_campaign_case`. Run one case per `run_julia` tool "
            "call and wait for its result before choosing the next action. Never "
            "emit multiple tool calls in one response, batch cases in a Julia "
            "loop, or repeat a successful case. "
            "Use the persistent Julia session and the returned workspace-relative "
            "ledger and artifact paths as durable campaign state. After all six "
            "result files exist, do not rerun any simulation. Use `read_file` on "
            "`campaign-index.csv` to select the run with the highest final recovery "
            "factor among runs whose final maximum reservoir pressure is at most "
            "320 bar. Copy its exact recovery factor, maximum pressure, and artifact "
            "path from that ledger row. Then use the `grep` file tool on only that "
            "artifact with the literal pattern `,150,` to isolate the single CSV "
            "row for `cell_id` 150; do not read the complete detailed artifact. "
            "Finally report the run ID, recovery factor, maximum pressure in bar, "
            "and cell 150 water saturation, each to at least six significant "
            "digits. If a required result is not present in tool output, say it is "
            "unavailable instead of guessing."
        ),
        target=CAMPAIGN_TARGET,
        metadata={
            "fixtures": {"campaign.jl": CAMPAIGN_FIXTURE},
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "context_window": 16_384,
            "interaction_strategy": "single-goal-autonomous-loop",
            "human_follow_up_count": 0,
            "memory_strategy": "structured-ledger-targeted-retrieval-stable-locators",
            "artifact_locator_strategy": "tool-returned-workspace-relative-paths-v1",
            "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
            "answer_precision_policy": "six-significant-digits-value-specific-v1",
            "numeric_tolerances": PRECISE_TOLERANCES,
            "path_policy": "agent-visible-to-real-path-resolution-v1",
        },
    )
    return Task(
        dataset=[campaign_sample(sample)],
        solver=jutul_agent_solver(),
        scorer=[
            *campaign_answer_scorers(
                tolerance=PRECISE_TOLERANCES,
                evidence_tools=("grep",),
            ),
            workspace_file_count("results/*.csv", len(CAMPAIGN_CASES)),
            workspace_file_exists("campaign-index.csv"),
            julia_case_call_coverage("run_autonomous_campaign_case", CAMPAIGN_CASES),
            used_tools(["run_julia"]),
            tool_call_matches("read_file", r"campaign-index\.csv"),
            tool_call_matches("grep", r"(?=.*run-falcon\.csv)(?=.*,150,)"),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )
