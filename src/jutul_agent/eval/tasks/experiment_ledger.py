"""Long-horizon experiment retrieval through a structured artifact ledger."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    used_any_tool,
    used_tools,
    workspace_file_count,
    workspace_file_exists,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.eval.tasks._campaign import (
    CAMPAIGN_CASES,
    CAMPAIGN_FIXTURE,
    CAMPAIGN_TARGET,
    LEDGER_INPUT,
    campaign_answer_scorers,
    campaign_efficiency_scorers,
    campaign_sample,
    ledger_follow_ups,
)

load_eval_credentials()


@task
def experiment_ledger() -> Task:
    """Run the campaign, then query a compact ledger and one detailed artifact."""
    follow_ups = ledger_follow_ups()
    follow_ups.append(
        "The campaign is complete. Do not rerun any simulation and do not infer "
        "or guess missing values. Use the filesystem search/read tools—not Julia "
        "or the shell—to inspect `campaign-index.csv`. Choose the run with the "
        "highest final recovery factor among those whose final maximum reservoir "
        "pressure is at most 320 bar. Then inspect only that run's artifact path "
        "from the ledger and retrieve the row whose `cell_id` is 150. Report the "
        "run ID, recovery factor, maximum pressure in bar, and cell 150 water "
        "saturation, each to at least six significant digits. Do not answer until "
        "you have read the artifact evidence."
    )
    sample = Sample(
        id="experiment-ledger-jutuldarcy",
        input=LEDGER_INPUT,
        target=CAMPAIGN_TARGET,
        metadata={
            "fixtures": {"campaign.jl": CAMPAIGN_FIXTURE},
            "follow_up_prompts": follow_ups,
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "context_window": 16_384,
            "memory_strategy": "structured-artifact-ledger",
        },
    )
    return Task(
        dataset=[campaign_sample(sample)],
        solver=jutul_agent_solver(),
        scorer=[
            *campaign_answer_scorers(),
            workspace_file_count("results/*.csv", len(CAMPAIGN_CASES)),
            workspace_file_exists("campaign-index.csv"),
            used_tools(["run_julia"]),
            used_any_tool(["read_file", "grep"]),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )
