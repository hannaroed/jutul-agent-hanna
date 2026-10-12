"""Multi-turn experiment-state evaluation with a real JutulDarcy campaign."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import used_tools, workspace_file_count
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


@task
def experiment_state() -> Task:
    follow_ups = [
        (
            "Continue the same campaign in the existing Julia session. Run "
            f'`run_campaign_case("{name}")` exactly once. Reply only `recorded` '
            "after the result file has been written."
        )
        for name in CAMPAIGN_CASES[1:]
    ]
    follow_ups.append(
        "The campaign is complete. Do not rerun any simulation. From the completed "
        "runs, choose the run with the highest final recovery factor among those "
        "whose final maximum reservoir pressure is at most 320 bar. Then retrieve "
        "the final water saturation for cell 150 of that run from the saved CSV. "
        "Report the run ID, recovery factor, maximum pressure in bar, and cell 150 "
        "water saturation, each to at least six significant digits."
    )
    sample = Sample(
        id="experiment-state-jutuldarcy",
        input=(
            "The workspace file `campaign.jl` defines a fixed JutulDarcy waterflood "
            "campaign. In the persistent Julia session, include the file and run "
            f'`run_campaign_case("{CAMPAIGN_CASES[0]}")` exactly once. Keep the summary '
            "available for a later cross-run decision. Reply only `recorded` after "
            "the result file has been written."
        ),
        target=CAMPAIGN_TARGET,
        metadata={
            "fixtures": {"campaign.jl": CAMPAIGN_FIXTURE},
            "follow_up_prompts": follow_ups,
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "memory_strategy": "conversation-summary-only",
            # The bridge cannot discover the server's context limit.
            "context_window": 16_384,
        },
    )
    return Task(
        dataset=[campaign_sample(sample)],
        solver=jutul_agent_solver(),
        scorer=[
            *campaign_answer_scorers(),
            workspace_file_count("results/*.csv", len(CAMPAIGN_CASES)),
            used_tools(["run_julia"]),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=80,
    )
