"""Long-horizon experiment retrieval with query-directed artifact access."""

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
def experiment_targeted() -> Task:
    """Run the campaign, then retrieve one exact row with a bounded query."""
    follow_ups = ledger_follow_ups()
    follow_ups.append(
        "The campaign is complete. Do not rerun a simulation and do not guess any "
        "value. First use `read_file` on `campaign-index.csv` to select the run with "
        "the highest final recovery factor among runs whose final maximum reservoir "
        "pressure is at most 320 bar. Copy its exact recovery factor, maximum "
        "pressure, and artifact path from that ledger row. Then use the `grep` file "
        "tool on only that artifact with the literal pattern `,150,` to return the "
        "single CSV row for `cell_id` 150. Do not read the entire detailed artifact: "
        "the targeted row must be isolated in grep output. Report the run ID, "
        "recovery factor, maximum pressure in bar, and cell 150 water saturation, "
        "each to at least six significant digits. If the matching row is not shown "
        "in tool output, report that it is unavailable rather than inventing it."
    )
    sample = Sample(
        id="experiment-targeted-jutuldarcy",
        input=LEDGER_INPUT,
        target=CAMPAIGN_TARGET,
        metadata={
            "fixtures": {"campaign.jl": CAMPAIGN_FIXTURE},
            "follow_up_prompts": follow_ups,
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "context_window": 16_384,
            "memory_strategy": "structured-ledger-targeted-retrieval-stable-locators",
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
            used_tools(["run_julia"]),
            used_any_tool(["read_file"]),
            used_any_tool(["grep"]),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )
