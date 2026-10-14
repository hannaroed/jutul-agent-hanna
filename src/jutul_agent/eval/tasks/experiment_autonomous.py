"""Single-prompt autonomous execution of a real JutulDarcy campaign."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.eval.tasks._campaign import (
    CAMPAIGN_CASES,
    CAMPAIGN_FIXTURE,
    CAMPAIGN_TARGET,
    PRECISE_TOLERANCES,
    campaign_sample,
    campaign_scorers,
)

load_eval_credentials()


def autonomous_sample(
    *,
    pressure_limit_bar: float = 320,
    query_cell_id: int = 150,
    fixture: str = CAMPAIGN_FIXTURE,
    target: str = CAMPAIGN_TARGET,
) -> Sample:
    """Build the campaign question directly from its parameters."""
    cases = ", ".join(f"`{name}`" for name in CAMPAIGN_CASES)
    sample = Sample(
        id="experiment-autonomous-jutuldarcy",
        input=(
            "Run the six supplied JutulDarcy simulations without follow-up messages. "
            f"Required cases: {cases}.\n"
            'First Julia call: use `run_julia` with exactly `include("campaign.jl")`. '
            "Wait for it to succeed. This loads the supplied Julia function "
            "`run_autonomous_campaign_case` into the persistent REPL; it is not a separate tool.\n"
            "Run one case per `run_julia` call, starting with the Julia code "
            '`run_autonomous_campaign_case("run-cedar")`, '
            "waiting for the result before continuing. Do not repeat successful cases.\n"
            "After all six cases have finished:\n"
            "1. Use `read_file` on `campaign-index.csv`. Choose the highest recovery "
            f"factor among runs with final maximum pressure at most {pressure_limit_bar:g} bar. "
            "Copy the recovery factor, maximum pressure, and artifact path from that row.\n"
            "2. Use `grep` on only that artifact with the literal pattern "
            f"`,{query_cell_id},` to retrieve the row for `cell_id` {query_cell_id}. "
            "Do not read the complete detailed artifact.\n"
            "Report the run ID, recovery factor, maximum pressure in bar, and cell "
            f"{query_cell_id} water saturation to at least six significant digits."
        ),
        target=target,
        metadata={
            "fixtures": {"campaign.jl": fixture},
            "query_cell_id": query_cell_id,
            "needs_env": True,
            "simulator": "jutuldarcy",
            "case_count": len(CAMPAIGN_CASES),
            "context_window": 16_384,
        },
    )
    return campaign_sample(
        sample,
        contract=dict(
            function="run_autonomous_campaign_case",
            cases=list(CAMPAIGN_CASES),
        ),
    )


@task
def experiment_autonomous() -> Task:
    """Run six supplied simulations, select from the ledger, and retrieve one row."""
    return Task(
        dataset=[autonomous_sample()],
        solver=jutul_agent_solver(),
        scorer=campaign_scorers(
            tolerance=PRECISE_TOLERANCES,
            evidence_tools=("grep",),
        ),
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )
