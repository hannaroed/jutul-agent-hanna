"""Experimental multi-turn eval."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    answer_cites,
    no_interpreters_via_execute,
    tool_call_count,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()


@task
def multiturn() -> Task:
    samples = [
        Sample(
            id="mt0-direct-control",
            input=(
                "The reference simulation run is named `case-amber-17` and its "
                "final pressure is `182.4 bar`. Return the run name and pressure "
                "as `<run>, <pressure> bar`."
            ),
            target="case-amber-17, 182.4 bar",
        ),
        Sample(
            id="mt1-reference-case",
            input=(
                "We are tracking a small synthetic simulation campaign. The reference "
                "run is named `case-amber-17` and its final pressure is `182.4 bar`. "
                "Keep these exact values for a later comparison. For now, reply only "
                "with `recorded`."
            ),
            target="case-amber-17, 182.4 bar",
            metadata={
                "follow_up_prompts": [
                    (
                        "An unrelated check reports that trial `case-blue-09` used "
                        "64 cells and took 7 nonlinear iterations. Reply only with "
                        "the number of cells."
                    ),
                    (
                        "Return the reference run name and final pressure from the "
                        "first turn, not the unrelated trial. Format the answer as "
                        "`<run>, <pressure> bar`."
                    ),
                ]
            },
        ),
    ]
    return Task(
        dataset=samples,
        solver=jutul_agent_solver(),
        scorer=[
            answer_cites(required=("case-amber-17", "182.4")),
            no_interpreters_via_execute(),
            tool_call_count(),
        ],
        time_limit=600,
        token_limit=200_000,
        message_limit=50,
    )
