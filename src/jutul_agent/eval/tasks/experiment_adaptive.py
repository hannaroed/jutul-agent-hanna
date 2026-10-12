"""Budgeted adaptive experiment selection with real JutulDarcy runs."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from jutul_agent.eval.scorers import (
    julia_function_call_count,
    tool_call_matches,
    used_tools,
    workspace_file_count,
    workspace_file_exists,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials
from jutul_agent.eval.tasks._campaign import (
    CAMPAIGN_FIXTURE,
    EXPECTED_METRICS,
    PRECISE_TOLERANCES,
    campaign_answer_scorers,
    campaign_efficiency_scorers,
    campaign_sample,
)

load_eval_credentials()

_RATES = (650, 800, 950, 1100, 1250, 1400)
_BUDGET = 3
_WINNER = "adaptive-1100"
_ADAPTIVE_FUNCTION = "run_adaptive_case"
_ADAPTIVE_FIXTURE = (
    CAMPAIGN_FIXTURE
    + f"""

const adaptive_candidate_rates = {_RATES!r}
const adaptive_run_budget = {_BUDGET}

function run_adaptive_case(rate::Integer)
    rate in adaptive_candidate_rates || error(
        "rate must be one of $(adaptive_candidate_rates)",
    )
    name = "adaptive-$rate"
    artifact = joinpath("results", "$name.csv")
    isfile(artifact) && error("duplicate adaptive run refused: $name")
    completed = if isdir("results")
        count(
            file -> startswith(file, "adaptive-") && endswith(file, ".csv"),
            readdir("results"),
        )
    else
        0
    end
    completed < adaptive_run_budget || error(
        "adaptive run budget exhausted: $completed/$adaptive_run_budget",
    )
    campaign_cases[name] = (porosity = 0.18, injection_rate = Float64(rate))
    summary = run_campaign_case_with_ledger(name)
    completed = count(
        file -> startswith(file, "adaptive-") && endswith(file, ".csv"),
        readdir("results"),
    )
    println(
        "adaptive_progress completed_count=$completed " *
        "expected_count=$adaptive_run_budget",
    )
    if completed == adaptive_run_budget
        println("verification_gate_id=adaptive-evidence-v1 design_complete=true")
        println(
            "verification_step_1=pending tool=read_file " *
            "path=$(summary.ledger_path) " *
            "purpose=select_pressure_feasible_boundary_and_copy_exact_values",
        )
        println(
            "verification_step_2=pending_after_step_1 tool=grep " *
            "path=selected_artifact_path pattern=,150, " *
            "purpose=retrieve_cell_150_water_saturation",
        )
        println(
            "completion_rule=do_not_answer_until_both_verification_steps_are_observed",
        )
    end
    return merge(summary, (completed_count = completed,))
end
"""
)


@task
def experiment_adaptive() -> Task:
    """Select three experiments to identify the pressure-feasible boundary."""
    rates = ", ".join(f"`{rate}`" for rate in _RATES)
    sample = Sample(
        id="experiment-adaptive-jutuldarcy",
        input=(
            "Act as an autonomous experimental-design agent. The workspace file "
            "`campaign.jl` defines a fixed-geology JutulDarcy waterflood and the "
            f"candidate injection rates {rates} m^3/day. For this fixed geology, "
            "final maximum reservoir pressure is monotone nondecreasing with "
            "injection rate. Your objective is to identify the highest candidate "
            "rate whose final maximum pressure is at most 315 bar. You have exactly "
            f"{_BUDGET} simulation runs. Include `campaign.jl` once, then choose "
            f"each rate adaptively and call `{_ADAPTIVE_FUNCTION}(rate)` in a "
            "separate `run_julia` response. Wait for and reason from each returned "
            "summary before selecting the next rate; never emit multiple tool calls "
            "in one response or batch rates in Julia. Use all three runs, do not "
            "repeat a rate, and establish the feasible boundary rather than merely "
            "finding any feasible point. Then read `campaign-index.csv`, and use "
            "`grep` with literal pattern `,150,` on only the selected artifact. "
            "Report the chosen run ID and injection rate, exact recovery factor, "
            "maximum pressure in bar, and cell 150 water saturation to at least six "
            "significant digits. Never guess values absent from tool output."
        ),
        target=f"{_WINNER}; injection_rate_m3_day=1100; {EXPECTED_METRICS}",
        metadata={
            "fixtures": {"campaign.jl": _ADAPTIVE_FIXTURE},
            "needs_env": True,
            "simulator": "jutuldarcy",
            "candidate_count": len(_RATES),
            "simulation_budget": _BUDGET,
            "context_window": 16_384,
            "interaction_strategy": "single-goal-budgeted-adaptive-design",
            "human_follow_up_count": 0,
            "memory_strategy": "durable-ledger-targeted-retrieval-stable-locators",
            "verification_strategy": "compaction-safe-ordered-verification-gate-v2",
            "answer_precision_policy": "six-significant-digits-value-specific-v1",
            "numeric_tolerances": PRECISE_TOLERANCES,
            "artifact_locator_strategy": "tool-returned-workspace-relative-paths-v1",
            "path_policy": "agent-visible-to-real-path-resolution-v1",
        },
    )
    return Task(
        dataset=[campaign_sample(sample)],
        solver=jutul_agent_solver(),
        scorer=[
            *campaign_answer_scorers(
                winner=_WINNER,
                tolerance=PRECISE_TOLERANCES,
                evidence_tools=("grep",),
            ),
            workspace_file_count("results/adaptive-*.csv", _BUDGET),
            workspace_file_exists("results/adaptive-1100.csv"),
            workspace_file_exists("results/adaptive-1250.csv"),
            workspace_file_exists("campaign-index.csv"),
            julia_function_call_count(_ADAPTIVE_FUNCTION, _BUDGET),
            used_tools(["run_julia"]),
            tool_call_matches("read_file", r"campaign-index\.csv"),
            tool_call_matches("grep", r"(?=.*adaptive-1100\.csv)(?=.*,150,)"),
            *campaign_efficiency_scorers(),
        ],
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )
