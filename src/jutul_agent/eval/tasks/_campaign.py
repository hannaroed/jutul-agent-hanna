"""Shared data and checks for the JutulDarcy campaign evaluations."""

import json
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer

from jutul_agent.eval.scorers import (
    campaign_winner,
    context_compaction_count,
    file_op_count,
    no_interpreters_via_execute,
    numeric_values_close,
    tool_call_count,
    tool_result_character_count,
    tool_result_numeric_evidence,
)

CAMPAIGN_FIXTURE = (Path(__file__).parent / "_fixtures" / "experiment_state.jl").read_text(
    encoding="utf-8"
)
CAMPAIGN_CASES = ("run-cedar", "run-falcon", "run-lotus", "run-maple", "run-raven", "run-sable")
CAMPAIGN_WINNER = "run-falcon"
# Recovery factor, final maximum pressure (bar), and cell 150 water saturation.
EXPECTED_VALUES = (0.5094961204297902, 303.2834880549588, 0.4751061563590662)
PRECISE_TOLERANCES = (5e-7, 5e-4, 5e-7)
EXPECTED_METRICS = (
    f"recovery_factor={EXPECTED_VALUES[0]}; "
    f"max_pressure_bar={EXPECTED_VALUES[1]}; "
    f"cell_150_water_saturation={EXPECTED_VALUES[2]}"
)
CAMPAIGN_TARGET = f"{CAMPAIGN_WINNER}; {EXPECTED_METRICS}"
CSV_COLUMNS = (
    "run_id",
    "porosity",
    "injection_rate_m3_day",
    "recovery_factor",
    "avg_pressure_bar",
    "max_pressure_bar",
    "cell_id",
    "cell_pressure_bar",
    "water_saturation",
)
ANSWER_FIELDS = ("recovery_factor", "max_pressure_bar", "cell_water_saturation")
SCORING_POLICY = "campaign-json-named-values-successful-csv-evidence-v2"

LEDGER_INPUT = (
    "The workspace file `campaign.jl` defines a fixed JutulDarcy waterflood "
    "campaign and a structured experiment ledger. In the persistent Julia "
    "session, include the file and run "
    f'`run_campaign_case_with_ledger("{CAMPAIGN_CASES[0]}")` exactly once. Reply '
    "only `recorded` after the result file and ledger row have been written."
)


def ledger_follow_ups() -> list[str]:
    """One follow-up per remaining case; each task adds its retrieval question."""
    return [
        (
            "Continue the same campaign in the existing Julia session. Run "
            f'`run_campaign_case_with_ledger("{name}")` exactly once. Reply only '
            "`recorded` after the result file and ledger row have been written."
        )
        for name in CAMPAIGN_CASES[1:]
    ]


def campaign_sample(sample: Sample) -> Sample:
    """Apply the versioned final-answer and direct-call protocol to each task."""
    if not sample.target.startswith("{"):
        parts = sample.target.split("; ")
        target = {"run_id": parts[0]}
        target.update(
            {key: float(value) for key, value in (part.split("=", 1) for part in parts[1:])}
        )
        sample.target = json.dumps(target, sort_keys=True)
    target = json.loads(sample.target)
    if "cell_150_water_saturation" in target:
        target["cell_water_saturation"] = target.pop("cell_150_water_saturation")
        sample.target = json.dumps(target, sort_keys=True)
    format_rule = (
        " Return the final answer as one JSON object with exactly these keys: "
        + ", ".join(f"`{key}`" for key in target)
        + ". Use numbers, not strings, for quantities; pressure is in bar. "
        "Do not add prose or a list of candidate answers."
    )
    direct_rule = (
        " Include/setup belongs in a separate Julia tool call. Each campaign call "
        "must be one direct literal function call (optionally assigned), with no loop, "
        "conditional, eval, or function definition."
    )
    sample.input += direct_rule
    metadata = dict(sample.metadata or {})
    if metadata.get("follow_up_prompts"):
        follow_ups = list(metadata["follow_up_prompts"])
        follow_ups[-1] += format_rule
        metadata["follow_up_prompts"] = follow_ups
    else:
        sample.input += format_rule
    metadata["scoring_policy"] = SCORING_POLICY
    metadata.setdefault("query_cell_id", 150)
    sample.metadata = metadata
    return sample


def campaign_answer_scorers(
    *,
    winner: str = CAMPAIGN_WINNER,
    tolerance: float | tuple[float, ...] = PRECISE_TOLERANCES,
    evidence_tools: tuple[str, ...] | None = None,
) -> list[Scorer]:
    """Keep the winner, answer-value, and observed-row checks consistent."""
    evidence_options = {} if evidence_tools is None else {"tools": evidence_tools}
    expected = EXPECTED_VALUES
    fields = ANSWER_FIELDS
    if winner.startswith("adaptive-"):
        expected = (1100, *expected)
        fields = ("injection_rate_m3_day", *fields)
        tolerance = (0, *(tolerance if isinstance(tolerance, tuple) else (tolerance,) * 3))
    return [
        campaign_winner(winner),
        numeric_values_close(expected, tolerance, fields=fields),
        tool_result_numeric_evidence(
            (EXPECTED_VALUES[-1],),
            0.001,
            required_text=(winner, ",150,"),
            csv_columns=CSV_COLUMNS,
            value_columns=("water_saturation",),
            answer_fields=("cell_water_saturation",),
            artifact=f"results/{winner}.csv",
            row_key={"run_id": winner, "cell_id": "150"},
            **evidence_options,
        ),
    ]


def campaign_efficiency_scorers() -> list[Scorer]:
    """Common execution-policy check and trace-backed efficiency measurements."""
    return [
        no_interpreters_via_execute(),
        tool_call_count(),
        file_op_count(),
        context_compaction_count(),
        tool_result_character_count(("glob", "read_file", "grep")),
    ]
