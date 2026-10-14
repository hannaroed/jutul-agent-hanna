"""Shared data and checks for the JutulDarcy campaign evaluations."""

import json
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer

from jutul_agent.eval.campaign_validation import CSV_COLUMNS
from jutul_agent.eval.scorers import (
    campaign_integrity,
    campaign_winner,
    context_compaction_count,
    file_op_count,
    numeric_values_close,
    tool_call_count,
    tool_result_character_count,
    tool_result_numeric_evidence,
)

CAMPAIGN_FIXTURE = (Path(__file__).parent / "_fixtures" / "campaign.jl").read_text(encoding="utf-8")
CAMPAIGN_CASES = ("run-cedar", "run-falcon", "run-lotus", "run-maple", "run-raven", "run-sable")
CAMPAIGN_WINNER = "run-falcon"
# Recovery factor, final maximum pressure (bar), and cell 150 water saturation.
EXPECTED_VALUES = (0.5118957827378053, 302.587035907436, 0.90042530855945)
PRECISE_TOLERANCES = (5e-7, 5e-4, 5e-7)
CAMPAIGN_ANSWER = {
    "run_id": CAMPAIGN_WINNER,
    "recovery_factor": EXPECTED_VALUES[0],
    "max_pressure_bar": EXPECTED_VALUES[1],
    "cell_water_saturation": EXPECTED_VALUES[2],
}
CAMPAIGN_TARGET = json.dumps(CAMPAIGN_ANSWER, sort_keys=True)
ANSWER_FIELDS = ("recovery_factor", "max_pressure_bar", "cell_water_saturation")
SCORING_POLICY = "campaign-mass-recovery-complete-artifacts-private-evidence-v5"


def campaign_sample(sample: Sample, *, contract: dict) -> Sample:
    """Add shared answer rules and an explicit execution contract."""
    target = json.loads(sample.target)
    format_rule = (
        " Return the final answer as one JSON object with exactly these keys: "
        + ", ".join(f"`{key}`" for key in target)
        + ". Use numbers for available quantities; pressure is in bar. "
        "Use null for unavailable values; never guess a missing number. "
        "Do not add prose or a list of candidate answers."
    )
    direct_rule = (
        " Include/setup belongs in a separate Julia tool call. Each campaign call "
        "must be one direct literal function call (optionally assigned), with no loop, "
        "conditional, eval, or function definition. "
        'Use `output_mode="content"` when grepping for a result row.'
    )
    sample.input += direct_rule
    metadata = dict(sample.metadata or {})
    sample.input += format_rule
    metadata["scoring_policy"] = SCORING_POLICY
    metadata.setdefault("query_cell_id", 150)
    metadata["campaign_contract"] = contract
    metadata["recovery_definition"] = "cumulative-produced-oil-mass/initial-reservoir-oil-mass-v1"
    metadata["pressure_constraint"] = "final-spatial-maximum-only"
    sample.input += (
        " Do not edit campaign.jl, manufacture artifacts, or use shell tools; "
        "use the supplied Julia function and file retrieval tools."
        " Do not read evaluation code, reference answers, run archives, "
        "or other trials' workspaces."
    )
    sample.metadata = metadata
    return sample


def campaign_scorers(
    *,
    winner: str = CAMPAIGN_WINNER,
    tolerance: float | tuple[float, ...] = PRECISE_TOLERANCES,
    evidence_tools: tuple[str, ...] | None = None,
) -> list[Scorer]:
    """Four correctness checks, followed by four cost measurements."""
    evidence_options = {} if evidence_tools is None else {"tools": evidence_tools}
    return [
        campaign_winner(winner),
        numeric_values_close(EXPECTED_VALUES, tolerance, fields=ANSWER_FIELDS),
        tool_result_numeric_evidence(
            (EXPECTED_VALUES[-1],),
            0.001,
            csv_columns=CSV_COLUMNS,
            value_columns=("water_saturation",),
            answer_fields=("cell_water_saturation",),
            artifact=f"results/{winner}.csv",
            row_key={"run_id": winner, "cell_id": "150"},
            **evidence_options,
        ),
        campaign_integrity(),
        tool_call_count(),
        file_op_count(),
        context_compaction_count(),
        tool_result_character_count(("glob", "read_file", "grep")),
    ]
