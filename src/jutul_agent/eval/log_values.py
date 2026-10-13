"""Read optional scorer values consistently across evaluation reports."""

from typing import Any

REQUIRED_SCORES = (
    "campaign_winner",
    "numeric_values_close",
    "tool_result_numeric_evidence",
    "campaign_integrity",
)


def score_value(sample: Any, name: str) -> Any:
    """Return a scorer's value, or ``None`` when the scorer is absent."""
    score = (sample.scores or {}).get(name)
    return getattr(score, "value", None)


def numeric_score(sample: Any, name: str) -> int | None:
    """Return a numeric scorer value as an integer, excluding boolean verdicts."""
    value = score_value(sample, name)
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
