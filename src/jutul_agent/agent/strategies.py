"""Explicit harness controls for controlled comparisons.

The defaults are the stock harness, so an agent built without strategies is
exactly the one a normal session gets. Only switches the harness can honour
live here; a benchmark condition joins them when its mechanism exists.

The named conditions (``HarnessStrategies.condition``):

- ``H0`` minimal: the model with one ``run_julia`` tool and a generic prompt.
- ``H1`` default: the stock harness, unchanged.
- ``H2`` budgeted: H1 plus an output budget on ``run_julia``, the
  ``describe_output`` tool and the output-semantics skill.
- ``H2-noskill`` and ``H2-nobudget``: the two ablations of H2.
"""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class HarnessStrategies:
    # Clear older tool results once the context grows (agent.context_editing).
    clear_tool_results: bool = True
    # H0: one run_julia tool and a generic prompt; none of the harness below.
    minimal: bool = False
    # Cap run_julia results at this many characters (agent.output_budget).
    output_budget_chars: int | None = None
    # Offer the describe_output tool (agent.describe_output).
    describe_output: bool = False
    # Load the output-semantics skill (agent/condition_skills).
    output_skill: bool = False

    def __post_init__(self) -> None:
        for name in ("clear_tool_results", "minimal", "describe_output", "output_skill"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        budget = self.output_budget_chars
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError("output_budget_chars must be a positive integer or None")
        if self.minimal and (budget is not None or self.describe_output or self.output_skill):
            raise ValueError("the minimal harness takes no other harness switches")

    def to_dict(self) -> dict[str, bool | int | None]:
        return asdict(self)

    @classmethod
    def condition(cls, name: str) -> "HarnessStrategies":
        """The strategies for a named benchmark condition (case-insensitive)."""
        from jutul_agent.agent.output_budget import DEFAULT_BUDGET_CHARS

        h2 = cls(output_budget_chars=DEFAULT_BUDGET_CHARS, describe_output=True, output_skill=True)
        conditions = {
            "h0": cls(minimal=True),
            "h1": cls(),
            "h2": h2,
            "h2-noskill": replace(h2, output_skill=False),
            "h2-nobudget": replace(h2, output_budget_chars=None),
        }
        try:
            return conditions[name.lower()]
        except KeyError:
            raise ValueError(f"unknown condition {name!r}; one of {sorted(conditions)}") from None
