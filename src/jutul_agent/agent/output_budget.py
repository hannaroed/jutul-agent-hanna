"""An output budget: large tool output never enters the context.

The principle this tests is that data stays in the Julia runtime and only
small, computed results reach the model. A ``run_julia`` result over the budget
is cut to its start and end, with a notice to compute the quantity in Julia and
print only that. This differs from the stock eviction, which offloads the full
text to a file the agent can page back in: under a budget, the bulk of the
output is gone, so context size no longer grows with the size of the data.

It is a harness condition for controlled comparisons (``HarnessStrategies``),
off in normal sessions.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

# About 2k tokens at ~4 characters per token.
DEFAULT_BUDGET_CHARS = 8_000
# Share of the budget kept from the start of the output; the rest is its end,
# where a REPL prints the return value.
_HEAD_SHARE = 0.75

_NOTICE = (
    "\n\n[output cut: {total} characters, over the {budget}-character output budget; "
    "{dropped} characters from the middle were dropped and are not available. Do not "
    "print large arrays or tables: compute the quantity you need in Julia (a sum, "
    "maximum, mean, a value at one time) and print only that result.]\n\n"
)


def cap_output(text: str, budget: int) -> str:
    """``text`` unchanged if within ``budget``, else its start and end with a notice."""
    if len(text) <= budget:
        return text
    head = int(budget * _HEAD_SHARE)
    tail = budget - head
    dropped = len(text) - head - tail
    notice = _NOTICE.format(total=len(text), budget=budget, dropped=dropped)
    return text[:head] + notice + text[len(text) - tail :]


class OutputBudgetMiddleware(AgentMiddleware):
    """Cap the text of the named tools' results at ``budget`` characters."""

    def __init__(
        self, budget: int = DEFAULT_BUDGET_CHARS, tool_names: tuple[str, ...] = ("run_julia",)
    ):
        super().__init__()
        self.budget = budget
        # Not `tools`: AgentMiddleware.tools is the tools a middleware contributes.
        self.tool_names = tool_names

    def _cap(self, request: Any, result: Any) -> Any:
        name = (getattr(request, "tool_call", None) or {}).get("name")
        if (
            name in self.tool_names
            and isinstance(result, ToolMessage)
            and isinstance(result.content, str)
        ):
            result.content = cap_output(result.content, self.budget)
        return result

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._cap(request, handler(request))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._cap(request, await handler(request))
