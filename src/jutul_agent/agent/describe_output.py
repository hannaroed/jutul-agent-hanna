"""``describe_output``: a compact map of saved simulation output.

Given a JLD2 file (or a folder of them) of plain dictionaries and arrays, it
returns the structure, not the data: every key with its type, each array's
element type, shape and value range, short strings and scalars as they are,
the time grid in summary, and, for a folder, what differs between the files.
It runs in the session's own Julia kernel, so it reads exactly what the agent's
``run_julia`` would.

A harness condition for controlled comparisons (``HarnessStrategies``), off in
normal sessions.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from jutul_agent.session import Session

_DESCRIBE_JL = (Path(__file__).parent / "describe_output.jl").read_text()


def make_describe_output_tool(session: Session):
    @tool
    async def describe_output(path: str) -> str:
        """Describe saved simulation output without loading it into the conversation.

        Returns the map of the data: every key, each array's type, shape and
        value range, metadata and units, the time grid, and for a folder of
        runs what differs between them. Use it first, then compute what you
        need with run_julia and print only the result.

        Args:
            path: A .jld2 file, or a folder of .jld2 files (one per run).
        """
        result = await session.julia.eval(_DESCRIBE_JL.replace("__PATH__", _julia_string(path)))
        if result.error:
            return f"ERROR: {result.error}"
        return result.output

    return describe_output


def _julia_string(text: str) -> str:
    """``text`` as a Julia string literal (raw, so backslashes survive)."""
    return 'raw"' + text.replace('"', '\\"') + '"'
