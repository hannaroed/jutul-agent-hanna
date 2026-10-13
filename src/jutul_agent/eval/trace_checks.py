"""Strict tool-result pairing and literal Julia campaign-call contracts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from jutul_agent.trace import Event, schema

_JULIA_ERROR = re.compile(r"(?m)^(?:\[stderr\]\s*)?(?:ERROR|LoadError):")
_DIRECT_CALL = re.compile(
    r"(?:[A-Za-z_]\w*\s*=\s*)?([A-Za-z_]\w*)\s*\(\s*"
    r'("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)\s*\)\s*;?'
)


def without_julia_comments(code: str) -> str:
    """Remove line/nested block comments, preserving quoted strings verbatim."""
    parts: list[str] = []
    i = 0
    while i < len(code):
        if code[i] in ('"', "'", "`"):
            start = i
            quote = code[i]
            delimiter = quote * 3 if code.startswith(quote * 3, i) else quote
            i += len(delimiter)
            while i < len(code) and not code.startswith(delimiter, i):
                i += 2 if code[i] == "\\" else 1
            i += len(delimiter)
            parts.append(code[start:i])
        elif code.startswith("#=", i):
            depth = 1
            i += 2
            while i < len(code) and depth:
                if code.startswith("#=", i):
                    depth += 1
                    i += 2
                elif code.startswith("=#", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                return ""  # An unterminated comment cannot execute a campaign call.
            parts.append(" ")
        elif code[i] == "#":
            end = code.find("\n", i)
            i = len(code) if end < 0 else end
            parts.append(" ")
        else:
            parts.append(code[i])
            i += 1
    return "".join(parts)


def literal_julia_call(code: str) -> tuple[str, Any] | None:
    """Parse one direct literal Julia call, optionally assigned."""
    match = _DIRECT_CALL.fullmatch(without_julia_comments(code).strip())
    if match is None:
        return None
    literal = match[2]
    if literal.startswith('"') and "$" in literal:
        return None
    try:
        argument = json.loads(literal)
    except ValueError:
        return None
    if not finite_number(argument) and not isinstance(argument, str):
        return None
    return match[1], argument


def result_succeeded(result: Event) -> bool:
    return (
        result.payload.get("status") == "success"
        and _JULIA_ERROR.search(str(result.payload.get("content", ""))) is None
        and re.match(r"(?i)\s*error\b", str(result.payload.get("content", ""))) is None
    )


@dataclass(frozen=True)
class Exchange:
    start: int
    end: int
    call: Event
    result: Event


def tool_exchanges(events: list[Event]) -> list[Exchange]:
    """Pair unique call/result IDs in order; discard orphans and duplicates."""
    calls: dict[str, list[tuple[int, Event]]] = {}
    results: dict[str, list[tuple[int, Event]]] = {}
    for index, event in enumerate(events):
        if event.kind == schema.TOOL_CALL:
            identifier = str(event.payload.get("id") or "")
            if identifier:
                calls.setdefault(identifier, []).append((index, event))
        elif event.kind == schema.TOOL_RESULT:
            identifier = str(event.payload.get("tool_call_id") or "")
            if identifier:
                results.setdefault(identifier, []).append((index, event))
    paired = []
    for identifier, matches in calls.items():
        outputs = results.get(identifier, [])
        if len(matches) != 1 or len(outputs) != 1:
            continue
        start, call = matches[0]
        end, result = outputs[0]
        if start < end and call.payload.get("name") == result.payload.get("name"):
            paired.append(Exchange(start, end, call, result))
    return sorted(paired, key=lambda pair: pair.start)


def campaign_calls(
    events: list[Event], function: str
) -> tuple[list[tuple[Any, Exchange]], list[str]]:
    """Recognize attempts and require a result before the next campaign call."""
    exchanges = {pair.start: pair for pair in tool_exchanges(events)}
    calls: list[tuple[Any, Exchange]] = []
    problems: list[str] = []
    prior_end = -1
    for index, event in enumerate(events):
        if event.kind != schema.TOOL_CALL or event.payload.get("name") not in (
            "run_julia",
            "plot_julia",
        ):
            continue
        code = str((event.payload.get("args") or {}).get("code", ""))
        literal = literal_julia_call(code)
        if literal is None or literal[0] != function:
            if re.search(rf"\b{re.escape(function)}\s*\(", without_julia_comments(code)):
                problems.append("campaign invocation is not one direct literal call")
            continue
        pair = exchanges.get(index)
        if pair is None:
            problems.append("campaign call has no unique matching result")
            continue
        if index < prior_end:
            problems.append("campaign call scheduled before the prior result")
        prior_end = max(prior_end, pair.end)
        calls.append((literal[1], pair))
    return calls, problems


def case_coverage_problems(
    events: list[Event], function: str, expected: dict[str, int]
) -> list[str]:
    calls, problems = campaign_calls(events, function)
    counts = Counter(argument for argument, _ in calls)
    if counts != Counter(expected):
        problems.append(f"unexpected case call counts: expected {expected}, found {dict(counts)}")
    for case in expected:
        attempts = [pair for argument, pair in calls if argument == case]
        if sum(result_succeeded(pair.result) for pair in attempts) != 1:
            problems.append(f"{case} must have exactly one successful result")
        if attempts and not result_succeeded(attempts[-1].result):
            problems.append(f"{case}'s final attempt failed")
    return problems


def finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
