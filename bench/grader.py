"""Unit-aware grading of benchmark answers.

An answer is a JSON object: ``{"value": 167.3, "unit": "bar"}`` for a
quantity, ``{"choice": "PROD2"}`` for a categorical answer. The grader converts
the value to the expected unit, so a correct answer in Pa is not marked wrong
for not being in bar. Default tolerance: 1% relative (bench plan: tolerances
follow physics, not digit matching).

The unit table below is deliberately small and explicit: only the units these
questions use, each with its dimension and its factor to SI. Nothing here
imports jutul_agent, so any harness can grade with it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

DAY = 86400.0
YEAR = 365.0 * DAY

# unit -> (dimension, factor to SI)
_UNITS: dict[str, tuple[str, float]] = {
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1e3),
    "mpa": ("pressure", 1e6),
    "bar": ("pressure", 1e5),
    "psi": ("pressure", 6894.757293168),
    "kg": ("mass", 1.0),
    "g": ("mass", 1e-3),
    "t": ("mass", 1e3),
    "kt": ("mass", 1e6),
    "mt": ("mass", 1e9),
    "kg/s": ("mass_rate", 1.0),
    "kg/day": ("mass_rate", 1.0 / DAY),
    "t/day": ("mass_rate", 1e3 / DAY),
    "t/yr": ("mass_rate", 1e3 / YEAR),
    "m3": ("volume", 1.0),
    "bbl": ("volume", 0.158987294928),
    "m3/s": ("volume_rate", 1.0),
    "m3/day": ("volume_rate", 1.0 / DAY),
    "bbl/day": ("volume_rate", 0.158987294928 / DAY),
    "s": ("time", 1.0),
    "h": ("time", 3600.0),
    "day": ("time", DAY),
    "yr": ("time", YEAR),
    "1": ("dimensionless", 1.0),
    "%": ("dimensionless", 0.01),
    "percentage_point": ("percentage_point", 1.0),
}

_ALIASES = {
    "pascal": "pa",
    "bars": "bar",
    "tonne": "t",
    "tonnes": "t",
    "ton": "t",
    "tons": "t",
    "megatonne": "mt",
    "megatonnes": "mt",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg/d": "kg/day",
    "t/d": "t/day",
    "m^3": "m3",
    "m³": "m3",
    "sm3": "m3",
    "sm^3": "m3",
    "m^3/day": "m3/day",
    "m³/day": "m3/day",
    "m3/d": "m3/day",
    "sm3/day": "m3/day",
    "m^3/s": "m3/s",
    "stb": "bbl",
    "stb/day": "bbl/day",
    "days": "day",
    "d": "day",
    "year": "yr",
    "years": "yr",
    "y": "yr",
    "a": "yr",
    "hour": "h",
    "hours": "h",
    "second": "s",
    "seconds": "s",
    "": "1",
    "-": "1",
    "none": "1",
    "fraction": "1",
    "dimensionless": "1",
    "percent": "%",
    "pct": "%",
    "pp": "percentage_point",
    "percentage points": "percentage_point",
    "percentage point": "percentage_point",
}


def canonical_unit(unit: str | None) -> str:
    raw = (unit or "").strip().lower()
    for key in (raw, raw.replace(" ", "")):
        key = _ALIASES.get(key, key)
        if key in _UNITS:
            return key
    raise ValueError(f"unknown unit {unit!r}")


def convert(value: float, unit: str, to: str) -> float:
    """Convert ``value`` from ``unit`` to ``to``; raise if the dimensions differ."""
    src, dst = canonical_unit(unit), canonical_unit(to)
    # A change in a percentage is often written "%" when percentage points are
    # meant; accept it for a percentage-point answer.
    if dst == "percentage_point" and src == "%":
        return value
    (sdim, sf), (ddim, df) = _UNITS[src], _UNITS[dst]
    if sdim != ddim:
        raise ValueError(f"cannot convert {unit!r} ({sdim}) to {to!r} ({ddim})")
    return value * sf / df


def _is_known_unit(unit: str | None) -> bool:
    try:
        canonical_unit(unit)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class Grade:
    correct: bool
    reason: str
    value_in_expected_unit: float | None = None


def _normalize_choice(text: str) -> str:
    parts = [re.sub(r"[\s_\-]", "", p).lower() for p in str(text).split(",")]
    return ",".join(sorted(p for p in parts if p))


def grade(answer: dict[str, Any] | None, expected: dict[str, Any], spec: dict[str, Any]) -> Grade:
    """Grade one parsed answer against the oracle's ``expected`` record.

    ``spec`` is the question's ``answer`` block (type, unit, optional
    tolerance ``{"rel": 0.01}`` or ``{"abs": 0}``).
    """
    if not isinstance(answer, dict):
        return Grade(False, "no JSON answer found")
    if spec["type"] == "choice":
        if "choice" not in answer:
            return Grade(False, "answer has no 'choice'")
        ok = _normalize_choice(answer["choice"]) == _normalize_choice(expected["choice"])
        return Grade(ok, "match" if ok else f"expected {expected['choice']!r}")
    if "value" not in answer:
        return Grade(False, "answer has no 'value'")
    unit = answer.get("unit", "")
    # A count may come with its noun as the unit ("8 scenarios", "12 wells").
    if spec["unit"] == "1" and not _is_known_unit(unit):
        unit = "1"
    try:
        value = float(answer["value"])
        got = convert(value, unit, spec["unit"])
    except (TypeError, ValueError) as exc:
        return Grade(False, f"unusable value or unit: {exc}")
    want = convert(float(expected["value"]), expected["unit"], spec["unit"])
    tol = spec.get("tolerance") or {"rel": 0.01}
    if "abs" in tol:
        ok = abs(got - want) <= float(tol["abs"])
    else:
        ok = math.isclose(got, want, rel_tol=float(tol["rel"]), abs_tol=0.0)
    return Grade(ok, "within tolerance" if ok else f"expected {want:.6g} {spec['unit']}", got)


def parse_answer(text: str) -> dict[str, Any] | None:
    """The last JSON object in ``text`` that has a value or a choice."""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("value" in obj or "choice" in obj):
            found = obj
    return found
