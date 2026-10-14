"""Choose from real simulation histories after a campaign update."""

from __future__ import annotations

import csv
import io
import json
import random
import re
from itertools import pairwise
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer

from jutul_agent.eval.campaign_validation import CSV_COLUMNS, artifact_rows, csv_rows
from jutul_agent.eval.campaigns import Campaign, load_reference
from jutul_agent.eval.scorers import (
    _answer_object,
    _trace_events,
    _workspace_tool_path,
    context_compaction_count,
    tool_call_count,
    tool_result_numeric_evidence,
)
from jutul_agent.eval.solver import STORE_TRACE_DB, STORE_WORKSPACE, jutul_agent_solver
from jutul_agent.eval.tasks._campaign import CAMPAIGN_CASES
from jutul_agent.eval.trace_checks import finite_number, result_succeeded, tool_exchanges

POLICY = "result-memory-cached-metrics-v5"
PEAK_LIMIT_BAR = 550.0
HISTORY_COLUMNS = ("time_days", "recovery_factor", "max_pressure_bar")
DIAGNOSTIC_COLUMNS = (
    "initial_oil_mass_kg",
    "produced_oil_mass_kg",
    "final_oil_mass_kg",
    "mass_balance_relative_error",
    "peak_pressure_bar",
)
TOLERANCES = {
    "late_oil_yield_kg_m3": 1e-3,
    "peak_pressure_bar": 5e-4,
    "cell_water_saturation": 5e-7,
}


def csv_text(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def reference_inputs(campaign: Campaign, reference: Path, project: Path, trial: int):
    saved = load_reference(campaign, reference, project)
    required = {f"{case}{suffix}.csv" for case in CAMPAIGN_CASES for suffix in ("", "-history")}
    if set(saved.get("diagnostic_sha256", {})) != required:
        raise ValueError("study requires newly calibrated, hashed histories and diagnostics")
    rng = random.Random(2000 + trial)
    names = [f"case-{index:02d}" for index in rng.sample(range(1, 100), 6)]
    cell = campaign.query_cell_id if trial == 1 else rng.randrange(1, 193)
    fixtures, facts, index_rows = {}, {}, []
    for case, name in zip(CAMPAIGN_CASES, names, strict=True):
        rows = artifact_rows(reference / "results" / f"{case}.csv", case)
        history = csv_rows(
            (reference / "diagnostics" / f"{case}-history.csv").read_text(), HISTORY_COLUMNS
        )
        diagnostics = csv_rows(
            (reference / "diagnostics" / f"{case}.csv").read_text(), DIAGNOSTIC_COLUMNS
        )
        if len(diagnostics) != 1:
            raise ValueError("expected one diagnostic row per case")
        diagnostic = diagnostics[0]
        if any(
            not finite_number(float(value))
            for row in [*history, diagnostic]
            for value in row.values()
        ):
            raise ValueError("non-finite reference diagnostic")
        times = [float(row["time_days"]) for row in history]
        if any(a >= b for a, b in pairwise(times)) or times[-1] != 720 or times.count(360) != 1:
            raise ValueError("history requires increasing times and exact days 360 and 720")
        recoveries = [float(row["recovery_factor"]) for row in history]
        if any(not 0 <= value <= 1 for value in recoveries) or any(
            a > b for a, b in pairwise(recoveries)
        ):
            raise ValueError("invalid recovery history")
        start = next(row for row in history if float(row["time_days"]) == 360)
        end = history[-1]
        mass = float(diagnostic["initial_oil_mass_kg"])
        produced = float(diagnostic["produced_oil_mass_kg"])
        final_mass = float(diagnostic["final_oil_mass_kg"])
        peak = float(diagnostic["peak_pressure_bar"])
        balance = abs(mass - final_mass - produced) / mass if mass > 0 else float("inf")
        if (
            mass <= 0
            or produced < 0
            or final_mass < 0
            or balance >= 1e-3
            or abs(balance - float(diagnostic["mass_balance_relative_error"])) > 1e-10
            or abs(produced / mass - float(end["recovery_factor"])) > 1e-7
            or abs(float(end["recovery_factor"]) - float(rows[0]["recovery_factor"])) > 1e-7
            or abs(float(end["max_pressure_bar"]) - float(rows[0]["max_pressure_bar"])) > 1e-4
            or abs(max(150, *(float(row["max_pressure_bar"]) for row in history)) - peak) > 1e-4
        ):
            raise ValueError("history, oil balance and final artifact disagree")
        rate = float(rows[0]["injection_rate_m3_day"])
        value = (
            (float(end["recovery_factor"]) - float(start["recovery_factor"])) * mass / (rate * 360)
        )
        renamed = [{**row, "run_id": name} for row in rows]
        fixtures[f"results/{name}.csv"] = csv_text(CSV_COLUMNS, renamed)
        facts[name] = {
            "late_oil_yield_kg_m3": value,
            "peak_pressure_bar": peak,
            "cell_water_saturation": float(
                next(row for row in rows if int(row["cell_id"]) == cell)["water_saturation"]
            ),
            "start": start,
            "end": end,
            "diagnostic": diagnostic,
        }
        index_rows.append(
            {
                "run_id": name,
                "injection_rate_m3_day": rows[0]["injection_rate_m3_day"],
                "summary_path": f"summaries/{name}.csv",
                "artifact_path": f"results/{name}.csv",
            }
        )
    fixtures["case-index.csv"] = csv_text(
        ("run_id", "injection_rate_m3_day", "summary_path", "artifact_path"),
        index_rows,
    )
    eligible = [name for name in names if facts[name]["peak_pressure_bar"] <= PEAK_LIMIT_BAR]
    ranked = sorted(eligible, key=lambda name: facts[name]["late_oil_yield_kg_m3"], reverse=True)
    if (
        len(ranked) < 2
        or facts[ranked[0]]["late_oil_yield_kg_m3"] - facts[ranked[1]]["late_oil_yield_kg_m3"]
        < 1e-3
    ):
        raise ValueError("study needs two feasible cases and an unambiguous winner")
    excluded, selected = ranked[:2]
    names = [excluded, *(name for name in names if name != excluded)]

    def status(updated):
        return csv_text(
            ("run_id", "status", "version"),
            [
                {
                    "run_id": name,
                    "status": "invalid" if updated and name == excluded else "valid",
                    "version": 2 if updated else 1,
                }
                for name in names
            ],
        )

    expected = {"run_id": selected, **{key: facts[selected][key] for key in TOLERANCES}}
    return fixtures, facts, names, cell, excluded, status(False), status(True), expected, saved


@scorer(metrics=[accuracy()])
def scientific_answer():
    async def score(state, target):
        root = state.store.get(STORE_WORKSPACE)
        path = Path(root) / "answer.json" if root else None
        text = path.read_text() if path and path.is_file() and not path.is_symlink() else ""
        answer = _answer_object(text) if text.lstrip().startswith("{") else None
        truth = json.loads(target.text)
        valid = (
            answer is not None
            and set(answer) == set(truth)
            and answer["run_id"] == truth["run_id"]
            and all(
                finite_number(answer[key]) and abs(answer[key] - truth[key]) <= tolerance
                for key, tolerance in TOLERANCES.items()
            )
        )
        return Score(
            value=CORRECT if valid else INCORRECT,
            explanation="answer.json matches the reference"
            if valid
            else "missing, malformed or incorrect answer.json",
        )

    return score


def observed_lines(state):
    observed = {}
    for pair in tool_exchanges(_trace_events(state)):
        tool = pair.call.payload.get("name")
        if tool not in ("read_file", "grep") or not result_succeeded(pair.result):
            continue
        path = _workspace_tool_path(state, pair.call, "path" if tool == "grep" else "file_path")
        lines = {
            re.sub(r"^\s*\d+(?::\s*|\t| {2})", "", line)
            for line in str(pair.result.payload.get("content", "")).splitlines()
        }
        observed.setdefault(path, set()).update(lines)
    return observed


@scorer(metrics=[accuracy()])
def scientific_sources():
    async def score(state, target):
        observed = observed_lines(state)
        required = state.metadata["source_rows"]
        valid = all(set(lines) <= observed.get(path, set()) for path, lines in required.items())
        return Score(
            value=CORRECT if valid else INCORRECT,
            explanation="canonical source rows retrieved"
            if valid
            else "required source evidence missing",
        )

    return score


@scorer(metrics=[accuracy()])
def study_integrity():
    async def score(state, target):
        workspace = Path(state.store.get(STORE_WORKSPACE)).resolve()
        problems = []
        for name, expected in state.metadata["final_fixtures"].items():
            path = workspace / name
            if path.is_symlink() or not path.is_file() or path.read_text() != expected:
                problems.append(f"input modified or missing: {name}")
        events = _trace_events(state)
        pairs = tool_exchanges(events)
        if len(pairs) != sum(event.kind == "tool_call" for event in events):
            problems.append("unpaired tool calls")
        allowed = {
            "read_file",
            "grep",
            "glob",
            "ls",
            "write_file",
            "edit_file",
            "write_todos",
        }
        trace = state.store.get(STORE_TRACE_DB)
        history_roots = (
            [Path(trace).parent / name for name in ("conversation_history", "large_tool_results")]
            if trace
            else []
        )
        from jutul_agent.paths import SHARED_SKILLS_DIR
        from jutul_agent.simulators.registry import get as simulator

        read_roots = [*history_roots, SHARED_SKILLS_DIR, simulator("jutuldarcy").skills_dir]
        for pair in pairs:
            name = pair.call.payload.get("name")
            args = pair.call.payload.get("args") or {}
            if name not in allowed:
                problems.append(f"unsupported tool: {name}")
            if name in ("read_file", "grep", "glob", "ls", "write_file", "edit_file"):
                key = "path" if name in ("grep", "glob", "ls") else "file_path"
                raw = args.get(key)
                relative = _workspace_tool_path(state, pair.call, key)
                if (
                    name in ("write_file", "edit_file")
                    and relative in state.metadata["final_fixtures"]
                ):
                    problems.append("attempted to edit a supplied input")
                if isinstance(raw, str) and relative is None:
                    path = Path(raw).resolve()
                    if name not in ("read_file", "grep") or not any(
                        path.is_relative_to(root.resolve()) for root in read_roots
                    ):
                        problems.append("file access outside this trial")
        status_lines = set(state.metadata["final_fixtures"]["campaign-status.csv"].splitlines())
        if not status_lines <= observed_lines(state).get("campaign-status.csv", set()):
            problems.append("current campaign status was not retrieved")
        return Score(
            value=INCORRECT if problems else CORRECT,
            explanation="; ".join(dict.fromkeys(problems)) or "inputs and current status verified",
        )

    return score


def result_memory_task(campaign, reference, project, *, phase, strategies, trial):
    if phase not in ("short", "long") or trial < 1:
        raise ValueError("invalid study phase or trial")
    fixtures, facts, names, cell, excluded, initial_status, final_status, expected, saved = (
        reference_inputs(campaign, reference, project, trial)
    )
    fixtures["campaign-status.csv"] = final_status if phase == "short" else initial_status
    objective = (
        "We are comparing six completed JutulDarcy waterflood simulations. Do not rerun them. "
        "Choose the VALID run with highest oil mass produced per prescribed water-injection volume "
        "during days 360-720, with peak pressure over the WHOLE simulation at most 550 bar. "
        "Yields and peak pressures have already been independently computed and checked. "
        "Use these supplied values directly; no calculations or Julia calls are needed. "
        "All task data are in this workspace. "
        "Use the read_file and grep TOOLS for retrieval, never execute or shell commands. "
        "Keep a small working.csv or your own notes here if useful. Do not edit supplied inputs, "
        "use Julia or shell tools, delegate, or read evaluation code, other workspaces "
        "or reference answers.\n"
    )
    final = (
        "Read the current campaign-status.csv. Apply the objective above and retrieve "
        f"cell_id {cell} "
        "from the selected results/<run_id>.csv by calling the tool named grep "
        "with pattern, path and output_mode='content', or the read_file tool. "
        f"grep uses literal text, not regex; pattern ',{cell},' matches the cell column. "
        "For a file header, use read_file with limit=5. "
        "Do not run a grep shell command through execute. Write answer.json in the workspace "
        "root, containing ONLY one "
        "JSON object with exactly these keys: run_id, late_oil_yield_kg_m3, peak_pressure_bar, "
        "cell_water_saturation. Use numerical values with at least six significant digits. "
        "Do not guess missing values. Then reply briefly that the file is ready."
    )
    follow_ups, updates = [], []
    columns = ("run_id", "late_oil_yield_kg_m3", "peak_pressure_bar", "artifact_path")
    summary = [
        {
            "run_id": name,
            "late_oil_yield_kg_m3": facts[name]["late_oil_yield_kg_m3"],
            "peak_pressure_bar": facts[name]["peak_pressure_bar"],
            "artifact_path": f"results/{name}.csv",
        }
        for name in names
    ]
    if phase == "short":
        fixtures["case-summary.csv"] = csv_text(columns, summary)
        prompt = (
            objective
            + "The independently computed yields and peaks are in case-summary.csv. "
            + final
        )
    else:
        for item in summary:
            fixtures[f"summaries/{item['run_id']}.csv"] = csv_text(columns, [item])

        def inspect(name):
            return (
                f"Inspect {name}: read summaries/{name}.csv. "
                "Record its supplied late oil yield and peak pressure. "
                "Keep any useful working notes in the workspace. "
                "We will make the final recommendation after all six cases; "
                "do not write answer.json yet."
            )

        prompt = objective + inspect(names[0])
        for index, name in enumerate(names[1:], 1):
            change = {"campaign-status.csv": final_status} if index == 3 else {}
            notice = (
                f"Campaign update: {excluded} is invalid because its setup is no longer accepted. "
                "campaign-status.csv has been updated. Read it and retain this change.\n"
                if change
                else ""
            )
            follow_ups.append(objective + notice + inspect(name))
            updates.append(change)
        follow_ups.append(objective + final)
        updates.append({})

    def row(item):
        return ",".join(str(value) for value in item.values())

    selected = expected["run_id"]
    metadata = {
        "fixtures": fixtures,
        "final_fixtures": {**fixtures, "campaign-status.csv": final_status},
        "follow_up_prompts": follow_ups,
        "follow_up_fixtures": updates,
        "needs_env": True,
        "julia_project": str(project),
        "simulator": "jutuldarcy",
        "context_window": 16384,
        "harness_strategies": strategies.to_dict(),
        "phase": phase,
        "campaign": campaign.to_dict(),
        "trial": trial,
        "query_cell_id": cell,
        "scoring_policy": POLICY,
        "reference": saved,
        "peak_pressure_limit_bar": PEAK_LIMIT_BAR,
        "time_interval_days": [360, 720],
        "injection_volume_basis": "prescribed-rate-times-duration",
        "metric_source": "independently-calibrated-cached-values",
        "source_rows": {"case-summary.csv": [row(item) for item in summary]}
        if phase == "short"
        else {f"summaries/{item['run_id']}.csv": [row(item)] for item in summary},
    }
    evidence = tool_result_numeric_evidence(
        (expected["cell_water_saturation"],),
        5e-7,
        csv_columns=CSV_COLUMNS,
        value_columns=("water_saturation",),
        answer_fields=("cell_water_saturation",),
        artifact=f"results/{selected}.csv",
        row_key={"run_id": selected, "cell_id": str(cell)},
    )
    sample = Sample(
        id=f"{phase}-{campaign.name}-{trial}",
        input=prompt,
        target=json.dumps(expected),
        metadata=metadata,
    )
    return Task(
        name="result_memory",
        version=POLICY,
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[
            scientific_answer(),
            scientific_sources(),
            evidence,
            study_integrity(),
            tool_call_count(),
            context_compaction_count(),
        ],
        time_limit=900,
        token_limit=750000,
        message_limit=90,
    )
