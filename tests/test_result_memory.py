"""Artifact answers, changing status and scientific evidence must agree."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Target
from inspect_ai.util import Store

from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.eval.archive import capture_source, sha256, write_json
from jutul_agent.eval.campaign_validation import CSV_COLUMNS
from jutul_agent.eval.campaigns import CAMPAIGNS, reference_answer
from jutul_agent.eval.result_memory import (
    DIAGNOSTIC_COLUMNS,
    HISTORY_COLUMNS,
    csv_text,
    reference_inputs,
    result_memory_task,
    scientific_answer,
    scientific_sources,
    study_integrity,
)
from jutul_agent.eval.runconfig import julia_environment_sha256
from jutul_agent.eval.solver import _run_prompts
from jutul_agent.eval.tasks._campaign import CAMPAIGN_CASES
from jutul_agent.trace import TraceLog


def test_meeting_notes_can_change_but_model_code_and_archives_are_frozen(tmp_path):
    from jutul_agent.eval.result_study import model_code_sha256

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "research").mkdir()
    code = repo / "src" / "task.py"
    code.write_text("metric = 1\n")
    notes = repo / "research" / "meeting.tex"
    notes.write_text("Initial meeting notes")
    before = capture_source(repo, tmp_path / "sources")
    notes.write_text("Updated meeting results")
    after = capture_source(repo, tmp_path / "sources")
    assert before["sha256"] != after["sha256"]
    assert model_code_sha256(before) == model_code_sha256(after)
    code.write_text("metric = 2\n")
    changed = capture_source(repo, tmp_path / "sources")
    assert model_code_sha256(changed) != model_code_sha256(before)
    Path(changed["archive"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="archive or manifest"):
        model_code_sha256(changed)


@pytest.fixture
def reference(tmp_path):
    campaign = CAMPAIGNS[0]
    project = tmp_path / "project"
    project.mkdir()
    (project / "Project.toml").write_text("[deps]\n")
    (project / "Manifest.toml").write_text("[deps]\n")
    root = tmp_path / "reference"
    root.mkdir()
    shutil.copytree(project, root / "julia-environment")
    (root / "results").mkdir()
    (root / "diagnostics").mkdir()
    (root / "campaign.jl").write_text(campaign.fixture())
    for index, case in enumerate(CAMPAIGN_CASES):
        porosity, rate = campaign.parameters[index]
        mass = 1e8 * porosity
        recovery = 0.4 + index * 0.02
        rows = [
            dict(
                zip(
                    CSV_COLUMNS,
                    (case, porosity, rate, recovery, 250, 250, cell, 250, cell / 193),
                    strict=True,
                )
            )
            for cell in range(1, 193)
        ]
        (root / "results" / f"{case}.csv").write_text(csv_text(CSV_COLUMNS, rows))
        history = [
            dict(zip(HISTORY_COLUMNS, values, strict=True))
            for values in ((360, 0.2, 400), (720, recovery, 250))
        ]
        (root / "diagnostics" / f"{case}-history.csv").write_text(
            csv_text(HISTORY_COLUMNS, history)
        )
        diagnostic = dict(
            zip(
                DIAGNOSTIC_COLUMNS,
                (mass, mass * recovery, mass * (1 - recovery), 0, 400),
                strict=True,
            )
        )
        (root / "diagnostics" / f"{case}.csv").write_text(
            csv_text(DIAGNOSTIC_COLUMNS, [diagnostic])
        )
    write_json(
        root / "reference.json",
        {
            "campaign": campaign.to_dict(),
            "fixture_sha256": sha256(root / "campaign.jl"),
            "manifest_sha256": sha256(project / "Manifest.toml"),
            "julia_environment_sha256": julia_environment_sha256(project),
            "answer": reference_answer(campaign, root),
            "artifact_sha256": {
                case: sha256(root / "results" / f"{case}.csv") for case in CAMPAIGN_CASES
            },
            "diagnostic_sha256": {
                path.name: sha256(path) for path in (root / "diagnostics").iterdir()
            },
        },
    )
    return campaign, root, project


def task(reference, phase="long"):
    campaign, root, project = reference
    return result_memory_task(
        campaign,
        root,
        project,
        phase=phase,
        strategies=HarnessStrategies(summarization="generic"),
        trial=1,
    )


def state(tmp_path, sample, *, answer=None, omit=None, failed=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, content in sample.metadata["final_fixtures"].items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if answer is not None:
        (tmp_path / "answer.json").write_text(answer)
    db = tmp_path / "trace.sqlite"
    source_paths = [*sample.metadata["source_rows"], "campaign-status.csv"]
    with TraceLog(db) as trace:
        for index, name in enumerate(source_paths):
            if name == omit:
                continue
            trace.append(
                "tool_call",
                {
                    "name": "read_file",
                    "id": str(index),
                    "args": {"file_path": str(tmp_path / name)},
                },
            )
            trace.append(
                "tool_result",
                {
                    "name": "read_file",
                    "tool_call_id": str(index),
                    "status": "error" if failed else "success",
                    "content": (tmp_path / name).read_text(),
                },
            )
    store = Store()
    store.set("jutul/workspace", str(tmp_path))
    store.set("jutul/trace_db", str(db))
    return SimpleNamespace(
        store=store, metadata=sample.metadata, output=SimpleNamespace(completion="File ready.")
    )


def test_metric_and_update_choose_the_second_feasible_case(reference):
    fixtures, facts, names, cell, excluded, initial, final, expected, _ = reference_inputs(
        *reference, 1
    )
    ranked = sorted(names, key=lambda name: facts[name]["late_oil_yield_kg_m3"], reverse=True)
    assert names[0] == excluded == ranked[0]
    assert expected["run_id"] == ranked[1]
    assert expected["late_oil_yield_kg_m3"] > 0
    assert cell == 150
    assert f"{excluded},valid,1" in initial
    assert f"{excluded},invalid,2" in final
    assert "case-summary.csv" not in fixtures


def test_short_and_long_have_same_final_question_and_answer(reference):
    short = task(reference, "short").dataset[0]
    long = task(reference).dataset[0]
    assert short.target == long.target
    assert not short.metadata["follow_up_prompts"]
    assert len(long.metadata["follow_up_prompts"]) == 6
    assert "case-summary.csv" not in long.metadata["fixtures"]
    assert len(long.metadata["source_rows"]) == 6
    assert all(name.startswith("summaries/") for name in long.metadata["source_rows"])
    assert all(
        "no calculations or Julia calls" in prompt
        for prompt in [long.input, *long.metadata["follow_up_prompts"]]
    )
    assert (
        short.metadata["final_fixtures"]["campaign-status.csv"]
        == long.metadata["final_fixtures"]["campaign-status.csv"]
    )
    assert "cell_id 150" not in long.input
    assert "cell_id 150" in long.metadata["follow_up_prompts"][-1]
    assert "Campaign update:" in long.metadata["follow_up_prompts"][2]
    assert all("never execute" in prompt for prompt in long.metadata["follow_up_prompts"])


@pytest.mark.parametrize("peak,eligible", [(550, True), (551, False)])
def test_peak_constraint_uses_whole_history_and_accepts_the_boundary(reference, peak, eligible):
    import csv

    campaign, root, project = reference
    before = reference_inputs(campaign, root, project, 1)
    case = CAMPAIGN_CASES[4]
    for file, columns, key in (
        (root / "diagnostics" / f"{case}.csv", DIAGNOSTIC_COLUMNS, "peak_pressure_bar"),
        (root / "diagnostics" / f"{case}-history.csv", HISTORY_COLUMNS, "max_pressure_bar"),
    ):
        rows = list(csv.DictReader(file.read_text().splitlines()))
        rows[0][key] = str(peak)
        file.write_text(csv_text(columns, rows))
    saved = json.loads((root / "reference.json").read_text())
    saved["diagnostic_sha256"] = {
        file.name: sha256(file) for file in (root / "diagnostics").iterdir()
    }
    write_json(root / "reference.json", saved)
    after = reference_inputs(campaign, root, project, 1)
    assert (after[4] == before[4]) == eligible
    assert (after[7]["run_id"] == before[7]["run_id"]) == eligible


@pytest.mark.parametrize(
    "change", ["corrupt", "missing", "unhashed", "inconsistent-mass", "wrong-time"]
)
def test_reference_rejects_bad_diagnostics(reference, change):
    campaign, root, project = reference
    file = root / "diagnostics" / f"{CAMPAIGN_CASES[0]}-history.csv"
    saved = json.loads((root / "reference.json").read_text())
    if change == "missing":
        file.unlink()
    elif change == "unhashed":
        saved.pop("diagnostic_sha256")
    elif change == "inconsistent-mass":
        file = root / "diagnostics" / f"{CAMPAIGN_CASES[0]}.csv"
        file.write_text(file.read_text().replace("6400000.0", "999.0"))
        saved["diagnostic_sha256"][file.name] = sha256(file)
    else:
        file.write_text(file.read_text().replace("360", "359"))
        if change == "wrong-time":
            saved["diagnostic_sha256"][file.name] = sha256(file)
    write_json(root / "reference.json", saved)
    with pytest.raises(ValueError):
        reference_inputs(campaign, root, project, 1)


@pytest.mark.parametrize(
    "change",
    [
        "wrong-winner",
        "wrong-units",
        "nan",
        "boolean",
        "extra-field",
        "duplicate",
        "prose",
        "fenced",
        "missing",
    ],
)
async def test_answer_file_rejects_adversarial_answers(reference, tmp_path, change):
    sample = task(reference).dataset[0]
    value = json.loads(sample.target)
    if change == "wrong-winner":
        value["run_id"] = "case-unknown"
    elif change == "wrong-units":
        value["peak_pressure_bar"] *= 100000
    elif change == "nan":
        value["late_oil_yield_kg_m3"] = float("nan")
    elif change == "boolean":
        value["cell_water_saturation"] = True
    elif change == "extra-field":
        value["other_answers"] = [value.copy()]
    text = json.dumps(value)
    if change == "duplicate":
        text = text[:-1] + ', "run_id": "case-other"}'
    elif change == "prose":
        text = "Maybe this: " + text
    elif change == "fenced":
        text = "```json\n" + text + "\n```"
    elif change == "missing":
        text = None
    assert (
        await scientific_answer()(
            state(tmp_path / "work", sample, answer=text), Target(sample.target)
        )
    ).value == "I"


async def test_answer_file_is_scored_independently_of_chat_prose(reference, tmp_path):
    sample = task(reference).dataset[0]
    trial = state(tmp_path / "work", sample, answer=sample.target)
    assert (await scientific_answer()(trial, Target(sample.target))).value == "C"
    assert (await scientific_sources()(trial, Target(sample.target))).value == "C"
    assert (await study_integrity()(trial, Target(sample.target))).value == "C"


@pytest.mark.parametrize(
    "attack",
    [
        "missing-source",
        "failed-read",
        "modified-input",
        "stale-status",
        "private-read",
        "restored-edit",
        "shell-retrieval",
    ],
)
async def test_evidence_and_integrity_reject_false_success(reference, tmp_path, attack):
    sample = task(reference).dataset[0]
    source = next(iter(sample.metadata["source_rows"]))
    trial = state(
        tmp_path / "work",
        sample,
        answer=sample.target,
        omit=source if attack == "missing-source" else None,
        failed=attack == "failed-read",
    )
    if attack == "modified-input":
        (Path(trial.store.get("jutul/workspace")) / source).write_text("fake")
    elif attack == "stale-status":
        path = Path(trial.store.get("jutul/workspace")) / "campaign-status.csv"
        path.write_text(sample.metadata["fixtures"]["campaign-status.csv"])
    elif attack == "private-read":
        with TraceLog(Path(trial.store.get("jutul/trace_db"))) as trace:
            trace.append(
                "tool_call",
                {
                    "id": "private",
                    "name": "read_file",
                    "args": {"file_path": str(tmp_path / "reference.json")},
                },
            )
            trace.append(
                "tool_result",
                {
                    "tool_call_id": "private",
                    "name": "read_file",
                    "status": "error",
                    "content": "Not found",
                },
            )
    if attack == "restored-edit":
        with TraceLog(Path(trial.store.get("jutul/trace_db"))) as trace:
            trace.append(
                "tool_call",
                {
                    "id": "edit",
                    "name": "write_file",
                    "args": {
                        "file_path": str(tmp_path / "work" / "case-index.csv"),
                        "content": "changed",
                    },
                },
            )
            trace.append(
                "tool_result",
                {
                    "tool_call_id": "edit",
                    "name": "write_file",
                    "status": "success",
                    "content": "restored afterwards",
                },
            )
    if attack == "shell-retrieval":
        with TraceLog(Path(trial.store.get("jutul/trace_db"))) as trace:
            trace.append(
                "tool_call",
                {
                    "id": "shell",
                    "name": "execute",
                    "args": {"command": "grep 150 results/case.csv"},
                },
            )
            trace.append(
                "tool_result",
                {"tool_call_id": "shell", "name": "execute", "status": "success", "content": "row"},
            )
    sources = await scientific_sources()(trial, Target(sample.target))
    integrity = await study_integrity()(trial, Target(sample.target))
    assert sources.value == "I" or integrity.value == "I"


async def test_follow_ups_reuse_runner_and_apply_update_before_prompt(tmp_path):
    calls = []

    async def run_prompt(text):
        calls.append(
            (
                text,
                (tmp_path / "status.csv").read_text()
                if (tmp_path / "status.csv").exists()
                else None,
            )
        )
        return SimpleNamespace(interrupts=[], messages=[])

    host = SimpleNamespace(runner=SimpleNamespace(run_prompt=run_prompt))
    await _run_prompts(host, tmp_path, "first", ["next", "last"], [{"status.csv": "updated"}, {}])
    assert calls == [("first", None), ("next", "updated"), ("last", "updated")]


async def test_follow_up_failure_stops_before_later_prompts(tmp_path):
    calls = []

    async def run_prompt(text):
        calls.append(text)
        if text == "next":
            raise RuntimeError("model failure")
        return SimpleNamespace(interrupts=[], messages=[])

    host = SimpleNamespace(runner=SimpleNamespace(run_prompt=run_prompt))
    with pytest.raises(RuntimeError):
        await _run_prompts(host, tmp_path, "first", ["next", "last"], [{}, {}])
    assert calls == ["first", "next"]


async def test_follow_up_escape_rejected_before_model_call(tmp_path):
    async def run_prompt(text):
        pytest.fail("must validate before calling the model")

    host = SimpleNamespace(runner=SimpleNamespace(run_prompt=run_prompt))
    with pytest.raises(ValueError):
        await _run_prompts(host, tmp_path, "first", ["next"], [{"../private": "bad"}])
