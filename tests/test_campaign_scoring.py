"""Campaign grading regressions: plausible answers and traces must not fool us."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Target
from inspect_ai.util import Store

from jutul_agent.eval.scorers import (
    _trace_events,
    campaign_winner,
    numeric_values_close,
    tool_result_numeric_evidence,
)
from jutul_agent.eval.tasks._campaign import CSV_COLUMNS
from jutul_agent.eval.trace_checks import case_coverage_problems
from jutul_agent.trace import TraceLog

TRUTH = {
    "run_id": "run-falcon",
    "recovery_factor": 0.51,
    "max_pressure_bar": 303.28,
    "cell_water_saturation": 0.4751,
}
FIELDS = ("recovery_factor", "max_pressure_bar", "cell_water_saturation")


def exchange(name, args, content="ok", *, status="success", identifier="a"):
    return [
        ("tool_call", {"id": identifier, "name": name, "args": args}),
        (
            "tool_result",
            {"tool_call_id": identifier, "name": name, "status": status, "content": content},
        ),
    ]


def state(tmp_path, events=(), answer=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "trace.sqlite"
    with TraceLog(database) as trace:
        for kind, payload in events:
            trace.append(kind, payload)
    store = Store()
    store.set("jutul/trace_db", str(database))
    store.set("jutul/workspace", str(tmp_path))
    return SimpleNamespace(store=store, metadata={}, output=SimpleNamespace(completion=answer))


@pytest.mark.parametrize(
    "answer",
    [
        "run-falcon: 0.51, 303.28, 0.4751",
        json.dumps({**TRUTH, "recovery_factor": 0.4751, "cell_water_saturation": 0.51}),
        json.dumps({**TRUTH, "max_pressure_bar": 30328000}),
        json.dumps({**TRUTH, "recovery_factor": True}),
        json.dumps({**TRUTH, "max_pressure_bar": float("nan")}),
        json.dumps({**TRUTH, "max_pressure_bar": float("inf")}),
        json.dumps({**TRUTH, "cell_water_saturation": None}),
        json.dumps({**TRUTH, "other_candidates": [0.51, 303.28, 0.4751]}),
        '{"run_id":"run-falcon","recovery_factor":0.9,"recovery_factor":0.51,'
        '"max_pressure_bar":303.28,"cell_water_saturation":0.4751}',
    ],
)
async def test_named_answer_rejects_adversarial_numbers(tmp_path, answer):
    score = await numeric_values_close((0.51, 303.28, 0.4751), 0.001, fields=FIELDS)(
        state(tmp_path, answer=answer), Target(json.dumps(TRUTH))
    )
    assert score.value == "I"


async def test_named_answer_and_winner_accept_exact_object(tmp_path):
    sample = state(tmp_path, answer=json.dumps(TRUTH))
    target = Target(json.dumps(TRUTH))
    assert (
        await numeric_values_close((0.51, 303.28, 0.4751), 0.001, fields=FIELDS)(sample, target)
    ).value == "C"
    assert (await campaign_winner("run-falcon")(sample, target)).value == "C"
    sample.output.completion = "The winner is not run-falcon. It is run-maple."
    assert (await campaign_winner("run-falcon")(sample, target)).value == "I"


@pytest.mark.parametrize("recovery,correct", [(0.50950, False), (0.509496, True)])
async def test_named_values_use_quantity_specific_tolerances(tmp_path, recovery, correct):
    truth = {**TRUTH, "recovery_factor": 0.5094961204297902, "max_pressure_bar": 303.2834880549588}
    answer = {**truth, "recovery_factor": recovery, "max_pressure_bar": 303.283}
    score = await numeric_values_close(
        tuple(truth[field] for field in FIELDS), (5e-7, 5e-4, 5e-7), fields=FIELDS
    )(state(tmp_path, answer=json.dumps(answer)), Target(json.dumps(truth)))
    assert score.value == ("C" if correct else "I")


@pytest.mark.parametrize(
    "code",
    [
        '# run_case("run-falcon")',
        '#= nested #= run_case("run-falcon") =# =#',
        'println("run_case(\\"run-falcon\\")")',
        'if false; run_case("run-falcon"); end',
        'for _ in 1:1; run_case("run-falcon"); end',
        'function fake(); run_case("run-falcon"); end',
        'run_case("run-falcon"); run_case("run-falcon")',
    ],
)
async def test_case_coverage_rejects_nonexecuted_or_batched_source(tmp_path, code):
    sample = state(tmp_path, exchange("run_julia", {"code": code}))
    assert case_coverage_problems(_trace_events(sample), "run_case", {"run-falcon": 1})


@pytest.mark.parametrize("status", ["error", None])
async def test_case_coverage_requires_explicit_success(tmp_path, status):
    sample = state(
        tmp_path, exchange("run_julia", {"code": 'run_case("run-falcon")'}, status=status)
    )
    assert case_coverage_problems(_trace_events(sample), "run_case", {"run-falcon": 1})


async def test_case_coverage_rejects_missing_duplicate_and_wrong_results(tmp_path):
    valid = exchange("run_julia", {"code": 'run_case("run-falcon")'})
    for index, events in enumerate(
        (
            valid[:1],
            [*valid, valid[-1]],
            [
                valid[0],
                (
                    "tool_result",
                    {"tool_call_id": "different", "name": "run_julia", "status": "success"},
                ),
            ],
        )
    ):
        sample = state(tmp_path / str(index), events)
        assert case_coverage_problems(_trace_events(sample), "run_case", {"run-falcon": 1})


@pytest.mark.parametrize(
    "attack",
    ["none", "error", "orphan", "wrong-file", "wrong-cell", "wrong-column", "fabricated-row"],
)
async def test_csv_evidence_binds_result_to_artifact_row(tmp_path, attack):
    canonical = "run-falcon,0.18,1100,0.51,250,303.28,150,254,0.4751"
    content = canonical
    if attack == "wrong-cell":
        content = content.replace(",150,", ",149,")
    if attack == "wrong-column":
        content = "run-falcon,0.18,1100,0.51,250,303.28,150,0.4751,0.9"
    if attack == "fabricated-row":
        content = content.replace(",250,", ",999,")
    events = exchange(
        "grep",
        {"path": "results/run-maple.csv" if attack == "wrong-file" else "results/run-falcon.csv"},
        "  151: " + content,
        status="error" if attack == "error" else "success",
    )
    if attack == "orphan":
        events = events[1:]
    sample = state(tmp_path, events)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/run-falcon.csv").write_text(
        ",".join(CSV_COLUMNS) + "\n" + canonical + "\n"
    )
    score = await tool_result_numeric_evidence(
        (0.4751,),
        0.001,
        tools=("grep",),
        csv_columns=CSV_COLUMNS,
        value_columns=("water_saturation",),
        answer_fields=("cell_water_saturation",),
        artifact="results/run-falcon.csv",
        row_key={"run_id": "run-falcon", "cell_id": "150"},
    )(sample, Target(json.dumps(TRUTH)))
    assert score.value == ("C" if attack == "none" else "I")


@pytest.mark.parametrize("literal", ["01", '"\\q"', '"$case"', '"$(run_case())"', "9" * 400])
def test_literal_parser_rejects_ambiguous_or_unrepresentable_values(literal):
    from jutul_agent.eval.trace_checks import literal_julia_call

    assert literal_julia_call(f"run_case({literal})") is None


def campaign_state(tmp_path, attack=None):
    from jutul_agent.eval.campaign_validation import LEDGER_COLUMNS

    fixture = "fixture source\n"
    (tmp_path / "campaign.jl").write_text(fixture)
    (tmp_path / "results").mkdir()
    case = "run-falcon"
    rows = [f"{case},0.18,1100,0.51,250,260,1,240,0.4", f"{case},0.18,1100,0.51,250,260,2,260,0.6"]
    artifact = ",".join(CSV_COLUMNS) + "\n" + "\n".join(rows) + "\n"
    (tmp_path / f"results/{case}.csv").write_text(artifact)
    ledger = ",".join(LEDGER_COLUMNS) + "\n" + f"{case},0.18,1100,0.51,250,260,results/{case}.csv\n"
    (tmp_path / "campaign-index.csv").write_text(ledger)
    setup = exchange("run_julia", {"code": 'include("campaign.jl")'}, identifier="setup")
    run = exchange("run_julia", {"code": 'run_case("run-falcon")'}, identifier="run")

    read = exchange(
        "read_file",
        {"file_path": "campaign-index.csv"},
        "@@ 7 total lines | showing lines 1-7 @@\n" + ledger,
        identifier="read",
    )
    grep = exchange(
        "grep",
        {"path": "results/run-falcon.csv", "pattern": ",2,"},
        "  3: " + rows[1],
        identifier="grep",
    )
    if attack == "early-read":
        events = setup + read + run + grep
    elif attack == "early-grep":
        events = setup + run + grep + read
    else:
        events = setup + run + read + grep
    if attack == "failed-run":
        run[-1][1]["status"] = "error"
    if attack == "fake-ledger-read":
        read[-1][1]["content"] = ",".join(LEDGER_COLUMNS) + "\n"
    if attack == "whole-artifact":
        grep[-1][1]["content"] = "\n".join(rows)
    if attack == "altered-artifact":
        (tmp_path / f"results/{case}.csv").write_text(artifact.replace(",260,0.6", ",260,0.9"))
    if attack == "extra-cell":
        (tmp_path / f"results/{case}.csv").write_text(artifact + rows[1] + "\n")
    if attack == "fixture-edit":
        (tmp_path / "campaign.jl").write_text("modified")
    if attack == "bypass":
        events += exchange("run_julia", {"code": 'run_campaign_case("extra")'}, identifier="extra")
    if attack == "partial-ledger":
        (tmp_path / "campaign-index.csv").write_text(",".join(LEDGER_COLUMNS) + "\n")
    if attack == "shell":
        events += exchange("execute", {"command": "echo fabricated"}, identifier="shell")
    if attack == "orphan":
        events += [("tool_call", {"id": "orphan", "name": "ls", "args": {}})]
    sample = state(tmp_path, events)
    sample.metadata = {
        "fixtures": {"campaign.jl": fixture},
        "query_cell_id": 2,
        "campaign_contract": {
            "function": "run_case",
            "cases": [case],
            "ledger": True,
            "targeted": True,
            "cell_count": 2,
        },
        "reference_artifacts": {
            case: [dict(zip(CSV_COLUMNS, row.split(","), strict=True)) for row in rows]
        },
    }
    return sample


@pytest.mark.parametrize(
    "attack",
    [
        None,
        "early-read",
        "early-grep",
        "failed-run",
        "fake-ledger-read",
        "whole-artifact",
        "altered-artifact",
        "extra-cell",
        "fixture-edit",
        "bypass",
        "partial-ledger",
        "shell",
        "orphan",
    ],
)
async def test_campaign_integrity_rejects_adversarial_workflows(tmp_path, attack):
    from jutul_agent.eval.scorers import campaign_integrity

    sample = campaign_state(tmp_path, attack)
    score = await campaign_integrity()(sample, Target(json.dumps({"run_id": "run-falcon"})))
    assert score.value == ("C" if attack is None else "I"), score.explanation


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_reference_comparison_rejects_nonfinite_values(value, side):
    from jutul_agent.eval.campaign_validation import compare_reference

    row = dict(
        zip(
            CSV_COLUMNS,
            ("run-falcon", "0.18", "1100", "0.51", "250", "303.28", "150", "254", "0.4751"),
            strict=True,
        )
    )
    actual, reference = dict(row), dict(row)
    (actual if side == "actual" else reference)["water_saturation"] = value
    with pytest.raises(ValueError, match="non-finite"):
        compare_reference([actual], [reference])


@pytest.mark.parametrize(
    "access",
    [
        "absolute-read",
        "relative-read",
        "recursive-grep",
        "other-trial",
        "research-notes",
        "previous-study",
    ],
)
async def test_campaign_rejects_access_to_answers_or_other_trial_data(tmp_path, access):
    from jutul_agent.eval.scorers import campaign_integrity

    (tmp_path / "workspace").mkdir()
    sample = campaign_state(tmp_path / "workspace")
    target = Target('{"run_id":"run-falcon"}')
    assert (await campaign_integrity()(sample, target)).value == "C"
    private = tmp_path / "reference"
    private.mkdir()
    answer = private / "answer.json"
    answer.write_text(json.dumps(TRUTH))
    sample.metadata["evaluation_private_roots"] = [str(private)]
    if access == "absolute-read":
        tool, args = "read_file", {"file_path": str(answer)}
    elif access == "relative-read":
        tool, args = "read_file", {"file_path": "../reference/answer.json"}
    elif access == "recursive-grep":
        tool, args = "grep", {"path": str(tmp_path), "pattern": "recovery_factor"}
    elif access in ("research-notes", "previous-study"):
        from jutul_agent.eval import scorers

        project = Path(scorers.__file__).resolve().parents[3]
        path = (
            "research/meeting-notes.txt"
            if access == "research-notes"
            else "output/experiments/old/reference.json"
        )
        tool, args = "read_file", {"file_path": str(project / path)}
    else:
        other = tmp_path / "other-trial"
        other.mkdir()
        sample.store.set("jutul/archive", str(other / "archive"))
        tool, args = "read_file", {"file_path": str(other / "result.json")}
    with TraceLog(Path(sample.store.get("jutul/trace_db"))) as trace:
        for kind, payload in exchange(tool, args, answer.read_text(), identifier="leak"):
            trace.append(kind, payload)
    score = await campaign_integrity()(sample, target)
    assert score.value == "I" and "private evaluation data" in score.explanation


async def test_campaign_can_read_its_own_offloaded_conversation(tmp_path):
    from jutul_agent.eval.scorers import campaign_integrity

    (tmp_path / "workspace").mkdir()
    sample = campaign_state(tmp_path / "workspace")
    private = tmp_path / "state"
    private.mkdir()
    database = private / "trace.sqlite"
    Path(sample.store.get("jutul/trace_db")).replace(database)
    sample.store.set("jutul/trace_db", str(database))
    sample.metadata["evaluation_private_roots"] = [str(private)]
    history = private / "conversation_history/history.md"
    history.parent.mkdir()
    history.write_text("Earlier results from this campaign")
    with TraceLog(database) as trace:
        for kind, payload in exchange(
            "read_file", {"file_path": str(history)}, history.read_text(), identifier="own-history"
        ):
            trace.append(kind, payload)
    score = await campaign_integrity()(sample, Target('{"run_id":"run-falcon"}'))
    assert score.value == "C", score.explanation


@pytest.mark.parametrize("kind", ["skill", "julia-source"])
async def test_campaign_can_read_bundled_simulator_guidance(tmp_path, kind):
    from jutul_agent.eval.scorers import campaign_integrity
    from jutul_agent.paths import PACKAGE_ROOT
    from jutul_agent.simulators.registry import get

    sample = campaign_state(tmp_path)
    path = (
        next(get("jutuldarcy").skills_dir.glob("*/SKILL.md"))
        if kind == "skill"
        else next((PACKAGE_ROOT / "julia_runtime").rglob("*.jl"))
    )
    with TraceLog(Path(sample.store.get("jutul/trace_db"))) as trace:
        for event, payload in exchange(
            "read_file", {"file_path": str(path)}, path.read_text(), identifier="guidance"
        ):
            trace.append(event, payload)
    score = await campaign_integrity()(sample, Target('{"run_id":"run-falcon"}'))
    assert score.value == "C", score.explanation


@pytest.mark.parametrize("failure", ["failed-run", "altered-artifact", "missing-file"])
async def test_valid_campaign_call_failure_is_distinct_from_an_incorrect_answer(tmp_path, failure):
    from jutul_agent.eval.scorers import campaign_integrity

    sample = campaign_state(tmp_path, attack=failure)
    if failure == "missing-file":
        (tmp_path / "results/run-falcon.csv").unlink()
    score = await campaign_integrity()(sample, Target('{"run_id":"run-falcon"}'))
    assert score.value == "I" and score.metadata["runtime_errors"]
