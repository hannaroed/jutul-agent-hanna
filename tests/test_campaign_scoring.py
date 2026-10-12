"""Campaign grading regressions: plausible answers and traces must not fool us."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.scorer import Target
from inspect_ai.util import Store

from jutul_agent.eval.scorers import (
    campaign_winner,
    julia_case_call_coverage,
    julia_case_recovered_from_ledger,
    julia_function_call_count,
    numeric_values_close,
    tool_result_numeric_evidence,
)
from jutul_agent.eval.tasks._campaign import CSV_COLUMNS
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
    assert (
        await julia_case_call_coverage("run_case", ("run-falcon",))(sample, Target(""))
    ).value == "I"


@pytest.mark.parametrize("status", ["error", None])
async def test_case_coverage_requires_explicit_success(tmp_path, status):
    sample = state(
        tmp_path, exchange("run_julia", {"code": 'run_case("run-falcon")'}, status=status)
    )
    assert (
        await julia_case_call_coverage("run_case", ("run-falcon",))(sample, Target(""))
    ).value == "I"


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
        assert (
            await julia_case_call_coverage("run_case", ("run-falcon",))(sample, Target(""))
        ).value == "I"


@pytest.mark.parametrize(
    "read_status,retry_status,late_read",
    [
        ("error", "success", False),
        ("success", "error", False),
        ("success", "success", True),
        (None, "success", False),
    ],
)
async def test_recovery_requires_successful_read_before_successful_retry(
    tmp_path, read_status, retry_status, late_read
):
    failed = exchange(
        "run_julia", {"code": 'run_case("run-maple")'}, status="error", identifier="first"
    )
    read = exchange(
        "read_file",
        {"file_path": "campaign-index.csv"},
        "run_id,...",
        status=read_status,
        identifier="read",
    )
    retry = exchange(
        "run_julia", {"code": 'run_case("run-maple")'}, status=retry_status, identifier="retry"
    )
    events = failed + (read[:1] + retry + read[1:] if late_read else read + retry)
    assert (
        await julia_case_recovered_from_ledger("run_case", "run-maple")(
            state(tmp_path, events), Target("")
        )
    ).value == "I"


async def test_successful_direct_calls_and_recovery_pass(tmp_path):
    events = exchange(
        "run_julia", {"code": 'run_case("run-maple")'}, status="error", identifier="first"
    )
    events += exchange(
        "read_file", {"file_path": "campaign-index.csv"}, "run_id,...", identifier="read"
    )
    events += exchange(
        "run_julia", {"code": 'summary = run_case("run-maple") # retry'}, identifier="retry"
    )
    sample = state(tmp_path, events)
    assert (
        await julia_case_call_coverage("run_case", ("run-maple",), (2,))(sample, Target(""))
    ).value == "C"
    assert (
        await julia_case_recovered_from_ledger("run_case", "run-maple")(sample, Target(""))
    ).value == "C"


async def test_adaptive_budget_rejects_failed_or_repeated_runs(tmp_path):
    events = exchange("run_julia", {"code": "run_adaptive_case(950)"})
    events += exchange("run_julia", {"code": "run_adaptive_case(950)"}, identifier="b")
    assert (
        await julia_function_call_count("run_adaptive_case", 2)(state(tmp_path, events), Target(""))
    ).value == "I"


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
