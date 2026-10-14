"""The comparison matrix keeps its intervention switches and budgets explicit."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")

from jutul_agent.eval.archive import capture_source
from jutul_agent.eval.campaigns import CAMPAIGNS
from jutul_agent.eval.comparison_report import REQUIRED_SCORES, summarize, trial_record
from jutul_agent.eval.comparisons import (
    CONDITIONS,
    advertised_model,
    pending_trials,
    require_batch_provenance,
    trial_order,
)


def test_endpoint_snapshot_ignores_dynamic_permissions_but_keeps_context_limit():
    model = {
        "id": "Qwen/Qwen3-8B",
        "max_model_len": 16384,
        "created": 1,
        "permission": [{"id": "a"}],
    }
    first = advertised_model({"data": [model]}, model["id"])
    model.update(created=2, permission=[{"id": "b"}])
    assert advertised_model({"data": [model]}, model["id"]) == first
    model["max_model_len"] = 32768
    assert advertised_model({"data": [model]}, model["id"]) != first


def server_configuration():
    return {
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "julia_version": "1.11.4",
        "vllm_version": "0.29.0",
        "launch_command": (
            f"vllm serve offline --revision {'a' * 40} --tokenizer-revision {'a' * 40}"
        ),
    }


@pytest.mark.parametrize(
    "missing",
    ["model_revision", "tokenizer_revision", "julia_version", "vllm_version", "launch_command"],
)
def test_batch_refuses_unrecorded_runtime_or_mutable_model_revision(missing):
    configuration = server_configuration()
    require_batch_provenance(configuration)
    configuration[missing] = "main" if "revision" in missing else None
    with pytest.raises(ValueError, match=missing):
        require_batch_provenance(configuration)


@pytest.mark.parametrize("form", ["space", "equals", "wrong-model", "missing-tokenizer"])
def test_recorded_launch_command_must_actually_pin_the_declared_revisions(form):
    configuration = server_configuration()
    if form == "equals":
        configuration["launch_command"] = configuration["launch_command"].replace(
            "revision ", "revision="
        )
    elif form == "wrong-model":
        configuration["model_revision"] = "b" * 40
    elif form == "missing-tokenizer":
        configuration["launch_command"] = f"vllm serve offline --revision {'a' * 40}"
    if form in ("space", "equals"):
        require_batch_provenance(configuration)
    else:
        with pytest.raises(ValueError, match="launch_command must pin"):
            require_batch_provenance(configuration)


def test_qwen_provider_preserves_sampling_and_model_id():
    from inspect_ai.model import GenerateConfig, get_model

    model = get_model(
        "openai-api/vllm/Qwen/Qwen3-8B",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-vllm",
        responses_api=False,
    )
    params = model.api.completion_params(
        GenerateConfig(temperature=0.6, top_p=0.95, max_tokens=4096), tools=True
    )
    assert params == {
        "model": "Qwen/Qwen3-8B",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
    }


def test_matrix_has_three_trials_per_campaign_and_condition():
    order = trial_order(3)
    assert order == trial_order(3)
    assert len(order) == 24
    assert Counter((campaign, condition) for _, campaign, condition in order) == {
        (campaign, condition): 3 for campaign in range(2) for condition in range(4)
    }
    assert {
        (condition.summarization, condition.clear_tool_results) for condition in CONDITIONS
    } == {
        ("generic", False),
        ("generic", True),
        ("scientific", False),
        ("scientific", True),
    }
    assert all(
        condition.sequential_tools and condition.sanitize_history for condition in CONDITIONS
    )
    assert CAMPAIGNS[0].parameters != CAMPAIGNS[1].parameters
    assert CAMPAIGNS[0].query_cell_id != CAMPAIGNS[1].query_cell_id
    from jutul_agent.eval.tasks.experiment_autonomous import autonomous_sample

    for campaign in CAMPAIGNS:
        prompt = autonomous_sample(
            pressure_limit_bar=campaign.pressure_limit_bar, query_cell_id=campaign.query_cell_id
        ).input
        assert f"{campaign.pressure_limit_bar:g} bar" in prompt
        assert f"`,{campaign.query_cell_id},`" in prompt


def test_fixture_and_source_changes_invalidate_saved_configuration(tmp_path):
    fixture = CAMPAIGNS[0].fixture()
    assert "output_substates = true" in fixture
    assert "Liquid_mass_rate" in fixture
    assert "initial_oil_mass" in fixture
    assert (
        hashlib.sha256(fixture.encode()).hexdigest()
        != hashlib.sha256(CAMPAIGNS[1].fixture().encode()).hexdigest()
    )
    (tmp_path / "src").mkdir()
    code = tmp_path / "src/model.py"
    code.write_text("one")
    first = capture_source(tmp_path, tmp_path / "snapshots")
    code.write_text("two")
    assert capture_source(tmp_path, tmp_path / "snapshots")["sha256"] != first["sha256"]


def report_fixture():
    plan = {
        "model": "openai-api/vllm/Qwen/Qwen3-8B",
        "endpoint": "http://127.0.0.1:8000/v1",
        "campaigns": [{"name": "a"}],
        "conditions": [CONDITIONS[0].to_dict()],
        "source": {"sha256": "source-hash"},
        "manifest_sha256": "manifest-hash",
        "context_window": 16384,
        "scoring_policy": "strict",
        "generation": {"temperature": 0.6},
        "limits": {"time_limit": 2400},
        "trials": 3,
    }
    sample = SimpleNamespace(
        metadata={"campaign": {"name": "a"}, "trial": 1, "scoring_policy": "strict"},
        store={
            "jutul/source": {"sha256": "source-hash"},
            "jutul/runconfig": {
                "harness_strategies": CONDITIONS[0].to_dict(),
                "manifest_sha256": "manifest-hash",
                "context_window": 16384,
            },
        },
        scores={name: SimpleNamespace(value="C") for name in REQUIRED_SCORES},
        error=None,
        limit=None,
        model_usage={},
        total_time=10,
    )
    log = SimpleNamespace(
        status="success",
        samples=[sample],
        eval=SimpleNamespace(
            model=plan["model"],
            model_base_url=plan["endpoint"],
            config=SimpleNamespace(time_limit=2400),
        ),
        plan=SimpleNamespace(config=SimpleNamespace(temperature=0.6, seed=1001)),
        location="trial.eval",
    )
    return log, plan


def test_campaign_check_survives_log_serialization_but_rejects_parameter_changes():
    log, plan = report_fixture()
    campaign = CAMPAIGNS[0]
    plan["campaigns"] = [asdict(campaign)]
    log.samples[0].metadata["campaign"] = json.loads(json.dumps(campaign.to_dict()))
    assert trial_record(log, (1, 0, 0), plan)["success"]
    for changed in (
        replace(campaign, pressure_limit_bar=300),
        replace(campaign, query_cell_id=73),
        replace(campaign, parameters=((0.17, 650), *campaign.parameters[1:])),
    ):
        log.samples[0].metadata["campaign"] = changed.to_dict()
        record = trial_record(log, (1, 0, 0), plan)
        assert record["configuration_mismatches"] == ["campaign"]
        assert not record["success"]


@pytest.mark.parametrize("reason", ["max_tokens", "model_length"])
def test_report_marks_truncated_responses_as_failures_even_with_correct_scores(reason):
    log, plan = report_fixture()
    log.samples[0].events = [
        SimpleNamespace(
            event="model", output=SimpleNamespace(choices=[SimpleNamespace(stop_reason=reason)])
        )
    ]
    record = trial_record(log, (1, 0, 0), plan)
    assert record["model_output_errors"] == [f"model response stopped with {reason}"]
    assert not record["success"]
    assert summarize([record], plan)[0]["model_output_failures"] == 1


def test_complete_model_responses_do_not_block_reports():
    log, plan = report_fixture()
    log.samples[0].events = [
        SimpleNamespace(event="tool"),
        SimpleNamespace(event="model", output=None),
        SimpleNamespace(
            event="model",
            output=SimpleNamespace(
                choices=[
                    SimpleNamespace(stop_reason="tool_calls"),
                    SimpleNamespace(stop_reason="stop"),
                ]
            ),
        ),
    ]
    record = trial_record(log, (1, 0, 0), plan)
    assert record["model_output_errors"] == [] and record["success"]


@pytest.mark.parametrize(
    "failure",
    [
        "error",
        "limit",
        "source",
        "sampling",
        "missing-score",
        "log-error",
        "environment",
        "deps",
        "runtime",
    ],
)
def test_report_rejects_false_success_and_retains_failed_attempt(failure):
    log, plan = report_fixture()
    assert trial_record(log, (1, 0, 0), plan)["success"]
    sample = log.samples[0]
    if failure == "error":
        sample.error = SimpleNamespace(message="connection failed")
    elif failure == "limit":
        sample.limit = SimpleNamespace(type="time")
    elif failure == "source":
        sample.store["jutul/source"]["sha256"] = "other-source"
    elif failure == "sampling":
        log.plan.config.temperature = 1.0
    elif failure == "log-error":
        log.status = "error"
    elif failure in ("environment", "deps"):
        name = "julia_environment_sha256" if failure == "environment" else "deps"
        plan[name] = "expected"
        sample.store["jutul/runconfig"][name] = "changed"
    elif failure == "runtime":
        sample.scores["campaign_integrity"].metadata = {"runtime_errors": ["helper failed"]}
    else:
        sample.scores.pop("campaign_integrity")
    failed = trial_record(log, (1, 0, 0), plan)
    assert failed["success"] is False
    group = summarize([failed], plan)[0]
    assert group["attempted"] == 1
    assert group["successes"] == 0
    assert group["pending"] == 2
    assert group["mean_wall_s"] == 10


def matrix_fixture():
    return {
        "campaigns": [{"name": c.name} for c in CAMPAIGNS],
        "conditions": [c.to_dict() for c in CONDITIONS],
        "order": trial_order(3),
    }


@pytest.mark.parametrize("output_tokens", [0, -1, 15564, 16384])
async def test_impossible_output_budget_stops_before_runtime_setup(tmp_path, output_tokens):
    from jutul_agent.eval.comparisons import run

    args = SimpleNamespace(output=tmp_path, julia_project=tmp_path, max_tokens=output_tokens)
    with pytest.raises(ValueError, match="leave room for model input"):
        await run(args)
    assert not (tmp_path / "plan.json").exists()


def completed_records(plan):
    return [
        {
            "trial": trial,
            "campaign": plan["campaigns"][campaign]["name"],
            "condition": condition,
            "success": True,
            "error": None,
            "limit": None,
            "configuration_mismatches": [],
            **{name: "C" for name in REQUIRED_SCORES},
        }
        for trial, campaign, condition in plan["order"]
        if trial == 1
    ]


def test_default_runs_one_trial_and_resumes_one_at_a_time():
    plan = matrix_fixture()
    assert pending_trials(plan, []) == plan["order"][:1]
    assert pending_trials(plan, completed_records(plan)[:1]) == plan["order"][1:2]


def test_batch_is_explicit_and_keeps_valid_incorrect_answers_in_the_study():
    plan = matrix_fixture()
    records = completed_records(plan)
    records[0]["success"] = False
    records[0]["campaign_winner"] = "I"
    assert pending_trials(plan, [], batch=True) == plan["order"]
    assert pending_trials(plan, records, batch=True) == plan["order"][8:]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "failure", ["error", "limit", "configuration", "missing-score", "runtime", "model-output"]
)
def test_technical_failure_blocks_both_default_and_batch_resume(batch, failure):
    plan = matrix_fixture()
    records = completed_records(plan)
    if failure == "configuration":
        records[-1]["configuration_mismatches"] = ["source"]
    elif failure == "missing-score":
        records[-1]["campaign_integrity"] = None
    elif failure == "runtime":
        records[-1]["runtime_errors"] = ["Julia helper failed"]
    elif failure == "model-output":
        records[-1]["model_output_errors"] = ["model response stopped with max_tokens"]
    else:
        records[-1][failure] = "failure"
    with pytest.raises(ValueError, match="previous trial had a technical failure"):
        pending_trials(plan, records, batch=batch)


@pytest.mark.parametrize(
    "failure",
    ["technical", "runtime", "model-output", "incorrect-answer", "exception", "unrecorded-runtime"],
)
async def test_runner_preserves_attempt_and_stops_before_more_compute(
    tmp_path, monkeypatch, failure
):
    import io
    import json

    from jutul_agent.eval import comparisons

    technical_error = failure in ("technical", "runtime", "model-output")

    project = tmp_path / "environment"
    project.mkdir()
    (project / "Manifest.toml").write_text("# offline fixture\n")
    (project / "Project.toml").write_text("[deps]\n")
    root = tmp_path / "study"
    root.mkdir()
    (root / "server.json").write_text(json.dumps(server_configuration()))
    args = SimpleNamespace(
        output=root,
        julia_project=project,
        calibrate=False,
        model="offline",
        endpoint="http://offline/v1",
        trials=3,
        max_tokens=4096,
        batch=True,
        log_model_api=True,
    )
    monkeypatch.setattr(comparisons, "trial_order", lambda _: [(1, 0, 0), (1, 1, 0)])
    monkeypatch.setattr(comparisons, "capture_source", lambda *_: {"sha256": "frozen"})
    monkeypatch.setattr(
        comparisons,
        "get_model",
        lambda *_, **__: SimpleNamespace(api=SimpleNamespace(service_model_name=lambda: "offline")),
    )
    endpoint = {"data": [{"id": "offline", "max_model_len": 16384}]}
    monkeypatch.setattr(
        comparisons, "urlopen", lambda *_, **__: io.BytesIO(json.dumps(endpoint).encode())
    )
    monkeypatch.setattr(
        comparisons,
        "comparison_task",
        lambda *_: SimpleNamespace(dataset=[SimpleNamespace(metadata={})]),
    )
    finalized = []
    monkeypatch.setattr(comparisons, "finalize_log", finalized.append)
    record = {
        "success": False,
        "error": "context overflow" if failure == "technical" else None,
        "runtime_errors": ["Julia helper failed"] if failure == "runtime" else [],
        "model_output_errors": ["max_tokens"] if failure == "model-output" else [],
        "limit": None,
        "configuration_mismatches": [],
        **{name: "I" for name in REQUIRED_SCORES},
    }
    monkeypatch.setattr(comparisons, "trial_record", lambda *_: record)
    calls = []

    async def evaluate(*_, **__):
        assert __["log_model_api"] is True
        assert __["max_tokens"] == 4096
        calls.append(1)
        assert (root / "active-trial.json").exists()
        if failure == "exception":
            raise RuntimeError("provider unavailable")
        sample = SimpleNamespace(error=record["error"], limit=None)
        return [
            SimpleNamespace(
                location=str(root / f"trial-{len(calls)}.eval"), status="success", samples=[sample]
            )
        ]

    monkeypatch.setattr(comparisons, "eval_async", evaluate)
    if failure == "unrecorded-runtime":
        configuration = server_configuration()
        configuration.pop("model_revision")
        (root / "server.json").write_text(json.dumps(configuration))

        def unexpected_model(*_, **__):
            raise AssertionError("model client initialized before provenance validation")

        monkeypatch.setattr(comparisons, "get_model", unexpected_model)
        with pytest.raises(ValueError, match="pinned model_revision"):
            await comparisons.run(args)
        assert not calls and not finalized
        return
    if failure == "exception":
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await comparisons.run(args)
        assert len(calls) == 1
        assert json.loads((root / "active-trial.json").read_text())["key"] == [1, 0, 0]
        with pytest.raises(ValueError, match="unfinished trial"):
            await comparisons.run(args)
        assert len(calls) == 1
        return
    if technical_error:
        with pytest.raises(RuntimeError, match="technical failure"):
            await comparisons.run(args)
        assert len(calls) == len(finalized) == 1
        assert json.loads((root / "stopped.json").read_text()) == record
    else:
        await comparisons.run(args)
        assert len(calls) == len(finalized) == 2
    attempts = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
    assert len(attempts) == len(calls)
    assert attempts[0]["key"] == [1, 0, 0]
    assert not (root / "active-trial.json").exists()
    if technical_error:
        monkeypatch.setattr(
            comparisons,
            "collect",
            lambda _: {
                "trials": [{**record, "trial": 1, "campaign": CAMPAIGNS[0].name, "condition": 0}]
            },
        )
        with pytest.raises(ValueError, match="previous trial had a technical failure"):
            await comparisons.run(args)
        assert len(calls) == 1


@pytest.mark.parametrize("batch", [False, True])
async def test_explicit_provenance_requirement_applies_before_any_model_client(
    tmp_path, monkeypatch, batch
):
    from jutul_agent.eval import comparisons

    project = tmp_path / "environment"
    project.mkdir()
    (project / "Manifest.toml").write_text("[deps]\n")
    (project / "Project.toml").write_text("[deps]\n")

    def unexpected(*args, **kwargs):
        pytest.fail("Model client created without required provenance")

    monkeypatch.setattr(comparisons, "get_model", unexpected)
    args = SimpleNamespace(
        output=tmp_path / "study",
        julia_project=project,
        calibrate=False,
        model="offline",
        endpoint="http://offline/v1",
        trials=1,
        batch=batch,
        require_provenance=True,
    )
    with pytest.raises(ValueError, match="pinned model_revision"):
        await comparisons.run(args)
