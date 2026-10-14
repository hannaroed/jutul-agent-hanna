"""Batch execution keeps the frozen trials and stops before unsafe continuation."""

import json
import shlex
from copy import deepcopy
from types import SimpleNamespace

import pytest

from jutul_agent.eval import result_study as study


def test_identical_code_can_resume_after_commit_without_rewriting_plan(monkeypatch):
    original = {
        "source": {"archive": "original"},
        "runtime": {
            "jutul_agent": {"commit": "before", "dirty": True, "version": "old"},
            "deps": {"deepagents": "0.7.18"},
        },
        "generation": {"temperature": 0.7},
    }
    current = deepcopy(original)
    current["source"] = {"archive": "new-snapshot"}
    current["runtime"]["jutul_agent"] = {"commit": "after", "dirty": False, "version": "new"}
    monkeypatch.setattr(study, "model_code_sha256", lambda source: "same-code")
    assert study.resume_plan(original, current) is original
    assert current["runtime"]["jutul_agent"]["commit"] == "after"
    assert original["source"]["archive"] == "original"


@pytest.mark.parametrize("change", ["source", "dependencies", "generation", "runtime-field"])
def test_commit_does_not_allow_changed_experiment_to_resume(monkeypatch, change):
    original = {
        "source": {"hash": "same"},
        "runtime": {
            "jutul_agent": {"commit": "before", "dirty": True, "version": "old"},
            "deps": {"deepagents": "0.7.18"},
        },
        "generation": {"temperature": 0.7},
    }
    current = deepcopy(original)
    if change == "source":
        current["source"]["hash"] = "different"
    elif change == "dependencies":
        current["runtime"]["deps"]["deepagents"] = "different"
    elif change == "generation":
        current["generation"]["temperature"] = 0.8
    else:
        current["runtime"]["jutul_agent"]["unexpected"] = True
    monkeypatch.setattr(study, "model_code_sha256", lambda source: source["hash"])
    with pytest.raises(ValueError, match="changed"):
        study.resume_plan(original, current)


@pytest.fixture
def server():
    revision = "a" * 40
    return {
        "model_revision": revision,
        "tokenizer_revision": revision,
        "julia_version": "1.11.4",
        "vllm_version": "0.29.0",
        "launch_command": (
            f"python vllm serve model --revision {revision} --tokenizer-revision {revision}"
        ),
        "packages": {"vllm": "0.29.0"},
        "cache_environment": {"HF_HOME": "/home/marcel/hanna/cache"},
        "gpu": "RTX 4090, driver, 24564 MiB, 20000 MiB, 4564 MiB",
        "snapshot_files": {"weights": {"bytes": 42, "sha256": "b" * 64}},
    }


def actual_server(server):
    return {
        "boot_id": "new-boot",
        "pid": 123,
        "launch_args": shlex.split(server["launch_command"]),
        "packages": deepcopy(server["packages"]),
        "cache_environment": deepcopy(server["cache_environment"]),
        "gpu": "RTX 4090, driver, 24564 MiB",
        "snapshot_files": deepcopy(server["snapshot_files"]),
    }


@pytest.mark.parametrize("change", ["command", "packages", "cache", "gpu", "weights"])
def test_live_configuration_changes_are_rejected(server, change):
    actual = actual_server(server)
    if change == "command":
        actual["launch_args"].extend(["--dtype", "float16"])
    elif change == "packages":
        actual["packages"]["vllm"] = "different"
    elif change == "cache":
        actual["cache_environment"]["HF_HOME"] = "/home/marcel/cache"
    elif change == "gpu":
        actual["gpu"] = "different GPU"
    else:
        actual["snapshot_files"]["weights"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="live server differs"):
        study.validate_server(actual, server)


def test_restarted_matching_server_does_not_rewrite_frozen_provenance(
    tmp_path, monkeypatch, server
):
    compile(study.REMOTE_PROBE, "remote-runtime-probe", "exec")
    original = json.dumps(server)
    (tmp_path / "server.json").write_text(original)
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 60 and kwargs["check"] is True
        assert json.loads(kwargs["input"]) == server
        assert "cd /home/marcel/hanna" in command[-1]
        return SimpleNamespace(stdout=json.dumps(actual_server(server)))

    monkeypatch.setattr(study.subprocess, "run", probe)
    assert study.verify_server(tmp_path)["pid"] == 123
    assert len(calls) == 1
    assert (tmp_path / "server.json").read_text() == original
    assert (tmp_path / "runtime/live-server-verification.json").is_file()


def batch_args(tmp_path):
    return SimpleNamespace(output=tmp_path, batch=True, calibrate=False, report=False)


async def test_batch_runs_one_trial_then_audits_and_keeps_incorrect_answers(tmp_path, monkeypatch):
    rows, calls = [], []

    def audit(root):
        calls.append(("audit", len(rows)))
        return {"trials": rows.copy(), "plan": {"order": [1, 2]}}

    async def run(args):
        calls.append(("run", len(rows)))
        rows.append({"success": len(rows) > 0})

    monkeypatch.setattr(study, "audit_completed", audit)
    monkeypatch.setattr(study, "verify_server", lambda root: calls.append(("server", 0)))
    monkeypatch.setattr(study, "run", run)
    await study.run_batch(batch_args(tmp_path))
    assert calls == [
        ("audit", 0),
        ("server", 0),
        ("run", 0),
        ("audit", 1),
        ("run", 1),
        ("audit", 2),
    ]
    assert [row["success"] for row in rows] == [False, True]


@pytest.mark.parametrize("failure", ["server", "trial", "audit"])
async def test_batch_stops_at_first_technical_problem(tmp_path, monkeypatch, failure):
    calls = []

    def audit(root):
        if calls and failure == "audit":
            raise RuntimeError("audit failed")
        return {"trials": [], "plan": {"order": [1, 2]}}

    def verify(root):
        if failure == "server":
            raise RuntimeError("server changed")

    async def run(args):
        calls.append("trial")
        if failure == "trial":
            raise RuntimeError("technical failure")

    monkeypatch.setattr(study, "audit_completed", audit)
    monkeypatch.setattr(study, "verify_server", verify)
    monkeypatch.setattr(study, "run", run)
    with pytest.raises(RuntimeError):
        await study.run_batch(batch_args(tmp_path))
    assert len(calls) == (0 if failure == "server" else 1)


@pytest.mark.parametrize("problem", ["budget", "generation", "reasoning", "empty", "technical"])
def test_request_audit_stops_before_more_model_calls(tmp_path, monkeypatch, problem):
    generation = {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20},
    }
    row = {"key": ["long", 0, 0, 1], "trial": 1, "log": "saved", "success": True}
    row["technical_failure"] = problem == "technical"
    request = {**generation, "model": study.MODEL_ID, "seed": 1001}
    usage = SimpleNamespace(input_tokens=14337 if problem == "budget" else 10000)
    usage.reasoning_tokens = 1 if problem == "reasoning" else 0
    if problem == "generation":
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    event = SimpleNamespace(
        event="model", call=SimpleNamespace(request=request), output=SimpleNamespace(usage=usage)
    )
    sample = SimpleNamespace(events=[] if problem == "empty" else [event])
    monkeypatch.setattr(
        study, "report", lambda root: {"trials": [row], "plan": {"generation": generation}}
    )
    monkeypatch.setattr(study, "read_eval_log", lambda path: SimpleNamespace(samples=[sample]))
    with pytest.raises(RuntimeError, match="trial audit failed"):
        study.audit_completed(tmp_path)
    assert json.loads((tmp_path / "runtime/batch-audits.json").read_text())[0]["problems"]
