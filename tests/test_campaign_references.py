"""Calibration must match the campaign, complete artifacts and local Julia code."""

import csv
import json
import shutil
from dataclasses import asdict, replace

import pytest

pytest.importorskip("inspect_ai")

from jutul_agent.eval.archive import sha256, write_json
from jutul_agent.eval.campaign_validation import CSV_COLUMNS, artifact_rows
from jutul_agent.eval.campaigns import CAMPAIGNS, calibrate, load_reference, reference_answer
from jutul_agent.eval.runconfig import julia_environment_sha256
from jutul_agent.eval.tasks._campaign import CAMPAIGN_CASES
from jutul_agent.julia.session import EvalResult


def artifact(directory, campaign, index):
    case = CAMPAIGN_CASES[index]
    path = directory / "results" / f"{case}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    porosity, rate = campaign.parameters[index]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(
            (case, porosity, rate, 0.55 if index == 1 else 0.4, 250, 250, cell, 250, 0.8)
            for cell in range(1, 193)
        )


@pytest.fixture
def reference(tmp_path):
    campaign = CAMPAIGNS[0]
    project = tmp_path / "environment"
    project.mkdir()
    (project / "Manifest.toml").write_text("[deps]\n")
    (project / "Project.toml").write_text("[deps]\n")
    (project / "helper.jl").write_text("x = 1\n")
    directory = tmp_path / "references" / campaign.name
    directory.mkdir(parents=True)
    shutil.copytree(project, directory / "julia-environment")
    (directory / "campaign.jl").write_text(campaign.fixture())
    for index in range(len(CAMPAIGN_CASES)):
        artifact(directory, campaign, index)
    saved = {
        "campaign": asdict(campaign),
        "fixture_sha256": sha256(directory / "campaign.jl"),
        "manifest_sha256": sha256(project / "Manifest.toml"),
        "julia_environment_sha256": julia_environment_sha256(project),
        "answer": reference_answer(campaign, directory),
        "artifact_sha256": {
            case: sha256(directory / "results" / f"{case}.csv") for case in CAMPAIGN_CASES
        },
    }
    write_json(directory / "reference.json", saved)
    return campaign, directory, project


@pytest.mark.parametrize(
    "change",
    ["campaign", "fixture", "parameters", "source", "project", "legacy", "archived-source"],
)
def test_reference_rejects_stale_or_inconsistent_configuration(reference, change):
    campaign, directory, project = reference
    load_reference(campaign, directory, project)
    if change == "campaign":
        campaign = replace(campaign, pressure_limit_bar=campaign.pressure_limit_bar + 1)
    elif change == "fixture":
        (directory / "campaign.jl").write_text("changed")
    elif change == "source":
        (project / "helper.jl").write_text("x = 2\n")
    elif change == "project":
        (project / "Project.toml").write_text('name = "changed"\n')
    elif change == "archived-source":
        (directory / "julia-environment/helper.jl").write_text("x = 2\n")
    else:
        path = directory / "reference.json"
        saved = json.loads(path.read_text())
        if change == "legacy":
            saved.pop("julia_environment_sha256")
        else:
            case = CAMPAIGN_CASES[0]
            artifact_path = directory / "results" / f"{case}.csv"
            artifact_path.write_text(artifact_path.read_text().replace(",0.16,", ",0.30,"))
            saved["artifact_sha256"][case] = sha256(artifact_path)
        write_json(path, saved)
    with pytest.raises(ValueError, match="reference"):
        load_reference(campaign, directory, project)


def test_environment_rejects_unarchived_external_path_dependency(reference):
    _, _, project = reference
    (project / "Manifest.toml").write_text('[[deps.External]]\npath = "../outside"\n')
    with pytest.raises(ValueError, match="inside the archived project"):
        julia_environment_sha256(project)


@pytest.mark.parametrize("pressure", [0, -1])
def test_artifacts_reject_nonpositive_absolute_pressure(reference, pressure):
    _, directory, _ = reference
    case = CAMPAIGN_CASES[0]
    path = directory / "results" / f"{case}.csv"
    path.write_text(path.read_text().replace(",250,", f",{pressure},"))
    with pytest.raises(ValueError, match="absolute pressure"):
        artifact_rows(path, case)


async def test_failed_calibration_is_not_published_and_can_be_retried(reference, monkeypatch):
    campaign, existing, project = reference
    root = existing.parent / "new-calibration"
    fail = True

    class Kernel:
        def __init__(self, config):
            self.directory = config.cwd

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def eval(self, code):
            if fail:
                return EvalResult(output="", error="injected failure")
            if code.startswith("run_campaign_case"):
                case = json.loads(code.partition("(")[2][:-1])
                artifact(self.directory, campaign, CAMPAIGN_CASES.index(case))
            return EvalResult(output="ok")

    monkeypatch.setattr("jutul_agent.juliakernel.JuliaKernel", Kernel)
    with pytest.raises(RuntimeError, match="calibration failed"):
        await calibrate(root, project, (campaign,))
    assert not (root / campaign.name).exists()
    assert list(root.glob(".*/calibration.jsonl"))
    fail = False
    await calibrate(root, project, (campaign,))
    assert (
        load_reference(campaign, root / campaign.name, project)["answer"]["run_id"] == "run-falcon"
    )
