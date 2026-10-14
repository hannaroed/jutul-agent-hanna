"""Predeclared campaign variants and model-independent Julia calibration."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from jutul_agent.eval.archive import archive_tree, capture_source, sha256, write_json
from jutul_agent.eval.runconfig import julia_environment_sha256
from jutul_agent.eval.tasks._campaign import CAMPAIGN_CASES, CAMPAIGN_FIXTURE


@dataclass(frozen=True)
class Campaign:
    name: str
    parameters: tuple[tuple[float, int], ...]
    pressure_limit_bar: float
    query_cell_id: int

    def to_dict(self) -> dict:
        """Use the same JSON representation in memory and saved logs."""
        return {**asdict(self), "parameters": [list(pair) for pair in self.parameters]}

    def fixture(self) -> str:
        entries = "\n".join(
            f'    "{case}" => (porosity = {porosity}, injection_rate = {float(rate)}),'
            for case, (porosity, rate) in zip(CAMPAIGN_CASES, self.parameters, strict=True)
        )
        _, function = CAMPAIGN_FIXTURE.split("function run_campaign_case(", 1)
        fixture = (
            "using JutulDarcy, Jutul\nusing Statistics\n\n"
            f"const campaign_cases = Dict(\n{entries}\n)\n\n"
            "function run_campaign_case(" + function
        )
        return fixture


CAMPAIGNS = (
    Campaign(
        "waterflood-a",
        ((0.16, 650), (0.18, 1100), (0.20, 850), (0.22, 1250), (0.25, 950), (0.28, 1400)),
        320.0,
        150,
    ),
    Campaign(
        "waterflood-b",
        ((0.21, 900), (0.17, 600), (0.19, 1150), (0.24, 750), (0.26, 1000), (0.22, 1300)),
        310.0,
        73,
    ),
)


def reference_answer(campaign: Campaign, directory: Path) -> dict:
    """Select from calibrated CSVs; never assume the same winner across tasks."""
    candidates = []
    for case in CAMPAIGN_CASES:
        with (directory / "results" / f"{case}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        first = rows[0]
        if float(first["max_pressure_bar"]) <= campaign.pressure_limit_bar:
            candidates.append((float(first["recovery_factor"]), case, rows))
    if not candidates:
        raise ValueError(f"{campaign.name}: no pressure-feasible case")
    _, case, rows = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    row = next(row for row in rows if int(row["cell_id"]) == campaign.query_cell_id)
    return {
        "run_id": case,
        "recovery_factor": float(row["recovery_factor"]),
        "max_pressure_bar": float(row["max_pressure_bar"]),
        "cell_water_saturation": float(row["water_saturation"]),
    }


async def calibrate(
    reference_dir: Path, julia_project: Path, campaigns: tuple[Campaign, ...] = CAMPAIGNS
) -> None:
    """Evaluate one case at a time in a persistent kernel, with no model calls."""
    from jutul_agent.juliakernel import JuliaKernel, KernelConfig

    source = capture_source(Path(__file__).resolve().parents[3], reference_dir / "source")
    for campaign in campaigns:
        directory = reference_dir / campaign.name
        if directory.exists():
            load_reference(campaign, directory, julia_project)
            continue
        reference_dir.mkdir(parents=True, exist_ok=True)
        destination = directory
        directory = Path(tempfile.mkdtemp(prefix=f".{campaign.name}-", dir=reference_dir))
        environment_hash = julia_environment_sha256(julia_project)
        archive_tree(julia_project, directory / "julia-environment")
        fixture = directory / "campaign.jl"
        fixture.write_text(campaign.fixture(), encoding="utf-8")
        config = KernelConfig(
            julia_project=julia_project,
            cwd=directory,
            threads="1",
            env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        write_json(directory / "configuration.json", {"kernel": asdict(config), "source": source})
        with (directory / "calibration.jsonl").open("w") as log:
            async with JuliaKernel(config) as kernel:
                for code in [
                    'include("campaign.jl")',
                    *(f'run_campaign_case("{case}")' for case in CAMPAIGN_CASES),
                ]:
                    result = await kernel.eval(code)
                    log.write(json.dumps({"code": code, "result": asdict(result)}) + "\n")
                    log.flush()
                    if result.error or result.interrupted:
                        raise RuntimeError(f"calibration failed: {code}: {result.error}")
                    print(f"calibrated {campaign.name}: {code}", flush=True)
        write_json(
            directory / "reference.json",
            {
                "campaign": campaign.to_dict(),
                "fixture_sha256": sha256(fixture),
                "manifest_sha256": sha256(julia_project / "Manifest.toml"),
                "julia_environment_sha256": environment_hash,
                "source": source,
                "answer": reference_answer(campaign, directory),
                "artifact_sha256": {
                    case: sha256(directory / "results" / f"{case}.csv") for case in CAMPAIGN_CASES
                },
                "diagnostic_sha256": {
                    path.name: sha256(path)
                    for path in sorted((directory / "diagnostics").glob("*.csv"))
                },
            },
        )
        load_reference(campaign, directory, julia_project)
        directory.rename(destination)


def load_reference(campaign: Campaign, directory: Path, julia_project: Path) -> dict:
    """Require matching fixture, Julia lockfile, and artifact hashes."""
    from jutul_agent.eval.campaign_validation import artifact_rows

    saved = json.loads((directory / "reference.json").read_text())
    if saved["campaign"] != campaign.to_dict():
        raise ValueError("reference campaign changed; recalibrate explicitly")
    fixture_hash = hashlib.sha256(campaign.fixture().encode()).hexdigest()
    if (
        saved["fixture_sha256"] != fixture_hash
        or sha256(directory / "campaign.jl") != fixture_hash
        or saved["manifest_sha256"] != sha256(julia_project / "Manifest.toml")
        or saved.get("julia_environment_sha256") != julia_environment_sha256(julia_project)
        or saved.get("julia_environment_sha256")
        != julia_environment_sha256(directory / "julia-environment")
    ):
        raise ValueError("reference fixture or Julia environment changed; recalibrate explicitly")
    if (source := saved.get("source")) and sha256(Path(source["archive"])) != source[
        "archive_sha256"
    ]:
        raise ValueError("reference source archive integrity failure")
    for case in CAMPAIGN_CASES:
        artifact = directory / "results" / f"{case}.csv"
        if sha256(artifact) != saved["artifact_sha256"][case]:
            raise ValueError("reference artifact integrity failure")
        rows = artifact_rows(artifact, case)
        index = CAMPAIGN_CASES.index(case)
        porosity, rate = campaign.parameters[index]
        if (
            float(rows[0]["porosity"]) != porosity
            or float(rows[0]["injection_rate_m3_day"]) != rate
        ):
            raise ValueError("reference artifact parameters disagree with the campaign")
    for name, digest in saved.get("diagnostic_sha256", {}).items():
        path = directory / "diagnostics" / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("reference diagnostic integrity failure")
    if saved["answer"] != reference_answer(campaign, directory):
        raise ValueError("reference answer disagrees with artifacts")
    return saved
