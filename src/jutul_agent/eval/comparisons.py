"""Run a fixed, blocked comparison of context strategies."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shlex
from pathlib import Path
from urllib.request import urlopen

from inspect_ai import Task, eval_async
from inspect_ai.model import get_model

from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.eval.archive import capture_source, finalize_log, sha256, write_json
from jutul_agent.eval.campaign_validation import artifact_rows
from jutul_agent.eval.campaigns import CAMPAIGNS, calibrate, load_reference
from jutul_agent.eval.comparison_report import REQUIRED_SCORES, collect, trial_record
from jutul_agent.eval.runconfig import build_runconfig, julia_environment_sha256
from jutul_agent.eval.solver import jutul_agent_solver
from jutul_agent.eval.tasks._campaign import (
    PRECISE_TOLERANCES,
    SCORING_POLICY,
    campaign_scorers,
)
from jutul_agent.eval.tasks.experiment_autonomous import autonomous_sample
from jutul_agent.paths import set_state_home

CONDITIONS = tuple(
    HarnessStrategies(summarization=summary, clear_tool_results=clear)
    for summary in ("generic", "scientific")
    for clear in (False, True)
)
CONTEXT_WINDOW = 16_384


def advertised_model(endpoint: dict, model_id: str) -> dict:
    model = next((item for item in endpoint["data"] if item["id"] == model_id), None)
    if model is None:
        raise ValueError("requested model is absent from the endpoint")
    return {key: model.get(key) for key in ("id", "root", "parent", "owned_by", "max_model_len")}


def require_batch_provenance(configuration: dict | None) -> None:
    """Validate pinned model revisions and recorded runtime details."""
    configuration = configuration or {}
    for name in ("model_revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(configuration.get(name, ""))):
            raise ValueError(f"comparisons require a pinned {name} in server.json")
    for name in ("julia_version", "vllm_version", "launch_command"):
        if not isinstance(configuration.get(name), str) or not configuration[name].strip():
            raise ValueError(f"comparisons require recorded {name} in server.json")
    command = shlex.split(configuration["launch_command"])
    for flag, name in (
        ("--revision", "model_revision"),
        ("--tokenizer-revision", "tokenizer_revision"),
    ):
        revision = configuration[name]
        if not any(
            token == f"{flag}={revision}"
            or (token == flag and command[index + 1 : index + 2] == [revision])
            for index, token in enumerate(command)
        ):
            raise ValueError(f"recorded launch_command must pin {name} with {flag}")


def comparison_task(campaign, reference: Path, julia_project: Path, strategies, trial: int) -> Task:
    saved = load_reference(campaign, reference, julia_project)
    sample = autonomous_sample(
        pressure_limit_bar=campaign.pressure_limit_bar,
        query_cell_id=campaign.query_cell_id,
        fixture=campaign.fixture(),
        target=json.dumps(saved["answer"], sort_keys=True),
    )
    sample.id = f"experiment-controlled-{campaign.name}"
    metadata = sample.metadata
    metadata.update(
        {
            "julia_project": str(julia_project),
            "harness_strategies": strategies.to_dict(),
            "campaign": campaign.to_dict(),
            "query_cell_id": campaign.query_cell_id,
            "context_window": CONTEXT_WINDOW,
            "trial": trial,
            "scoring_policy": SCORING_POLICY,
            "reference": saved,
            "evaluation_private_roots": [str(reference.parent.resolve())],
            "reference_artifacts": {
                case: artifact_rows(reference / "results" / f"{case}.csv", case)
                for case in metadata["campaign_contract"]["cases"]
            },
        }
    )
    return Task(
        name="experiment_controlled",
        version=SCORING_POLICY,
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=campaign_scorers(tolerance=PRECISE_TOLERANCES, evidence_tools=("grep",)),
        time_limit=2400,
        token_limit=1_000_000,
        message_limit=100,
    )


def trial_order(trials: int) -> list[tuple[int, int, int]]:
    order = []
    for trial in range(1, trials + 1):
        block = [
            (trial, campaign, condition)
            for campaign in range(len(CAMPAIGNS))
            for condition in range(len(CONDITIONS))
        ]
        random.Random(20261012 + trial).shuffle(block)
        order.extend(block)
    return order


def technical_failure(record: dict) -> bool:
    return bool(
        record["error"]
        or record.get("runtime_errors")
        or record.get("model_output_errors")
        or record["limit"]
        or record["configuration_mismatches"]
        or any(record[name] is None for name in REQUIRED_SCORES)
    )


def pending_trials(plan: dict, records: list[dict], *, batch: bool = False) -> list[tuple]:
    completed = {
        (
            row["trial"],
            next(i for i, c in enumerate(plan["campaigns"]) if c["name"] == row["campaign"]),
            row["condition"],
        )
        for row in records
    }
    if any(technical_failure(row) for row in records):
        raise ValueError("a previous trial had a technical failure; inspect it before more runs")
    pending = [tuple(key) for key in plan["order"] if tuple(key) not in completed]
    return pending if batch else pending[:1]


async def run(args) -> None:
    root = args.output.resolve()
    project = args.julia_project.resolve()
    active = root / "active-trial.json"
    if active.exists():
        raise ValueError("an unfinished trial exists; inspect its logs before resuming")
    output_tokens = getattr(args, "max_tokens", 2048)
    if not 0 < output_tokens < int(CONTEXT_WINDOW * 0.95):
        raise ValueError("max_tokens must be positive and leave room for model input")
    import os

    for key in (
        "JUTUL_AGENT_JULIA_THREADS",
        "JUTUL_AGENT_HYPRE_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[key] = "1"
    set_state_home(root / "state")
    if args.calibrate:
        await calibrate(root / "reference", project)
        return
    source_root = Path(__file__).resolve().parents[3]
    source = capture_source(source_root, root / "state/eval-sources")
    from jutul_agent.simulators.registry import get as simulator

    runtime = build_runconfig(simulator("jutuldarcy"), julia_project=project)
    server_config = root / "server.json"
    server_configuration = json.loads(server_config.read_text()) if server_config.exists() else None
    if args.batch or getattr(args, "require_provenance", False):
        require_batch_provenance(server_configuration)
    model = get_model(args.model, base_url=args.endpoint, api_key="local-vllm", responses_api=False)
    with urlopen(args.endpoint.rstrip("/") + "/models", timeout=5) as response:
        endpoint = json.load(response)
    server_snapshot = advertised_model(endpoint, model.api.service_model_name())
    plan = {
        "model": args.model,
        "endpoint": args.endpoint,
        "model_api": "chat-completions",
        "context_window": CONTEXT_WINDOW,
        "server_context_window": 16_384,
        "server_snapshot": server_snapshot,
        "server_configuration": server_configuration,
        "budget_policy": "Deep Agents reserves output and 5% headroom from the context window",
        "scoring_policy": SCORING_POLICY,
        "limits": {"time_limit": 2400, "token_limit": 1_000_000, "message_limit": 100},
        "trials": args.trials,
        "campaigns": [campaign.to_dict() for campaign in CAMPAIGNS],
        "conditions": [condition.to_dict() for condition in CONDITIONS],
        "order": trial_order(args.trials),
        "source": source,
        "manifest_sha256": sha256(project / "Manifest.toml"),
        "julia_environment_sha256": runtime["julia_environment_sha256"],
        "deps": runtime["deps"],
        "runtime": runtime["runtime"],
        "context_policy": runtime["context_policy"],
        "model_output_policy": runtime["model_output_policy"],
        "small_window_token_margin": runtime["small_window_token_margin"],
        "summary_policy": runtime["summary_policy"],
        "compute_threads": 1,
        "generation": {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": output_tokens,
            "max_retries": 2,
            "timeout": 180,
            "attempt_timeout": 120,
        },
        "seed_policy": "1000 + trial, matched across campaigns and settings",
        "execution_policy": "one-trial-default-stop-on-technical-failure-v1",
        "log_model_api": getattr(args, "log_model_api", False),
    }
    plan_path = root / "plan.json"
    plan = json.loads(json.dumps(plan))
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("comparison plan changed; use a new output directory")
    write_json(plan_path, plan)
    journal = root / "runs.jsonl"
    records = collect(root)["trials"] if journal.exists() else []
    pending = pending_trials(plan, records, batch=args.batch)
    for trial, campaign_index, condition_index in pending:
        if capture_source(source_root, root / "state/eval-sources")["sha256"] != source["sha256"]:
            raise ValueError("source changed during the comparison")
        if julia_environment_sha256(project) != plan["julia_environment_sha256"]:
            raise ValueError("Julia environment changed during the comparison")
        with urlopen(args.endpoint.rstrip("/") + "/models", timeout=5) as response:
            server = json.load(response)
        available = advertised_model(server, model.api.service_model_name())
        if available.get("max_model_len") != plan["server_context_window"]:
            raise ValueError("server context window differs from the declared plan")
        if available != plan["server_snapshot"]:
            raise ValueError("advertised model configuration changed during the comparison")
        write_json(root / "endpoint.json", server)
        campaign = CAMPAIGNS[campaign_index]
        strategies = CONDITIONS[condition_index]
        task = comparison_task(
            campaign, root / "reference" / campaign.name, project, strategies, trial
        )
        task.dataset[0].metadata.update(
            server_snapshot=plan["server_snapshot"],
            server_configuration=plan["server_configuration"],
        )
        print(
            f"START trial={trial} campaign={campaign.name} strategies={strategies.to_dict()}",
            flush=True,
        )
        write_json(active, {"key": [trial, campaign_index, condition_index]})
        logs = await eval_async(
            task,
            model=args.model,
            model_base_url=args.endpoint,
            model_args={"api_key": "local-vllm", "responses_api": False},
            log_dir=str(root / "logs"),
            max_samples=1,
            max_tasks=1,
            fail_on_error=False,
            score_on_error=True,
            log_model_api=plan["log_model_api"],
            seed=1000 + trial,
            **plan["generation"],
        )
        if len(logs) != 1:
            raise RuntimeError("expected one evaluation log; inspect the output before resuming")
        log = logs[0]
        finalize_log(Path(log.location))
        entry = {
            "key": [trial, campaign_index, condition_index],
            "log": log.location,
            "status": log.status,
            "samples": len(log.samples or []),
            "sample_errors": sum(sample.error is not None for sample in log.samples or []),
            "limits": sum(sample.limit is not None for sample in log.samples or []),
        }
        with journal.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"FINISHED {entry}", flush=True)
        record = trial_record(log, (trial, campaign_index, condition_index), plan)
        active.unlink()
        if technical_failure(record):
            write_json(root / "stopped.json", record)
            raise RuntimeError(
                f"technical failure; results preserved in {log.location}; inspect before more runs"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--julia-project", type=Path, required=True)
    parser.add_argument("--model", default="openai-api/vllm/Qwen/Qwen3-8B")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048, help="output tokens per response")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="require recorded setup for a single trial",
    )
    parser.add_argument(
        "--log-model-api", action="store_true", help="save model API details for inspection"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--first", action="store_true", help="run one pending trial (the default)")
    mode.add_argument(
        "--batch",
        action="store_true",
        help="explicitly run remaining trials; stop on technical failure",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
