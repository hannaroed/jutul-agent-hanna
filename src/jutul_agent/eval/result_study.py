"""Run one trial at a time in the frozen result-memory study."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from urllib.request import urlopen

from inspect_ai import eval_async
from inspect_ai.log import read_eval_log

from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.eval.archive import capture_source, finalize_log, sha256, write_json
from jutul_agent.eval.campaigns import CAMPAIGNS, calibrate
from jutul_agent.eval.comparisons import advertised_model, require_batch_provenance
from jutul_agent.eval.result_memory import POLICY, result_memory_task
from jutul_agent.eval.runconfig import build_runconfig
from jutul_agent.paths import set_state_home
from jutul_agent.simulators.registry import get as simulator

MODEL = "openai-api/vllm/Qwen/Qwen3.5-9B"
MODEL_ID = "Qwen/Qwen3.5-9B"
REQUIRED = (
    "scientific_answer",
    "scientific_sources",
    "tool_result_numeric_evidence",
    "study_integrity",
)
CONDITIONS = tuple(
    HarnessStrategies(summarization=name, clear_tool_results=True)
    for name in ("generic", "scientific")
)

REMOTE_PROBE = """
import hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path

expected = json.load(sys.stdin)
os.chdir('/home/marcel/hanna')
matches = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().decode().rstrip('\\0').split('\\0')
        if 'vllm' in ' '.join(args) and 'serve' in args and expected['model'] in args:
            matches.append((path.parent, args))
    except (OSError, UnicodeError):
        pass
if len(matches) != 1:
    raise RuntimeError('expected exactly one matching vLLM server')
process, args = matches[0]
environment = dict(item.split('=', 1) for item in
                   (process / 'environ').read_bytes().decode().split('\\0') if '=' in item)
snapshot = Path(expected['snapshot'])
snapshot.resolve().relative_to('/home/marcel/hanna')
files = {}
for name in expected['snapshot_files']:
    path = snapshot / name
    path.resolve().relative_to('/home/marcel/hanna')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    files[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
result = {
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'pid': int(process.name),
    'process_start_ticks': (process / 'stat').read_text().rsplit(')', 1)[1].split()[19],
    'launch_args': args,
    'packages': {name: importlib.metadata.version(name) for name in expected['packages']},
    'cache_environment': {name: environment.get(name) for name in expected['cache_environment']},
    'snapshot_files': files,
    'gpu': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        text=True).strip(),
}
print(json.dumps(result))
"""


def verify_server(root):
    """Check the running server without changing the frozen provenance record."""
    expected = json.loads((root / "server.json").read_text())
    require_batch_provenance(expected)
    response = subprocess.run(
        [
            "ssh",
            "-oBatchMode=yes",
            "-oConnectTimeout=8",
            "marcel@100.111.80.113",
            "cd /home/marcel/hanna && PYTHONDONTWRITEBYTECODE=1 "
            "/home/marcel/hanna/vllm-env/bin/python -c " + shlex.quote(REMOTE_PROBE),
        ],
        input=json.dumps(expected),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    actual = json.loads(response.stdout)
    validate_server(actual, expected)
    write_json(root / "runtime" / "live-server-verification.json", actual)
    return actual


def validate_server(actual, expected):
    checks = {
        "launch command": actual["launch_args"] == shlex.split(expected["launch_command"]),
        "packages": actual["packages"] == expected["packages"],
        "cache directories": actual["cache_environment"] == expected["cache_environment"],
        "GPU": actual["gpu"] == ",".join(expected["gpu"].split(",")[:3]).strip(),
        "model snapshot": actual["snapshot_files"]
        == {
            name: {key: saved[key] for key in ("bytes", "sha256")}
            for name, saved in expected["snapshot_files"].items()
        },
    }
    if failed := [name for name, passed in checks.items() if not passed]:
        raise ValueError(f"live server differs from frozen configuration: {', '.join(failed)}")


def audit_completed(root):
    """Verify archives and every actual request before continuing the batch."""
    result = report(root)
    audits = []
    generation = result["plan"]["generation"]
    for row in result["trials"]:
        (sample,) = read_eval_log(row["log"]).samples
        requests = [
            event for event in sample.events if event.event == "model" and event.output is not None
        ]
        problems = []
        for event in requests:
            request = event.call.request
            usage = event.output.usage
            if usage is None or usage.input_tokens + generation["max_tokens"] > 16384:
                problems.append("request exceeds context budget or has no measured usage")
            for name in ("temperature", "top_p", "presence_penalty", "max_tokens", "extra_body"):
                if request.get(name) != generation[name]:
                    problems.append(f"actual request changed {name}")
            if request.get("seed") != 1000 + row["trial"] or request.get("model") != MODEL_ID:
                problems.append("actual model or seed changed")
            if usage is not None and usage.reasoning_tokens:
                problems.append("unexpected reasoning tokens")
        if not requests:
            problems.append("no recorded model requests")
        if row["technical_failure"]:
            problems.append("recorded technical failure")
        audit = {
            "key": row["key"],
            "success": row["success"],
            "requests": len(requests),
            "problems": sorted(set(problems)),
        }
        audits.append(audit)
    write_json(root / "runtime" / "batch-audits.json", audits)
    if any(audit["problems"] for audit in audits):
        raise RuntimeError("trial audit failed; preserved and stopped before further model calls")
    return result


async def run_batch(args):
    if not args.batch:
        await run(args)
        return
    if args.calibrate or args.report:
        raise ValueError("--batch cannot be combined with --calibrate or --report")
    root = args.output.resolve()
    result = audit_completed(root)
    if len(result["trials"]) == len(result["plan"]["order"]):
        print("All planned trials are complete.")
        return
    verify_server(root)
    while len(result["trials"]) < len(result["plan"]["order"]):
        before = len(result["trials"])
        await run(args)
        result = audit_completed(root)
        if len(result["trials"]) != before + 1:
            raise RuntimeError("expected exactly one new trial; stopped")
        print(f"Audited {len(result['trials'])}/{len(result['plan']['order'])} trials.", flush=True)


def model_code_sha256(source):
    archive = Path(source["archive"])
    files = json.loads((archive.parent / "files.json").read_text())

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    if sha256(archive) != source["archive_sha256"] or digest(files) != source["sha256"]:
        raise ValueError("source archive or manifest changed")
    relevant = {
        name: value
        for name, value in files.items()
        if (name.startswith("src/") and name != "src/jutul_agent/eval/result_study.py")
        or name in ("pyproject.toml", "uv.lock", "AGENTS.md")
    }
    return digest(relevant)


def resume_plan(previous, current):
    if model_code_sha256(previous["source"]) != model_code_sha256(current["source"]):
        raise ValueError("model-facing code changed; use a new output directory")
    comparison = deepcopy(current)
    comparison["source"] = previous["source"]
    for name in ("commit", "dirty", "version"):
        comparison["runtime"]["jutul_agent"][name] = previous["runtime"]["jutul_agent"][name]
    if previous != comparison:
        raise ValueError("study plan changed; use a new output directory")
    return previous


def trial_order(trials):
    order = [["short", campaign, 0, 1] for campaign in range(len(CAMPAIGNS))]
    for trial in range(1, trials + 1):
        block = [
            ["long", campaign, condition, trial]
            for campaign in range(len(CAMPAIGNS))
            for condition in range(len(CONDITIONS))
        ]
        random.Random(3000 + trial).shuffle(block)
        order.extend(block)
    return order


def record(log, key):
    (sample,) = log.samples
    scores = {name: score.value for name, score in (sample.scores or {}).items()}
    output_errors = [
        str(choice.stop_reason)
        for event in sample.events
        if event.event == "model" and event.output is not None
        for choice in event.output.choices
        if choice.stop_reason in ("max_tokens", "model_length")
    ]
    technical = (
        log.status != "success"
        or sample.error is not None
        or sample.limit is not None
        or bool(output_errors)
        or any(scores.get(name) is None for name in REQUIRED)
    )
    answer_path = Path(sample.store["jutul/archive"]) / "workspace/answer.json"
    return {
        "key": key,
        "phase": key[0],
        "campaign": CAMPAIGNS[key[1]].name,
        "summary": CONDITIONS[key[2]].summarization,
        "trial": key[3],
        "success": not technical and all(scores.get(name) == "C" for name in REQUIRED),
        "technical_failure": technical,
        "error": sample.error.message if sample.error else None,
        "limit": str(sample.limit.type) if sample.limit else None,
        "output_errors": output_errors,
        "scores": scores,
        "tokens": sum(usage.total_tokens for usage in sample.model_usage.values()),
        "seconds": sample.working_time,
        "log": log.location,
        "log_sha256": sha256(Path(log.location)),
        "archive": sample.store["jutul/archive"],
        "answer_file": answer_path.read_text() if answer_path.is_file() else None,
        "expected": json.loads(sample.target),
    }


def report(root):
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    journal = root / "runs.jsonl"
    for line in journal.read_text().splitlines() if journal.exists() else []:
        row = json.loads(line)
        if row["key"] not in plan["order"] or any(item["key"] == row["key"] for item in rows):
            raise ValueError("undeclared or duplicate trial")
        if sha256(Path(row["log"])) != row["log_sha256"]:
            raise ValueError("trial log changed")
        archive = Path(row["archive"])
        manifest = json.loads((archive / "manifest.json").read_text())
        if any(sha256(archive / name) != digest for name, digest in manifest.items()):
            raise ValueError("trial archive changed")
        log = read_eval_log(row["log"])
        current = record(log, row["key"])
        if current != row:
            raise ValueError("journal disagrees with graded log")
        (sample,) = log.samples
        if model_code_sha256(sample.store["jutul/source"]) != model_code_sha256(plan["source"]):
            raise ValueError("trial source differs from plan")
        expected_strategy = plan["conditions"][row["key"][2]]
        if (
            log.eval.model != plan["model"]
            or log.eval.model_base_url != plan["endpoint"]
            or sample.metadata["phase"] != row["key"][0]
            or sample.metadata["trial"] != row["key"][3]
            or sample.metadata["campaign"] != CAMPAIGNS[row["key"][1]].to_dict()
            or sample.store["jutul/runconfig"]["harness_strategies"] != expected_strategy
            or log.plan.config.seed != 1000 + row["key"][3]
            or any(
                getattr(log.plan.config, name) != value
                for name, value in plan["generation"].items()
            )
        ):
            raise ValueError("trial configuration differs from plan")
        rows.append(row)
    summary = []
    for phase in ("short", "long"):
        for name in ("generic", "scientific"):
            group = [row for row in rows if row["phase"] == phase and row["summary"] == name]
            if group:
                compactions = [row["scores"].get("context_compaction_count") for row in group]
                compactions = [value for value in compactions if isinstance(value, (int, float))]
                seconds = [row["seconds"] for row in group if row["seconds"] is not None]
                summary.append(
                    {
                        "phase": phase,
                        "summary": name,
                        "attempted": len(group),
                        "passed": sum(row["success"] for row in group),
                        "technical_failures": sum(row["technical_failure"] for row in group),
                        "mean_tokens": fmean(row["tokens"] for row in group),
                        "mean_seconds": fmean(seconds) if seconds else None,
                        "mean_compactions": fmean(compactions) if compactions else None,
                        "compaction_measurements": len(compactions),
                    }
                )
    result = {"plan": plan, "trials": rows, "summary": summary}
    write_json(root / "report.json", result)
    if summary:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return result


async def run(args):
    root, project = args.output.resolve(), args.julia_project.resolve()
    set_state_home(root / "state")
    for name in (
        "JUTUL_AGENT_JULIA_THREADS",
        "JUTUL_AGENT_HYPRE_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if args.calibrate:
        await calibrate(root / "reference", project)
        return
    if args.report:
        print(json.dumps(report(root)["summary"], indent=2))
        return
    if (root / "active.json").exists():
        raise ValueError("unfinished trial; inspect before resuming")
    server = json.loads((root / "server.json").read_text())
    require_batch_provenance(server)
    with urlopen(args.endpoint + "/models", timeout=10) as response:
        endpoint = advertised_model(json.load(response), MODEL_ID)
    if endpoint["max_model_len"] != 16384 or server.get("model") != MODEL_ID:
        raise ValueError("expected the pinned Qwen3.5 model with a 16k context")
    source = capture_source(Path(__file__).resolve().parents[3], root / "source")
    runtime = build_runconfig(
        simulator("jutuldarcy"),
        julia_project=project,
        context_window=16384,
        strategies=CONDITIONS[0],
    )
    plan = {
        "policy": POLICY,
        "model": MODEL,
        "endpoint": args.endpoint,
        "server": server,
        "endpoint_snapshot": endpoint,
        "source": source,
        "runtime": runtime,
        "conditions": [item.to_dict() for item in CONDITIONS],
        "trials": args.trials,
        "order": trial_order(args.trials),
        "generation": {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
            },
            "max_tokens": 2048,
            "max_retries": 2,
            "timeout": 180,
            "attempt_timeout": 120,
        },
        "seed_policy": "1000 + trial, paired across campaigns and conditions",
        "scope": (
            "Two calibrated campaigns, repeated model seeds per condition; "
            "a pilot, not independent evidence across many scientific problems."
        ),
    }
    plan = json.loads(json.dumps(plan))
    path = root / "plan.json"
    if path.exists():
        plan = resume_plan(json.loads(path.read_text()), plan)
    write_json(path, plan)
    rows = report(root)["trials"]
    if any(row["technical_failure"] for row in rows):
        raise ValueError("previous technical failure; review before further trials")
    pending = [key for key in plan["order"] if not any(row["key"] == key for row in rows)]
    if not pending:
        print("All planned trials are complete.")
        return
    key = pending[0]
    phase, campaign, condition, trial = key
    task = result_memory_task(
        CAMPAIGNS[campaign],
        root / "reference" / CAMPAIGNS[campaign].name,
        project,
        phase=phase,
        strategies=CONDITIONS[condition],
        trial=trial,
    )
    write_json(root / "active.json", {"key": key})
    logs = await eval_async(
        task,
        model=MODEL,
        model_base_url=args.endpoint,
        model_args={"api_key": "local-vllm", "responses_api": False},
        log_dir=str(root / "logs"),
        max_samples=1,
        max_tasks=1,
        fail_on_error=False,
        score_on_error=True,
        log_model_api=True,
        seed=1000 + trial,
        **plan["generation"],
    )
    (log,) = logs
    finalize_log(Path(log.location))
    row = record(log, key)
    with (root / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    (root / "active.json").unlink()
    report(root)
    print(json.dumps(row, indent=2))
    if row["technical_failure"]:
        write_json(root / "stopped.json", row)
        raise RuntimeError("technical failure; archived and stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--julia-project", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--batch", action="store_true", help="run pending trials serially, auditing each"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
