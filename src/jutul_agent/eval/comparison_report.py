"""Audit a frozen comparison plan without dropping failed attempts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from jutul_agent.eval.archive import sha256, write_json
from jutul_agent.eval.log_values import REQUIRED_SCORES, numeric_score, score_value


def trial_record(log, key: tuple[int, int, int], plan: dict) -> dict:
    trial, campaign, condition = key
    if len(log.samples or []) != 1:
        raise ValueError("each comparison log must contain one sample")
    sample = log.samples[0]
    metadata = sample.metadata or {}
    store = sample.store or {}
    config = store.get("jutul/runconfig", {})
    mismatches = []
    checks = {
        "log status": log.status == "success",
        "model": log.eval.model == plan["model"],
        "endpoint": log.eval.model_base_url == plan["endpoint"],
        "campaign": json.dumps(metadata.get("campaign"), sort_keys=True)
        == json.dumps(plan["campaigns"][campaign], sort_keys=True),
        "trial": metadata.get("trial") == trial,
        "strategies": config.get("harness_strategies") == plan["conditions"][condition],
        "source": store.get("jutul/source", {}).get("sha256") == plan["source"]["sha256"],
        "Julia manifest": config.get("manifest_sha256") == plan["manifest_sha256"],
        "context window": config.get("context_window") == plan["context_window"],
        "scoring policy": metadata.get("scoring_policy") == plan["scoring_policy"],
        "server snapshot": metadata.get("server_snapshot") == plan.get("server_snapshot"),
        "server configuration": metadata.get("server_configuration")
        == plan.get("server_configuration"),
    }
    for name in (
        "julia_environment_sha256",
        "deps",
        "runtime",
        "context_policy",
        "model_output_policy",
        "small_window_token_margin",
        "summary_policy",
    ):
        if name in plan:
            checks[name] = config.get(name) == plan[name]
    for name, expected in {**plan["generation"], "seed": 1000 + trial}.items():
        checks[name] = getattr(log.plan.config, name) == expected
    for name, expected in plan["limits"].items():
        checks[name] = getattr(log.eval.config, name) == expected
    mismatches.extend(name for name, valid in checks.items() if not valid)
    scores = {name: score_value(sample, name) for name in REQUIRED_SCORES}
    integrity = (sample.scores or {}).get("campaign_integrity")
    runtime_errors = (getattr(integrity, "metadata", None) or {}).get("runtime_errors", [])
    limit = getattr(sample, "limit", None)
    usage = sample.model_usage or {}
    model_output_errors = [
        f"model response stopped with {choice.stop_reason}"
        for event in getattr(sample, "events", ())
        if event.event == "model" and event.output is not None
        for choice in event.output.choices
        if choice.stop_reason in ("max_tokens", "model_length")
    ]
    return {
        "trial": trial,
        "campaign": plan["campaigns"][campaign]["name"],
        "condition": condition,
        **plan["conditions"][condition],
        "configuration_mismatches": mismatches,
        "error": sample.error.message if sample.error else None,
        "limit": str(limit.type) if limit else None,
        "runtime_errors": runtime_errors,
        "model_output_errors": model_output_errors,
        "success": not mismatches
        and sample.error is None
        and limit is None
        and not runtime_errors
        and not model_output_errors
        and all(value == "C" for value in scores.values()),
        **scores,
        "compactions": numeric_score(sample, "context_compaction_count"),
        "tool_calls": numeric_score(sample, "tool_call_count"),
        "file_ops": numeric_score(sample, "file_op_count"),
        "total_tokens": sum(value.total_tokens for value in usage.values()),
        "wall_s": sample.total_time,
        "log": log.location,
        "archive": store.get("jutul/archive"),
    }


def summarize(records: list[dict], plan: dict) -> list[dict]:
    groups = []
    for campaign in plan["campaigns"]:
        for condition, strategy in enumerate(plan["conditions"]):
            runs = [
                row
                for row in records
                if row["campaign"] == campaign["name"] and row["condition"] == condition
            ]
            group = {
                "campaign": campaign["name"],
                "condition": condition,
                **strategy,
                "planned": plan["trials"],
                "attempted": len(runs),
                "pending": plan["trials"] - len(runs),
                "successes": sum(row["success"] for row in runs),
                "sample_errors": sum(row["error"] is not None for row in runs),
                "limits": sum(row["limit"] is not None for row in runs),
                "runtime_failures": sum(bool(row.get("runtime_errors")) for row in runs),
                "model_output_failures": sum(bool(row.get("model_output_errors")) for row in runs),
                "configuration_mismatches": sum(
                    bool(row["configuration_mismatches"]) for row in runs
                ),
            }
            for metric in ("compactions", "tool_calls", "file_ops", "total_tokens", "wall_s"):
                values = [row[metric] for row in runs if row[metric] is not None]
                group[f"mean_{metric}"] = fmean(values) if values else None
                group[f"missing_{metric}"] = len(runs) - len(values)
            groups.append(group)
    return groups


def collect(directory: Path) -> dict:
    from inspect_ai.log import read_eval_log

    plan = json.loads((directory / "plan.json").read_text())
    if sha256(Path(plan["source"]["archive"])) != plan["source"]["archive_sha256"]:
        raise ValueError("source archive integrity failure")
    journal = directory / "runs.jsonl"
    records = []
    seen = set()
    for line in journal.read_text().splitlines() if journal.exists() else []:
        entry = json.loads(line)
        key = tuple(entry["key"])
        if key in seen or list(key) not in plan["order"]:
            raise ValueError("duplicate or undeclared comparison trial")
        seen.add(key)
        log = read_eval_log(entry["log"])
        record = trial_record(log, key, plan)
        archive = Path(record["archive"])
        manifest = json.loads((archive / "manifest.json").read_text())
        for name, expected in manifest.items():
            if sha256(archive / name) != expected:
                raise ValueError(f"run archive integrity failure: {name}")
        saved_log = json.loads((archive / "eval-log.json").read_text())
        if saved_log["sha256"] != sha256(Path(entry["log"])):
            raise ValueError("evaluation log integrity failure")
        records.append(record)
    return {"plan": plan, "trials": records, "summary": summarize(records, plan)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = collect(args.directory)
    write_json(args.directory / "report.json", report)
    for name in ("trials", "summary"):
        rows = report[name]
        if rows:
            with (args.directory / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Audited {len(report['trials'])}/{len(report['plan']['order'])} planned trials")


if __name__ == "__main__":
    main()
