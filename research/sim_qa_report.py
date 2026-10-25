"""Summarise sim_qa eval logs: one row per (harness, question).

    uv run --no-sync python research/sim_qa_report.py [LOG ...] > report.md

With no arguments, reads every sim_qa log in .jutul-agent/idun/eval-logs. Token
counts are the agent's own model_usage events (input includes cached tokens).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

LOG_DIR = Path(__file__).resolve().parents[1] / ".jutul-agent" / "idun" / "eval-logs"


HEADER = "| Harness | Question | Tier | Correct | Answer | Grader | Input tok | Cached | Peak | Tools | Time s |"  # noqa: E501


def _failure(sample) -> str:
    if not sample.error:
        return "no score"
    message = sample.error.message
    for marker in ("ContextWindowExceeded", "CancelledError", "time limit"):
        if marker in message:
            return f"error: {marker}"
    return "error: " + " ".join(message.split())[:60]


def _trace_tokens(sample) -> dict:
    """Token use from the session trace, for a sample that errored before scoring."""
    db = sample.store.get("jutul/trace_db")
    if not db or not Path(db).exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    query = "select payload_json from events where kind = 'model_usage'"
    usage = [json.loads(p) for (p,) in con.execute(query)]
    con.close()
    return {
        "input": sum(u.get("input_tokens") or 0 for u in usage),
        "cached": sum((u.get("input_token_details") or {}).get("cache_read") or 0 for u in usage),
        "output": sum(u.get("output_tokens") or 0 for u in usage),
        "peak_input": max((u.get("input_tokens") or 0 for u in usage), default=0),
    }


def rows(paths: list[Path]):
    for path in paths:
        log = read_eval_log(str(path))
        if log.eval.task != "sim_qa" or not log.samples:
            continue
        harness = log.eval.task_args.get("harness", "H1")
        model = log.eval.model.split("/", 1)[-1]
        for s in log.samples:
            scores = s.scores or {}
            answer = scores.get("bench_answer")
            usage = scores.get("token_usage")
            calls = scores.get("tool_call_count")
            tokens = usage.value if usage and isinstance(usage.value, dict) else _trace_tokens(s)
            yield {
                "harness": harness,
                "model": model,
                "id": s.id,
                "tier": s.metadata.get("tier"),
                "correct": answer is not None and answer.value == "C",
                "answer": answer.answer if answer else None,
                "why": answer.explanation if answer else _failure(s),
                "input": tokens.get("input", 0),
                "cached": tokens.get("cached", 0),
                "output": tokens.get("output", 0),
                "peak": tokens.get("peak_input", 0),
                "tools": int(calls.value) if calls else 0,
                "time_s": round(s.total_time or 0),
            }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(LOG_DIR.glob("*.eval"))
    table = list(rows(paths))
    print(HEADER)
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        print(
            f"| {r['harness']} | {r['id']} | {r['tier']} | {'yes' if r['correct'] else 'NO'} | "
            f"{r['answer']} | {r['why']} | {r['input']:,} | {r['cached']:,} | {r['peak']:,} | "
            f"{r['tools']} | {r['time_s']} |"
        )
    print("\n| Harness | Tier | Correct | Mean input tok | Mean tools |")
    print("|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in table:
        groups[(r["harness"], r["tier"])].append(r)
    for (harness, tier), rs in sorted(groups.items()):
        n_ok = sum(r["correct"] for r in rs)
        mean_input = sum(r["input"] for r in rs) // len(rs)
        mean_tools = sum(r["tools"] for r in rs) / len(rs)
        print(f"| {harness} | {tier} | {n_ok}/{len(rs)} | {mean_input:,} | {mean_tools:.1f} |")


if __name__ == "__main__":
    main()
