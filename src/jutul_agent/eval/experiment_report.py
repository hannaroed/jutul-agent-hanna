"""Report campaign results and recorded settings for each sample."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from jutul_agent.eval.log_values import numeric_score, score_value
from jutul_agent.paths import state_home


@dataclass
class ExperimentRecord:
    run: str
    trial: int | None
    condition: str
    task: str
    model: str
    completed: bool  # Legacy record key: Inspect reported no sample error.
    protocol_correct: bool | None
    winner_correct: bool | None
    numeric_correct: bool | None
    grounded_evidence: bool | None
    tool_calls: int | None
    file_ops: int | None
    compactions: int | None
    filesystem_result_chars: int | None
    total_tokens: int
    wall_s: float | None
    context_window: int | None
    summary_prompt_sha256: str | None
    summary_sanitization: str | None
    memory_strategy: str | None
    path_policy: str | None
    tool_scheduling: str | None


def _correct(sample: Any, name: str) -> bool | None:
    value = score_value(sample, name)
    return None if value is None else str(value) == "C"


def _protocol_correct(sample: Any) -> bool | None:
    """Combine the task's trace-level execution and recovery contracts."""
    names = (
        "julia_case_call_coverage",
        "julia_case_recovered_from_ledger",
        "julia_function_call_count",
    )
    values = [score_value(sample, name) for name in names]
    observed = [value for value in values if value is not None]
    return None if not observed else all(str(value) == "C" for value in observed)


def _condition(task: str, metadata: dict[str, Any], runconfig: dict[str, Any]) -> str:
    """Human-readable intervention derived only from recorded run metadata."""
    context_window = runconfig.get("context_window") or metadata.get("context_window")
    summary = runconfig.get("summarization_prompt_sha256")
    memory = metadata.get("memory_strategy")
    path_policy = runconfig.get("path_policy") or metadata.get("path_policy")
    tool_scheduling = runconfig.get("tool_scheduling")
    verification = metadata.get("verification_strategy")
    precision = metadata.get("answer_precision_policy")
    recovery = metadata.get("recovery_strategy")
    summary_sanitization = runconfig.get("summary_sanitization")

    if task.endswith("experiment_state"):
        if context_window is None:
            return "Unaware 16k baseline"
        if summary is None:
            return "Window-aware generic compaction"
        return "Scientific-state compaction"
    if task.endswith("experiment_ledger"):
        return "Structured artifact ledger"
    if task.endswith("experiment_targeted"):
        if memory == "structured-ledger-targeted-retrieval-stable-locators":
            return "Targeted retrieval + verified artifact locators"
        if path_policy == "agent-visible-to-real-path-resolution-v1":
            return "Targeted retrieval + path-contract fix"
        if path_policy == "real-paths-with-workspace-alias-recovery":
            return "Targeted retrieval + /workspace alias"
        if memory == "structured-ledger-targeted-retrieval":
            return "Targeted retrieval (unresolved paths)"
    if task.endswith("experiment_autonomous"):
        if (
            verification == "compaction-safe-ordered-verification-gate-v2"
            and precision == "six-significant-digits-value-specific-v1"
        ):
            return "Autonomous campaign + action-aware compaction + 6-digit check"
        if verification == "compaction-safe-ordered-verification-gate-v2":
            return "Autonomous campaign + action-aware compaction"
        if verification == "tool-returned-completion-gate-v1":
            return "Autonomous campaign + completion marker"
        if tool_scheduling == "sequential-provider-enforced-v1":
            return "Autonomous campaign + sequential scheduling"
        return "Autonomous campaign (parallel scheduling)"
    if task.endswith("experiment_recovery"):
        if (
            recovery == "checkpoint-read-retry-gate-v2"
            and summary_sanitization == "strip-internal-reasoning-capsules-v1"
        ):
            return "Autonomous fault recovery + sanitized compaction"
        if recovery == "checkpoint-read-retry-gate-v2":
            return "Autonomous fault recovery + compaction-safe retry gate"
        if verification == "compaction-safe-ordered-verification-gate-v2":
            return "Autonomous fault recovery + action-aware verification"
        return "Autonomous campaign + fault recovery"
    if task.endswith("experiment_adaptive"):
        return "Budgeted adaptive experimental design"
    return memory or task.rsplit("/", 1)[-1]


_FILE_TOOLS = frozenset({"read_file", "write_file", "edit_file", "ls", "glob", "grep"})
_RETRIEVAL_TOOLS = frozenset({"glob", "read_file", "grep"})


def _trace_metrics(sample: Any) -> dict[str, int] | None:
    """Recover comparable efficiency metrics directly from the saved trace."""
    trace_path = (sample.store or {}).get("jutul/trace_db")
    if not trace_path or not Path(trace_path).exists():
        return None
    metrics = {"tool_calls": 0, "file_ops": 0, "compactions": 0, "filesystem_chars": 0}
    uri = f"file:{trace_path}?immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        for kind, payload_json in database.execute(
            "select kind,payload_json from events order by id"
        ):
            payload = json.loads(payload_json)
            name = str(payload.get("name", ""))
            if kind == "tool_call":
                metrics["tool_calls"] += 1
                if name in _FILE_TOOLS:
                    metrics["file_ops"] += 1
            elif kind == "context_compaction":
                metrics["compactions"] += 1
            elif kind == "tool_result" and name in _RETRIEVAL_TOOLS:
                metrics["filesystem_chars"] += len(str(payload.get("content", "")))
    return metrics


def collect(prefixes: list[str], log_dir: Path | None = None) -> list[ExperimentRecord]:
    """Collect one record per sample without aggregating intervention runs."""
    from inspect_ai.log import read_eval_log

    root = log_dir or (state_home() / "eval-logs")
    paths = sorted({path for prefix in prefixes for path in root.glob(f"{prefix}*.eval")})
    records: list[ExperimentRecord] = []
    for path in paths:
        log = read_eval_log(path)
        for sample in log.samples or []:
            metadata = sample.metadata or {}
            if not str(sample.id).startswith("experiment-"):
                continue
            runconfig = (sample.store or {}).get("jutul/runconfig") or {}
            metrics = _trace_metrics(sample)
            if metrics is None:
                metrics = {
                    "tool_calls": numeric_score(sample, "tool_call_count"),
                    "file_ops": numeric_score(sample, "file_op_count"),
                    "compactions": numeric_score(sample, "context_compaction_count"),
                    "filesystem_chars": numeric_score(sample, "tool_result_character_count"),
                }
            total_tokens = 0
            for usage in (sample.model_usage or {}).values():
                total_tokens += usage.input_tokens or 0
                total_tokens += usage.input_tokens_cache_read or 0
                total_tokens += usage.input_tokens_cache_write or 0
                total_tokens += usage.output_tokens or 0
            task_name = log.eval.task
            records.append(
                ExperimentRecord(
                    run=path.stem,
                    trial=getattr(sample, "epoch", None),
                    condition=_condition(task_name, metadata, runconfig),
                    task=task_name.rsplit("/", 1)[-1],
                    model=log.eval.model,
                    completed=getattr(sample, "error", None) is None,
                    protocol_correct=_protocol_correct(sample),
                    winner_correct=_correct(sample, "campaign_winner")
                    if score_value(sample, "campaign_winner") is not None
                    else _correct(sample, "answer_cites"),
                    numeric_correct=_correct(sample, "numeric_values_close"),
                    grounded_evidence=_correct(sample, "tool_result_numeric_evidence"),
                    tool_calls=metrics["tool_calls"],
                    file_ops=metrics["file_ops"],
                    compactions=metrics["compactions"],
                    filesystem_result_chars=metrics["filesystem_chars"],
                    total_tokens=total_tokens,
                    wall_s=getattr(sample, "total_time", None),
                    context_window=runconfig.get("context_window")
                    or metadata.get("context_window"),
                    summary_prompt_sha256=runconfig.get("summarization_prompt_sha256"),
                    summary_sanitization=runconfig.get("summary_sanitization"),
                    memory_strategy=metadata.get("memory_strategy"),
                    path_policy=runconfig.get("path_policy") or metadata.get("path_policy"),
                    tool_scheduling=runconfig.get("tool_scheduling"),
                )
            )
    return records


def _mark(value: bool | None) -> str:
    if value is None:
        return "—"
    return "pass" if value else "fail"


def _number(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _rate(records: list[ExperimentRecord], field: str) -> str:
    values = [getattr(record, field) for record in records]
    observed = [value for value in values if value is not None]
    if not observed:
        return "—"
    return f"{sum(observed)}/{len(observed)} ({100 * sum(observed) / len(observed):.0f}%)"


def _mean(records: list[ExperimentRecord], field: str) -> str:
    values = [getattr(record, field) for record in records]
    observed = [value for value in values if value is not None]
    if not observed:
        return "—"
    return f"{fmean(observed):,.1f}"


def _no_error_rate(records: list[ExperimentRecord]) -> str:
    returned = sum(record.completed for record in records)
    return f"{returned}/{len(records)} ({100 * returned / len(records):.0f}%)"


def render_markdown(records: list[ExperimentRecord]) -> str:
    """Render a compact, auditable intervention table."""
    lines = [
        "# Long-horizon experiment-state trajectory",
        "",
        (
            "Each audit row is one evaluation sample. Fixed campaigns request six real "
            "JutulDarcy simulations; adaptive campaigns allow three. Conditions may be successive "
            "harness interventions or independent trials of the same intervention; repeated "
            "conditions are summarized before the sample-level audit table. Total tokens "
            "are cumulative across model calls, not the per-call context-window size. "
            "A returned execution has no recorded sample error; it can still leave the "
            "campaign unfinished or fail its scoring criteria."
        ),
        "",
    ]

    grouped: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[record.condition].append(record)
    repeated = {condition: group for condition, group in grouped.items() if len(group) > 1}
    if repeated:
        lines.extend(
            [
                "## Replicated-condition summary",
                "",
                (
                    "| Condition | n | No sample error | Protocol | Winner | Numerical criterion | "
                    "File-tool row evidence | "
                    "Mean tool calls | Mean file ops | Mean compactions | Mean FS chars |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, group in repeated.items():
            lines.append(
                f"| {condition} | {len(group)} | {_no_error_rate(group)} | "
                f"{_rate(group, 'protocol_correct')} | "
                f"{_rate(group, 'winner_correct')} | "
                f"{_rate(group, 'numeric_correct')} | {_rate(group, 'grounded_evidence')} | "
                f"{_mean(group, 'tool_calls')} | {_mean(group, 'file_ops')} | "
                f"{_mean(group, 'compactions')} | {_mean(group, 'filesystem_result_chars')} |"
            )
        lines.extend(["", "## Sample-level audit", ""])

    lines.extend(
        [
            (
                "| Condition | Trial | Execution | Protocol | Winner | Numerical criterion | "
                "File-tool row evidence | "
                "Tool calls | File ops | Compactions | FS chars | Total tokens | Wall |"
            ),
            "|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        wall = "—" if record.wall_s is None else f"{record.wall_s / 60:.1f} min"
        lines.append(
            f"| {record.condition} | {_number(record.trial)} | "
            f"{'returned' if record.completed else 'error'} | {_mark(record.protocol_correct)} | "
            f"{_mark(record.winner_correct)} | "
            f"{_mark(record.numeric_correct)} | {_mark(record.grounded_evidence)} | "
            f"{_number(record.tool_calls)} | {_number(record.file_ops)} | "
            f"{_number(record.compactions)} | {_number(record.filesystem_result_chars)} | "
            f"{record.total_tokens:,} | {wall} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility keys",
            "",
            (
                "| Run | Trial | Context | Summary prompt | Summary sanitization | "
                "Memory strategy | Path policy | Tool scheduling |"
            ),
            "|---|---:|---:|---|---|---|---|---|",
        ]
    )
    for record in records:
        summary = record.summary_prompt_sha256[:10] if record.summary_prompt_sha256 else "—"
        lines.append(
            f"| `{record.run}` | {_number(record.trial)} | {_number(record.context_window)} | "
            f"`{summary}` | "
            f"{record.summary_sanitization or '—'} | "
            f"{record.memory_strategy or '—'} | {record.path_policy or '—'} | "
            f"{record.tool_scheduling or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="+", help="eval-log filename prefixes")
    parser.add_argument("-o", "--output", type=Path, help="write Markdown here")
    parser.add_argument("--json", type=Path, help="write one JSON record per line")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    records = collect(args.prefixes, log_dir=args.log_dir)
    if not records:
        print("no experiment-state samples found", file=sys.stderr)
        return 1
    if args.json:
        args.json.write_text(
            "\n".join(json.dumps(asdict(record)) for record in records) + "\n",
            encoding="utf-8",
        )
    markdown = render_markdown(records)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.output} ({len(records)} runs)")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
