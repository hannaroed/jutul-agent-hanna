"""Run the sim_qa benchmark task on an NTNU-hosted model through Inspect.

    uv run --no-sync python research/ntnu_eval.py --model Qwen/Qwen3.8-27B-FP8 \
        --harness H2 --ids egg-prod2-final-water-cut

Uses the key `research/idun_connection.py setup` saved, scoped to this process
(NTNU_API_KEY), so personal OPENAI_* settings are left alone. The ntnu provider
(src/jutul_agent/eval/ntnu_provider.py) is registered here, before Inspect
resolves the model. Needs the NTNU VPN or campus network.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEY_FILE = REPO / ".jutul-agent" / "idun" / "api-key"
TASK = REPO / "src" / "jutul_agent" / "eval" / "tasks" / "sim_qa.py"
LOG_DIR = REPO / ".jutul-agent" / "idun" / "eval-logs"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="Exact NTNU model id, e.g. Qwen/Qwen3.8-27B-FP8"
    )
    parser.add_argument("--harness", default="H1", help="H0, H1, H2, H2-noskill or H2-nobudget")
    parser.add_argument("--dataset", help="Only this dataset's questions")
    parser.add_argument("--tier", choices=["easy", "medium", "hard"])
    parser.add_argument("--ids", help="Comma-separated question ids")
    parser.add_argument("--epochs", type=int, default=1, help="Repeats per question")
    args = parser.parse_args()
    if not KEY_FILE.is_file():
        print("No NTNU key yet: run research/idun_connection.py setup.", file=sys.stderr)
        return 1
    os.environ["NTNU_API_KEY"] = KEY_FILE.read_text().strip()

    from inspect_ai import eval as inspect_eval

    from jutul_agent.eval import ntnu_provider  # noqa: F401  (registers ntnu/...)

    task_args = {"harness": args.harness}
    for name in ("dataset", "tier", "ids"):
        if getattr(args, name):
            task_args[name] = getattr(args, name)
    logs = inspect_eval(
        f"{TASK}@sim_qa",
        model=f"ntnu/{args.model}",
        task_args=task_args,
        epochs=args.epochs,
        # One sample at a time: the solver runs one session at a time anyway, and
        # a queued sample's wait would count against its time limit.
        max_samples=1,
        # A sample that errors (e.g. its prompt outgrew the model's window) is a
        # failed sample, not a reason to stop the run.
        fail_on_error=False,
        log_dir=str(LOG_DIR),
        display="plain",
    )
    return 0 if all(log.results is not None for log in logs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
