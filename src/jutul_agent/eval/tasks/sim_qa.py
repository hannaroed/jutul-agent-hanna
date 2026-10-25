"""Simulation-output questions: the thesis benchmark run through jutul-agent.

The benchmark lives in the repository's ``bench/`` folder, apart from the agent
code: saved simulation output (``bench/data``), scientist-phrased questions
(``bench/questions``), oracle answers computed in Julia (``bench/answers.json``)
and a unit-aware grader (``bench/grader.py``). This module only adapts it to
Inspect so the stock harness can be measured on it. Every harness gets the same
prompt: the dataset's context, where its files are and how to read them, the
question, and the answer format.

    inspect eval src/jutul_agent/eval/tasks/sim_qa.py@sim_qa --model ... \
        -T dataset=co2_storage -T tier=easy -T harness=H2
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.eval import ntnu_provider  # noqa: F401  (registers --model ntnu/...)
from jutul_agent.eval.scorers import (
    context_compaction_count,
    token_usage,
    tool_call_count,
    tool_result_character_count,
)
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()

BENCH_ENV_VAR = "JUTUL_BENCH_DIR"

_ANSWER_FORMAT = {
    "quantity": '{"value": <number>, "unit": "<unit>"}',
    "choice": '{"choice": "<name>"}',
}

PROMPT = """{context}

The saved output is in {location}. It is JLD2 data of plain Julia dictionaries \
and arrays, all in SI units; read it with `Jutul.JLD2.load(path)`. Analyse the \
saved output; do not re-run the simulation.

Question: {question}

End your reply with the answer as one JSON object on its own line: {answer_format}"""


def bench_dir() -> Path:
    """The benchmark folder: ``$JUTUL_BENCH_DIR`` or ``bench/`` in this checkout."""
    configured = os.environ.get(BENCH_ENV_VAR)
    return Path(configured) if configured else Path(__file__).resolve().parents[4] / "bench"


@cache
def _grader() -> Any:
    spec = importlib.util.spec_from_file_location("bench_grader", bench_dir() / "grader.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before running it: dataclasses look their module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _location(data: Path, dataset: str) -> str:
    path = data / dataset
    if path.is_dir():
        files = sorted(p.name for p in path.glob("*.jld2"))
        return f"the folder {path} ({len(files)} files: {files[0]} ... {files[-1]})"
    return f"the file {path}.jld2"


def load_samples(
    dataset: str | None = None, tier: str | None = None, ids: str | list[str] | None = None
) -> list[Sample]:
    """One sample per bench question, optionally filtered."""
    root = bench_dir()
    answers = json.loads((root / "answers.json").read_text())
    wanted = set(ids.split(",") if isinstance(ids, str) else ids or [])
    samples = []
    for path in sorted((root / "questions").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if dataset and doc["dataset"] != dataset:
            continue
        for q in doc["questions"]:
            if (tier and q["tier"] != tier) or (wanted and q["id"] not in wanted):
                continue
            prompt = PROMPT.format(
                context=doc["context"].strip(),
                location=_location(root / "data", doc["dataset"]),
                question=q["question"].strip(),
                answer_format=_ANSWER_FORMAT[q["answer"]["type"]],
            )
            samples.append(
                Sample(
                    id=q["id"],
                    input=prompt,
                    metadata={
                        "needs_env": True,
                        "simulator": "jutuldarcy",
                        "dataset": doc["dataset"],
                        "tier": q["tier"],
                        "tags": q.get("tags", {}),
                        "answer_spec": q["answer"],
                        "oracle": answers[q["id"]],
                        "expected": json.dumps(answers[q["id"]]),
                    },
                )
            )
    return samples


@scorer(metrics=[accuracy(), stderr()])
def bench_answer() -> Scorer:
    """Grade the final reply's JSON answer against the oracle, unit-aware."""

    async def score(state: TaskState, target: Target) -> Score:
        grader = _grader()
        text = state.output.completion if state.output else ""
        answer = grader.parse_answer(text)
        result = grader.grade(answer, state.metadata["oracle"], state.metadata["answer_spec"])
        return Score(
            value=CORRECT if result.correct else INCORRECT,
            answer=json.dumps(answer) if answer is not None else None,
            explanation=result.reason,
            metadata={"tier": state.metadata["tier"], "value": result.value_in_expected_unit},
        )

    return score


@task
def sim_qa(
    dataset: str | None = None,
    tier: str | None = None,
    ids: str | None = None,
    harness: str = "H1",
) -> Task:
    """``harness`` is a condition name: H0, H1, H2, H2-noskill or H2-nobudget."""
    return Task(
        dataset=load_samples(dataset, tier, ids),
        solver=jutul_agent_solver(harness_strategies=HarnessStrategies.condition(harness)),
        scorer=[
            bench_answer(),
            token_usage(),
            tool_call_count(),
            tool_result_character_count(),
            context_compaction_count(),
        ],
        time_limit=1800,
        message_limit=120,
    )
