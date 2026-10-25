"""Feed the frozen pilot prompts to an untouched upstream agent snapshot."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "output/experiments/qwen35-memory-pilot-2026-10-13"
FOLLOWUP = REPO / "output/experiments/sintef-result-retrieval-followup-2026-10-13"
OUTPUT = REPO / "output/experiments/qwen35-original-baseline-2026-10-13"
PROJECT = REPO / "output/experiments/post-cleanup-one-trial-2026-10-12/runtime/julia-project"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def verify_plan(identifier):
    plan = json.loads((OUTPUT / "plan.json").read_text())
    if identifier not in plan["order"]:
        raise ValueError("sample is not in the frozen comparison plan")
    for path, expected in (
        (PILOT / "plan.json", plan["fork_reference_plan_sha256"]),
        (PILOT / "runs.jsonl", plan["fork_reference_journal_sha256"]),
        (OUTPUT / "samples.json", plan["paired_samples_sha256"]),
        (Path(__file__), plan["driver_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen comparison input changed: {path}")
    pilot = json.loads((PILOT / "plan.json").read_text())
    for name in ("policy", "model", "generation", "endpoint", "seed_policy"):
        if plan[name] != pilot[name]:
            raise ValueError(f"comparison differs from pilot: {name}")


def prepare():
    baseline = json.loads((FOLLOWUP / "plan.json").read_text())["baseline"]
    archive = Path(baseline["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == baseline["archive_sha256"]
    source = OUTPUT / "runtime/upstream"
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            (source / member.name).resolve().relative_to(source.resolve())
            if not (member.isfile() or member.isdir()):
                raise ValueError("upstream snapshot must contain only regular files")
        if not source.exists():
            source.mkdir(parents=True)
            bundle.extractall(source, filter="data")
        files = {}
        for member in members:
            if member.isfile():
                expected = bundle.extractfile(member).read()
                assert (source / member.name).read_bytes() == expected
                files[member.name] = hashlib.sha256(expected).hexdigest()
    for name in ("pyproject.toml", "uv.lock"):
        assert (source / name).read_bytes() == (REPO / name).read_bytes()
    write(OUTPUT / "runtime/upstream-files.json", files)
    write(OUTPUT / "runtime/source.json", baseline)
    sys.path.insert(0, str(source / "src"))
    return source


def model():
    from langchain.chat_models import init_chat_model

    result = init_chat_model(
        "openai:inspect", disable_streaming=True, api_key="inspect-agent-bridge", max_tokens=2048
    )
    result.profile = {**(result.profile or {}), "max_input_tokens": 16384}
    return result


async def start(workspace):
    from jutul_agent.paths import eval_sessions_state_root, set_state_home, set_workspace_root
    from jutul_agent.session_host import SessionHost
    from jutul_agent.simulators.registry import get

    set_state_home(OUTPUT / "state")
    set_workspace_root(workspace)
    return await SessionHost.start(
        simulator=get("jutuldarcy"),
        surface="tui",
        model=model(),
        approval_mode="auto",
        workspace=workspace,
        state_root=eval_sessions_state_root("jutuldarcy"),
        julia_project=PROJECT,
        ephemeral_memory=True,
        prepare_env=False,
        discover=False,
        warmup=False,
        virtual_display=False,
        threads="1",
    )


def verify_imports(source):
    imported = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name.startswith("jutul_agent") and getattr(module, "__file__", None)
    }
    assert imported
    assert all(Path(path).is_relative_to(source / "src") for path in imported.values())
    return imported


async def preflight(source):
    workspace = OUTPUT / "runtime/preflight-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    host = await start(workspace)
    try:
        write(
            OUTPUT / "runtime/preflight.json",
            {
                "model_calls": 0,
                "prepared_julia_project": str(PROJECT),
                "loaded_modules": verify_imports(source),
                "native_agent_constructed": True,
                "model_context_window": 16384,
                "model_max_output_tokens": 2048,
            },
        )
    finally:
        await host.aclose()
    print("Original agent constructed successfully; no model calls or simulations.")


async def run_trial(source, identifier):
    from inspect_ai import Task, eval_async
    from inspect_ai.agent import AgentState, agent_bridge
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.solver import solver

    # This observer only copies artifacts. All agent modules come from upstream.
    path = REPO / "src/jutul_agent/eval/archive.py"
    spec = importlib.util.spec_from_file_location("baseline_archive_observer", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    saved = json.loads((OUTPUT / "samples.json").read_text())
    sample = Sample(**next(item for item in saved if item["id"] == identifier))
    sample.metadata.pop("harness_strategies", None)
    trial = sample.metadata["trial"]
    output = OUTPUT / "trials" / identifier
    if output.exists():
        raise ValueError("trial already exists; inspect it instead of overwriting")
    workspace = output / "workspace"
    workspace.mkdir(parents=True)
    archive = output / "archive"
    for name, content in sample.metadata["fixtures"].items():
        target = workspace / name
        target.resolve().relative_to(workspace.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    @solver
    def original_session():
        async def solve(state, generate):
            from jutul_agent.eval.solver import _final_text

            state.store.set("jutul/workspace", str(workspace))
            state.store.set("jutul/archive", str(archive))
            agent_state = AgentState(messages=list(state.messages))
            host = None
            outcome = "error"
            try:
                async with agent_bridge(agent_state):
                    host = await start(workspace)
                    state.store.set("jutul/trace_db", str(host.session.state_dir / "trace.sqlite"))
                    state.store.set("jutul/output_dir", str(host.session.output_dir))
                    observer.write_json(archive / "loaded-modules.json", verify_imports(source))
                    result = await host.runner.run_prompt(state.input_text)
                    for prompt, updates in zip(
                        state.metadata["follow_up_prompts"],
                        state.metadata["follow_up_fixtures"],
                        strict=True,
                    ):
                        if result.interrupts:
                            break
                        for name, content in updates.items():
                            target = workspace / name
                            target.resolve().relative_to(workspace.resolve())
                            target.write_text(content)
                        result = await host.runner.run_prompt(prompt)
                    if result.interrupts:
                        raise RuntimeError("original agent paused in auto approval mode")
                    final = _final_text(result.messages)
                    verify_imports(source)
                    outcome = "returned"
            finally:
                if host is not None:
                    memory = host.session._ephemeral_memory_dir
                    if memory is not None and memory.exists():
                        observer.archive_tree(memory, archive / "memory")
                    await host.aclose()
                observer.write_json(archive / "execution.json", {"outcome": outcome})
                observer.finish_archive(
                    archive,
                    workspace,
                    session_dir=host.session.state_dir if host is not None else None,
                )
            state.messages = agent_state.messages
            state.output = ModelOutput.from_content(model=get_model().name, content=final)
            return state

        return solve

    plan = json.loads((PILOT / "plan.json").read_text())
    write(output / "sample.json", sample.model_dump(mode="json"))
    write(
        output / "instrumentation.json",
        {
            "upstream_revision": json.loads((FOLLOWUP / "plan.json").read_text())["baseline"][
                "revision"
            ],
            "bridge": "OpenAI Inspect bridge; 16k input profile and 2048 output, as in the pilot",
            "staged_turns": "same seven user prompts and fixture updates as the pilot",
            "observer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "agent_source_modified": False,
        },
    )
    logs = await eval_async(
        Task(
            name="original-result-memory",
            version=plan["policy"],
            dataset=[sample],
            solver=original_session(),
            time_limit=900,
            token_limit=750000,
            message_limit=90,
        ),
        model=plan["model"],
        model_base_url=plan["endpoint"],
        model_args={"api_key": "local-vllm", "responses_api": False},
        log_dir=str(output / "logs"),
        max_samples=1,
        max_tasks=1,
        fail_on_error=False,
        log_model_api=True,
        seed=1000 + trial,
        **plan["generation"],
    )
    write(output / "log.json", {"path": logs[0].location})


async def grade(log_path, destination, original=False):
    from dataclasses import replace

    from inspect_ai import score_async
    from inspect_ai.log import read_eval_log, write_eval_log

    from jutul_agent import paths
    from jutul_agent.eval.campaigns import CAMPAIGNS
    from jutul_agent.eval.result_memory import result_memory_task
    from jutul_agent.eval.result_study import CONDITIONS
    from jutul_agent.simulators import registry

    log = read_eval_log(log_path)
    (sample,) = log.samples
    campaign = next(c for c in CAMPAIGNS if c.name == sample.metadata["campaign"]["name"])
    task = result_memory_task(
        campaign,
        PILOT / "reference" / campaign.name,
        PROJECT,
        phase=sample.metadata["phase"],
        strategies=CONDITIONS[0],
        trial=sample.metadata["trial"],
    )
    expected = task.dataset[0]
    assert sample.input == expected.input and sample.target == expected.target
    for name in (
        "fixtures",
        "final_fixtures",
        "follow_up_prompts",
        "follow_up_fixtures",
        "source_rows",
        "query_cell_id",
        "scoring_policy",
    ):
        assert sample.metadata[name] == expected.metadata[name], name

    shared, lookup = paths.SHARED_SKILLS_DIR, registry.get
    try:
        if original:
            # The unchanged integrity scorer permits the evaluated agent's own skills.
            source = OUTPUT / "runtime/upstream/src/jutul_agent/simulators"
            paths.SHARED_SKILLS_DIR = source / "shared_skills"
            registry.get = lambda name: replace(lookup(name), module_dir=source / lookup(name).name)
        scored = await score_async(log, task.scorer, model="mockllm/model", action="overwrite")
    finally:
        paths.SHARED_SKILLS_DIR, registry.get = shared, lookup
    write_eval_log(scored, destination)
    (result,) = scored.samples
    return {name: score.value for name, score in result.scores.items()}


async def audit_pilot():
    rows = [json.loads(line) for line in (PILOT / "runs.jsonl").read_text().splitlines()]
    results = []
    for index, row in enumerate(rows, 1):
        target = OUTPUT / "runtime/scorer-check" / f"trial-{index:02d}.eval"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores = await grade(Path(row["log"]), target)
        assert scores == row["scores"], row["key"]
        results.append({"key": row["key"], "unchanged_scores": True})
    write(OUTPUT / "runtime/scorer-compatibility.json", {"model_calls": 0, "checks": results})
    print("Common scoring reproduced all 14 archived outcomes exactly; no model calls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", help="one prepared sample ID; default only checks construction")
    parser.add_argument("--grade", type=Path, help="grade one original-agent log offline")
    parser.add_argument("--audit-pilot", action="store_true", help="rescore archived pilot copies")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "JUTUL_AGENT_JULIA_THREADS",
        "JUTUL_AGENT_HYPRE_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if args.audit_pilot:
        asyncio.run(audit_pilot())
        return
    if args.grade:
        target = args.grade.with_name("graded.eval")
        print(json.dumps(asyncio.run(grade(args.grade, target, original=True)), indent=2))
        return
    if args.trial:
        verify_plan(args.trial)
        # Keep the verifier in a separate process so fork modules cannot enter the agent.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; "
                "from jutul_agent.eval.result_study import verify_server; "
                "verify_server(Path(sys.argv[1]))",
                str(OUTPUT),
            ],
            cwd=REPO,
            check=True,
            timeout=90,
        )
    source = prepare()
    asyncio.run(run_trial(source, args.trial) if args.trial else preflight(source))


if __name__ == "__main__":
    main()
