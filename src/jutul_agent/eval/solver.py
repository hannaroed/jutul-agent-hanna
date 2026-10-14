"""Run one full jutul-agent session per Inspect sample.

The solver builds the same stack a real session uses (workspace, Julia
kernel, :class:`Session`, ``build_agent``, ``TurnRunner``) inside Inspect's
``agent_bridge``. The bridge intercepts the model client, so the eval's
``--model`` decides which provider actually answers while the agent's tools,
skills, prompt, and trace run unchanged. Scorers grade through the sample
store, which records where the session's workspace and trace database ended
up (see :data:`STORE_WORKSPACE`, :data:`STORE_TRACE_DB`).

Sample metadata understood by the solver:

- ``fixtures``: mapping of workspace-relative path -> file content, written
  before the turn starts.
- ``needs_env``: instantiate the simulator's Julia environment in the
  workspace first (slow on a cold depot; off by default).
- ``needs_display``: start a managed virtual display so GLMakie plotting
  works headless.
- ``context_window``: input-context budget for bridged models.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_ai.agent import AgentState, agent_bridge
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from jutul_agent.agent.text import strip_internal_capsules

STORE_WORKSPACE = "jutul/workspace"
STORE_TRACE_DB = "jutul/trace_db"
STORE_SESSION_ID = "jutul/session_id"
STORE_OUTPUT_DIR = "jutul/output_dir"
STORE_RUNCONFIG = "jutul/runconfig"
STORE_ARCHIVE = "jutul/archive"
STORE_SOURCE = "jutul/source"

# One session at a time: the workspace root is process-global state
# (``paths.set_workspace_root``), so concurrent samples would race on it.
_SESSION_LOCK = asyncio.Lock()


def load_eval_credentials() -> None:
    """Load provider keys the way the app does (global + cwd ``.env``).

    Call at task-module import time: Inspect resolves the eval's model (and
    checks its key) before any solver runs.
    """
    from dotenv import load_dotenv

    from jutul_agent.credentials import load_user_credentials

    load_user_credentials()
    load_dotenv()


def _bridge_model(context_window: int | None = None) -> Any:
    """The model instance the agent under test calls.

    The literal model name ``inspect`` is what the agent bridge intercepts;
    requests never reach the network, so the API key only has to satisfy the
    client constructor. The client family has to match the eval's target
    provider in two cases:

    - Gemini targets use the google client: Gemini 3 requires its
      ``thought_signature`` to be replayed with function calls, and only the
      google-format round-trip preserves it.
    - OpenAI targets use the Responses client so encrypted reasoning items
      survive tool-call replay. An Anthropic-shaped bridge replaces those
      opaque items with signatures, which OpenAI cannot decrypt on the next turn.
    - Other providers use the Anthropic client, whose plain ``messages.create``
      calls the bridge understands.

    Streaming is disabled because the bridge rejects streaming requests; the
    agent's event stream still works, the HTTP call just resolves in one piece.
    """
    from langchain.chat_models import init_chat_model

    target = get_model()
    provider = str(target).partition("/")[0]
    if provider in ("google", "vertex"):
        from jutul_agent.eval import _gemini_compat

        _gemini_compat.apply()
        spec = "google_genai:inspect"
    elif provider in ("openai", "openai-api"):
        spec = "openai:inspect"
    else:
        spec = "anthropic:inspect"
    options = (
        {"use_responses_api": provider == "openai"} if provider in ("openai", "openai-api") else {}
    )
    max_tokens = getattr(getattr(target, "config", None), "max_tokens", None)
    if isinstance(max_tokens, int) and max_tokens > 0:
        options["max_tokens"] = max_tokens
    model = init_chat_model(
        spec,
        disable_streaming=True,
        api_key="inspect-agent-bridge",
        **options,
    )
    if context_window is not None:
        if context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        profile = getattr(model, "profile", None)
        updated = dict(profile) if isinstance(profile, dict) else {}
        updated["max_input_tokens"] = context_window
        model.profile = updated
    return model


# The bridge annotates assistant text with internal capsules that must not
# reach scorers: a <content-internal> metadata tag, and (for Gemini) a redacted
# <think> reasoning signature. Both carry base64 a number-extracting check would
# happily mine digits out of, so they are stripped before the answer is graded.
def _final_text(messages: list[Any]) -> str:
    """Text of the last assistant message, flattened from content blocks."""
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        content = message.content
        if isinstance(content, str):
            return strip_internal_capsules(content).strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            ]
            return strip_internal_capsules("\n".join(p for p in parts if p)).strip()
        return strip_internal_capsules(str(content)).strip()
    return ""


# Simulators whose golden env has been built or re-resolved by this process;
# the alignment check runs once per run, not once per sample.
_ALIGNED_ENVS: set[str] = set()


def _golden_env(adapter: Any, simulator: str) -> Path:
    """A prepared workspace env for this simulator, built once and kept aligned.

    Instantiating a simulator env from scratch takes minutes; one prepared
    copy lives under the jutul-agent home (``eval-envs/<sim>``) and each
    sample starts from a copy of it. Copying preserves mtimes, so the
    precompile marker stays current and the per-sample env prep is a cheap
    reconcile instead of a full instantiate.

    Envs carry no version pins, so a cached copy freezes whatever upstream
    served when it was built and would silently grade against old versions
    after a new release. On its first use in a run, a cached env is therefore
    re-resolved against the registry (``update_env``); if that fails, the run
    fails rather than grade a misaligned environment.

    The golden is then brought fully current once with ``prepare_workspace_env``
    (the normal launch reconcile): the in-env JutulAgent runtime is refreshed and
    marked, and the env precompiled. Without this, a golden whose warm-source
    marker is stale (e.g. cached before the runtime changed) makes *every*
    per-sample copy re-copy the runtime and re-bake the env — turning a one-time
    cost into a per-sample one. Aligning the golden lets each copy's reconcile be
    the intended cheap no-op.
    """
    from jutul_agent.paths import state_home
    from jutul_agent.simulators.env_setup import (
        _workspace_env_lock,
        bootstrap_workspace,
        prepare_workspace_env,
        update_env,
    )

    golden = state_home() / "eval-envs" / simulator
    env = golden / ".jutul-agent" / "julia-env"
    if not (env / "Manifest.toml").exists():
        golden.mkdir(parents=True, exist_ok=True)
        bootstrap_workspace(adapter, workspace=golden, precompile=True)
    elif simulator not in _ALIGNED_ENVS:
        with _workspace_env_lock(golden):
            update_env(env)
    if simulator not in _ALIGNED_ENVS:
        prepare_workspace_env(adapter, workspace=golden, julia_project=env, sim_name=simulator)
    _ALIGNED_ENVS.add(simulator)
    return env


async def _run_jutul_session(
    *,
    simulator: str,
    prompt: str,
    fixtures: Mapping[str, str],
    needs_env: bool,
    needs_display: bool,
    context_window: int | None,
    harness_strategies: Any,
    julia_project_override: str | None,
    scratch: Path,
    store: Any,
    ground_truth: str | None = None,
    follow_up_prompts: list[str] | None = None,
    follow_up_fixtures: list[dict[str, str]] | None = None,
) -> str:
    from jutul_agent.paths import set_workspace_root
    from jutul_agent.simulators import registry
    from jutul_agent.trace.schema import EVAL_TARGET

    adapter = registry.get(simulator)
    workspace = scratch / "workspace"
    workspace.mkdir(parents=True)
    store.set(STORE_WORKSPACE, str(workspace))
    for rel, content in fixtures.items():
        path = (workspace / rel).resolve()
        path.relative_to(workspace.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    set_workspace_root(workspace)

    julia_project = None
    if needs_env:
        import shutil

        from jutul_agent.simulators.env_setup import prepare_workspace_env
        from jutul_agent.workspace import resolve_julia_project

        julia_project = (
            Path(julia_project_override).resolve()
            if julia_project_override
            else resolve_julia_project(workspace)
        )
        if julia_project_override and not (julia_project / "Manifest.toml").is_file():
            raise ValueError("the supplied Julia project must have an instantiated Manifest.toml")
        if not julia_project_override and julia_project == workspace / ".jutul-agent" / "julia-env":
            golden = await asyncio.to_thread(_golden_env, adapter, simulator)
            await asyncio.to_thread(shutil.copytree, golden, julia_project, dirs_exist_ok=True)
        if not julia_project_override:
            await asyncio.to_thread(
                prepare_workspace_env,
                adapter,
                workspace=workspace,
                julia_project=julia_project,
                sim_name=simulator,
            )

    from jutul_agent.eval.runconfig import build_runconfig

    store.set(
        STORE_RUNCONFIG,
        build_runconfig(
            adapter,
            julia_project=julia_project,
            context_window=context_window,
            strategies=harness_strategies,
        ),
    )

    from jutul_agent.eval.archive import archive_tree, write_json

    archive = Path(store.get(STORE_ARCHIVE))
    write_json(archive / "runconfig.json", store.get(STORE_RUNCONFIG))
    if julia_project is not None:
        archive_tree(julia_project, archive / "julia-environment")

    # Session traces remain discoverable through `jutul-agent review`; the
    # solver also archives the complete workspace and trace on every exit path.
    from jutul_agent.paths import eval_sessions_state_root
    from jutul_agent.session_host import SessionHost

    # The shared bootstrap, parameterized for a hermetic run: the env was
    # prepared above (golden copy), no ambient entry-point capabilities, no
    # background warm-up competing with the sample's own Julia work, and a
    # virtual display only when the task declares it needs one.
    host = await SessionHost.start(
        simulator=adapter,
        surface="tui",
        model=_bridge_model(context_window),
        approval_mode="auto",
        workspace=workspace,
        state_root=eval_sessions_state_root(simulator),
        julia_project=julia_project,
        ephemeral_memory=True,
        prepare_env=False,
        discover=False,
        warmup=False,
        virtual_display=needs_display,
        harness_strategies=harness_strategies,
    )
    try:
        store.set(STORE_WORKSPACE, str(workspace))
        store.set(STORE_TRACE_DB, str(host.session.state_dir / "trace.sqlite"))
        store.set(STORE_SESSION_ID, host.session.session_id)
        store.set(STORE_OUTPUT_DIR, str(host.session.output_dir))
        from jutul_agent.eval.runconfig import _sha256_text
        from jutul_agent.trace import TraceLog

        with TraceLog.open_readonly(host.session.state_dir / "trace.sqlite") as trace:
            actual = trace.first_payload("eval_harness_config")
        config = store.get(STORE_RUNCONFIG)
        if actual:
            config["prompt_sha256"] = _sha256_text(actual["system_prompt"])
            config["summarization_prompt_sha256"] = _sha256_text(actual["summary_prompt"])
            write_json(archive / "harness.json", actual)
        store.set(STORE_RUNCONFIG, config)
        write_json(archive / "runconfig.json", config)
        # Record the known answer so a later review can judge the result against
        # ground truth, the sharpest signal an eval run offers.
        if ground_truth:
            host.session.trace.append(EVAL_TARGET, {"expected": ground_truth})
        result = await _run_prompts(
            host, workspace, prompt, follow_up_prompts or [], follow_up_fixtures or []
        )
    finally:
        try:
            memory = host.session._ephemeral_memory_dir
            if memory is not None and memory.exists():
                archive_tree(memory, Path(store.get(STORE_ARCHIVE)) / "memory")
        finally:
            await host.aclose()

    if result.interrupts:
        return "[eval-error] the turn paused for tool approval in auto mode"
    return _final_text(result.messages)


async def _run_prompts(host, workspace, prompt, follow_ups, updates):
    if not isinstance(follow_ups, list) or any(not isinstance(text, str) for text in follow_ups):
        raise TypeError("follow_up_prompts must be a list of strings")
    if updates and len(updates) != len(follow_ups):
        raise ValueError("follow_up_fixtures must match follow_up_prompts")
    if not isinstance(updates, list) or any(not isinstance(item, dict) for item in updates):
        raise TypeError("follow_up_fixtures must be a list of dictionaries")
    changes = updates or [{} for _ in follow_ups]
    for fixtures in changes:
        for name, content in fixtures.items():
            (workspace / name).resolve().relative_to(workspace.resolve())
            if not isinstance(content, str):
                raise TypeError("follow-up fixture content must be text")
    result = await host.runner.run_prompt(prompt)
    for text, fixtures in zip(follow_ups, changes, strict=True):
        if result.interrupts:
            break
        for name, content in fixtures.items():
            path = (workspace / name).resolve()
            path.relative_to(workspace.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        result = await host.runner.run_prompt(text)
    return result


@solver
def jutul_agent_solver(simulator: str = "jutuldarcy") -> Solver:
    """Solve each sample by handing it to a fresh jutul-agent session."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        fixtures = state.metadata.get("fixtures", {})
        needs_env = bool(state.metadata.get("needs_env", False))
        needs_display = bool(state.metadata.get("needs_display", False))
        context_window = state.metadata.get("context_window")
        if context_window is not None and (
            not isinstance(context_window, int) or isinstance(context_window, bool)
        ):
            raise TypeError("sample metadata context_window must be an integer")
        sim = state.metadata.get("simulator", simulator)
        from jutul_agent.agent.strategies import HarnessStrategies
        from jutul_agent.eval.archive import capture_source, finish_archive, write_json
        from jutul_agent.paths import state_home

        strategies = HarnessStrategies(**(state.metadata.get("harness_strategies") or {}))
        identifier = uuid.uuid4().hex
        archive = state_home() / "eval-archives" / identifier
        store = state.store
        store.set(STORE_ARCHIVE, str(archive))
        source_root = Path(__file__).resolve().parents[3]
        if not (source_root / "src" / "jutul_agent").exists():
            source_root = Path(__file__).resolve().parents[2]
        source = await asyncio.to_thread(capture_source, source_root, state_home() / "eval-sources")
        store.set(STORE_SOURCE, source)
        model = get_model()
        write_json(
            archive / "sample.json",
            {
                "sample_id": state.sample_id,
                "epoch": state.epoch,
                "input": state.input_text,
                "target": state.target.text,
                "metadata": state.metadata,
                "source": source,
                "model": model.name,
                "generation": model.config.model_dump(exclude={"extra_headers"}),
            },
        )
        scratch = Path(tempfile.gettempdir()) / "jutul-agent-eval" / uuid.uuid4().hex[:8]
        scratch.mkdir(parents=True)

        agent_state = AgentState(messages=list(state.messages))
        outcome = "error"
        try:
            async with _SESSION_LOCK, agent_bridge(agent_state):
                from jutul_agent.paths import set_workspace_root, workspace_root

                previous_workspace = workspace_root()
                try:
                    final = await _run_jutul_session(
                        simulator=sim,
                        prompt=state.input_text,
                        fixtures=fixtures,
                        needs_env=needs_env,
                        needs_display=needs_display,
                        context_window=context_window,
                        harness_strategies=strategies,
                        julia_project_override=state.metadata.get("julia_project"),
                        scratch=scratch,
                        store=state.store,
                        ground_truth=state.metadata.get("expected"),
                        follow_up_prompts=state.metadata.get("follow_up_prompts"),
                        follow_up_fixtures=state.metadata.get("follow_up_fixtures"),
                    )
                finally:
                    set_workspace_root(previous_workspace)
            outcome = "returned"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            trace_path = store.get(STORE_TRACE_DB)
            write_json(archive / "execution.json", {"outcome": outcome})
            try:
                await asyncio.to_thread(
                    finish_archive,
                    archive,
                    scratch / "workspace",
                    session_dir=Path(trace_path).parent if trace_path else None,
                )
            except Exception as error:
                if outcome == "returned":
                    raise
                logging.getLogger(__name__).exception("Failed to finish evaluation archive")
                write_json(archive / "archive-error.json", {"error": str(error)})

        # The bridge tracked the real conversation; the completion graded by
        # scorers is the agent's own final message.
        state.messages = agent_state.messages
        state.output = ModelOutput.from_content(model=get_model().name, content=final)
        return state

    return solve
