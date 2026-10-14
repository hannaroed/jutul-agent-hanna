"""Recorded inputs for reproducible evaluation runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str | None:
    repo = Path(__file__).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo.parent), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def julia_environment_sha256(project: Path) -> str:
    """Hash the lockfiles and local sources that the run archive preserves."""
    root = project.resolve()
    manifest = tomllib.loads((root / "Manifest.toml").read_text())
    for entries in manifest.get("deps", {}).values():
        for entry in entries:
            if "path" in entry and not (root / entry["path"]).resolve().is_relative_to(root):
                raise ValueError("Julia path dependencies must be inside the archived project")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Julia environment refuses symlink: {relative}")
        if path.is_file():
            files[relative.as_posix()] = _sha256_file(path)
    if "Project.toml" not in files:
        raise ValueError("Julia environment requires Project.toml")
    return _sha256_text(json.dumps(files, sort_keys=True))


def _skill_hashes(adapter: Any) -> dict[str, str]:
    """One hash per active SKILL.md, keyed by its directory name."""
    from jutul_agent.paths import SHARED_SKILLS_DIR

    hashes: dict[str, str] = {}
    for root in (SHARED_SKILLS_DIR, adapter.skills_dir):
        if not root.exists():
            continue
        for skill in sorted(root.glob("*/SKILL.md")):
            hashes[skill.parent.name] = _sha256_file(skill)
    return hashes


def build_runconfig(
    adapter: Any,
    *,
    julia_project: Path | None = None,
    context_window: int | None = None,
    strategies: Any | None = None,
) -> dict[str, Any]:
    """The attribution record for one eval run (JSON-serializable)."""
    from jutul_agent import __version__
    from jutul_agent.agent.context_editing import (
        CLEARED_RESULT_PREFIX_LINES,
        CLEARED_RESULT_PREFIX_MAX_CHARS,
        SMALL_TOOL_RESULT_MAX_CHARS,
        filesystem_tool_result_limit,
    )
    from jutul_agent.agent.prompts import assemble_session_prompt
    from jutul_agent.agent.recovery import MODEL_OUTPUT_POLICY
    from jutul_agent.agent.strategies import HarnessStrategies
    from jutul_agent.agent.summarization import (
        GENERIC_SUMMARY_PROMPT,
        MESSAGE_HISTORY_SANITIZATION_POLICY,
        SCIENTIFIC_SUMMARY_PROMPT,
        SMALL_WINDOW_TOKEN_MARGIN,
        SUMMARY_POLICY,
        SUMMARY_SANITIZATION_POLICY,
    )
    from jutul_agent.agent.tool_scheduling import TOOL_SCHEDULING_POLICY

    strategies = strategies or HarnessStrategies()
    summary_prompt = (
        SCIENTIFIC_SUMMARY_PROMPT
        if strategies.summarization == "scientific"
        else GENERIC_SUMMARY_PROMPT
    )
    manifest = (julia_project / "Manifest.toml") if julia_project else None
    return {
        "jutul_agent": {
            "version": __version__,
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"]) or None),
        },
        "prompt_sha256": _sha256_text(assemble_session_prompt(adapter, open_windows=False)),
        "summarization_prompt_sha256": _sha256_text(summary_prompt),
        "summary_sanitization": SUMMARY_SANITIZATION_POLICY
        if strategies.sanitize_history
        else None,
        "message_history_sanitization": MESSAGE_HISTORY_SANITIZATION_POLICY
        if strategies.sanitize_history
        else None,
        "harness_strategies": strategies.to_dict(),
        "skills_sha256": _skill_hashes(adapter),
        "simulator": adapter.name,
        "context_window": context_window,
        "context_policy": "unscaled-full-request-budget-with-small-window-margin-v7",
        "small_window_token_margin": SMALL_WINDOW_TOKEN_MARGIN,
        "summary_policy": SUMMARY_POLICY,
        "tool_result_retention": {
            "filesystem_token_limit": filesystem_tool_result_limit(context_window),
            "small_result_max_chars": SMALL_TOOL_RESULT_MAX_CHARS,
            "cleared_prefix_lines": CLEARED_RESULT_PREFIX_LINES,
            "cleared_prefix_max_chars": CLEARED_RESULT_PREFIX_MAX_CHARS,
        },
        "path_policy": "agent-visible-to-real-path-resolution-v1",
        "model_output_policy": MODEL_OUTPUT_POLICY,
        "tool_scheduling": TOOL_SCHEDULING_POLICY
        if strategies.sequential_tools
        else "provider-default",
        "manifest_sha256": (
            _sha256_file(manifest) if manifest is not None and manifest.exists() else None
        ),
        "julia_environment_sha256": julia_environment_sha256(julia_project)
        if manifest is not None and manifest.exists()
        else None,
        "deps": {
            name: _dist_version(name)
            for name in (
                "deepagents",
                "langchain",
                "langchain-core",
                "langchain-openai",
                "langchain-anthropic",
                "langgraph",
                "inspect-ai",
            )
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.system().lower(),
        },
    }
