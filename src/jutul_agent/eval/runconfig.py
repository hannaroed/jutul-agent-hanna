"""RunConfig: hash every input that can change a bench score.

A score delta is only attributable when you can show what changed between
two runs. The RunConfig records a hash of each tunable input (assembled
system prompt, every active skill file, the instantiated Julia manifest)
plus the code version and dependency versions. Two runs that differ in
exactly one hash answer "did that change help?" mechanically; runs from a
dirty working tree are labelled so they are never mistaken for clean
comparisons.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
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
    from deepagents.middleware.summarization import DEEPAGENTS_DEFAULT_SUMMARY_PROMPT

    from jutul_agent import __version__
    from jutul_agent.agent.prompts import assemble_session_prompt
    from jutul_agent.agent.strategies import HarnessStrategies
    from jutul_agent.agent.summarization import (
        MESSAGE_HISTORY_SANITIZATION_POLICY,
        SCIENTIFIC_SUMMARY_PROMPT,
        SUMMARY_SANITIZATION_POLICY,
    )
    from jutul_agent.agent.tool_scheduling import TOOL_SCHEDULING_POLICY

    strategies = strategies or HarnessStrategies()
    summary_prompt = (
        SCIENTIFIC_SUMMARY_PROMPT
        if strategies.summarization == "scientific"
        else DEEPAGENTS_DEFAULT_SUMMARY_PROMPT
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
        "path_policy": "agent-visible-to-real-path-resolution-v1",
        "tool_scheduling": TOOL_SCHEDULING_POLICY
        if strategies.sequential_tools
        else "provider-default",
        "manifest_sha256": (
            _sha256_file(manifest) if manifest is not None and manifest.exists() else None
        ),
        "deps": {
            name: _dist_version(name)
            for name in ("deepagents", "langchain", "langgraph", "inspect-ai")
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.system().lower(),
        },
    }
