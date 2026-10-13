"""Content-addressed source snapshots and durable per-sample run artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

_SOURCE_PATHS = (
    "src",
    "tests",
    "docs",
    "research",
    "pyproject.toml",
    "uv.lock",
    "AGENTS.md",
    "README.md",
    "jutul_agent",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def capture_source(root: Path, snapshots: Path) -> dict[str, str]:
    """Snapshot project source, including local changes; refuse symlinks."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *_SOURCE_PATHS,
        ],
        capture_output=True,
    )
    names = set(result.stdout.decode().split("\0")) - {""}
    if result.returncode:
        names = {
            path.relative_to(root).as_posix()
            for name in _SOURCE_PATHS
            for path in ([root / name] if (root / name).is_file() else (root / name).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
    files = {}
    for name in sorted(names):
        path = root / name
        relative = Path(name)
        if any(part.startswith(".env") for part in relative.parts):
            continue
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root):
            raise ValueError(f"source snapshot refuses symlink: {name}")
        if not path.exists():
            continue
        path.resolve().relative_to(root.resolve())
        if path.is_file():
            files[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    identifier = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    destination = snapshots / identifier
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "source.tar"
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as temporary_file:
            temporary = Path(temporary_file.name)
        with tarfile.open(temporary, "w") as bundle:
            for name, data in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o644
                bundle.addfile(entry, io.BytesIO(data))
        temporary.replace(archive)
        write_json(destination / "files.json", manifest)
    with tarfile.open(archive) as bundle:
        saved = {
            member.name: hashlib.sha256(bundle.extractfile(member).read()).hexdigest()
            for member in bundle.getmembers()
            if member.isfile()
        }
    if saved != manifest:
        raise ValueError("existing source snapshot failed integrity verification")
    return {"sha256": identifier, "archive": str(archive), "archive_sha256": sha256(archive)}


def archive_tree(source: Path, destination: Path) -> None:
    """Copy regular files only; SQLite backups include any uncheckpointed WAL."""
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("archive destination must be outside the source tree")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(
            part in (".git", "__pycache__", "run-archive") or part.startswith(".env")
            for part in relative.parts
        ):
            continue
        if path.is_symlink() or not path.is_file() or path.name.endswith(("-wal", "-shm")):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".sqlite":
            reader = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            writer = sqlite3.connect(target)
            try:
                reader.backup(writer)
            finally:
                writer.close()
                reader.close()
        else:
            shutil.copy2(path, target)


def finish_archive(
    destination: Path, workspace: Path, *, session_dir: Path | None = None
) -> dict[str, str]:
    """Preserve artifacts on success, failure, or cancellation, then inventory."""
    if workspace.exists():
        archive_tree(workspace, destination / "workspace")
    if session_dir is not None and session_dir.exists():
        archive_tree(session_dir, destination / "session")
    return inventory_archive(destination)


def inventory_archive(destination: Path) -> dict[str, str]:
    manifest = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path != destination / "manifest.json"
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def finalize_log(path: Path) -> None:
    """Save graded sample results beside the execution archive."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(path)
    for sample in log.samples or []:
        archive = (sample.store or {}).get("jutul/archive")
        if not archive:
            continue
        destination = Path(archive)
        write_json(destination / "result.json", sample.model_dump(mode="json"))
        write_json(
            destination / "eval-log.json",
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "eval": log.eval.model_dump(mode="json"),
            },
        )
        inventory_archive(destination)
