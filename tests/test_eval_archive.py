"""Run archives preserve dirty source, WAL data, and artifact integrity."""

import json
import sqlite3
import subprocess
import tarfile

import pytest

from jutul_agent.eval.archive import capture_source, finish_archive
from jutul_agent.trace import TraceLog


def test_source_snapshot_distinguishes_dirty_untracked_and_deleted_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src").mkdir()
    code = repo / "src/model.py"
    code.write_text("version = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "src/model.py"], check=True)
    (repo / ".env").write_text("SECRET=excluded")
    (repo / "src/.env.local").write_text("SECRET=also-excluded")
    first = capture_source(repo, tmp_path / "snapshots")
    assert capture_source(repo, tmp_path / "snapshots") == first
    code.write_text("version = 2\n")
    (repo / "src/new.py").write_text("new = True\n")
    second = capture_source(repo, tmp_path / "snapshots")
    assert first["sha256"] != second["sha256"]
    with tarfile.open(second["archive"]) as bundle:
        assert set(bundle.getnames()) == {"src/model.py", "src/new.py"}
        assert bundle.extractfile("src/model.py").read() == b"version = 2\n"
    code.unlink()
    third = capture_source(repo, tmp_path / "snapshots")
    assert third["sha256"] != second["sha256"]
    with tarfile.open(third["archive"]) as bundle:
        assert bundle.getnames() == ["src/new.py"]


def test_source_archive_detects_corruption_and_refuses_symlink(tmp_path):
    (tmp_path / "src").mkdir()
    code = tmp_path / "src/model.py"
    code.write_text("safe\n")
    archive = capture_source(tmp_path, tmp_path / "snapshots")
    code.unlink()
    code.symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("outside")
    with pytest.raises(ValueError, match="symlink"):
        capture_source(tmp_path, tmp_path / "snapshots")
    code.unlink()
    code.write_text("safe\n")
    with tarfile.open(archive["archive"], "w"):
        pass
    with pytest.raises(ValueError, match="integrity"):
        capture_source(tmp_path, tmp_path / "snapshots")


def test_run_archive_keeps_artifacts_and_live_wal_trace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.csv").write_text("run_id,value\nrun-1,42\n")
    (workspace / ".env").write_text("SECRET=excluded")
    session = tmp_path / "session"
    with TraceLog(session / "trace.sqlite") as trace:
        trace.append("tool_call", {"id": "latest-in-wal"})
        destination = tmp_path / "archive"
        manifest = finish_archive(destination, workspace, session_dir=session)
        with sqlite3.connect(destination / "session/trace.sqlite") as connection:
            assert connection.execute("select count(*) from events").fetchone()[0] == 1
    assert "workspace/result.csv" in manifest
    assert "workspace/.env" not in manifest
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    (workspace / "result.csv").unlink()
    assert "42" in (destination / "workspace/result.csv").read_text()
