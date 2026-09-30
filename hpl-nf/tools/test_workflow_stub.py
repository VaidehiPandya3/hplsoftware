#!/usr/bin/env python3
"""The hpl-nf workflow itself: its DAG, its config, and its watchdog.

  * The real pipeline, in Nextflow's stub mode, runs every process — shards
    fanned out and collected — and writes every stage marker the server reads.
  * nextflow.config finishes the run on a refusal (65) and retries everything
    else; conf/beatson.config waits long enough for CephFS.
  * tools/nf_supervise.sh is ANORAK's watchdog, byte for byte — one script,
    because two stall detectors are two chances for one to be stale.

Runs under pytest and standalone.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
REPO = PIPELINE.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(PIPELINE / "bin"))
sys.path.insert(0, str(BACKEND))

import hpl_common  # noqa: E402
import hpl_nf_state as state  # noqa: E402


def test_the_config_finishes_on_a_refusal_and_retries_everything_else(_tmp=None):
    config = (PIPELINE / "nextflow.config").read_text(encoding="utf-8")
    assert f"task.exitStatus == {hpl_common.REFUSAL_EXIT_CODE} ? 'finish'" in config
    assert "task.attempt <= 2 ? 'retry'" in config
    beatson = (PIPELINE / "conf" / "beatson.config").read_text(encoding="utf-8")
    assert re.search(r"exitReadTimeout\s*=\s*'15 min'", beatson)
    # The watchdog reads the monitor's silence; its heartbeat must be pinned.
    assert re.search(r"dumpInterval\s*=\s*'5 min'", beatson)


def test_the_watchdog_is_anoraks_byte_for_byte(_tmp=None):
    ours = PIPELINE / "tools" / "nf_supervise.sh"
    theirs = REPO / "anorak-nf" / "tools" / "nf_supervise.sh"
    assert ours.is_file(), f"{ours} is missing — the head job cannot start without it"
    assert ours.read_bytes() == theirs.read_bytes(), "the two pipelines' watchdogs differ"

    import submit_hpl_nf
    assert submit_hpl_nf.supervisor_drift(ours) is None


def test_a_drifted_watchdog_is_refused_at_submission(tmp_path):
    import submit_hpl_nf
    stale = tmp_path / "nf_supervise.sh"
    stale.write_text("#!/bin/bash\nexec \"$@\"\n")
    assert submit_hpl_nf.supervisor_drift(stale), "a different watchdog passed as the same one"


# --- the real pipeline, stubbed ------------------------------------------------------------

def test_the_stub_pipeline_marks_every_stage(tmp_path):
    if not shutil.which("nextflow"):
        print("  (skipped: nextflow not on PATH)")
        return
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("/data/a b.svs\n/data/c.svs\n")
    config = tmp_path / "run_config.json"
    config.write_text("{}")
    out = tmp_path / "out"
    result = subprocess.run(
        ["nextflow", "-q", "run", str(PIPELINE), "-profile", "stub", "-stub",
         "-work-dir", str(tmp_path / "work"),
         "--config", str(config), "--manifest", str(manifest), "--outdir", str(out),
         "--python", sys.executable, "--backend_dir", str(BACKEND),
         "--stub_extract_shards", "3", "--stub_assign_shards", "2"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    for stage in state.STAGES:
        assert state.stage_state(out, stage, "RUNNING") == "COMPLETED", stage
    trace = (out / "pipeline_info" / "trace.txt").read_text()
    assert trace.count("EXTRACT_SHARD") == 3 and trace.count("ASSIGN_SHARD") == 2


def test_a_finished_stage_asks_for_no_gpu_and_still_verifies(tmp_path):
    """A plan answering "skip" (output already validates) must queue no shard —
    a GPU job to print "nothing to do" — and FINISH must still run, because it
    is the step that re-validates and writes the done marker."""
    if not shutil.which("nextflow"):
        print("  (skipped: nextflow not on PATH)")
        return
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("/data/a.svs\n")
    config = tmp_path / "run_config.json"
    config.write_text("{}")
    out = tmp_path / "out"
    result = subprocess.run(
        ["nextflow", "-q", "run", str(PIPELINE), "-profile", "stub", "-stub",
         "-work-dir", str(tmp_path / "work"),
         "--config", str(config), "--manifest", str(manifest), "--outdir", str(out),
         "--python", sys.executable, "--backend_dir", str(BACKEND),
         "--stub_extract_shards", "0", "--stub_assign_shards", "0"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    trace = (out / "pipeline_info" / "trace.txt").read_text()
    assert "EXTRACT_SHARD" not in trace and "ASSIGN_SHARD" not in trace, trace
    assert "EXTRACT_FINISH" in trace and "ASSIGN_FINISH" in trace, trace
    for stage in state.STAGES:
        assert state.stage_state(out, stage, "RUNNING") == "COMPLETED", stage


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_nf_stub_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
