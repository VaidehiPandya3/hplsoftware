#!/usr/bin/env python3
"""The server and the stepper's side of a pipeline run (hpl-nf/).

  * A stage is COMPLETED only on its done marker. A head job that exited 0,
    or aged out of accounting, without a stage finishing reads FAILED — never
    COMPLETED, which would hide the Resume button behind a stage that did not
    happen.
  * Sentinels never reach sacct/squeue, and a finished stage resolves from its
    run directory alone.
  * The stepper's pipeline view says "fails validation" rather than "done"
    when the task and the server disagree.

The pipeline's own per-stage guards and its stub run are tested beside it, in
hpl-nf/tools/test_workflow_*.py. Runs under pytest and standalone.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import hpl_nf_state as state  # noqa: E402


# --- stage state ------------------------------------------------------------------

def test_a_stage_is_completed_only_by_its_marker(tmp_path):
    out = tmp_path
    for head in ("COMPLETED", "", "FAILED", "TIMEOUT"):
        got = state.stage_state(out, "packaging", head)
        assert got != "COMPLETED", f"head {head!r} with no marker read as COMPLETED"
    assert state.stage_state(out, "packaging", "COMPLETED") == "FAILED"
    assert state.stage_state(out, "packaging", "") == "FAILED"
    assert state.stage_state(out, "packaging", None) is None
    assert state.stage_state(out, "packaging", "RUNNING") == "PENDING"
    state.mark_started(out, "packaging")
    assert state.stage_state(out, "packaging", "RUNNING") == "RUNNING"
    state.mark_done(out, "packaging", {"tiles": 3})
    # The marker is the evidence; Slurm being unreachable does not undo it.
    assert state.stage_state(out, "packaging", None) == "COMPLETED"
    assert state.read_done(out, "packaging") == {"tiles": 3}


def test_a_chain_folds_to_one_state(_tmp=None):
    fold = state.combine_head_states
    assert fold(["TIMEOUT", "COMPLETED", "CANCELLED"]) == "COMPLETED"
    assert fold(["TIMEOUT", "RUNNING", "PENDING"]) == "RUNNING"
    assert fold(["RUNNING", "PENDING"]) == "RUNNING"
    # A standby cleared for an invalid dependency is not how the run ended.
    assert fold(["FAILED", "CANCELLED", "CANCELLED"]) == "FAILED"
    assert fold(["FAILED", None]) is None
    assert fold([]) is None


def test_a_sentinel_cannot_name_another_directory(_tmp=None):
    assert state.parse_nf_job_id("nf:abc-123:tiling") == ("abc-123", "tiling")
    for bad in ("nf:../etc:tiling", "nf:a/b:tiling", "nf:abc:registration", "local:tiling", "123"):
        try:
            state.parse_nf_job_id(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted as a pipeline sentinel")


# --- the stepper's pipeline view ------------------------------------------------------

def _stage_states_fn():
    source = (REPO / "app" / "app_v28.py").read_text()
    start = source.index("_PIPELINE_STAGES = (")
    end = source.index("def _render_pipeline_overview")
    namespace = {"_SLURM_IN_FLIGHT": {"PENDING", "RUNNING", "CONFIGURING"}}
    exec(compile(source[start:end], "probe", "exec"), namespace)
    return namespace["_pipeline_stage_states"]


def _pipeline_status(states: dict, **ready) -> dict:
    return {
        "pipeline": {"stages": {k: {"state": v} for k, v in states.items()}},
        **ready,
    }


def test_the_stepper_never_calls_an_unvalidated_stage_done(_tmp=None):
    fn = _stage_states_fn()
    computed = tuple(("done", f"summary {i}") for i in range(4))

    # Every task said COMPLETED, but the server rejects the .h5.
    status = _pipeline_status(
        {"tiling": "COMPLETED", "packaging": "COMPLETED",
         "extraction": "COMPLETED", "assignment": "COMPLETED"},
        tiling_complete=True, h5_ready=False, extraction_ready=True, assignment_ready=True,
    )
    tiling, packaging, extraction, assignment = fn(status, computed)
    assert tiling == ("done", "summary 0")
    assert packaging[0] == "attention" and "validation" in packaging[1]

    # A run that stopped at extraction: that stage failed, the next not reached.
    status = _pipeline_status(
        {"tiling": "COMPLETED", "packaging": "COMPLETED",
         "extraction": "FAILED", "assignment": "FAILED"},
        tiling_complete=True, h5_ready=True,
    )
    states = [s for s, _ in fn(status, computed)]
    assert states == ["done", "done", "failed", "blocked"], states

    # Running: the next stage waits for this one.
    status = _pipeline_status(
        {"tiling": "RUNNING", "packaging": "PENDING",
         "extraction": "PENDING", "assignment": "PENDING"},
    )
    states = fn(status, computed)
    assert states[0][0] == "running" and states[1] == ("blocked", "waits for tiling")


# --- the server never asks Slurm about a sentinel ----------------------------------------

def test_the_server_resolves_pipeline_stages_without_asking_slurm(tmp_path):
    try:
        import tile_server_v2_ as srv
    except Exception as e:  # noqa: BLE001 - no openslide/DB on this machine
        print(f"  (skipped: tile server not importable here: {e})")
        return

    real, sentinels = srv._split_local_job_ids(["123", "nf:abc:tiling", "local:tiling"])
    assert real == ["123"] and sentinels == ["nf:abc:tiling", "local:tiling"]

    original_root, original_run = srv.HPL_NF_RESULTS_ROOT, srv.subprocess.run
    srv.HPL_NF_RESULTS_ROOT = tmp_path
    state.mark_done(tmp_path / "abc", "tiling", {"slides": 1})

    def no_slurm(*a, **k):
        raise AssertionError(f"a sentinel reached Slurm: {a}")

    srv.subprocess.run = no_slurm
    try:
        assert srv._get_slurm_array_state_counts(["nf:abc:tiling"]) == {"COMPLETED": 1}
        assert srv._get_slurm_job_state("nf:abc:tiling") == "COMPLETED"
    finally:
        srv.HPL_NF_RESULTS_ROOT, srv.subprocess.run = original_root, original_run


def test_a_path_alone_is_a_complete_pipeline_request(_tmp=None):
    """The UI sends only the dataset path. Everything else is the server's:
    the checkpoint (HPL_CHECKPOINT), 50 slides at a time, and earlier outputs
    moved aside rather than refused, since a one-click run has no other way
    forward — never deleted."""
    try:
        import tile_server_v2_ as srv
        import submit_hpl_nf
    except Exception as e:  # noqa: BLE001 - no openslide/DB on this machine
        print(f"  (skipped: tile server not importable here: {e})")
        return
    req = srv.PipelineRunRequest(dataset_path="/data/NewCohort")
    assert req.checkpoint is None and submit_hpl_nf.DEFAULT_CHECKPOINT
    assert req.max_concurrent == submit_hpl_nf.DEFAULT_MAX_TILING
    assert req.move_existing_outputs is True
    assert req.allow_incomplete is False, "a one-click run must not drop slides unasked"
    defaults = srv.pipeline_defaults()
    assert defaults["checkpoint"] == submit_hpl_nf.DEFAULT_CHECKPOINT
    assert defaults["reference"].endswith(".npz")


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_nf_pipeline_"))
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
