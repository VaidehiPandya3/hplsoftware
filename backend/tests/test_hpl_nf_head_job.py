#!/usr/bin/env python3
"""The HPL pipeline's head job runs under the same watchdog as ANORAK's, and
must hand it the same things.

submit_hpl_nf.py used to build its own copy of the supervisor command, and the
copy fell behind: no --stop-marker, so a real pipeline failure left nothing to
stop the chain and every --chain standby started, resumed and failed in turn;
no --signal-lead-seconds, so the TERM budget matched Slurm's --signal only
because both happened to be 120. And hpl-nf finished its run on the two
failures CephFS produces most (an unreadable .exitcode, and exit 1), while its
own refusals exited 1 too, so the config could tell neither apart — that
half is now tested beside the pipeline, in hpl-nf/tools/test_workflow_*.py.

These run the real builders. Runs under pytest and
standalone.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import submit_anorak_nf  # noqa: E402
import submit_hpl_nf  # noqa: E402


def _config(tmp_path: Path) -> dict:
    return {
        "out_dir": str(tmp_path / "run"), "manifest": str(tmp_path / "manifest.csv"),
        "backend_dir": str(BACKEND), "tiling": {}, "packaging": {},
        "extraction": {}, "assignment": {},
    }


def _supervisor_argv(sbatch_command: str) -> list[str]:
    """The nf_supervise.sh arguments inside the sbatch --wrap string."""
    tokens = shlex.split(sbatch_command)
    wrap = tokens[tokens.index("--wrap") + 1]
    inner = shlex.split(wrap.splitlines()[-1])
    assert inner[0] == "exec", inner
    return inner[1:]


def _option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def test_the_head_job_passes_the_stop_marker_and_signal_lead(tmp_path):
    info = submit_hpl_nf.submit_pipeline(_config(tmp_path), {}, dry_run=True)
    argv = _supervisor_argv(info["sbatch_command"])
    out_dir = Path(info["out_dir"])
    assert _option(argv, "--stop-marker") == str(out_dir / submit_anorak_nf.SUPERVISOR_STOP_MARKER)
    lead = _option(argv, "--signal-lead-seconds")
    signal = re.search(r"--signal=B:TERM@(\d+)", info["sbatch_command"])
    assert lead is not None and signal and lead == signal.group(1), (lead, signal)
    # HPL keeps its own stall limit (HPL_NF_WATCHDOG_STALL_SECONDS).
    assert _option(argv, "--stall-seconds") == str(submit_hpl_nf.WATCHDOG_STALL_SECONDS)


def test_a_resubmission_clears_the_previous_stop_marker(tmp_path):
    config = _config(tmp_path)
    marker = Path(config["out_dir"]) / submit_anorak_nf.SUPERVISOR_STOP_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("nextflow exited 1 on its own\n")

    class _Result:
        stdout = "Submitted batch job 4242\n"
        stderr = ""

    original = submit_hpl_nf._run_sbatch_with_retry
    submit_hpl_nf._run_sbatch_with_retry = lambda command: _Result()
    try:
        info = submit_hpl_nf.submit_pipeline(config, {}, chain=1)
    finally:
        submit_hpl_nf._run_sbatch_with_retry = original
    assert not marker.exists(), "a head job would find the old marker and stop at once"
    assert info["cleared_stop_marker"].startswith("nextflow exited 1")


def test_a_resume_submits_with_the_runs_own_settings(tmp_path):
    """Resume used to rebuild the Nextflow params from defaults, so a run started
    with 50 concurrent tiling tasks resumed with 10 — and on whatever partition
    was current. The submission records them; resume reads them back."""
    recorded = {"max_tiling_forks": 50, "cpu_partition": "compute-low-priority",
                "gpu_partition": "gpu", "gpu_gres": "gpu:nvidia_a100_80gb_pcie:1",
                "assign_partition": "gpu", "assign_device": "cpu",
                "allow_incomplete": "false", "tiling_time": "172800s"}
    config = {**_config(tmp_path), "nf_params": recorded, "allow_incomplete": False}
    resumed, nf = submit_hpl_nf.resume_params(config)
    for key in ("max_tiling_forks", "cpu_partition", "assign_partition", "tiling_time"):
        assert nf[key] == recorded[key], key
    assert resumed["allow_incomplete"] is False and nf["allow_incomplete"] == "false"

    resumed, nf = submit_hpl_nf.resume_params(config, allow_incomplete=True)
    assert resumed["allow_incomplete"] is True and nf["allow_incomplete"] == "true"
    assert config["allow_incomplete"] is False, "resume changed the caller's recorded config"


def test_a_config_without_recorded_settings_is_not_resumed_on_defaults(tmp_path):
    try:
        submit_hpl_nf.resume_params(_config(tmp_path))
    except ValueError:
        return
    raise AssertionError("resumed a run whose settings were never recorded")


def test_the_submission_records_its_settings_for_a_resume(tmp_path):
    config = _config(tmp_path)
    submit_hpl_nf.submit_pipeline(config, {"max_tiling_forks": 50}, dry_run=True)
    written = json.loads((Path(config["out_dir"]) / "run_config.json").read_text())
    assert written["nf_params"] == {"max_tiling_forks": 50}


def test_moving_outputs_aside_keeps_them_and_only_them(tmp_path):
    """A full run over a dataset an earlier run packaged is refused, or — when
    asked — moves that run's outputs into superseded-<stamp>/. Nothing is
    deleted, the stage's leftovers go with their output, and a neighbour that
    only shares the stem (a test .h5) is left where it is."""
    ds = tmp_path / "model_input" / "DS"
    ds.mkdir(parents=True)
    h5 = ds / "hdf5_DS_he_train.h5"
    family = [h5, ds / "hdf5_DS_he_train.h5.partial", ds / "hdf5_DS_he_train.h5.completed.txt"]
    neighbour = ds / "hdf5_DS_he_train_test_sample_ab12.h5"
    for f in (*family, neighbour):
        f.write_text("x")
    res = tmp_path / "results"
    res.mkdir()
    csv = res / "DS_hpc_assignments.csv"
    parts = [csv, res / "DS_hpc_assignments.rows0-5.csv", res / "DS_hpc_assignments.query_mean.npy"]
    for f in parts:
        f.write_text("x")
    (res / "DS_hpc_assignments.csv.chunks").mkdir()

    moved = submit_hpl_nf.move_outputs_aside(
        {"h5": h5, "projections": res / "absent.h5", "assignments": csv}, "20260929-120000")

    assert neighbour.exists(), "a file that only shares the stem was moved"
    for f in (*family, *parts, res / "DS_hpc_assignments.csv.chunks"):
        assert not f.exists(), f"{f.name} was left in the new run's way"
        assert (f.parent / "superseded-20260929-120000" / f.name).exists(), f"{f.name} was lost"
    assert len(moved) == len(family) + len(parts) + 1


def test_a_complete_earlier_output_is_refused_without_the_move(tmp_path):
    import h5py
    import numpy as np

    h5 = tmp_path / "hdf5_DS_he_train.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("img", (2, 4, 4, 3), dtype="uint8")
        for name in ("samples", "slides", "tiles"):
            f.create_dataset(name, data=np.array([b"x", b"y"]))
    paths = {"h5": h5, "projections": tmp_path / "p.h5", "assignments": tmp_path / "a.csv"}
    try:
        submit_hpl_nf.refuse_foreign_outputs(paths)
    except FileExistsError as e:
        assert "Move earlier outputs aside" in str(e), "the refusal does not name the way out"
        return
    raise AssertionError("a fresh run was allowed to adopt another run's complete .h5")


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_nf_head_job_"))
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
