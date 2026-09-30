#!/usr/bin/env python3
"""Every stage of hpl-nf checks its output the way the server's gate does —
and these make each check fail, not only pass.

A stage's last task writes stages/<stage>.done.json only after the validator
the server's own gate uses accepts the output, bound to the row count of the
stage before: a short tile CSV, an .h5 of disagreeing lengths, a projections
file short of the packaged tile count and an assignments CSV of the wrong
length are each refused with exit 65 (bin/hpl_common.REFUSAL_EXIT_CODE, the
code nextflow.config finishes on) and no marker. A step whose output already
validates does nothing, and a stale extraction output is cleared before the
GPU is asked for.

These call the real bin/ wrappers. Runs under pytest and standalone.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

PIPELINE = Path(__file__).resolve().parent.parent
BIN = PIPELINE / "bin"
REPO = PIPELINE.parent
BACKEND = REPO / "backend"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(BACKEND))

import hpl_assign  # noqa: E402
import hpl_common  # noqa: E402
import hpl_extract  # noqa: E402
import hpl_nf_state as state  # noqa: E402
import hpl_package  # noqa: E402
import hpl_tile  # noqa: E402


# --- fixtures -----------------------------------------------------------------

def _config(tmp: Path, *, shards: int = 1) -> dict:
    out = tmp / "run"
    out.mkdir(parents=True, exist_ok=True)
    manifest = tmp / "manifest.txt"
    if not manifest.exists():
        manifest.write_text(f"{tmp / 'raw' / 'SLIDE1.svs'}\n")
    return {
        "out_dir": str(out),
        "backend_dir": str(BACKEND),
        "manifest": str(manifest),
        "mask_dir": str(tmp / "masks"),
        "tile_dir": str(tmp / "tiles"),
        "dataset_name": "DS",
        "tiling": {"min_tissue": 30.0},
        "packaging": {
            "output_root": str(tmp / "model_input"), "h5_dataset_name": "DS",
            "h5_path": str(tmp / "model_input" / "DS" / "hdf5_DS_he_train.h5"),
            "marker": "he", "split": "train", "tile_size": 224, "threads_per_process": 1,
        },
        "extraction": {
            "shards": shards, "checkpoint": "/ckpt/BarlowTwins_3.ckt",
            "output_path": str(tmp / "results" / "hdf5_DS_he_train.h5"),
            "hpl_repo_dir": "/nowhere", "singularity_image": "/nowhere.sif",
            "singularity_bin": "singularity", "extras_dir": "/nowhere",
            "dataset_name": "DS", "model": "BarlowTwins_3", "marker": "he",
            "z_dim": 128, "img_size": 224, "batch_size": 256,
        },
        "assignment": {
            "shards": shards, "rep_key": "z_latent", "reference": "/ref.npz",
            "out_csv": str(tmp / "results" / "DS_hpc_assignments.csv"), "vote": "tuned",
        },
    }


def _refused(fn, *args) -> bool:
    try:
        fn(*args)
    except SystemExit as e:
        assert e.code == hpl_common.REFUSAL_EXIT_CODE, f"exit {e.code}, not a refusal"
        return True
    return False


def _write_tiles(config: dict, saved: int, csv_rows: int) -> None:
    slide_dir = Path(config["tile_dir"]) / "DS" / "SLIDE1"
    slide_dir.mkdir(parents=True, exist_ok=True)
    (slide_dir / "SLIDE1_tiling_summary.json").write_text(json.dumps({"saved_tiles": saved}))
    rows = "".join(f"{i}_0.jpeg,{i},0\n" for i in range(csv_rows))
    (slide_dir / "SLIDE1_tile_metadata.csv").write_text("tile,col,row\n" + rows)


def _write_h5(path: Path, rows: int, slides_rows: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (rows, 4, 4, 3), dtype="uint8")
        f.create_dataset("samples", data=np.array([b"S"] * rows))
        f.create_dataset("slides", data=np.array([b"SLIDE1"] * (slides_rows or rows)))
        f.create_dataset("tiles", data=np.array([f"{i}_0.jpeg".encode() for i in range(rows)]))


def _write_projections(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("img_z_latent", data=np.zeros((rows, 8), "float32"))
        f.create_dataset("img_h_latent", data=np.zeros((rows, 8), "float32"))
        for name in ("samples", "slides", "tiles"):
            f.create_dataset(name, data=np.array([b"x"] * rows))


def _write_assignments(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "samples,slides,tiles,leiden_2.5,vote_margin,neighbor_distance,hpc_reference\n"
    path.write_text(header + "".join(f"S,SLIDE1,{i}_0.jpeg,3,0.5,0.1,ref\n" for i in range(rows)))


# --- Stage 1 ------------------------------------------------------------------------

def test_the_tiling_gate_refuses_a_csv_that_stopped_short(tmp_path):
    config = _config(tmp_path)
    _write_tiles(config, saved=3, csv_rows=2)
    assert _refused(hpl_tile.step_tiling_gate, config), "a short tile CSV passed the gate"
    assert state.read_done(config["out_dir"], "tiling") is None

    _write_tiles(config, saved=3, csv_rows=3)
    hpl_tile.step_tiling_gate(config)
    assert state.read_done(config["out_dir"], "tiling")["succeeded"] == 1


def test_the_tiling_gate_refuses_a_cohort_with_no_tiles_at_all(tmp_path):
    config = _config(tmp_path)
    _write_tiles(config, saved=0, csv_rows=0)
    assert _refused(hpl_tile.step_tiling_gate, config)
    assert state.read_done(config["out_dir"], "tiling") is None


def test_left_out_slides_are_refused_unless_the_run_allows_them(tmp_path):
    """One unreadable slide stops the run by default. Allowed, it is left out of
    what packaging reads — and named in the marker, not silently dropped."""
    (tmp_path / "manifest.txt").write_text(
        f"{tmp_path / 'raw' / 'SLIDE1.svs'}\n{tmp_path / 'raw' / 'BROKEN.svs'}\n")
    config = _config(tmp_path)
    _write_tiles(config, saved=3, csv_rows=3)          # SLIDE1 fine, BROKEN never tiled
    assert _refused(hpl_tile.step_tiling_gate, config)
    assert not hpl_common.packaged_manifest_path(config).exists()

    config["allow_incomplete"] = True
    hpl_tile.step_tiling_gate(config)
    done = state.read_done(config["out_dir"], "tiling")
    assert done["excluded"] == 1 and done["excluded_slides"] == ["BROKEN.svs"], done
    packaged = hpl_common.packaged_manifest_path(config).read_text().split()
    assert packaged == [str(tmp_path / "raw" / "SLIDE1.svs")], packaged


# --- Stage 2 ------------------------------------------------------------------------

def _gate_manifest(config: dict) -> None:
    """What the tiling gate leaves for packaging."""
    hpl_common.packaged_manifest_path(config).write_text(Path(config["manifest"]).read_text())


def test_packaging_refuses_to_run_without_a_verified_tiling_stage(tmp_path):
    config = _config(tmp_path)
    assert _refused(hpl_package.step_package, config, 1), \
        "packaging ran with no manifest from the tiling gate"


def test_packaging_refuses_an_h5_of_disagreeing_lengths(tmp_path, monkeypatch=None):
    import make_hpl_hdf5

    config = _config(tmp_path)
    _gate_manifest(config)
    h5 = Path(config["packaging"]["h5_path"])

    def fake_package(**kwargs):
        _write_h5(h5, rows=5, slides_rows=4)
        return {"output_h5_path": str(h5)}

    original = make_hpl_hdf5.package_to_h5
    make_hpl_hdf5.package_to_h5 = fake_package
    try:
        assert _refused(hpl_package.step_package, config, 1), "an inconsistent .h5 was marked packaged"
        assert state.read_done(config["out_dir"], "packaging") is None
    finally:
        make_hpl_hdf5.package_to_h5 = original


def test_packaging_skips_an_h5_that_already_validates(tmp_path):
    import make_hpl_hdf5

    config = _config(tmp_path)
    _gate_manifest(config)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=5)

    def must_not_run(**kwargs):
        raise AssertionError("repackaged over a complete .h5")

    original = make_hpl_hdf5.package_to_h5
    make_hpl_hdf5.package_to_h5 = must_not_run
    try:
        hpl_package.step_package(config, 1)
    finally:
        make_hpl_hdf5.package_to_h5 = original
    assert state.read_done(config["out_dir"], "packaging")["tiles"] == 5


# --- Stage 3 ------------------------------------------------------------------------

def test_extraction_refuses_output_short_of_the_packaged_tiles(tmp_path):
    config = _config(tmp_path)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=10)
    output = Path(config["extraction"]["output_path"])
    _write_projections(output, rows=8)
    assert _refused(hpl_extract.step_extract_finish, config), "8 of 10 embeddings passed as complete"
    assert state.read_done(config["out_dir"], "extraction") is None

    _write_projections(output, rows=10)
    hpl_extract.step_extract_finish(config)
    assert state.read_done(config["out_dir"], "extraction")["embeddings"] == 10


def test_the_extraction_plan_clears_a_stale_output_and_skips_a_complete_one(tmp_path):
    import submit_feature_extraction as sfe

    config = _config(tmp_path)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=10)
    output = Path(config["extraction"]["output_path"])
    ranges = tmp_path / "ranges.txt"

    _write_projections(output, rows=10)
    hpl_extract.step_extract_plan(config, ranges)
    assert ranges.read_text().split() == ["skip"]

    # A leftover the encoder would treat as done and crash on: cleared first.
    _write_projections(output, rows=4)
    checks = {name: getattr(sfe, name) for name in (
        "_check_hpl_repo_dir", "_check_checkpoint", "_check_singularity_image",
        "_check_container_extras")}
    for name in checks:
        setattr(sfe, name, lambda *a, **k: None)
    try:
        hpl_extract.step_extract_plan(config, ranges)
    finally:
        for name, fn in checks.items():
            setattr(sfe, name, fn)
    assert not output.exists(), "a stale extraction output survived into the GPU job"
    assert ranges.read_text().split() == ["all"]


def test_the_extraction_plan_refuses_before_the_gpu_on_a_bad_setup(tmp_path):
    """The runtime checks run in the plan step, so a missing repo or image is
    a refusal on a CPU node, not a failure after a GPU allocation."""
    config = _config(tmp_path)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=10)
    try:
        hpl_extract.step_extract_plan(config, tmp_path / "ranges.txt")
    except (NotADirectoryError, FileNotFoundError, SystemExit):
        return
    raise AssertionError("an HPL_REPO_DIR of /nowhere was accepted")


def test_a_retried_shard_clears_what_the_failed_attempt_left(tmp_path):
    """The encoder treats any existing output as finished and crashes on it, so
    a shard retried after an OOM or preemption must remove the half-written
    part itself — and keep one that is whole (a lost exit code, not a failure).
    """
    import submit_feature_extraction as sfe

    config = _config(tmp_path, shards=2)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=10)
    output = Path(config["extraction"]["output_path"])
    part = sfe.shard_output_path(output, 0, 5)
    ran = []
    original = hpl_extract.run_shell
    hpl_extract.run_shell = lambda command: ran.append(command)
    try:
        _write_projections(part, rows=3)                  # attempt 1 died at row 3
        hpl_extract.step_extract(config, "0 5")
        assert not part.exists(), "the half-written part survived into the retry"
        assert ran, "the retry did not run the encoder"

        ran.clear()
        _write_projections(part, rows=5)                  # attempt finished, exit lost
        hpl_extract.step_extract(config, "0 5")
        assert part.exists() and not ran, "a complete part was encoded again"
    finally:
        hpl_extract.run_shell = original


def test_a_retried_unsharded_encode_clears_a_partial_output(tmp_path):
    config = _config(tmp_path)
    _write_h5(Path(config["packaging"]["h5_path"]), rows=10)
    output = Path(config["extraction"]["output_path"])
    _write_projections(output, rows=6)
    original = hpl_extract.run_shell
    hpl_extract.run_shell = lambda command: None
    try:
        hpl_extract.step_extract(config, "all")
    finally:
        hpl_extract.run_shell = original
    assert not output.exists(), "a partial output was left for the encoder to crash on"


# --- Stage 4 ------------------------------------------------------------------------

def test_assignment_refuses_a_csv_of_the_wrong_length(tmp_path):
    config = _config(tmp_path)
    _write_projections(Path(config["extraction"]["output_path"]), rows=10)
    out_csv = Path(config["assignment"]["out_csv"])
    _write_assignments(out_csv, rows=9)
    assert _refused(hpl_assign.step_assign_finish, config), "9 of 10 assignments passed as complete"
    assert state.read_done(config["out_dir"], "assignment") is None

    _write_assignments(out_csv, rows=10)
    hpl_assign.step_assign_finish(config)
    assert state.read_done(config["out_dir"], "assignment")["assignments"] == 10



# --- refusals exit 65, through the real script -----------------------------

def test_a_refusal_exits_with_the_code_the_config_finishes_on(tmp_path):
    bad = tmp_path / "run_config.json"
    bad.write_text(json.dumps({"out_dir": str(tmp_path)}))   # missing every stage key
    result = subprocess.run(
        [sys.executable, str(BIN / "hpl_tile.py"), "gate", "--config", str(bad)],
        capture_output=True, text=True)
    assert result.returncode == hpl_common.REFUSAL_EXIT_CODE == 65, result.stderr
    assert "REFUSED:" in result.stderr


def test_a_config_naming_the_wrong_backend_is_refused(tmp_path):
    config = _config(tmp_path)
    config["backend_dir"] = str(tmp_path)          # no hpl_nf_state.py there
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(BIN / "hpl_tile.py"), "gate", "--config", str(path)],
        capture_output=True, text=True)
    assert result.returncode == 65 and "backend/" in result.stderr, result.stderr


def test_no_refusal_is_left_exiting_1(_tmp=None):
    """A `raise SystemExit("message")` exits 1, which the config retries as if
    it were a crash. Every message exit goes through refuse()."""
    for script in sorted(BIN.glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        offenders = [n.lineno for n in ast.walk(tree)
                     if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                     and getattr(n.exc.func, "id", None) == "SystemExit" and n.exc.args
                     and isinstance(n.exc.args[0], (ast.JoinedStr, ast.Constant))
                     and isinstance(getattr(n.exc.args[0], "value", ""), str)]
        assert not offenders, f"{script.name}: SystemExit(<message>) at lines {offenders}"


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_nf_steps_"))
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
