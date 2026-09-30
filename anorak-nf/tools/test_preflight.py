#!/usr/bin/env python3
"""Tests for tools/preflight.sh, run for real against a fake cluster.

The user runs preflight once, on the cluster, before a 7,221-slide run, and
both ways of it being wrong are expensive: a guard that cannot fail lets the
run start and die a day later, and a guard that fails wrongly blocks a launch
that was fine. So every check here is exercised both ways — the good cluster
in fake_cluster.py must pass with nothing skipped, and each test then breaks
one thing and asserts that the check responsible, and only by its own section,
says FAIL. Paths throughout carry spaces, and the raw directory a quote and
brackets, because the pre-rewrite check 10 pasted that path into Python source.

    PREFLIGHT_UNDER_TEST=<other preflight.sh> python3 tools/test_preflight.py

runs the same suite against another copy — which is how these were shown to
fail against the preflight they replace.

Runs under pytest and standalone. Needs bash and python3; no Slurm.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_cluster import SLIDE_TILES, FakeCluster  # noqa: E402

UNDER_TEST = Path(os.environ.get("PREFLIGHT_UNDER_TEST", HERE / "preflight.sh"))


def make(tmp_path) -> FakeCluster:
    return FakeCluster(Path(tmp_path), preflight=UNDER_TEST)


def section(result, number: str) -> str:
    """The output of one numbered step, e.g. "6b", up to the next step."""
    # Up to the next step heading, not the next unindented line: checks 9 and
    # 10 stream their tools' own output at column 0.
    match = re.search(r"^%s\. .*?(?=^(?:\d+[a-z]?(?:-\d+)?\. |Summary$|Effective configuration$)|\Z)"
                      % re.escape(number), result.out, re.M | re.S)
    assert match, f"no step {number} in the output:\n{result.out}"
    return match.group(0)


def fails(result, number: str, *words: str) -> str:
    text = section(result, number)
    lines = [l for l in text.splitlines() if l.lstrip().startswith("FAIL")]
    assert lines, f"step {number} did not fail:\n{text}"
    for word in words:
        assert any(word in l for l in lines), f"step {number} failed, but not naming {word!r}:\n{text}"
    assert result.returncode == 1, f"a failed check must exit 1, got {result.returncode}"
    return text


def passes(result, number: str) -> str:
    text = section(result, number)
    assert "FAIL" not in text, f"step {number} failed:\n{text}"
    return text


# --- the baseline --------------------------------------------------------------

def test_a_good_cluster_passes_with_every_check_run(tmp_path):
    c = make(tmp_path)
    r = c.preflight()
    assert r.returncode == 0, r.out
    assert re.search(r"\b0 failed\b.*\b0 skipped\b", r.out), r.out
    for step in ("5", "5b", "5c", "5d", "6", "6b", "6c", "7", "8", "9", "9b", "9c", "10", "11"):
        passes(r, step)


def test_a_skipped_check_is_not_a_pass(tmp_path):
    # Without a slide list checks 5, 5b, 5c and 11 cannot run; that must not
    # read as "Ready".
    c = make(tmp_path)
    r = c.preflight("--raw-dir", str(c.raw), default_inputs=False)
    assert r.returncode == 2, r.out
    assert "Ready" not in r.out


# --- 5: the slide list -----------------------------------------------------------

def test_a_column_that_merely_contains_slide_id_fails(tmp_path):
    c = make(tmp_path)
    c.write_list([("S1", "T1", "true")], header=("slide_ids", "samples", "is_tumour"))
    fails(c.preflight(), "5", "slide_id")


def test_a_blank_sample_fails(tmp_path):
    c = make(tmp_path)
    c.write_list([("S1", "T1", "true"), ("S2", "", "true")])
    fails(c.preflight(), "5", "blank")


def test_a_normal_slide_fails(tmp_path):
    c = make(tmp_path)
    c.write_list([("S1", "T1", "true"), ("S2", "T2", "false")])
    fails(c.preflight(), "5", "is_tumour")


def test_the_slide_column_option_is_honoured(tmp_path):
    c = make(tmp_path)
    c.write_list([("S1", "T1", "true"), ("S2", "T2", "true")], header=("image", "samples", "is_tumour"))
    r = c.preflight("--slide-column", "image")
    passes(r, "5")
    passes(r, "10")


def test_main_nf_refusing_the_launch_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["preview_error"] = "2 of 2 slides in slide_id have no file under raw"
    c.save()
    fails(c.preflight(), "5b", "refuses")


# --- 5c: .mrxs -------------------------------------------------------------------

def _with_mrxs(c, data_dir=True):
    c.add_slide("S3.mrxs", {"props": {"openslide.objective-power": "40", "openslide.mpp-x": "0.25"}})
    if data_dir:
        (c.raw / "S3").mkdir()
        (c.raw / "S3" / "Slidedat.ini").write_text("[GENERAL]\n")
    c.write_list([("S1", "T1", "true"), ("S3", "T3", "true")])


def test_mrxs_fails_when_main_nf_does_not_support_it(tmp_path):
    c = make(tmp_path)
    _with_mrxs(c)
    main_nf = c.pipeline / "main.nf"
    main_nf.write_text(re.sub(r"'\.mrxs',\s*", "", main_nf.read_text()))
    fails(c.preflight(), "5c", ".mrxs")


def test_mrxs_fails_when_main_nf_refuses_it(tmp_path):
    c = make(tmp_path)
    _with_mrxs(c)
    main_nf = c.pipeline / "main.nf"
    text = main_nf.read_text()
    at = text.index("\n", text.index("def SUPPORTED")) + 1
    main_nf.write_text(text[:at] + "    def REFUSED = ['.mrxs']   // stand-in for a refusal\n" + text[at:])
    fails(c.preflight(), "5c", "special-cases")


def test_mrxs_without_its_data_directory_fails(tmp_path):
    c = make(tmp_path)
    if ".mrxs" not in (c.pipeline / "main.nf").read_text():
        return   # main.nf no longer accepts .mrxs; the two tests above cover that
    _with_mrxs(c, data_dir=False)
    fails(c.preflight(), "5c", "data directory")


# --- 5d: outdir and work directory --------------------------------------------

def test_an_outdir_on_another_filesystem_fails(tmp_path):
    """PUBLISH_SLIDE hard-links tiles from work/ into outdir; across two
    filesystems every slide's publish job refuses."""
    c = make(tmp_path)
    work = tmp_path / "elsewhere" / "work"
    work.mkdir(parents=True)
    # A df that reports the work directory on its own mount, as a second
    # filesystem would; everything else goes to the real df.
    _write_df = c.bin / "df"
    _write_df.write_text(
        "#!/bin/sh\n"
        f'case "$2" in "{tmp_path / "elsewhere"}"*) '
        'echo "Filesystem 1024-blocks Used Available Capacity Mounted on"; '
        'echo "other 1 1 1 1% /mnt/other"; exit 0 ;; esac\n'
        'exec /bin/df "$@"\n')
    _write_df.chmod(0o755)
    fails(c.preflight("--work-dir", str(work)), "5d", "/mnt/other")


def test_the_submitters_layout_passes(tmp_path):
    # work/ inside the outdir, not yet created — the submitter's layout on a
    # first launch.
    c = make(tmp_path)
    assert "both on" in passes(c.preflight(), "5d")


def test_without_an_outdir_the_check_is_skipped_not_passed(tmp_path):
    c = make(tmp_path)
    r = c.preflight("--slides-csv", str(c.slides_csv), "--raw-dir", str(c.raw),
                    default_inputs=False)
    assert r.returncode == 2, r.out
    assert "needs --outdir" in section(r, "5d")


# --- 6: partition limits ---------------------------------------------------------

def test_a_retry_over_a_stitch_partitions_time_fails(tmp_path):
    # The step the hard-coded version never looked at.
    c = make(tmp_path)
    c.cluster["partitions"]["short"] = dict(c.cluster["partitions"]["compute"], default=False, maxtime="06:00:00")
    c.save()
    c.set_config("process.'withLabel:process_stitch'.queue", "'short'")
    fails(c.preflight(), "6", "process_stitch")


def test_a_gpu_retry_over_the_partition_time_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["partitions"]["gpu"]["maxtime"] = "1-00:00:00"
    c.save()
    fails(c.preflight(), "6", "process_gpu")


def test_memory_more_than_any_node_has_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["partitions"]["gpu"]["nodes"][0]["mem"] = 65536
    c.save()
    fails(c.preflight(), "6", "MB")


def test_a_ceiling_no_attempt_reaches_does_not_fail(tmp_path):
    # min(8h x 3, 5d) is 24h, inside compute's 2 days: failing on the 5d
    # would block a launch that is fine.
    c = make(tmp_path)
    c.set_config("params.max_cpu_time", "'5d'")
    passes(c.preflight(), "6")


def test_a_gpu_type_the_partition_lacks_fails(tmp_path):
    c = make(tmp_path)
    c.set_config("params.gpu_type", "'nvidia_h100'")
    fails(c.preflight(), "6", "nvidia_h100")


def test_a_gpu_partition_with_no_gpus_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["partitions"]["gpu"]["nodes"][0]["gres"] = "(null)"
    c.save()
    fails(c.preflight(), "6", "no GPUs")


def test_a_head_job_over_its_partition_fails(tmp_path):
    c = make(tmp_path)
    r = c.preflight(ANORAK_HEAD_TIME_LIMIT="3-00:00:00")
    fails(r, "6", "head job")


# --- 6b: submit limits -----------------------------------------------------------

def test_a_qos_submit_limit_below_queue_size_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["qos"]["normal"]["MaxSubmitPU"] = "150"
    c.save()
    fails(c.preflight(), "6b", "150")


def test_an_association_submit_limit_below_queue_size_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["assoc"][0]["MaxSubmitJobs"] = "100"
    c.save()
    fails(c.preflight(), "6b", "100")


def test_an_empty_limit_is_unlimited(tmp_path):
    c = make(tmp_path)
    c.cluster["qos"]["normal"]["MaxSubmitPU"] = ""
    c.save()
    text = passes(c.preflight(), "6b")
    assert "unlimited" in text


def test_the_chain_counts_against_the_limit(tmp_path):
    # 200 tasks + 2 head jobs fits 202; + 3 does not.
    c = make(tmp_path)
    c.cluster["qos"]["normal"]["MaxSubmitPU"] = "202"
    c.save()
    passes(c.preflight("--chain", "2"), "6b")
    fails(c.preflight("--chain", "3"), "6b", "202")


def test_a_partition_qos_limit_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["partitions"]["gpu"]["qos"] = "gpu_qos"
    c.cluster["qos"]["gpu_qos"] = {"MaxSubmitPU": "50", "MaxJobsPU": "25"}
    c.save()
    fails(c.preflight(), "6b", "gpu_qos")


def test_the_chain_default_is_read_from_the_submitter(tmp_path):
    c = make(tmp_path)
    backend = c.pipeline.parent / "backend"
    backend.mkdir()
    backend.joinpath("submit_anorak_nf.py").write_text(
        'parser.add_argument("--chain", type=int, default=4, metavar="N")\n')
    c.cluster["qos"]["normal"]["MaxSubmitPU"] = "203"
    c.save()
    fails(c.preflight(), "6b", "203")   # 200 + 4 > 203


# --- 6c, 9c: GPU isolation -------------------------------------------------------

def test_a_whitelist_without_cuda_visible_devices_fails(tmp_path):
    c = make(tmp_path)
    c.set_config("singularity.envWhitelist", "''")
    r = c.preflight()
    fails(r, "6c", "envWhitelist")
    fails(r, "9c", "sees 4 GPUs")


def test_a_whitelist_given_as_a_list_passes(tmp_path):
    c = make(tmp_path)
    c.set_config("singularity.envWhitelist", "['TMPDIR', 'CUDA_VISIBLE_DEVICES']")
    r = c.preflight()
    passes(r, "6c")
    passes(r, "9c")


def test_unconfined_devices_warn_but_do_not_fail(tmp_path):
    c = make(tmp_path)
    del c.cluster["config"]["ConstrainDevices"]
    c.save()
    r = c.preflight()
    assert "warn" in passes(r, "6c")
    assert r.returncode == 0, r.out


# --- 7, 8: reaching a node, and paths inside the container -----------------------

def test_a_node_that_cannot_submit_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["node_has_sbatch"] = False
    c.save()
    fails(c.preflight(), "7", "could not run sbatch")


def test_an_srun_that_never_ran_is_not_reported_as_missing_paths(tmp_path):
    c = make(tmp_path)
    c.cluster["srun_broken"] = True
    c.save()
    r = c.preflight()
    text = fails(r, "8", "did not run")
    assert "invisible" not in text
    fails(r, "7", "srun itself failed")


def test_a_path_no_bind_covers_is_invisible(tmp_path):
    c = make(tmp_path)
    c.set_config("singularity.runOptions", "'--nv --bind /nonexistent:/nonexistent'")
    fails(c.preflight(), "8", "invisible", str(c.anorak))


# --- 9, 9b: the inference container ------------------------------------------------

def test_run_options_without_nv_fail_the_gpu_check(tmp_path):
    # The old check added --nv itself, so it passed a config whose tasks
    # would never see a GPU.
    c = make(tmp_path)
    c.set_config("singularity.runOptions", f"'--bind {c.root}:{c.root}'")
    fails(c.preflight(), "9")


def test_a_checkpoint_that_does_not_load_fails(tmp_path):
    c = make(tmp_path)
    ckpt = c.anorak / "models" / "AIgrading_anorak.h5"
    ckpt.write_bytes(ckpt.read_bytes() + b"CORRUPT")
    fails(c.preflight(), "9")


def test_extras_missing_cv2_fail_the_import_check(tmp_path):
    import shutil
    c = make(tmp_path)
    shutil.rmtree(c.extras / "cv2")
    fails(c.preflight(), "9b", "predict_gp")


def test_the_images_own_pythonpath_is_kept(tmp_path):
    # PREDICT_GP prepends the extras to the image's PYTHONPATH. Replacing it
    # instead (SINGULARITYENV_PYTHONPATH) tests an interpreter no task gets.
    c = make(tmp_path)
    ngc = c.root / "imgsite" / "ngc"
    (ngc / "ngc_only").mkdir(parents=True)
    (ngc / "ngc_only" / "__init__.py").write_text("")
    c.image_meta["env_pythonpath"] = str(ngc)
    c.save_image()
    predict = c.anorak / "inference_slide" / "predict_gp.py"
    predict.write_text("import ngc_only\n" + predict.read_text())
    passes(c.preflight(), "9b")


# --- 4: the extras lock ----------------------------------------------------------

def test_extras_that_drifted_from_their_lock_fail(tmp_path):
    c = make(tmp_path)
    c.write_lock(["opencv-python-headless==4.10.0.84"])
    fails(c.preflight(), "4", "drifted")


def test_a_lock_from_another_image_fails(tmp_path):
    c = make(tmp_path)
    c.write_lock(["opencv-python-headless==4.8.1.78"], image="/elsewhere/old.sif")
    fails(c.preflight(), "4", "old.sif")


def test_no_lock_fails(tmp_path):
    c = make(tmp_path)
    (c.extras / "anorak_extras.lock").unlink()
    fails(c.preflight(), "4", "anorak_extras.lock")


# --- 10, 11: slides, and node-local scratch ----------------------------------------

def test_a_later_slide_with_no_mpp_fails(tmp_path):
    # Not the first slide, and with an upper-case extension: the old check
    # opened only the first file under --raw-dir, by lower-case extension.
    c = make(tmp_path)
    c.add_slide("sub/S2.SVS", {"props": {"openslide.objective-power": "40"}})
    fails(c.preflight(), "10")


def test_scratch_smaller_than_the_largest_slide_fails(tmp_path):
    c = make(tmp_path)
    c.cluster["tmp_avail_kb"] = int(SLIDE_TILES * 0.4 * 1024)   # below even 0.5 MB/tile
    c.save()
    fails(c.preflight(), "11", "process_tiling")


def test_scratch_within_the_estimate_range_warns(tmp_path):
    c = make(tmp_path)
    c.cluster["tmp_avail_kb"] = int(SLIDE_TILES * 0.75 * 1024)  # between 0.5 and 1 MB/tile
    c.save()
    r = c.preflight()
    assert "warn" in passes(r, "11")


def test_a_label_without_scratch_is_not_held_to_it(tmp_path):
    c = make(tmp_path)
    c.cluster["tmp_avail_kb"] = 1024
    c.save()
    for label in ("tiling", "gpu", "stitch", "light", "publish"):
        c.set_config(f"process.'withLabel:process_{label}'.scratch", "false")
    passes(c.preflight(), "11")


# --- standalone runner -------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, test in tests:
        with tempfile.TemporaryDirectory(prefix="anorak_preflight_test_") as tmp:
            try:
                test(Path(tmp))
                print(f"PASS  {name}")
            except Exception as error:
                failures.append(name)
                print(f"FAIL  {name}: {type(error).__name__}: {str(error)[:300]}")
                if os.environ.get("VERBOSE"):
                    traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed (preflight under test: {UNDER_TEST})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
