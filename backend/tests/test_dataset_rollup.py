"""Tests for the dataset rollup: how several runs become one pipeline state.

Runs under pytest, and also standalone with `python3 test_dataset_rollup.py`
for the same reason dataset_rollup.py takes its inputs as arguments — this
project has no test framework installed and no CI, and a suite nobody can run
is a suite that stops being true. Every case here is pure data: no Postgres, no
Slurm, no filesystem, no HDF5.

The cases worth keeping are the ones guarding against a wrong answer that still
*looks* right on screen: a subset run's .h5 being claimed as the dataset's, a
cancelled run dragging a finished dataset to "failed", or "tiling complete"
being asserted from Slurm alone when nothing checked the disk. Each of those
renders as a perfectly plausible line of text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_rollup import (  # noqa: E402
    coarse_run_state,
    dataset_name_for,
    group_runs_by_dataset,
    job_ids,
    rollup_dataset,
    states_for_jobs,
)

DIR = "/scratch/TCGA_LUAD"


def run(submission_id, **overrides):
    """A slurm_dataset_runs row with sane defaults, overridden per case."""
    row = {
        "submission_id": submission_id,
        "raw_dir": DIR,
        "dataset_name": "TCGA_LUAD",
        "tile_dir": "/scratch/processed_tiles",
        "manifest_path": f"/scratch/manifests/{submission_id}.txt",
        "status": "submitted",
        "is_subset": False,
        "total_slides": 2431,
        "job_id": None,
        "h5_job_id": None,
        "h5_output_path": None,
        "extraction_job_id": None,
        "extraction_output_path": None,
        "test_h5_job_id": None,
        "test_h5_output_path": None,
        "resumed_from_submission_id": None,
        "submitted_at": "2026-08-01T09:00:00",
    }
    row.update(overrides)
    return row


def roll(runs, states=None, on_disk=(), scopes=None, coverage=None):
    return rollup_dataset(
        DIR,
        "TCGA_LUAD",
        runs,
        job_states=states,
        path_exists=lambda p: p in set(on_disk),
        packaging_scopes=scopes,
        coverage=coverage,
    )


def step(rollup, key):
    return next(s for s in rollup["steps"] if s["key"] == key)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def test_resume_fork_groups_into_one_dataset():
    """A resume mints a new submission_id but reuses raw_dir and dataset_name,
    which is the only reason these can be put back together at all."""
    rows = [
        run("first", submitted_at="2026-08-01T09:00:00"),
        run("resume-1", submitted_at="2026-08-02T09:00:00",
            is_subset=True, total_slides=43, resumed_from_submission_id="first"),
        run("resume-2", submitted_at="2026-08-03T09:00:00",
            is_subset=True, total_slides=4, resumed_from_submission_id="resume-1"),
    ]
    grouped = group_runs_by_dataset(rows)
    assert len(grouped) == 1
    assert [r["submission_id"] for r in grouped[0]["runs"]] == ["first", "resume-1", "resume-2"]


def test_same_directory_different_output_folder_stays_separate():
    """Two runs deliberately tiling one directory into different folders are
    two datasets — merging them would report one's .h5 as the other's."""
    rows = [
        run("a", dataset_name="TCGA_LUAD"),
        run("b", dataset_name="TCGA_LUAD_retile"),
    ]
    assert len(group_runs_by_dataset(rows)) == 2


def test_null_dataset_name_falls_back_to_directory_name():
    """Rows predating the dataset_name column must group with the runs that
    followed them, not split off into a nameless dataset of their own."""
    old = run("old", dataset_name=None)
    assert dataset_name_for(old) == "TCGA_LUAD"
    grouped = group_runs_by_dataset([old, run("new")])
    assert len(grouped) == 1


def test_runs_come_back_oldest_first_newest_last():
    """rollup_dataset documents runs[-1] as the newest, and _resume_target
    relies on it to pick which run to resume."""
    rows = [run("c", submitted_at="2026-08-03T00:00:00"),
            run("a", submitted_at="2026-08-01T00:00:00"),
            run("b", submitted_at="2026-08-02T00:00:00")]
    assert [r["submission_id"] for r in group_runs_by_dataset(rows)[0]["runs"]] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Cancelled runs: artifacts count, verdicts don't
# ---------------------------------------------------------------------------

def test_cancelled_run_h5_still_counts():
    """The whole reason the rollup reads cancelled runs at all. Someone
    cancelled the run after packaging finished; that .h5 is still good, and
    telling them to package again would waste hours re-decoding it."""
    rows = [run("a", status="cancelled", h5_job_id="200",
                h5_output_path="/h5/TCGA_LUAD.h5", job_id="100")]
    got = roll(rows, states={"100": {"CANCELLED"}, "200": {"COMPLETED"}},
               on_disk=["/h5/TCGA_LUAD.h5"])
    assert step(got, "packaging")["state"] == "done"
    assert got["packaging"]["output_path"] == "/h5/TCGA_LUAD.h5"


def test_cancelled_tiling_does_not_poison_a_finished_dataset():
    """A cancelled run's array reads CANCELLED forever. Folding that into the
    rollup would report a dataset as broken because of a run its successor
    already superseded."""
    rows = [
        run("cancelled-one", status="cancelled", job_id="100"),
        run("finished-one", job_id="101"),
    ]
    got = roll(rows, states={"100": {"CANCELLED"}, "101": {"COMPLETED"}})
    assert step(got, "tiling")["state"] == "done"


def test_cancelled_runs_are_counted_and_still_listed():
    """Hidden in the UI's run picker, but the count is reported so a filtered
    list doesn't look like data went missing."""
    rows = [run("a", status="cancelled"), run("b")]
    got = roll(rows)
    assert got["run_count"] == 2
    assert got["cancelled_run_count"] == 1
    assert {r["submission_id"] for r in got["runs"]} == {"a", "b"}
    assert next(r for r in got["runs"] if r["submission_id"] == "a")["slurm_state"] == "cancelled"


def test_errored_run_still_counts_as_live():
    """Unlike a cancelled run, an errored one is normally the thing you
    resume, so its state must reach the rollup."""
    got = roll([run("a", status="error", job_id="100")], states={"100": {"FAILED"}})
    assert step(got, "tiling")["state"] == "attention"
    assert got["next_action"]["kind"] == "resume_tiling"


# ---------------------------------------------------------------------------
# Full vs subset
# ---------------------------------------------------------------------------

def test_subset_h5_does_not_satisfy_the_full_track():
    """A `_subset_N` .h5 is real output, but it is not the dataset's .h5.
    Claiming otherwise sends a 30-slide sample into feature extraction
    presented as 2431 slides."""
    rows = [run("a", is_subset=True, total_slides=30, job_id="100",
                h5_job_id="200", h5_output_path="/h5/TCGA_LUAD_subset_2.h5")]
    got = roll(rows, states={"100": {"COMPLETED"}, "200": {"COMPLETED"}},
               on_disk=["/h5/TCGA_LUAD_subset_2.h5"])
    assert step(got, "packaging")["state"] != "done"
    assert "1 other .h5 also exists" in step(got, "packaging")["summary"]


def test_subset_run_packaging_with_scope_tiled_does_satisfy_it():
    """scope="tiled" packages every slide with tiles on disk and gets the
    plain unsuffixed name, so a subset run can legitimately produce the
    dataset's real .h5. Only the recorded params say so."""
    rows = [run("a", is_subset=True, total_slides=30, job_id="100",
                h5_job_id="200", h5_output_path="/h5/TCGA_LUAD.h5")]
    got = roll(rows, states={"100": {"COMPLETED"}, "200": {"COMPLETED"}},
               on_disk=["/h5/TCGA_LUAD.h5"], scopes={"a": {"tiled"}})
    assert step(got, "packaging")["state"] == "done"


def test_only_subset_runs_will_not_claim_tiling_is_done():
    """Every job COMPLETED, and the directory may still be mostly untiled —
    the subset runs finished their own manifests, which says nothing about
    the rest."""
    rows = [run("a", is_subset=True, total_slides=30, job_id="100")]
    got = roll(rows, states={"100": {"COMPLETED"}})
    assert step(got, "tiling")["state"] == "attention"
    assert got["next_action"]["kind"] == "check_coverage"


def test_total_slides_ignores_subset_run_totals():
    """Reporting "30 slides" for a 2431-slide directory is worse than
    reporting nothing."""
    rows = [run("a", is_subset=True, total_slides=30),
            run("b", total_slides=2431)]
    assert roll(rows)["total_slides"] == 2431
    assert roll([run("a", is_subset=True, total_slides=30)])["total_slides"] is None


# ---------------------------------------------------------------------------
# Coverage overrides the Slurm guess
# ---------------------------------------------------------------------------

def test_coverage_verdict_wins_over_slurm():
    """The filesystem is the only authority on how much is tiled. A complete
    coverage check promotes the subset-only case straight to done."""
    rows = [run("a", is_subset=True, total_slides=30, job_id="100")]
    got = roll(rows, states={"100": {"COMPLETED"}},
               coverage={"verdict": "complete", "slides_in_directory": 2431,
                         "slides_tiled": 2431, "slides_untiled": 0})
    assert step(got, "tiling")["state"] == "done"
    assert got["coverage_checked"] is True
    assert got["total_slides"] == 2431


def test_coverage_stalled_asks_for_a_resume():
    rows = [run("a", job_id="100")]
    got = roll(rows, states={"100": {"COMPLETED"}},
               coverage={"verdict": "stalled", "slides_in_directory": 2431,
                         "slides_tiled": 2388, "slides_untiled": 43})
    assert step(got, "tiling")["state"] == "attention"
    assert "43" in step(got, "tiling")["summary"]
    assert got["next_action"]["kind"] == "resume_tiling"


# ---------------------------------------------------------------------------
# Slurm being unreachable is not "finished"
# ---------------------------------------------------------------------------

def test_unreachable_sacct_never_reads_as_done():
    """None means sacct could not be reached. Rendering that as complete would
    invent a finished pipeline out of a controller outage."""
    got = roll([run("a", job_id="100")], states=None)
    assert step(got, "tiling")["state"] == "attention"
    assert "can't reach Slurm" in step(got, "tiling")["summary"]


def test_coarse_state_distinguishes_unknown_from_no_record():
    assert coarse_run_state(None) == "unknown"
    assert coarse_run_state(set()) == "no record"
    assert coarse_run_state({"COMPLETED"}) == "complete"
    assert coarse_run_state({"COMPLETED", "TIMEOUT"}) == "failed"
    assert coarse_run_state({"COMPLETED", "RUNNING"}) == "running"
    assert coarse_run_state({"PENDING", "COMPLETED"}) == "pending"


def test_array_tasks_and_steps_fold_into_their_parent_job():
    """sacct reports 990_5 and 990.batch against one array; the caller wants
    the array's states together, not one entry per task."""
    assert job_ids("990,991") == ["990", "991"]
    assert states_for_jobs(["990_5", "991"], {"990": {"RUNNING"}, "991": {"COMPLETED"}}) == {
        "RUNNING", "COMPLETED"
    }
    assert states_for_jobs(["990"], None) is None


# ---------------------------------------------------------------------------
# next_action across the lifecycle
# ---------------------------------------------------------------------------

def test_next_action_walks_the_whole_pipeline():
    """One ordered pass through a dataset's life, asserting the single next
    step at each point. The pipeline is strictly ordered, so there is always
    exactly one — that is what makes a single button honest."""
    completed = {"COMPLETED"}

    nothing = roll([])
    assert nothing["next_action"]["kind"] == "submit"

    tiling = roll([run("a", job_id="100")], states={"100": {"RUNNING"}})
    assert tiling["next_action"]["kind"] == "wait"

    failed_tiling = roll([run("a", job_id="100")], states={"100": {"TIMEOUT"}})
    assert failed_tiling["next_action"]["kind"] == "resume_tiling"
    assert failed_tiling["next_action"]["submission_id"] == "a"

    tiled = roll([run("a", job_id="100")], states={"100": completed})
    assert tiled["next_action"]["kind"] == "start_packaging"

    packaging = roll(
        [run("a", job_id="100", h5_job_id="200", h5_output_path="/h5/x.h5")],
        states={"100": completed, "200": {"RUNNING"}},
    )
    assert packaging["next_action"]["kind"] == "wait"

    interrupted = roll(
        [run("a", job_id="100", h5_job_id="200", h5_output_path="/h5/x.h5")],
        states={"100": completed, "200": {"TIMEOUT"}},
    )
    assert interrupted["next_action"]["kind"] == "resume_packaging"

    packaged = roll(
        [run("a", job_id="100", h5_job_id="200", h5_output_path="/h5/x.h5")],
        states={"100": completed, "200": completed}, on_disk=["/h5/x.h5"],
    )
    assert packaged["next_action"]["kind"] == "start_extraction"

    done = roll(
        [run("a", job_id="100", h5_job_id="200", h5_output_path="/h5/x.h5",
             extraction_job_id="300", extraction_output_path="/feat/x.h5")],
        states={"100": completed, "200": completed, "300": completed},
        on_disk=["/h5/x.h5", "/feat/x.h5"],
    )
    assert done["next_action"]["kind"] == "complete"
    assert [s["state"] for s in done["steps"]] == ["done", "done", "done"]


def test_resume_targets_the_newest_live_run():
    """Resume re-reads the target row's manifest and tiling_params, so it must
    land on the newest run that is still a live opinion — not the cancelled
    one that happens to be most recent."""
    rows = [
        run("old", submitted_at="2026-08-01T00:00:00", job_id="100"),
        run("newer", submitted_at="2026-08-02T00:00:00", job_id="101"),
        run("cancelled", submitted_at="2026-08-03T00:00:00", status="cancelled", job_id="102"),
    ]
    got = roll(rows, states={"100": {"FAILED"}, "101": {"FAILED"}, "102": {"CANCELLED"}})
    assert got["next_action"]["kind"] == "resume_tiling"
    assert got["next_action"]["submission_id"] == "newer"


def test_extraction_stays_blocked_until_a_full_h5_exists():
    """The step is listed either way — that is the point of the stepper — but
    it must not offer to start."""
    rows = [run("a", is_subset=True, job_id="100", h5_job_id="200",
                h5_output_path="/h5/TCGA_LUAD_subset_1.h5")]
    got = roll(rows, states={"100": {"COMPLETED"}, "200": {"COMPLETED"}},
               on_disk=["/h5/TCGA_LUAD_subset_1.h5"])
    assert step(got, "extraction")["state"] == "blocked"


# ---------------------------------------------------------------------------
# Artifacts: presence at the final path is the signal
# ---------------------------------------------------------------------------

def test_completed_job_without_its_file_is_interrupted_not_ready():
    """Packaging publishes atomically: the final path only appears on success.
    A COMPLETED job with no file means the rename never happened."""
    rows = [run("a", job_id="100", h5_job_id="200", h5_output_path="/h5/x.h5")]
    got = roll(rows, states={"100": {"COMPLETED"}, "200": {"COMPLETED"}}, on_disk=[])
    assert step(got, "packaging")["state"] == "attention"
    assert got["next_action"]["kind"] == "resume_packaging"


def test_test_packaging_is_reported_but_never_gates():
    """A trial .h5 should stop a dataset looking untouched without ever
    standing in for the real packaging output."""
    rows = [run("a", job_id="100", test_h5_job_id="900",
                test_h5_output_path="/h5/test.h5")]
    got = roll(rows, states={"100": {"COMPLETED"}, "900": {"COMPLETED"}},
               on_disk=["/h5/test.h5"])
    assert len(got["tests"]) == 1
    assert got["tests"][0]["status"] == "ready"
    assert step(got, "packaging")["state"] == "action"
    assert got["next_action"]["kind"] == "start_packaging"


def test_newest_attempt_decides_the_stage_state():
    """Repackaging after a failure must not leave the stage reading failed."""
    rows = [
        run("a", submitted_at="2026-08-01T00:00:00", job_id="100",
            h5_job_id="200", h5_output_path="/h5/x.h5"),
        run("b", submitted_at="2026-08-02T00:00:00", job_id="101",
            h5_job_id="201", h5_output_path="/h5/x.h5"),
    ]
    got = roll(rows, states={"100": {"COMPLETED"}, "101": {"COMPLETED"},
                             "200": {"TIMEOUT"}, "201": {"COMPLETED"}},
               on_disk=["/h5/x.h5"])
    assert step(got, "packaging")["state"] == "done"


# ---------------------------------------------------------------------------
# Deliverables — every .h5 the dataset has, not just the one the steps name
# ---------------------------------------------------------------------------

def test_every_distinct_h5_is_listed():
    """The reason this list exists. The steps above name one .h5 per stage, so
    a subset packaged in July vanished behind later attempts at the full
    dataset — and a finished .h5 nobody can see is one somebody repackages, at
    hours of cluster time for a file already on disk."""
    rows = [
        run("a", submitted_at="2026-07-21T00:00:00", total_slides=3644,
            job_id="100", h5_job_id="200", h5_output_path="/h5/sub_3.h5"),
        run("b", submitted_at="2026-08-02T00:00:00", total_slides=14044,
            job_id="101", h5_job_id="201", h5_output_path="/h5/full.h5"),
    ]
    got = roll(rows, states={}, on_disk=["/h5/sub_3.h5", "/h5/full.h5"])
    assert [d["output_path"] for d in got["deliverables"]] == [
        "/h5/full.h5", "/h5/sub_3.h5",
    ]
    assert all(d["status"] == "ready" for d in got["deliverables"])


def test_deliverable_ignores_a_cancelled_later_attempt_at_the_same_path():
    """Three runs wrote to the same subset .h5: a 3,644-slide one that produced
    it, then two 30-slide ones cancelled the next day. Describing the file by
    the newest attempt labelled it "30 slides" — not what is in it, and not
    what anyone looking for their 3,644-slide packaging would recognise."""
    rows = [
        run("a", submitted_at="2026-07-21T00:00:00", total_slides=3644,
            h5_job_id="200", h5_output_path="/h5/sub_3.h5"),
        run("b", submitted_at="2026-07-22T00:00:00", total_slides=30,
            status="cancelled", h5_job_id="201", h5_output_path="/h5/sub_3.h5"),
    ]
    got = roll(rows, states={}, on_disk=["/h5/sub_3.h5"])
    only = got["deliverables"][0]
    assert only["total_slides"] == 3644
    assert only["attempts"] == 2
    # The counts disagreed, and saying so is the honest version of a guess:
    # nothing here opened the file.
    assert only["slides_seen"] == [30, 3644]


def test_deliverable_falls_back_to_the_newest_when_all_were_cancelled():
    """No non-cancelled attempt to prefer. Still listed — cancelling a run does
    not delete what it had already written."""
    rows = [
        run("a", submitted_at="2026-07-21T00:00:00", total_slides=100,
            status="cancelled", h5_job_id="200", h5_output_path="/h5/x.h5"),
        run("b", submitted_at="2026-07-22T00:00:00", total_slides=200,
            status="cancelled", h5_job_id="201", h5_output_path="/h5/x.h5"),
    ]
    got = roll(rows, states={}, on_disk=["/h5/x.h5"])
    assert got["deliverables"][0]["total_slides"] == 200
    assert got["deliverables"][0]["status"] == "ready"


def test_deliverable_without_a_file_is_not_reported_ready():
    """An h5 job that ran and left nothing is the case worth catching: it is
    the one that looks identical to success in a list of names."""
    rows = [run("a", h5_job_id="200", h5_output_path="/h5/x.h5")]
    got = roll(rows, states={"200": {"FAILED"}}, on_disk=[])
    assert got["deliverables"][0]["status"] == "interrupted"


def test_packaging_summary_counts_the_other_h5_files():
    """So the step line admits there is more to see than the one path it names."""
    rows = [
        run("a", submitted_at="2026-07-01T00:00:00", is_subset=True,
            total_slides=10, h5_job_id="200", h5_output_path="/h5/s1.h5"),
        run("b", submitted_at="2026-07-02T00:00:00", is_subset=True,
            total_slides=20, h5_job_id="201", h5_output_path="/h5/s2.h5"),
        run("c", submitted_at="2026-07-03T00:00:00", job_id="100",
            h5_job_id="202", h5_output_path="/h5/full.h5"),
    ]
    got = roll(rows, states={}, on_disk=["/h5/s1.h5", "/h5/s2.h5", "/h5/full.h5"])
    summary = step(got, "packaging")["summary"]
    assert summary.startswith(".h5 ready")
    assert "2 other .h5 files also exist" in summary


def test_a_dataset_with_no_packaging_has_no_deliverables():
    rows = [run("a", job_id="100")]
    assert roll(rows, states={"100": {"COMPLETED"}})["deliverables"] == []


def _main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as e:
            failures.append(name)
            print(f"  FAIL {name}: {e or 'assertion failed'}")
        except Exception as e:
            failures.append(name)
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())


def test_a_pipeline_run_is_never_offered_a_per_stage_button(_tmp=None):
    """A pipeline run's Stages 1-4 belong to its Nextflow run, and the per-stage
    endpoints refuse it — so "Start packaging" or "Resume tiling" here would be
    a button that can only fail. The run's own panel offers Resume."""
    sentinels = {"job_id": "nf:a:tiling", "h5_job_id": "nf:a:packaging",
                 "h5_output_path": "/h5/x.h5", "extraction_job_id": "nf:a:extraction",
                 "extraction_output_path": "/r/x.h5"}
    completed = {"COMPLETED"}
    cases = {
        "tiling running": {"nf:a:tiling": {"RUNNING"}},
        "tiling failed": {"nf:a:tiling": {"FAILED"}},
        "tiled, packaging queued": {"nf:a:tiling": completed, "nf:a:packaging": {"PENDING"}},
        "packaging failed": {"nf:a:tiling": completed, "nf:a:packaging": {"FAILED"}},
    }
    for name, states in cases.items():
        got = roll([run("a", **sentinels)], states=states)["next_action"]
        assert got["kind"] == "wait", f"{name}: offered {got['kind']!r}"
