"""Roll a dataset's several Slurm runs up into one pipeline state.

A dataset is tiled, packaged and feature-extracted by however many runs it
takes, because POST /dataset-jobs/{id}/resume mints a *new* submission rather
than continuing the old one. Each of those rows carries its own single-slot job
columns (job_id, h5_job_id, extraction_job_id), so no individual run can answer
"is this dataset finished": the run that tiled the last 43 slides knows nothing
about the .h5 an earlier run produced, and the run that produced the .h5 was
reporting its own manifest complete while thousands of slides sat untiled
beside it.

Everything here is a plain function over row dicts. No database, no Slurm, no
filesystem — the caller resolves those and passes the results in. That is what
makes the classification testable without a cluster, and it is also what keeps
the expensive parts (a per-slide filesystem walk, an HDF5 read) out of a code
path the UI polls.

Two rules run through all of it:

  Artifacts count from every run, verdicts only from live ones.
    Cancelling a run sets status='cancelled' and scancels its jobs, but never
    clears h5_job_id or deletes what already finished. A .h5 completed before
    the cancel is still a perfectly good .h5. So a cancelled run's *output*
    still counts towards the dataset, while its Slurm *state* (CANCELLED) is
    ignored — otherwise one cancelled run would poison the rollup into
    reporting a dataset as failed when its successor finished the job.

  A subset run finishing is not a dataset being finished.
    is_subset means the run's manifest covered only a sample of the directory.
    Such a run reports 30/30 slides tiled and produces a `_subset_N` .h5, and
    neither statement is about the dataset. Only `coverage` (the caller's
    filesystem walk) can settle full tiling; without it this module says so
    rather than guessing.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

# sacct states meaning "still queued or actively running". Mirrors
# IN_FLIGHT_SLURM_STATES in tile_server_v2_.py; COMPLETING counts as in-flight
# because Slurm's epilogue and any writes still flushing to a network
# filesystem are not done yet.
IN_FLIGHT_SLURM_STATES = frozenset({
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING",
    "CONFIGURING",
})

# Step states, matching the vocabulary app_v28.py's _pipeline_steps already
# emits and _STEP_ICON already renders, so a rolled-up step and a per-run step
# draw identically.
DONE = "done"
RUNNING = "running"
ACTION = "action"        # ready for the user to start
ATTENTION = "attention"  # finished or stalled, but needs a decision
FAILED = "failed"
BLOCKED = "blocked"      # an earlier step must finish first

# Run statuses that mean the run is no longer a live opinion about the
# pipeline. Cancelled runs are excluded from state rollups (but not from
# artifact collection); errored runs stay in, because an errored run is
# normally the thing you resume.
DEAD_RUN_STATUSES = frozenset({"cancelled"})


def coarse_run_state(job_states: set[str] | None) -> str:
    """One word for a whole run's worth of Slurm tasks.

    Ordered by what a reader needs to act on first: something still going
    (running/pending) outranks a failure that already happened, because the run
    is not finished being judged yet; a failure outranks completion, because a
    run where 1 of 15 batches failed is not "complete" however many succeeded.

    None means sacct could not be reached, which is "unknown" and must never be
    rendered as finished. An empty set means sacct ran and had no rows — aged
    out of retention, or too fresh to have been written yet.
    """
    if job_states is None:
        return "unknown"
    if not job_states:
        return "no record"
    if job_states & {"RUNNING", "COMPLETING", "RESIZING"}:
        return "running"
    if job_states & {"PENDING", "CONFIGURING", "REQUEUED", "SUSPENDED"}:
        return "pending"
    if job_states - {"COMPLETED"}:
        # Anything terminal that is not COMPLETED: FAILED, TIMEOUT,
        # OUT_OF_MEMORY, CANCELLED, NODE_FAIL.
        return "failed"
    return "complete"


def job_ids(value) -> list[str]:
    """Split a comma-joined job_id column into individual Slurm job IDs.

    Tiling submits one array job per ~1000-slide batch and records them joined
    by commas, so this field holds one ID for a small dataset and fourteen for
    a large one.
    """
    return [part for part in str(value or "").split(",") if part]


def states_for_jobs(
    ids: Sequence[str], job_states: dict[str, set[str]] | None
) -> set[str] | None:
    """Every Slurm state seen across these job IDs, or None if unreachable.

    job_states is keyed by base job ID, so an array task ("123_5") and a job
    step ("123.batch") both fold into "123" — the caller wants the array's
    states together, not one entry per task.
    """
    if job_states is None:
        return None
    seen: set[str] = set()
    for job_id in ids:
        seen |= job_states.get(job_id.split("_", 1)[0], set())
    return seen


def is_live_run(run: dict) -> bool:
    """Whether this run's Slurm state should count towards the rollup.

    False for cancelled runs. Their outputs still count (see the module
    docstring) — this only governs whose CANCELLED array states get to make
    the dataset look failed.
    """
    return (run.get("status") or "") not in DEAD_RUN_STATUSES


def dataset_name_for(run: dict) -> str:
    """The dataset folder a run's tiles live under.

    Rows created before the dataset_name column existed have it NULL; those
    runs used raw_dir's own folder name at the time, so that is the correct
    fallback rather than a blank.
    """
    name = run.get("dataset_name")
    if name:
        return str(name)
    raw_dir = str(run.get("raw_dir") or "").rstrip("/")
    return raw_dir.rsplit("/", 1)[-1] if raw_dir else ""


def group_runs_by_dataset(rows: Iterable[dict]) -> list[dict]:
    """Bucket run rows into datasets, keyed by (raw_dir, dataset_name).

    That pair is the grouping key because resume reuses both: it passes the
    original row's raw_dir and its resolved dataset_name straight through to
    the new submission. Grouping on raw_dir alone would merge two runs that
    deliberately tiled the same directory into different output folders.

    Runs come back oldest-first within each dataset, so runs[-1] is the most
    recent — the one a resume should be issued against.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("raw_dir") or ""), dataset_name_for(row))
        buckets.setdefault(key, []).append(row)

    grouped = []
    for (raw_dir, dataset_name), runs in buckets.items():
        runs.sort(key=lambda r: (str(r.get("submitted_at") or ""), str(r.get("submission_id") or "")))
        grouped.append({"raw_dir": raw_dir, "dataset_name": dataset_name, "runs": runs})

    # Newest activity first, matching how every other listing in the UI orders.
    grouped.sort(
        key=lambda d: str(d["runs"][-1].get("submitted_at") or ""), reverse=True
    )
    return grouped


def _artifact_status(
    output_path: str | None,
    ids: Sequence[str],
    job_states: dict[str, set[str]] | None,
    path_exists: Callable[[str], bool],
) -> tuple[str, str]:
    """Classify one packaging/extraction output as (status, slurm_state).

    status is one of "none" (never submitted), "running", "ready",
    "interrupted".

    Readiness here is the cheap two-of-three: the output file is present, and
    Slurm is not still working on it. The third condition the per-run /status
    applies — actually opening the .h5 and reading its first and last rows —
    is deliberately not done here, because this runs for every dataset on a
    polled listing. That is safe rather than sloppy: packaging stages to
    `<final>.h5.partial` and only os.replace()s on success, so a file existing
    at the final path already means an attempt completed. And nothing
    destructive hangs off this verdict — POST /extract-features re-checks
    packaging state and re-reads the .h5 itself before submitting a GPU job,
    so an over-optimistic "ready" here shows the user a button, not a
    corrupted run.
    """
    seen = states_for_jobs(ids, job_states)
    slurm_state = coarse_run_state(seen)

    if not ids and not output_path:
        return "none", slurm_state
    if slurm_state in ("running", "pending"):
        return "running", slurm_state
    if output_path and path_exists(output_path):
        return "ready", slurm_state
    if not ids:
        return "none", slurm_state
    return "interrupted", slurm_state


def _roll_up_tiling(
    runs: list[dict],
    job_states: dict[str, set[str]] | None,
    coverage: dict | None,
) -> tuple[str, str]:
    """(state, summary) for the dataset's tiling, across all its runs.

    With coverage the answer is authoritative: the caller has walked the
    filesystem and knows how many slides in the directory actually have tiles.
    Without it, this reports what Slurm says about the runs' own manifests and
    is careful not to promote that into a claim about the whole dataset — a
    directory tiled only by subset runs can have every job COMPLETED and still
    be mostly untiled.
    """
    live = [run for run in runs if is_live_run(run)]

    every_id: list[str] = []
    for run in live:
        every_id.extend(job_ids(run.get("job_id")))

    seen = states_for_jobs(every_id, job_states) if every_id else set()
    slurm_state = coarse_run_state(seen) if every_id else "no record"

    if coverage is not None:
        total = coverage.get("slides_in_directory") or 0
        tiled = coverage.get("slides_tiled") or 0
        untiled = coverage.get("slides_untiled") or 0
        verdict = coverage.get("verdict")
        if verdict == "complete":
            return DONE, f"{tiled:,}/{total:,} slides have tiles"
        if verdict == "no_slides":
            return ATTENTION, "no slides found in this directory"
        if verdict == "in_progress":
            return RUNNING, f"{tiled:,}/{total:,} slides · tiling still running"
        if verdict == "unknown":
            return ATTENTION, f"{tiled:,}/{total:,} slides · can't reach Slurm"
        if verdict == "not_started":
            return ACTION, f"{tiled:,}/{total:,} slides · no run submitted yet"
        # "stalled"
        return ATTENTION, f"{tiled:,}/{total:,} slides · {untiled:,} still need tiling"

    if not runs:
        return ACTION, "no runs submitted yet"
    if not every_id:
        # Rows exist but no Slurm job was ever recorded against a live one —
        # still discovering slides, or every run that got as far as sbatch has
        # since been cancelled.
        statuses = {str(run.get("status") or "") for run in live}
        if statuses & {"queued", "discovering", "submitting"}:
            return RUNNING, "discovering slides"
        return ACTION, "no tiling job submitted yet"

    if slurm_state in ("running", "pending"):
        return RUNNING, f"tiling jobs {slurm_state}"
    if slurm_state == "unknown":
        return ATTENTION, "can't reach Slurm — tiling state unknown"
    if slurm_state == "failed":
        return ATTENTION, "some tiling tasks did not complete"

    # Every recorded tiling job finished cleanly. Whether that finished the
    # *dataset* depends on what those runs covered, and only a full run can
    # even claim to have aimed at all of it.
    if any(not run.get("is_subset") for run in live):
        return DONE, "tiling jobs complete"
    return ATTENTION, "only subset runs — full coverage unverified"


def _deliverable(path: str, attempts: list[tuple[dict, dict, bool]]) -> dict:
    """One .h5 output, summarised from every run that aimed at that path.

    attempts arrive newest-first. The newest is not automatically the one
    described, because a cancelled attempt says nothing about the file: three
    runs wrote to hdf5_Radiogenomics_subset_3_he_train.h5, a 3,644-slide one
    that produced it and two 30-slide ones cancelled the following day. Taking
    the newest labelled that file "30 slides", which is not what is in it and
    not what anyone would search for. The newest attempt that was not cancelled
    is the best available guess at what actually wrote the file, so it supplies
    the slide count and date; the cancelled ones still count towards attempts,
    because "tried four times" is the useful part of their existence.

    A guess is all it is — nothing here opens the file to check. slides_seen
    carries every distinct count attempted so a caller can show that the
    attempts disagreed rather than implying this one is confirmed.
    """
    representative = next(
        (a for a in attempts if (a[0].get("status") or "") not in DEAD_RUN_STATUSES),
        attempts[0],
    )
    run, record, covers_full = representative
    slides_seen = sorted({
        int(a[0]["total_slides"]) for a in attempts
        if a[0].get("total_slides") is not None
    })
    return {
        **record,
        "name": path.rsplit("/", 1)[-1],
        "covers_full": covers_full,
        "total_slides": run.get("total_slides"),
        "submitted_at": _as_text(run.get("submitted_at")),
        "run_status": _as_text(run.get("status")),
        "attempts": len(attempts),
        "last_attempt_at": _as_text(attempts[0][0].get("submitted_at")),
        "slides_seen": slides_seen,
    }


def rollup_dataset(
    raw_dir: str,
    dataset_name: str,
    runs: list[dict],
    *,
    job_states: dict[str, set[str]] | None = None,
    path_exists: Callable[[str], bool] = lambda _p: False,
    packaging_scopes: dict[str, set[str]] | None = None,
    coverage: dict | None = None,
) -> dict:
    """One dataset's whole pipeline, rolled up from all of its runs.

    runs must be oldest-first (as group_runs_by_dataset returns them), so
    runs[-1] is the most recent.

    job_states is one batched sacct result keyed by base job ID, covering every
    job across every dataset in the response — resolving states per run would
    be one cluster round-trip each, on a listing the UI polls.

    path_exists and packaging_scopes are injected so this stays free of the
    filesystem and the database. packaging_scopes maps a submission_id to the
    scopes its packaging attempts used; a run flagged is_subset still counts
    towards the full-dataset track if it packaged with scope="tiled", because
    that scope packages every slide with tiles on disk and gets the plain,
    unsuffixed .h5 name.

    coverage, when given, is the authoritative filesystem answer from
    _tiling_readiness. It is optional because it stats two files per slide.
    """
    packaging_scopes = packaging_scopes or {}
    live = [run for run in runs if is_live_run(run)]

    tiling_state, tiling_summary = _roll_up_tiling(runs, job_states, coverage)

    # --- Packaging ---------------------------------------------------------
    # Newest first: the most recent attempt is the one whose state matters.
    full_packaging = None
    attempts_by_path: dict[str, list[tuple[dict, dict, bool]]] = {}
    for run in reversed(runs):
        status, slurm_state = _artifact_status(
            run.get("h5_output_path"),
            job_ids(run.get("h5_job_id")),
            job_states,
            path_exists,
        )
        if status == "none":
            continue
        record = {
            "submission_id": run.get("submission_id"),
            "status": status,
            "slurm_state": slurm_state,
            "output_path": run.get("h5_output_path"),
        }
        covers_full = (not run.get("is_subset")) or (
            "tiled" in packaging_scopes.get(str(run.get("submission_id")), set())
        )
        if covers_full and full_packaging is None:
            full_packaging = record

        # Every distinct .h5 this dataset has ever produced, keyed by where it
        # was written. Without this, full_packaging is the entire packaging
        # story and it holds exactly one record: a 3,644-slide .h5 packaged in
        # July went missing from the UI purely because later attempts at the
        # full dataset were newer, and one of those won the single slot.
        # Rebuilding it would have cost hours of cluster time for a file
        # already sitting on disk.
        path = _as_text(run.get("h5_output_path"))
        if path:
            attempts_by_path.setdefault(path, []).append((run, record, covers_full))

    deliverables = {
        path: _deliverable(path, attempts) for path, attempts in attempts_by_path.items()
    }

    # Other finished .h5 files are real output and worth showing, but none of
    # them is the dataset's .h5 — appended as a note so a dataset that has only
    # ever packaged a sample cannot read as "packaging done", and so one that
    # has packaged several says so instead of silently listing one.
    full_path = full_packaging["output_path"] if full_packaging else None
    others_ready = sum(
        1 for d in deliverables.values()
        if d["status"] == "ready" and d["output_path"] != full_path
    )
    if others_ready == 1:
        subset_note = " · 1 other .h5 also exists"
    elif others_ready > 1:
        subset_note = f" · {others_ready} other .h5 files also exist"
    else:
        subset_note = ""

    if full_packaging is None:
        packaging_path = None
        if tiling_state == DONE:
            packaging = (ACTION, f"ready to start{subset_note}")
        elif tiling_state in (ATTENTION, FAILED):
            # Tiling is unresolved, but the server still lets you package what
            # is on disk (scope="tiled", or allow_incomplete), so this is a
            # decision rather than a lockout.
            packaging = (ATTENTION, f"tiling unresolved — packaging not started{subset_note}")
        else:
            packaging = (BLOCKED, f"waiting on tiling{subset_note}")
    else:
        packaging_path = full_packaging["output_path"]
        if full_packaging["status"] == "ready":
            packaging = (DONE, f".h5 ready{subset_note}")
        elif full_packaging["status"] == "running":
            packaging = (RUNNING, f"running ({full_packaging['slurm_state']})")
        else:
            packaging = (
                ATTENTION,
                f"interrupted ({full_packaging['slurm_state']})",
            )

    # --- Feature extraction ------------------------------------------------
    extraction_record = None
    for run in reversed(runs):
        status, slurm_state = _artifact_status(
            run.get("extraction_output_path"),
            job_ids(run.get("extraction_job_id")),
            job_states,
            path_exists,
        )
        if status == "none":
            continue
        extraction_record = {
            "submission_id": run.get("submission_id"),
            "status": status,
            "slurm_state": slurm_state,
            "output_path": run.get("extraction_output_path"),
        }
        break

    if extraction_record is None:
        extraction_path = None
        if packaging[0] == DONE:
            extraction = (ACTION, "ready to start")
        else:
            extraction = (BLOCKED, "waiting on packaging")
    else:
        extraction_path = extraction_record["output_path"]
        if extraction_record["status"] == "ready":
            extraction = (DONE, "features ready")
        elif extraction_record["status"] == "running":
            extraction = (RUNNING, f"running ({extraction_record['slurm_state']})")
        else:
            extraction = (
                ATTENTION,
                f"did not finish ({extraction_record['slurm_state']})",
            )

    # --- Test / trial artifacts -------------------------------------------
    # Never gate anything. Surfaced so a dataset someone has only trialled
    # doesn't look untouched, and so an existing test .h5 can be reused for a
    # checkpoint trial instead of being packaged again.
    tests = []
    for run in reversed(runs):
        if not run.get("test_h5_job_id"):
            continue
        status, slurm_state = _artifact_status(
            run.get("test_h5_output_path"),
            job_ids(run.get("test_h5_job_id")),
            job_states,
            path_exists,
        )
        tests.append({
            "submission_id": run.get("submission_id"),
            "status": status,
            "slurm_state": slurm_state,
            "output_path": run.get("test_h5_output_path"),
        })

    steps = [
        {"key": "tiling", "title": "1. Tiling", "state": tiling_state, "summary": tiling_summary},
        {"key": "packaging", "title": "2. Packaging (.h5)", "state": packaging[0], "summary": packaging[1]},
        {"key": "extraction", "title": "3. Feature extraction", "state": extraction[0], "summary": extraction[1]},
    ]

    return {
        "runs": [_run_summary(run, job_states) for run in reversed(runs)],
        "raw_dir": raw_dir,
        "dataset_name": dataset_name,
        "run_count": len(runs),
        "cancelled_run_count": len(runs) - len(live),
        "has_full_run": any(not run.get("is_subset") for run in runs),
        "has_subset_run": any(run.get("is_subset") for run in runs),
        "total_slides": _best_total_slides(runs, coverage),
        "slides_tiled": (coverage or {}).get("slides_tiled"),
        "slides_untiled": (coverage or {}).get("slides_untiled"),
        "coverage_checked": coverage is not None,
        "first_submitted_at": _as_text(runs[0].get("submitted_at")) if runs else None,
        "last_submitted_at": _as_text(runs[-1].get("submitted_at")) if runs else None,
        "steps": steps,
        "tiling": steps[0],
        "packaging": {**steps[1], "output_path": packaging_path},
        "extraction": {**steps[2], "output_path": extraction_path},
        "tests": tests,
        # Newest .h5 first. Includes outputs from cancelled runs: cancelling a
        # run does not delete what it had already written, and a finished .h5 is
        # the single most expensive thing in this pipeline to reproduce.
        "deliverables": list(deliverables.values()),
        "next_action": _next_action(runs, live, steps, coverage),
    }


def _run_summary(run: dict, job_states: dict[str, set[str]] | None) -> dict:
    """One run, reduced to what a picker row needs.

    slurm_state follows the same precedence /dataset-jobs?with_state=true uses:
    a row status of cancelled or error is a decision already recorded about the
    run and outranks whatever Slurm remembers, since a cancelled run's batches
    may well read COMPLETED.
    """
    status = str(run.get("status") or "")
    ids = job_ids(run.get("job_id"))
    if status in ("cancelled", "error"):
        slurm_state = status
    elif not ids:
        slurm_state = status or "queued"
    else:
        slurm_state = coarse_run_state(states_for_jobs(ids, job_states))

    return {
        "submission_id": run.get("submission_id"),
        "status": status,
        "slurm_state": slurm_state,
        "is_subset": bool(run.get("is_subset")),
        "total_slides": run.get("total_slides"),
        "job_id": run.get("job_id"),
        "submitted_at": _as_text(run.get("submitted_at")),
        "raw_dir": run.get("raw_dir"),
        "dataset_name": dataset_name_for(run),
        "h5_output_path": run.get("h5_output_path"),
        "extraction_output_path": run.get("extraction_output_path"),
        "test_h5_output_path": run.get("test_h5_output_path"),
        # True when this run was created by resuming another. NULL for every
        # row predating the lineage column, which is why the UI must not treat
        # its absence as "this was a deliberate separate run".
        "resumed_from_submission_id": run.get("resumed_from_submission_id"),
    }


def _best_total_slides(runs: list[dict], coverage: dict | None) -> int | None:
    """How many slides the dataset actually holds.

    Coverage counted the directory, so it wins outright. Failing that, only a
    full run's total_slides describes the dataset — a subset run's is the size
    of its sample, and reporting "30 slides" for a 14,000-slide directory is
    worse than reporting nothing.
    """
    if coverage and coverage.get("slides_in_directory"):
        return int(coverage["slides_in_directory"])
    totals = [
        int(run["total_slides"])
        for run in runs
        if run.get("total_slides") and not run.get("is_subset")
    ]
    return max(totals) if totals else None


def _as_text(value) -> str | None:
    """Timestamps arrive as datetimes from SQLAlchemy and as strings from a
    pandas round-trip, and only ever get rendered."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _resume_target(runs: list[dict], live: list[dict]) -> str | None:
    """Which run a "resume" should be issued against.

    The newest live run, because resume re-reads that row's manifest and
    tiling_params to reproduce the original submission. Cancelled runs are a
    last resort rather than excluded outright: resuming one is still valid (it
    only reads the manifest), and a dataset whose every run was cancelled has
    no other way back to a running pipeline.
    """
    for run in reversed(live or runs):
        if run.get("manifest_path"):
            return str(run.get("submission_id"))
    for run in reversed(runs):
        if run.get("manifest_path"):
            return str(run.get("submission_id"))
    return str(runs[-1].get("submission_id")) if runs else None


def _next_action(
    runs: list[dict], live: list[dict], steps: list[dict], coverage: dict | None
) -> dict:
    """The single thing to do next to move this dataset towards finished.

    One action, not a menu: the pipeline is strictly ordered, so at any moment
    there is exactly one stage that can be advanced. Everything else on screen
    is context for this line.
    """
    tiling, packaging, extraction = steps
    target = _resume_target(runs, live)

    # A pipeline run (POST /pipeline-runs) drives its own Stages 1-4, and the
    # per-stage endpoints refuse it — so "Start packaging" here would be a
    # button that can only fail. Until its stages are done, the one thing to do
    # is follow (or resume) the pipeline, which the run's own panel offers.
    target_run = next((r for r in runs if r.get("submission_id") == target), None)
    if (target_run and str(target_run.get("job_id") or "").startswith("nf:")
            and not all(step["state"] == DONE for step in steps)):
        stopped = any(step["state"] in (FAILED, ATTENTION) for step in steps)
        return {
            "kind": "wait",
            "submission_id": target,
            "label": "Pipeline stopped — resume it from the run below" if stopped
                     else "Nextflow pipeline in progress",
            "detail": "Stages 1-4 run as one Nextflow pipeline; each stage is "
                      "verified before the next starts.",
        }

    if not runs:
        return {
            "kind": "submit",
            "submission_id": None,
            "label": "Submit a tiling run",
            "detail": "Nothing has been submitted for this dataset yet.",
        }

    if tiling["state"] == RUNNING:
        return {
            "kind": "wait",
            "submission_id": target,
            "label": "Tiling in progress",
            "detail": tiling["summary"],
        }

    if tiling["state"] in (ATTENTION, FAILED):
        # Only subset runs, and nobody has checked the directory — the honest
        # next step is to find out, not to resubmit work that may be done.
        if coverage is None and not any(not run.get("is_subset") for run in live):
            return {
                "kind": "check_coverage",
                "submission_id": target,
                "label": "Check how much of this dataset is tiled",
                "detail": (
                    "Only subset runs exist here, so their finishing says nothing "
                    "about the rest of the directory."
                ),
            }
        return {
            "kind": "resume_tiling",
            "submission_id": target,
            "label": "Resume tiling",
            "detail": tiling["summary"],
        }

    if tiling["state"] == BLOCKED:
        return {
            "kind": "none",
            "submission_id": target,
            "label": "Nothing to do",
            "detail": tiling["summary"],
        }

    if packaging["state"] == RUNNING:
        return {
            "kind": "wait",
            "submission_id": target,
            "label": "Packaging in progress",
            "detail": packaging["summary"],
        }
    if packaging["state"] == ATTENTION:
        return {
            "kind": "resume_packaging",
            "submission_id": target,
            "label": "Resume packaging",
            "detail": packaging["summary"],
        }
    if packaging["state"] in (ACTION, BLOCKED):
        return {
            "kind": "start_packaging",
            "submission_id": target,
            "label": "Start packaging",
            "detail": packaging["summary"],
        }

    if extraction["state"] == RUNNING:
        return {
            "kind": "wait",
            "submission_id": target,
            "label": "Feature extraction in progress",
            "detail": extraction["summary"],
        }
    if extraction["state"] == ATTENTION:
        return {
            "kind": "start_extraction",
            "submission_id": target,
            "label": "Retry feature extraction",
            "detail": extraction["summary"],
        }
    if extraction["state"] in (ACTION, BLOCKED):
        return {
            "kind": "start_extraction",
            "submission_id": target,
            "label": "Start feature extraction",
            "detail": extraction["summary"],
        }

    return {
        "kind": "complete",
        "submission_id": target,
        "label": "Pipeline complete",
        "detail": "Tiling, packaging and feature extraction have all finished.",
    }
