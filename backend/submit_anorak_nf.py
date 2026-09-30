#!/usr/bin/env python3
"""Submit the ANORAK Nextflow pipeline to Slurm, over a full cohort or a sample.

    python submit_anorak_nf.py --slides-csv radiogenomics_tumour_slides_min10.csv \
        --raw-dir /mnt/.../Radiogenomics --out-dir /mnt/.../anorak/Radiogenomics \
        --pipeline-dir /path/to/anorak-nf --anorak-dir /path/to/AIgrading \
        --scope subset --sample-size 10 --dry-run

What gets submitted is one small, long-lived job: `nextflow run`, which then
submits a job per slide per stage itself. That head process is nearly idle —
the resources below are for waiting, not working — but it must live as long as
the whole pipeline, which is why its time limit is days rather than hours.

**The head job has to be able to run sbatch.** A Nextflow head process on a
compute node that cannot reach the Slurm controller starts, submits nothing,
and sits there until its own time limit — no error, no child jobs, a `work/`
directory with nothing in it. Nothing about the login node's ability to submit
says whether a compute node can. So `--check-submit` (a one-second `srun` that
asks a real compute node whether `sbatch` answers) exists for the same reason
`submit_kb_write.py --check-db` does, and is worth running once per cluster.

**The head job supervises Nextflow rather than trusting it.** Nextflow
notices finished tasks on a single thread that reads each one's `.exitcode`
off CephFS, and one such read can block in the kernel forever — observed
2026-09-23, five idle hours, nothing failed. `anorak-nf/tools/nf_supervise.sh`
restarts it with `-resume` once that thread has been silent for
WATCHDOG_STALL_SECONDS. See the comment at the top of that script.

**A chain stops on a failure a successor cannot fix.** Standbys wait on
`afternotok`, so every non-zero exit starts the next one, and a deterministic
failure used to run through the whole chain in seconds. The supervisor writes
SUPERVISOR_STOP_MARKER into out_dir on such a failure (Nextflow's own non-zero
exit, a manual `scancel`, a Nextflow that survived KILL) and every head job
exits 0 without running while it is there. A fresh submission clears it, and
reports what it said. To stop a run by hand, `scancel` the running head job:
the marker is what keeps its standbys from resuming the cohort.

Scope is `full` or `subset`, and a subset is a *random* sample with a recorded
seed, not the first N. The first N of a cohort sorted by slide id is one or two
patients, which is the least informative way to spend a test: it shares a
scanner, a batch, a stain run and often a block. The seed is written into the
run's slide list so the same sample can be asked for again — and so a result
that differs from a later full run can be traced to which slides it saw.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submit_mask_tile_slurm import _run_sbatch_with_retry  # noqa: E402

#: Where the head job's own stdout/stderr land, matching every other stage.
LOG_DIR = Path(__file__).resolve().parent / "slurm_logs"

#: The head process orchestrates and waits. It holds the run's task graph and
#: one JVM, so it is not free, but it does no science.
HEAD_CPUS = 2
HEAD_MEMORY = "8G"

#: How long the head job may live. This is the one resource here that is not a
#: property of the work: it is a property of the partition, and a cluster whose
#: MaxTime is below it rejects the submission outright ("Requested time limit is
#: invalid"). So it is configurable, and the default is a guess rather than a
#: measurement.
#:
#: Longer is better up to that ceiling, because the head job has to outlive
#: every job it submits — one killed at its limit leaves its children running
#: with nothing collecting them. It is not a disaster when it happens:
#: resubmitting with resume=True picks the run up from Nextflow's cache and
#: re-runs only what had not finished. But each expiry costs a resubmission and
#: whatever was in flight at the time.
#:
#: What the partition actually allows:
#:     sinfo -o "%P %l %L"      # partition, MaxTime, DefaultTime
#:
#: Two days because that is the Beatson default partition's ceiling (compute,
#: MaxTime 2-00:00:00) and no --partition is passed, so that is where the head
#: job lands. The gpu* partitions and compute-low-priority are uncapped, but a
#: head job belongs on neither: a preempted one is the same problem as an
#: expired one, more often.
HEAD_TIME_LIMIT_ENV = "ANORAK_HEAD_TIME_LIMIT"
HEAD_TIME_LIMIT = os.getenv(HEAD_TIME_LIMIT_ENV, "2-00:00:00")

#: Seconds of warning Slurm gives the head job before killing it at its limit.
#: This is what turns an expiry from a mess into a stop, and it matters more
#: than the limit itself — see the comment on the sbatch command below.
HEAD_SIGNAL_SECONDS = 120

#: The watchdog the head job runs Nextflow under (anorak-nf/tools/nf_supervise.sh).
#: Nextflow has one thread that notices finished tasks, by reading each task's
#: .exitcode off CephFS, and on 2026-09-23 one such read blocked in the kernel
#: and never returned: every job after it finished in Slurm, none was collected,
#: and the run sat idle for five hours with nothing failing. No Nextflow setting
#: can time out a read stuck below the JVM, so the supervisor restarts the
#: process with -resume when that thread has logged nothing for this long.
#:
#: The monitor logs a summary every executor.dumpInterval (5 min, pinned in
#: conf/beatson.config) whenever a task is running, so 30 minutes is six missed
#: heartbeats — silence that long means stuck, not quiet. A stall then costs
#: this plus a minute or two of restart, instead of the rest of the walltime.
WATCHDOG_STALL_SECONDS_ENV = "ANORAK_WATCHDOG_STALL_SECONDS"
WATCHDOG_STALL_SECONDS = int(os.getenv(WATCHDOG_STALL_SECONDS_ENV, "1800"))

#: Restarts before the head job gives up and exits non-zero (so a --chain
#: successor, if there is one, takes over). A run that stalls this often has a
#: filesystem problem to report, not a pipeline one to wait out.
WATCHDOG_MAX_RESTARTS = 10

#: Where the supervisor lives inside the pipeline directory.
SUPERVISOR_RELATIVE_PATH = Path("tools") / "nf_supervise.sh"

#: The file in out_dir that tells every later head job of this run not to
#: start (nf_supervise.sh --stop-marker). It holds the reason.
SUPERVISOR_STOP_MARKER = "nf_supervise.stop"

#: Shell run before `nextflow`, for clusters where it arrives via modules.
#: Set once in the server's environment rather than typed per submission.
PRELUDE_ENV = "HPL_NEXTFLOW_PRELUDE"

#: Columns tumour_grade.py writes. Used to tell a finished run from a file that
#: exists — see validate_anorak_output.
_REQUIRED_GRADE_COLUMNS = ("sample", "slides", "predominant_pattern", "iaslc_grade")

#: The slide list's columns main.nf reads (params.slide_column and
#: params.sample_column there), and the verdict select_tumour_slides.py writes.
SLIDE_COLUMN = "slide_id"
SAMPLE_COLUMN = "samples"
TUMOUR_COLUMN = "is_tumour"

#: How pandas writes True, and what a hand-edited list might say instead.
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "t"})

#: Options build_nextflow_command sets itself. Nextflow refuses a repeated
#: launcher option outright ("Can only specify option -profile once"), so a
#: duplicate in extra_args is a head job that dies at launch; and a repeated
#: --param silently takes the last value, so an extra --outdir would publish
#: the grading table somewhere validate_anorak_output never looks. The two
#: column params are here because the validator assumes their defaults.
_OWNED_NEXTFLOW_OPTIONS = (
    "-log", "-profile", "-work-dir", "-w", "-ansi-log", "-resume",
    "--slides_csv", "--raw_dir", "--anorak_dir", "--outdir",
    "--slide_column", "--sample_column",
)


# --- choosing the slides ---------------------------------------------------

def select_slide_rows(
    frame: pd.DataFrame,
    *,
    scope: str = "full",
    sample_size: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The rows this run should process, plus a record of how they were chosen.

    scope="full" takes every row. scope="subset" takes `sample_size` at random.

    Asking for more slides than the list holds is refused rather than quietly
    narrowed to fit — the same rule, for the same reason, as
    `submit_mask_tile_slurm.select_slides`: a request for 500 that silently
    returns 30 produces a completed run, a grading table, and no indication
    that 470 slides were never considered.
    """
    if scope not in ("full", "subset"):
        raise ValueError(f"scope must be 'full' or 'subset', got {scope!r}")

    if scope == "full":
        return frame, {"scope": "full", "slides": len(frame)}

    if not sample_size or sample_size <= 0:
        raise ValueError("A subset run needs --sample-size greater than zero.")
    if sample_size > len(frame):
        raise ValueError(
            f"Asked for {sample_size} slides but the list holds {len(frame)}. "
            f"Ask for at most {len(frame)}, or run the full list."
        )

    # A seed is always recorded, even when the caller did not choose one, so
    # every subset run can be reproduced from its own record. Without this the
    # sample is unrepeatable and a disagreement with a later run has no
    # explanation to reach for.
    resolved_seed = random.randrange(2 ** 31) if seed is None else seed
    if sample_size == len(frame):
        chosen = frame
    else:
        indices = random.Random(resolved_seed).sample(range(len(frame)), sample_size)
        chosen = frame.iloc[sorted(indices)]
    return chosen, {"scope": "subset", "slides": len(chosen),
                    "sample_size": sample_size, "seed": resolved_seed,
                    "pool": len(frame)}


def read_slide_csv(path: Path) -> pd.DataFrame:
    """A slide list, every cell as the string that is in the file.

    Left to infer types, pandas read sample `007` as 7, `1001` as 1001.0 once a
    blank sat in the same column, and `NA` as missing — so the run's own copy
    named tumours that were not in the source, and the grading table then
    failed to match its slide list or, worse, matched a different tumour.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


#: What slide_list_from_directory writes in is_tumour: not a verdict, a
#: statement that nobody has made one. check_slide_list accepts it only when
#: the caller says the list is unverified on purpose (tumour_verified=False).
UNVERIFIED = "unverified"

#: Extensions ANORAK's main.nf resolves (its SUPPORTED list). A slide in any
#: other format is left out of a directory list and counted, rather than handed
#: to a pipeline that would refuse the whole cohort at launch.
ANORAK_EXTENSIONS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".png", ".qptiff")


def slide_list_from_directory(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Every slide under raw_dir as an ANORAK slide list, tumour status unverified.

    For running ANORAK on a dataset nobody has selected tumour slides from yet —
    a brand-new cohort before HPL has run. Each slide is grouped into a tumour
    by the rule HPL's packaging uses for its samples column
    (make_hpl_hdf5.sample_from_slide_id), and slide_id is the file's stem,
    which is what main.nf resolves a slide by. Refuses two files with one stem:
    main.nf would refuse the whole cohort for the ambiguity at launch.
    """
    from make_hpl_hdf5 import sample_from_slide_id
    from slide_naming import slide_id_from_raw_path

    files = sorted(
        (p for p in Path(raw_dir).rglob("*") if p.is_file()),
        key=lambda p: str(p).lower(),
    )
    usable = [p for p in files if p.suffix.lower() in ANORAK_EXTENSIONS]
    # Slides HPL tiles but ANORAK cannot open: counted, not silently absent.
    skipped = [p.name for p in files if p.suffix.lower() == ".scn"]
    by_stem: dict[str, list[Path]] = {}
    for path in usable:
        by_stem.setdefault(path.stem, []).append(path)
    clashes = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if clashes:
        example = next(iter(clashes.items()))
        raise ValueError(
            f"{len(clashes)} slide name(s) under {raw_dir} belong to more than one "
            f"file (e.g. {example[0]!r}: {', '.join(str(p) for p in example[1][:3])}). "
            f"ANORAK finds a slide by its name, so it would refuse the cohort; give "
            f"it a tumour-slide list that names one of each."
        )
    if not usable:
        raise ValueError(f"No slides ANORAK can read ({', '.join(ANORAK_EXTENSIONS)}) under {raw_dir}.")
    frame = pd.DataFrame({
        SLIDE_COLUMN: [p.stem for p in usable],
        SAMPLE_COLUMN: [sample_from_slide_id(slide_id_from_raw_path(p)) for p in usable],
        TUMOUR_COLUMN: [UNVERIFIED] * len(usable),
    })
    return frame, {"slides": len(usable), "skipped_unsupported": skipped}


def check_slide_list(frame: pd.DataFrame, source: Path, *, tumour_verified: bool = True) -> None:
    """Refuse a slide list that would grade the wrong thing without failing.

    Non-tumour rows: select_tumour_slides.py writes every slide by default,
    with is_tumour as its verdict, and nothing downstream reads that column —
    so a list straight from it grades every slide in the cohort. Blank
    samples: main.nf keeps a blank sample as '', and tumour_grade.py then
    pools every such slide into one tumour that does not exist.
    """
    for column in (SLIDE_COLUMN, SAMPLE_COLUMN, TUMOUR_COLUMN):
        if column not in frame.columns:
            raise ValueError(
                f"{source} has no {column!r} column (it has "
                f"{', '.join(frame.columns)}). The pipeline grades per "
                f"{SAMPLE_COLUMN!r} and must only be given slides whose "
                f"{TUMOUR_COLUMN!r} is true — use select_tumour_slides.py's "
                f"output, optionally filtered by filter_slides_by_tile_count.py."
            )

    def _examples(mask) -> str:
        ids = frame.loc[mask, SLIDE_COLUMN].str.strip().tolist()
        return ", ".join(repr(i) for i in ids[:5]) + (" ..." if len(ids) > 5 else "")

    blank_slide = frame[SLIDE_COLUMN].str.strip() == ""
    if blank_slide.any():
        raise ValueError(
            f"{source} has {int(blank_slide.sum())} row(s) with a blank "
            f"{SLIDE_COLUMN!r}. main.nf skips such a row without a word, so "
            f"the run would be short a slide nobody named."
        )
    not_tumour = ~frame[TUMOUR_COLUMN].str.strip().str.lower().isin(_TRUE_STRINGS)
    if not tumour_verified:
        # A directory-wide list, run on purpose before any tumour selection
        # exists. Only its own "unverified" is excused: a list that says some
        # slides are NOT tumour is a verdict, and still refused.
        not_tumour &= frame[TUMOUR_COLUMN].str.strip().str.lower() != UNVERIFIED
    if not_tumour.any():
        raise ValueError(
            f"{source} has {int(not_tumour.sum())} of {len(frame)} slide(s) whose "
            f"{TUMOUR_COLUMN!r} is not true (e.g. {_examples(not_tumour)}). "
            f"Nothing downstream reads that column, so every one of them would "
            f"be graded as a tumour. Regenerate the list with "
            f"select_tumour_slides.py --tumour-only."
        )
    blank_sample = frame[SAMPLE_COLUMN].str.strip() == ""
    if blank_sample.any():
        raise ValueError(
            f"{source} has {int(blank_sample.sum())} slide(s) with a blank "
            f"{SAMPLE_COLUMN!r} (e.g. {_examples(blank_sample)}). Grading "
            f"aggregates per sample, so these would be pooled into one tumour "
            f"that does not exist. Fill them in, or drop those slides."
        )


def write_slide_list(frame: pd.DataFrame, path: Path, selection: dict) -> None:
    """The exact list this run was given, beside the run's own outputs.

    Written rather than passed by reference to the original CSV: that file can
    be regenerated with a different threshold, and a run whose inputs can
    change after the fact cannot be explained afterwards.

    Under a .tmp name and renamed, so an interrupted write leaves the previous
    list whole rather than a truncated one that Nextflow reads as a shorter
    cohort — and that validate_anorak_output would then check a short table
    against and pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    path.with_suffix(".selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )


# --- is the output real? ---------------------------------------------------

def validate_anorak_output(grades_csv: Path) -> tuple[bool, str]:
    """(ok, reason) for the pipeline's final grading table.

    A Nextflow run that dies partway can still have published earlier stages'
    outputs, and a `TUMOUR_GRADE` killed mid-write leaves a readable CSV. So
    this checks the columns are the ones the grader writes and that its tumours
    are exactly the samples of the run's own slide_list.csv beside it — a table
    short of some tumours has the right columns and rows, and read as finished.
    Compared as strings, both sides read without type inference, so `007` is
    not `7` and `NA` is a sample rather than a gap.
    """
    if not grades_csv.is_file():
        # "does not exist" is true and useless: for a pipeline that failed at
        # launch it names the file that was never going to be written instead
        # of the reason. The head job's own log sits beside it and holds the
        # actual error, so point there when it is present.
        log = grades_csv.parent / "nextflow.log"
        if log.is_file():
            return False, (
                f"the pipeline did not produce a grading table. Its log is at "
                f"{log} — the failure is in there, usually in the last 40 lines."
            )
        return False, (
            f"{grades_csv} does not exist, and there is no nextflow.log beside "
            f"it — the head job most likely never started Nextflow at all. "
            f"Check the Slurm log in backend/slurm_logs/anorak_nf_<jobid>.out."
        )
    if grades_csv.stat().st_size == 0:
        return False, f"{grades_csv} is empty"
    try:
        frame = pd.read_csv(grades_csv, dtype=str, keep_default_na=False)
    except Exception as error:
        return False, f"{grades_csv.name} could not be read: {error}"

    missing = [c for c in _REQUIRED_GRADE_COLUMNS if c not in frame.columns]
    if missing:
        return False, (f"{grades_csv.name} is missing {', '.join(missing)} — "
                       f"this is not a finished grading table")
    if frame.empty:
        return False, f"{grades_csv.name} has a header but no tumours"

    slide_list = grades_csv.parent / "slide_list.csv"
    if not slide_list.is_file():
        return False, (f"there is no {slide_list.name} beside {grades_csv.name}, "
                       f"so nothing says which tumours it should hold — it "
                       f"cannot be told apart from a partial table")
    try:
        slides = read_slide_csv(slide_list)
    except Exception as error:
        return False, f"{slide_list} could not be read: {error}"
    if SAMPLE_COLUMN not in slides.columns:
        return False, f"{slide_list} has no {SAMPLE_COLUMN!r} column"

    graded = frame["sample"].str.strip()
    repeated = sorted(set(graded[graded.duplicated()]))
    if repeated:
        return False, (f"{grades_csv.name} lists {len(repeated)} tumour(s) more "
                       f"than once (e.g. {repeated[:5]})")
    expected = set(slides[SAMPLE_COLUMN].str.strip())
    absent = sorted(expected - set(graded))
    extra = sorted(set(graded) - expected)
    if absent or extra:
        parts = []
        if absent:
            parts.append(f"{len(absent)} of the run's {len(expected)} tumours "
                         f"are missing (e.g. {absent[:5]})")
        if extra:
            parts.append(f"{len(extra)} are not in its slide list "
                         f"(e.g. {extra[:5]})")
        return False, (f"{grades_csv.name} does not match {slide_list.name}: "
                       + "; ".join(parts))
    return True, ""


def grades_csv_path(out_dir: Path) -> Path:
    """Where the pipeline publishes its grading table."""
    return out_dir / "anorak_tumour_grades.csv"


# --- can a compute node submit jobs? ---------------------------------------

def check_submit_from_compute_node(partition: str | None = None,
                                   timeout: int = 120) -> dict:
    """Ask an actual compute node whether it can reach the Slurm controller.

    The failure this catches is silent by construction: a head process that
    cannot submit does not crash, it waits. Checked here with a one-second
    allocation rather than inferred from the login node, which always can.
    """
    probe = "command -v sbatch >/dev/null && squeue --version >/dev/null && echo SUBMIT_OK"
    command = ["srun", "-t", "1", "-n", "1",
               *(["-p", partition] if partition else []),
               "bash", "-lc", probe]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
    except FileNotFoundError:
        return {"ok": False, "reason": "srun is not on PATH here"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"no compute node allocated within {timeout}s"}

    ok = "SUBMIT_OK" in (result.stdout or "")
    return {
        "ok": ok,
        "command": shlex.join(command),
        "reason": "" if ok else (
            (result.stderr or result.stdout or "").strip()[:400]
            or "the compute node produced no output"
        ),
    }


# --- submission ------------------------------------------------------------

def build_nextflow_command(
    *,
    pipeline_dir: Path,
    slides_csv: Path,
    raw_dir: Path,
    anorak_dir: Path,
    out_dir: Path,
    work_dir: Path,
    profile: str = "beatson",
    resume: bool = True,
    extra_args: list[str] | None = None,
) -> list[str]:
    """The `nextflow run` invocation, as a list.

    -ansi-log false because the log is being written to a file: the live
    progress display rewrites its own lines and is unreadable afterwards.

    extra_args may not repeat anything set here — see _OWNED_NEXTFLOW_OPTIONS.
    """
    clashes = [arg for arg in (extra_args or [])
               if any(arg == option or arg.startswith(option + "=")
                      for option in _OWNED_NEXTFLOW_OPTIONS)]
    if clashes:
        raise ValueError(
            f"extra Nextflow arguments repeat {', '.join(clashes)}, which the "
            f"submitter already sets. Nextflow refuses a repeated option at "
            f"launch (\"Can only specify option ... once\"), and a repeated "
            f"--param silently wins over the one the rest of this stage reads. "
            f"Change it through this submitter's own arguments instead."
        )
    return [
        "nextflow", "-log", str(out_dir / "nextflow.log"),
        "run", str(pipeline_dir),
        "-profile", profile,
        "-work-dir", str(work_dir),
        "-ansi-log", "false",
        *(["-resume"] if resume else []),
        "--slides_csv", str(slides_csv),
        "--raw_dir", str(raw_dir),
        "--anorak_dir", str(anorak_dir),
        "--outdir", str(out_dir),
        *(extra_args or []),
    ]


def build_supervised_command(
    *,
    supervisor: Path,
    out_dir: Path,
    work_dir: Path,
    nextflow_command: list[str],
    stall_seconds: int | None = None,
) -> list[str]:
    """The head job's command: `nextflow_command` under nf_supervise.sh.

    Shared with submit_hpl_nf.py, which passes its own stall_seconds; the
    rest — the stop marker, the signal lead — must be the same for both.

    --signal-lead-seconds is HEAD_SIGNAL_SECONDS, passed rather than assumed:
    the supervisor fits its whole TERM shutdown inside that lead, and a lead
    it had to guess at would be two copies of one number in two files.
    """
    return [
        "bash", str(supervisor),
        "--log", str(out_dir / "nextflow.log"),
        "--work-dir", str(work_dir),
        "--stop-marker", str(out_dir / SUPERVISOR_STOP_MARKER),
        "--stall-seconds", str(WATCHDOG_STALL_SECONDS if stall_seconds is None else stall_seconds),
        "--max-restarts", str(WATCHDOG_MAX_RESTARTS),
        "--signal-lead-seconds", str(HEAD_SIGNAL_SECONDS),
        "--",
        *nextflow_command,
    ]


def _sbatch_advice(reason: str, time_limit: str, partition: str | None) -> str:
    """Extra sentences for the sbatch rejections that have an obvious fix.

    sbatch's own messages are accurate and say nothing about which of this
    submitter's knobs produced the number it rejected. The time limit is the
    one that actually bites: it is a partition property, the default here is a
    guess, and the rejection names neither the value asked for nor where it
    came from.
    """
    lowered = reason.lower()
    if "time limit" in lowered:
        partition_flag = f" -p {partition}" if partition else ""
        return (
            f"\n\nThe head job asked for --time={time_limit}, which this "
            f"partition will not allow. Find the ceiling with\n"
            f"    sinfo{partition_flag} -o \"%P %l %L\"\n"
            f"(MaxTime is the second column) and set {HEAD_TIME_LIMIT_ENV} below "
            f"it, then restart the tile server — or pass --time-limit here.\n\n"
            f"Pick the largest the partition allows. The head job has to outlive "
            f"every job it submits, and one killed at its limit leaves its "
            f"children running with nothing collecting them. That is recoverable "
            f"— resubmitting with resume picks the run up from Nextflow's cache "
            f"— but it costs whatever was in flight."
        )
    if "invalid partition" in lowered or "partition" in lowered and "specified" in lowered:
        return (f"\n\nPartitions on this cluster:\n    sinfo -o \"%P %a %l\"")
    return ""


def head_sbatch_command(
    name: str,
    *,
    inner: str,
    partition: str | None,
    time_limit: str,
    log_stem: str,
    chdir: Path,
    notify_email: str | None = None,
    dependency: str | None = None,
) -> list[str]:
    """The sbatch line for a Nextflow head job, and for each --chain standby.

    Shared with submit_hpl_nf.py: both pipelines run one long-lived,
    supervised head process, and the reasons behind every flag below apply to
    any such process, not to ANORAK in particular.
    """
    return [
        "sbatch",
        f"--job-name={name}",
        *([f"--partition={partition}"] if partition else []),
        f"--cpus-per-task={HEAD_CPUS}",
        f"--mem={HEAD_MEMORY}",
        f"--time={time_limit}",
        # The head job's children are independent Slurm jobs; killing it does
        # not kill them. So a head job that simply hits its wall leaves tasks
        # running, and a resubmission then resumes into the *same* work
        # directories — Nextflow derives them from the task hash — putting two
        # processes in one directory with no error from either.
        #
        # This asks Slurm for SIGTERM two minutes before the kill. Nextflow
        # treats that as a shutdown and cancels what it has submitted, so the
        # wall becomes a clean stop and `-resume` is then safe with nothing to
        # tidy up first.
        f"--signal=B:TERM@{HEAD_SIGNAL_SECONDS}",
        f"--output={LOG_DIR}/{log_stem}_%j.out",
        f"--error={LOG_DIR}/{log_stem}_%j.err",
        f"--chdir={chdir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        # afternotok, not afterany: a successor exists to take over from a head
        # job that did NOT finish the cohort. When one succeeds, the rest of
        # the chain has nothing to do, and --kill-on-invalid-dep makes Slurm
        # clear them instead of leaving them pending on a dependency that can
        # never be satisfied.
        #
        # afternotok cannot tell a time limit from a pipeline failure or a
        # scancel, and would start a successor for all three. The supervisor
        # makes that distinction instead: on the last two it writes a stop
        # marker, and a successor that finds it exits 0 without running —
        # which also ends the rest of the chain, by the same rule as a success.
        *([f"--dependency=afternotok:{dependency}", "--kill-on-invalid-dep=yes"]
          if dependency else []),
        "--wrap", inner,
    ]


def submit_anorak_job(
    *,
    slides_csv: Path,
    raw_dir: Path,
    out_dir: Path,
    pipeline_dir: Path,
    anorak_dir: Path,
    scope: str = "full",
    sample_size: int | None = None,
    seed: int | None = None,
    profile: str = "beatson",
    resume: bool = True,
    chain: int = 1,
    partition: str | None = None,
    job_name: str = "anorak_nf",
    time_limit: str | None = None,
    prelude: str | None = None,
    notify_email: str | None = None,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
    tumour_verified: bool = True,
) -> dict:
    """Choose the slides, write the run's own list, and submit the head job.

    tumour_verified=False is for a list from slide_list_from_directory: every
    slide in a cohort nobody has selected tumour slides from yet. It is
    recorded in the run's selection, so the grading table can always say so.
    """
    # Each missing path gets the sentence that fixes it. Two of the four are
    # deployment-level locations the caller never typed — they come from the
    # server's environment — so naming the path alone leaves someone looking at
    # a directory they did not choose with no indication of where the choice
    # was made or how to change it.
    _MISSING_PATH_ADVICE = {
        "AIgrading clone": (
            f"Clone it there with\n"
            f"    git clone https://github.com/xi11/AIgrading.git <path>\n"
            f"or set ANORAK_REPO_DIR to wherever it already is, and restart the "
            f"tile server — that variable is read at import.\n"
            f"The clone must also hold models/AIgrading_anorak.h5 "
            f"(https://zenodo.org/records/15272883); the pipeline refuses at "
            f"launch without it."
        ),
        "pipeline directory": (
            f"This is the anorak-nf directory of this repository. Set "
            f"ANORAK_PIPELINE_DIR if the checkout on the cluster is somewhere "
            f"other than beside backend/, and restart the tile server."
        ),
    }
    if chain < 1:
        raise ValueError(f"--chain must be at least 1, got {chain}")
    if chain > 1 and not resume:
        raise ValueError(
            "--chain needs resume. Every job in a chain runs the same command, "
            "so without -resume each successor would start the cohort again "
            "from nothing — several runs writing the same output directory, "
            "each believing it is the only one."
        )

    for path, what in ((slides_csv, "slide list"), (raw_dir, "raw slide directory"),
                       (pipeline_dir, "pipeline directory"), (anorak_dir, "AIgrading clone")):
        if not path.exists():
            advice = _MISSING_PATH_ADVICE.get(what)
            raise ValueError(
                f"No such {what}: {path}" + (f"\n\n{advice}" if advice else "")
            )

    # Absolute from here on. The head job runs with --chdir set to out_dir, so
    # a relative path that resolved fine at submit time resolves against a
    # different directory inside the job — and `nextflow run anorak-nf` then
    # fails looking for a pipeline that is sitting right where it was typed.
    slides_csv, raw_dir, out_dir, pipeline_dir, anorak_dir = (
        slides_csv.resolve(), raw_dir.resolve(), out_dir.resolve(),
        pipeline_dir.resolve(), anorak_dir.resolve(),
    )
    if not (pipeline_dir / "main.nf").is_file():
        raise ValueError(f"{pipeline_dir} has no main.nf — point --pipeline-dir "
                         f"at the anorak-nf directory.")
    # Refused rather than run without: a head job that is not supervised is the
    # one that sat idle for five hours. Its absence means the cluster's copy of
    # anorak-nf predates the watchdog, which is a deployment to finish.
    supervisor = pipeline_dir / SUPERVISOR_RELATIVE_PATH
    if not supervisor.is_file():
        raise ValueError(
            f"{supervisor} does not exist. The head job runs Nextflow under it, "
            f"so that a stalled run restarts itself instead of idling until its "
            f"time limit. Copy the whole anorak-nf directory to the cluster "
            f"again — this file is newer than the copy there."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    run_list = out_dir / "slide_list.csv"

    # The run's own copy of the list can never be the source it is copied from.
    # Reading and writing one path looks harmless — pandas has the frame in
    # memory before the write — but it makes the input mutable by the process
    # consuming it: a second submission truncates the file while the first
    # run's head job is reading it, and Nextflow gets zero bytes and reports a
    # missing CSV header. Nothing about that failure points back here.
    #
    # A source anywhere under out_dir is refused for the same reason, not just
    # this exact name.
    if slides_csv.resolve() == run_list.resolve() or out_dir.resolve() in slides_csv.resolve().parents:
        raise ValueError(
            f"The slide list {slides_csv} is inside this run's own output "
            f"directory, so it is an output of a previous attempt rather than "
            f"a source. Point this at the list the cohort actually came from — "
            f"select_tumour_slides.py's output, optionally filtered by "
            f"filter_slides_by_tile_count.py — and keep it outside "
            f"{out_dir}."
        )

    frame = read_slide_csv(slides_csv)
    if frame.empty:
        raise ValueError(f"{slides_csv} lists no slides")
    check_slide_list(frame, slides_csv, tumour_verified=tumour_verified)

    chosen, selection = select_slide_rows(
        frame, scope=scope, sample_size=sample_size, seed=seed
    )
    selection["source_csv"] = str(slides_csv)
    selection["tumour_verified"] = bool(tumour_verified)
    write_slide_list(chosen, run_list, selection)

    work_dir = out_dir / "work"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    nextflow_command = build_nextflow_command(
        pipeline_dir=pipeline_dir, slides_csv=run_list, raw_dir=raw_dir,
        anorak_dir=anorak_dir, out_dir=out_dir, work_dir=work_dir,
        profile=profile, resume=resume, extra_args=extra_args,
    )
    prelude = prelude if prelude is not None else os.environ.get(PRELUDE_ENV, "")
    # exec, so the batch shell is *replaced* by the supervisor rather than
    # waiting on it as a child. Slurm's --signal=B: targets the batch shell's
    # own pid, and a plain `sh -c` would take the signal without passing it on
    # — which is the whole mechanism below. The supervisor traps it and
    # forwards it to Nextflow. `bash <script>` rather than running the script
    # directly, for the same reason main.nf calls `python3 bin/x.py`: a copy
    # that drops the exec bit must not matter.
    supervised = build_supervised_command(
        supervisor=supervisor, out_dir=out_dir, work_dir=work_dir,
        nextflow_command=nextflow_command,
    )
    inner = ((f"{prelude}\n" if prelude.strip() else "")
             + "exec " + shlex.join(supervised))

    def _sbatch_for(name: str, dependency: str | None = None) -> list[str]:
        return head_sbatch_command(
            name, inner=inner, partition=partition,
            time_limit=time_limit or HEAD_TIME_LIMIT, log_stem="anorak_nf",
            chdir=out_dir, notify_email=notify_email, dependency=dependency,
        )

    sbatch_command = _sbatch_for(job_name)

    info = {
        "slides_csv": str(run_list),
        "source_csv": str(slides_csv),
        "out_dir": str(out_dir),
        "work_dir": str(work_dir),
        "grades_csv": str(grades_csv_path(out_dir)),
        "selection": selection,
        "time_limit": time_limit or HEAD_TIME_LIMIT,
        "nextflow_command": shlex.join(nextflow_command),
        "watchdog": {"stall_seconds": WATCHDOG_STALL_SECONDS,
                     "max_restarts": WATCHDOG_MAX_RESTARTS},
        "sbatch_command": shlex.join(sbatch_command),
        "anorak_job_id": None,
        "chain": chain,
        "chain_job_ids": [],
    }
    if dry_run:
        return info

    # A marker left by an earlier attempt stopped that attempt's chain; this
    # submission is someone deciding to go again. Cleared only now, after
    # every refusal above, and its reason handed back rather than lost.
    marker = out_dir / SUPERVISOR_STOP_MARKER
    if marker.exists():
        info["cleared_stop_marker"] = marker.read_text(encoding="utf-8",
                                                       errors="replace").strip()
        marker.unlink()

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as error:
        reason = ((error.stderr or "").strip() or (error.stdout or "").strip()
                  or "no output from sbatch")
        raise RuntimeError(
            f"sbatch failed (exit {error.returncode}): {reason}"
            + _sbatch_advice(reason, time_limit or HEAD_TIME_LIMIT, partition)
            # Cleared before sbatch — a head job can start the moment it is
            # queued, and would find the marker and stop — so if nothing was
            # queued, this is the only place its reason still exists.
            + (f"\n\nThe previous attempt's stop marker was cleared; it said:\n"
               f"{info['cleared_stop_marker']}" if info.get("cleared_stop_marker") else "")
        ) from error

    stdout = (result.stdout or "").strip()
    info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info["anorak_job_id"] = match.group(1)

    # The successors, submitted now rather than when they are needed: the
    # point is that nobody has to be watching at 03:00 on day 9. Each waits on
    # the one before it and carries the same -resume, so the cohort continues
    # in the same work directory rather than starting again.
    #
    # A failure here leaves a SHORTER chain, not a broken one — the jobs
    # already submitted still run in order — so it is reported and not raised.
    if chain > 1:
        if not info["anorak_job_id"]:
            info["chain_error"] = (
                "sbatch did not report a job id, so no successors were "
                "submitted — there is nothing for them to depend on."
            )
            return info
        previous = info["anorak_job_id"]
        for position in range(2, chain + 1):
            follower = _sbatch_for(f"{job_name}_{position}", dependency=previous)
            try:
                result = _run_sbatch_with_retry(follower)
            except subprocess.CalledProcessError as error:
                reason = ((error.stderr or "").strip()
                          or (error.stdout or "").strip() or "no output")
                info["chain_error"] = (
                    f"submitted {len(info['chain_job_ids']) + 1} of {chain} head "
                    f"jobs; the {position}th was refused: {reason}"
                )
                break
            match = re.search(r"Submitted batch job (\d+)",
                              (result.stdout or "").strip())
            if not match:
                info["chain_error"] = (
                    f"the {position}th head job reported no job id, so the "
                    f"chain stops there."
                )
                break
            previous = match.group(1)
            info["chain_job_ids"].append(previous)
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-submit", action="store_true",
                        help="ask a compute node whether it can run sbatch, then exit")
    parser.add_argument("--slides-csv", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--pipeline-dir", type=Path,
                        help="the anorak-nf directory (holds main.nf)")
    parser.add_argument("--anorak-dir", type=Path,
                        help="clone of github.com/xi11/AIgrading")
    parser.add_argument("--scope", choices=("full", "subset"), default="full",
                        help="full cohort, or a random sample of it (default: full)")
    parser.add_argument("--sample-size", type=int,
                        help="slides to sample when --scope subset")
    parser.add_argument("--seed", type=int,
                        help="sampling seed; one is chosen and recorded if omitted")
    parser.add_argument("--profile", default="beatson")
    parser.add_argument("--chain", type=int, default=1, metavar="N",
                        help="submit N head jobs, each waiting on the one "
                             "before it and running only if it did not finish "
                             "the cohort. For a run longer than the partition's "
                             "MaxTime: the head job is replaced automatically "
                             "instead of by hand. Default 1 (no chain).")
    parser.add_argument("--no-resume", action="store_true",
                        help="start a fresh run instead of continuing the cached one")
    parser.add_argument("--partition", help="partition for the head job")
    parser.add_argument("--time-limit",
                        help=f"walltime for the head job, Slurm format "
                             f"(default: ${HEAD_TIME_LIMIT_ENV} or "
                             f"{HEAD_TIME_LIMIT}). Must be within the "
                             f"partition's MaxTime — see sinfo -o '%P %l'")
    parser.add_argument("--job-name", default="anorak_nf")
    parser.add_argument("--prelude", help=f"shell run before nextflow "
                                          f"(default: ${PRELUDE_ENV})")
    parser.add_argument("--notify-email")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_submit:
        result = check_submit_from_compute_node(args.partition)
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            print("\nA compute node could not run sbatch. The Nextflow head job "
                  "submits every task itself, so it would start, submit nothing, "
                  "and wait until its time limit.", file=sys.stderr)
        return 0 if result["ok"] else 1

    required = {"--slides-csv": args.slides_csv, "--raw-dir": args.raw_dir,
                "--out-dir": args.out_dir, "--pipeline-dir": args.pipeline_dir,
                "--anorak-dir": args.anorak_dir}
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    try:
        info = submit_anorak_job(
            slides_csv=args.slides_csv, raw_dir=args.raw_dir, out_dir=args.out_dir,
            pipeline_dir=args.pipeline_dir, anorak_dir=args.anorak_dir,
            scope=args.scope, sample_size=args.sample_size, seed=args.seed,
            profile=args.profile, resume=not args.no_resume, chain=args.chain,
            partition=args.partition, job_name=args.job_name,
            time_limit=args.time_limit,
            prelude=args.prelude, notify_email=args.notify_email,
            dry_run=args.dry_run,
        )
    except (ValueError, RuntimeError) as refusal:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 1

    selection = info["selection"]
    print(f"  scope:      {selection['scope']} ({selection['slides']} slides"
          + (f" sampled from {selection['pool']}, seed {selection['seed']}"
             if selection["scope"] == "subset" else "") + ")")
    print(f"  slide list: {info['slides_csv']}")
    print(f"  head job:   {info['time_limit']} walltime")
    print(f"  out dir:    {info['out_dir']}")
    if args.dry_run:
        print(f"\nDry run — nothing submitted:\n  {info['sbatch_command']}")
    else:
        if info.get("cleared_stop_marker"):
            print(f"  cleared:    the previous attempt's stop marker, which said:\n"
                  + "\n".join(f"                {line}" for line in
                              info["cleared_stop_marker"].splitlines()))
        print(f"  job id:     {info['anorak_job_id'] or 'none returned'}")
        if info.get("chain_job_ids"):
            print(f"  chain:      {len(info['chain_job_ids'])} standby head "
                  f"job(s) queued behind it: "
                  f"{', '.join(info['chain_job_ids'])}")
        if info.get("chain_error"):
            print(f"  chain:      {info['chain_error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
