"""Where a Nextflow run of Stages 1-4 is, read off disk and one Slurm state.

The pipeline (hpl-nf/) runs as one Slurm head job that submits everything
else itself, so the per-stage job ids the rest of the server gates on do not
exist. Rather than grow a second set of gates beside the existing ones, each
stage column on slurm_dataset_runs records a sentinel —

    nf:<submission_id>:<stage>

— and this module answers "what Slurm state is that stage in", in the same
vocabulary sacct uses, so _get_slurm_job_state, _job_output_ready and the
stepper treat a Nextflow stage exactly like a real job. The output validators
still have the last word, exactly as for a real job: COMPLETED here means the
stage's own task said so, never that a file is trustworthy.

The evidence is two files per stage, written by the tasks themselves under the
run's output directory:

    stages/<stage>.started     — the first task of the stage began
    stages/<stage>.done.json   — the stage's final task validated its output

and the head job's state. A stage is COMPLETED only on its done marker, which
the task writes after the same validator the server's gate uses
(stage_outputs.py). Without one it is RUNNING or PENDING while the head job is
alive, and otherwise takes the head job's terminal state — reported as FAILED
when the head job itself says COMPLETED, because a pipeline that exited cleanly
without finishing a stage stopped early, and "COMPLETED" would put a Retry
button out of reach.

No imports from the tile server, and no Postgres: the tasks call this too.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

NF_JOB_ID_PREFIX = "nf:"

#: Stages the pipeline runs, in order. The names are the ones the server's
#: status payload and the UI stepper already use.
STAGES = ("tiling", "packaging", "extraction", "assignment")

#: The run's head job id(s), written by the submitter after sbatch answers.
#: One per line: the first is the head job, the rest its --chain standbys.
HEAD_JOB_IDS_FILE = "head_job_ids"

_IN_FLIGHT = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING",
}

_SENTINEL_RE = re.compile(r"^nf:(?P<submission_id>[A-Za-z0-9_.-]+):(?P<stage>[a-z_]+)$")


def nf_job_id(submission_id: str, stage: str) -> str:
    """The sentinel recorded in a stage's job-id column."""
    if stage not in STAGES:
        raise ValueError(f"Unknown pipeline stage {stage!r}; expected one of {STAGES}")
    return f"{NF_JOB_ID_PREFIX}{submission_id}:{stage}"


def is_nf_job_id(job_id) -> bool:
    return str(job_id or "").startswith(NF_JOB_ID_PREFIX)


def parse_nf_job_id(job_id: str) -> tuple[str, str]:
    """(submission_id, stage), or ValueError for anything malformed.

    Strict, because the submission id becomes a directory name below: a
    sentinel carrying a slash would read state from somewhere else entirely.
    """
    match = _SENTINEL_RE.match(str(job_id or ""))
    if not match or match.group("stage") not in STAGES:
        raise ValueError(f"Not a pipeline sentinel: {job_id!r}")
    return match.group("submission_id"), match.group("stage")


# --- markers -------------------------------------------------------------

def stages_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "stages"


def started_marker(out_dir: Path, stage: str) -> Path:
    return stages_dir(out_dir) / f"{stage}.started"


def done_marker(out_dir: Path, stage: str) -> Path:
    return stages_dir(out_dir) / f"{stage}.done.json"


def mark_started(out_dir: Path, stage: str) -> None:
    path = started_marker(out_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def mark_done(out_dir: Path, stage: str, payload: dict) -> Path:
    """Record a stage as finished. Written under a temporary name and renamed,
    so a reader never sees half a marker — the rule every output here follows."""
    path = done_marker(out_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_done(out_dir: Path, stage: str) -> dict | None:
    path = done_marker(out_dir, stage)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- the head job ----------------------------------------------------------

def write_head_job_ids(out_dir: Path, job_ids: list[str]) -> None:
    path = Path(out_dir) / HEAD_JOB_IDS_FILE
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text("".join(f"{j}\n" for j in job_ids if j), encoding="utf-8")
    os.replace(tmp, path)


def read_head_job_ids(out_dir: Path) -> list[str]:
    try:
        text = (Path(out_dir) / HEAD_JOB_IDS_FILE).read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip().isdigit()]


def combine_head_states(states: list[str | None]) -> str | None:
    """One state for a head job and its --chain standbys.

    A chain is one run: a standby only starts if the job before it did not
    finish (afternotok), and is killed as an invalid dependency when it did. So
    any COMPLETED means the run completed, anything live means the run is live
    (RUNNING over PENDING — a standby pending on its dependency is not news
    while the head runs), and otherwise the last job to have run is the answer.

    None anywhere with nothing decisive beside it stays None: an unreachable
    Slurm is not evidence of anything.
    """
    if not states:
        return None
    if "COMPLETED" in states:
        return "COMPLETED"
    live = [s for s in states if s in _IN_FLIGHT]
    if live:
        return "RUNNING" if "RUNNING" in live else live[0]
    if any(s is None for s in states):
        return None
    # A standby killed for an invalid dependency reads CANCELLED, which is not
    # what happened to the run; prefer the state of a job that actually ran.
    ran = [s for s in states if s and s != "CANCELLED"]
    if ran:
        return ran[-1]
    return states[-1]


def stage_state(out_dir: Path, stage: str, head_state: str | None) -> str | None:
    """Slurm-vocabulary state for one stage. See the module docstring."""
    if done_marker(out_dir, stage).is_file():
        return "COMPLETED"
    if head_state is None:
        return None
    if head_state in _IN_FLIGHT:
        if head_state == "RUNNING" and started_marker(out_dir, stage).is_file():
            return "RUNNING"
        return "PENDING"
    if head_state in ("", "COMPLETED"):
        # "" is a head job that has aged out of accounting: long over, and
        # this stage never finished. COMPLETED without a marker is a pipeline
        # that stopped before reaching it.
        return "FAILED"
    return head_state


def stage_summary(out_dir: Path, head_state: str | None) -> dict:
    """Per-stage state plus whatever the done markers recorded, for /status."""
    return {
        stage: {
            "state": stage_state(out_dir, stage, head_state),
            "started": started_marker(out_dir, stage).is_file(),
            "done": read_done(out_dir, stage),
        }
        for stage in STAGES
    }
