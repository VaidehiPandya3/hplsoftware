#!/usr/bin/env python3
"""Submit HPL Stages 1-4 as one Nextflow run (hpl-nf/): the one-click pipeline.

The tile server calls submit_pipeline() from POST /pipeline-runs; nothing
about it needs the server, so it is a module of its own and testable without
one. What gets submitted is the same shape as Stage 7's ANORAK run — one
small, long-lived head job running `nextflow run` under the stall watchdog
(anorak-nf/tools/nf_supervise.sh), which then submits a job per slide, per
shard, per step itself. See submit_anorak_nf.py for why the head job looks the
way it does; head_sbatch_command() is shared with it.

Everything is decided HERE, before the queue, and written to the run's
run_config.json for the tasks to read:

  * **Every refusal the per-stage submitters make, made at once.** A missing
    checkpoint, an HPL_REPO_DIR that is not the clone, an absent container or
    extras directory, a reference that is really an .h5ad, a vote configuration
    that does nothing, a GPU request on a partition without GPUs, a walltime the
    partition cannot grant. Clicked through by hand these surfaced one stage at
    a time, each after the last had finished; a pipeline that discovered the
    checkpoint typo after eight hours of tiling would be strictly worse than
    what it replaces.

  * **Every output path.** The .h5, the projections file and the assignments
    CSV land exactly where the per-stage path puts them, so registration, the
    viewer and the KB load find them where they always have, and the server
    records them on the run row at submission.

  * **Refusing to start over someone else's output.** A complete .h5 or
    projections file already at one of those paths belongs to another run —
    the per-stage endpoints refuse to overwrite it, and so does this. It is also
    what lets each task treat "my output exists and validates" as "this run
    already made it" and skip, which is what makes -resume safe for tasks that
    write outside their work directory.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hpl_nf_state  # noqa: E402
from stage_outputs import validate_assignment_csv, validate_packaged_h5  # noqa: E402
from submit_anorak_nf import (  # noqa: E402
    HEAD_TIME_LIMIT,
    PRELUDE_ENV,
    SUPERVISOR_STOP_MARKER,
    WATCHDOG_MAX_RESTARTS,
    _sbatch_advice,
    build_supervised_command,
    check_submit_from_compute_node,
    head_sbatch_command,
)
from submit_mask_tile_slurm import _run_sbatch_with_retry  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent

#: The pipeline directory (holds main.nf). Beside backend/ in this repository;
#: set when the cluster's copy lives elsewhere.
PIPELINE_DIR = Path(os.getenv("HPL_NF_PIPELINE_DIR", str(BACKEND_DIR.parent / "hpl-nf")))

#: The watchdog, at <pipeline dir>/tools/nf_supervise.sh — the same place
#: ANORAK's lives in its own pipeline, so the pipeline folder is complete by
#: itself. In this repository that path is a symlink to ANORAK's copy, not a
#: second copy: the script is written against Nextflow's task monitor, not
#: against either pipeline, and two stall detectors are two chances for one to
#: be stale. A copy that dereferenced the link is fine as long as it is the same
#: bytes, which supervisor_drift() checks at every submission.
SUPERVISOR_RELATIVE_PATH = Path("tools") / "nf_supervise.sh"
SUPERVISOR = Path(os.getenv("HPL_NF_SUPERVISOR", str(PIPELINE_DIR / SUPERVISOR_RELATIVE_PATH)))
ANORAK_SUPERVISOR = BACKEND_DIR.parent / "anorak-nf" / SUPERVISOR_RELATIVE_PATH


def supervisor_drift(supervisor: Path, reference: Path = ANORAK_SUPERVISOR) -> str | None:
    """Why this supervisor differs from ANORAK's, or None if it does not (or
    there is no ANORAK copy here to compare with)."""
    try:
        if not reference.is_file() or Path(supervisor).resolve() == reference.resolve():
            return None
        if Path(supervisor).read_bytes() == reference.read_bytes():
            return None
    except OSError as e:
        return f"could not compare {supervisor} with {reference}: {e}"
    return (f"{supervisor} is not the same script as {reference}. The two pipelines "
            f"share one watchdog; one of these copies is stale. Copy "
            f"anorak-nf/tools/nf_supervise.sh over it, or restore the symlink.")

#: How long one head job may live — a partition property; see HEAD_TIME_LIMIT
#: in submit_anorak_nf.py. Separately settable because this pipeline and ANORAK
#: may reasonably be put on different partitions.
HEAD_TIME_LIMIT_ENV = "HPL_NF_HEAD_TIME_LIMIT"

#: Head jobs submitted per run: the head plus standbys that take over, with
#: -resume, only if the one before did not finish. Three by default because a
#: large cohort is tiling (hours) + packaging (up to a day) + encoding (up to a
#: day per shard) + assignment, which does not fit one 2-day head job, and a
#: single click should not need a second one days later. A standby behind a
#: head job that finished is cleared by Slurm (--kill-on-invalid-dep) at no cost.
DEFAULT_CHAIN = int(os.getenv("HPL_NF_CHAIN", "3"))

WATCHDOG_STALL_SECONDS = int(os.getenv("HPL_NF_WATCHDOG_STALL_SECONDS", "1800"))

#: Walltimes per step, Slurm format, checked against the partition before
#: anything is queued (sbatch's own refusal names neither value nor limit, and
#: from inside a pipeline it would arrive hours in). Defaults are the
#: per-stage submitters' own, except assignment: 4 days there assumed the GPU
#: partition's 5-day ceiling, and a sharded run needs nothing like it.
TIME_LIMITS = {
    "tiling": "2-00:00:00",
    "package": "2-00:00:00",
    "extract": "2-00:00:00",
    "mean": "08:00:00",
    "assign": "2-00:00:00",
}

RUN_CONFIG_NAME = "run_config.json"

#: The encoder weights every one-click run uses. A deployment setting, not a
#: per-run choice: the reference's clusters were built from embeddings of this
#: checkpoint, so a run under different weights would classify into the wrong
#: space. Checked at every submission (and by tools/preflight.py).
DEFAULT_CHECKPOINT = os.getenv(
    "HPL_CHECKPOINT",
    "/hpc-home/home/users/vpandya/long-term-scratch/Vaidehi/weights/BarlowTwins_3.ckt",
)

#: Slides tiled at once. 50 rather than the per-stage form's 10: tiling is one
#: CPU per slide and throughput-bound across the cohort, and at 10 a
#: 14,000-slide cohort's tiling alone is the better part of two weeks.
DEFAULT_MAX_TILING = int(os.getenv("HPL_NF_MAX_TILING", "50"))


# --- where a run's outputs go ----------------------------------------------

def output_paths(
    *,
    output_root: Path,
    hpl_repo_dir: Path,
    tile_dataset_name: str,
    h5_dataset_name: str,
    model: str,
    marker: str = "he",
    split: str = "train",
    tile_size: int = 224,
    z_dim: int = 128,
) -> dict[str, Path]:
    """The three artifacts, at the paths the per-stage endpoints use.

    Each path is computed by the function that stage's own code uses, so a
    change there moves this too rather than leaving the pipeline writing
    somewhere the server no longer looks.
    """
    from make_hpl_hdf5 import hpl_h5_output_path
    from submit_feature_extraction import expected_extraction_output_path

    h5 = hpl_h5_output_path(output_root, h5_dataset_name, marker, split, tile_size)
    projections = expected_extraction_output_path(
        hpl_repo_dir, model, h5_dataset_name, h5, z_dim=z_dim, img_size=tile_size,
    )
    # The tile server's _assignment_output_path: beside the projections, named
    # after the run's tile dataset.
    assignments = projections.parent / f"{tile_dataset_name}_hpc_assignments.csv"
    return {"h5": h5, "projections": projections, "assignments": assignments}


def refuse_foreign_outputs(paths: dict[str, Path], expected_h5_rows: int | None = None) -> None:
    """Refuse to start a fresh run over complete outputs it did not produce.

    The per-stage endpoints refuse the same thing one stage at a time
    ("A complete .h5 already exists at ..."). Here it matters twice over: every
    task skips an output that already validates, so a fresh run over someone
    else's .h5 would not rebuild it — it would adopt it, and carry another
    run's slides into this run's classification.
    """
    from submit_feature_extraction import validate_extraction_output

    found = []
    if paths["h5"].is_file() and validate_packaged_h5(paths["h5"])[0]:
        found.append(f"a complete .h5 at {paths['h5']}")
    if paths["projections"].is_file() and validate_extraction_output(paths["projections"])[0]:
        found.append(f"a complete projections file at {paths['projections']}")
    if paths["assignments"].is_file() and validate_assignment_csv(paths["assignments"])[0]:
        found.append(f"a complete assignments CSV at {paths['assignments']}")
    if found:
        raise FileExistsError(
            "This run's outputs already exist from an earlier run: "
            + "; ".join(found)
            + ". A pipeline run adopts outputs that already validate rather than "
              "rebuilding them, so starting here would mix another run's results "
              "into this one. Continue the run that made them (its Resume "
              "button), or start this one with 'Move earlier outputs aside', "
              "which moves them into a superseded-<date> folder beside them — "
              "nothing is deleted."
        )


def _output_family(target: Path) -> list[Path]:
    """The target and the files a stage leaves beside it: .partial, packaging's
    checkpoint sidecars (<name>.completed.txt ...), shard parts
    (<stem>.rows<lo>-<hi>.*), the assigner's <name>.chunks/ and the shared
    query mean. Matched by exact prefix, so a neighbour that merely shares the
    stem — a test .h5 named <stem>_test_sample_<sig>.h5 — is left alone."""
    if not target.parent.is_dir():
        return []
    name, stem = target.name, target.stem
    return sorted(
        entry for entry in target.parent.iterdir()
        if entry.name == name
        or entry.name.startswith(name + ".")
        or entry.name.startswith(stem + ".rows")
        or entry.name.startswith(stem + ".query_mean")
    )


def move_outputs_aside(paths: dict[str, Path], stamp: str) -> list[dict]:
    """Move a dataset's earlier outputs out of this run's way, keeping them.

    For a fresh run that would otherwise be refused by refuse_foreign_outputs —
    typically a full-cohort run over a dataset whose earlier (smaller) run left
    a complete .h5 at the same path. Each family goes into
    <its directory>/superseded-<stamp>/ on the same filesystem, so it is one
    rename, reversible by moving it back, and nothing is deleted. The caller
    must first establish no live job is writing these (the server does, from
    the run table).
    """
    moved = []
    for target in paths.values():
        family = _output_family(Path(target))
        if not family:
            continue
        dest = Path(target).parent / f"superseded-{stamp}"
        dest.mkdir(exist_ok=True)
        for entry in family:
            os.replace(entry, dest / entry.name)
            moved.append({"from": str(entry), "to": str(dest / entry.name)})
    return moved


# --- resolving everything up front --------------------------------------------

def _nf_duration(slurm_walltime: str) -> str:
    """'2-00:00:00' -> '172800s', the form Nextflow's time directive takes."""
    from submit_cluster_assignment import parse_slurm_walltime

    seconds = parse_slurm_walltime(slurm_walltime)
    if seconds is None:
        raise ValueError(f"A task walltime cannot be unlimited: {slurm_walltime!r}")
    return f"{seconds}s"


def resolve_run(
    *,
    submission_id: str,
    out_dir: Path,
    manifest: Path,
    raw_dir: Path,
    mask_dir: Path,
    tile_dir: Path,
    output_root: Path,
    tile_dataset_name: str,
    h5_dataset_name: str,
    tiling_params: dict,
    checkpoint: str,
    model: str = "BarlowTwins_3",
    marker: str = "he",
    reference: Path | None = None,
    vote_preset: str | None = None,
    vote_overrides: dict | None = None,
    extraction_shards: int = 1,
    assignment_shards: int = 1,
    device: str = "auto",
    cpu_partition: str | None = None,
    max_tiling_forks: int = 10,
    allow_incomplete: bool = False,
    check_partitions: bool = True,
) -> tuple[dict, dict]:
    """(run_config, nextflow params) for one run, or a refusal.

    Raises ValueError / FileNotFoundError / NotADirectoryError / SystemExit
    with the message the matching per-stage submitter would have given —
    SystemExit because resolve_vote/vote_flags refuse that way, and callers
    already map it to a 400.
    """
    import submit_cluster_assignment as sca
    import submit_feature_extraction as sfe

    if extraction_shards < 1 or assignment_shards < 1:
        raise ValueError("Shard counts must be at least 1.")

    # Stage 3's refusals.
    checkpoint = (checkpoint or "").strip()
    sfe._check_checkpoint(checkpoint)
    sfe._check_hpl_repo_dir(sfe.HPL_REPO_DIR)
    sfe._check_singularity_image(sfe.SINGULARITY_IMAGE, sfe.SINGULARITY_BIN)
    sfe._check_container_extras(sfe.CONTAINER_EXTRAS, sfe.SINGULARITY_IMAGE, sfe.SINGULARITY_BIN)
    gres, gres_reason = sfe.select_gpu_gres(sfe.GPU_PARTITION)

    # Stage 4's refusals.
    reference_path = sca._reference_path(reference)
    reference_info = sca.check_reference(reference_path)
    overrides = dict(vote_overrides or {})
    resolved_vote = sca.resolve_vote(vote_preset or sca.DEFAULT_VOTE_PRESET, **overrides)
    vote = sca.vote_flags(**resolved_vote)
    vote_description = sca.describe_vote(resolved_vote, vote_preset or sca.DEFAULT_VOTE_PRESET)
    device, device_reason = sca.resolve_device(device)
    assign_partition = sfe.GPU_PARTITION if device == "gpu" else sca.MERGE_PARTITION
    if device == "gpu" and sca.partition_has_gpus(assign_partition) is False:
        raise ValueError(
            f"Assignment on the GPU needs a GPU partition, but {assign_partition!r} "
            f"advertises none — the job would pend forever as ReqNodeNotAvail. "
            f"Use device 'cpu', or set HPL_GPU_PARTITION."
        )
    assign_extras = sfe.CONTAINER_EXTRAS_GPU if device == "gpu" else sfe.CONTAINER_EXTRAS
    sfe._check_container_extras(assign_extras, sfe.SINGULARITY_IMAGE, sfe.SINGULARITY_BIN)

    # Walltimes against their partitions, before the queue rather than hours in.
    if check_partitions:
        for step, partition in (("extract", sfe.GPU_PARTITION), ("assign", assign_partition),
                                ("tiling", cpu_partition), ("package", cpu_partition)):
            if partition:
                sca.check_time_limit(partition, TIME_LIMITS[step])

    paths = output_paths(
        output_root=output_root, hpl_repo_dir=sfe.HPL_REPO_DIR,
        tile_dataset_name=tile_dataset_name, h5_dataset_name=h5_dataset_name,
        model=model, marker=marker,
    )

    config = {
        "submission_id": submission_id,
        # Package without slides that fail to tile (see hpl-nf/bin/hpl_tile.py).
        # Recorded here, not only as a Nextflow param, so the tiling gate — the
        # step that decides what packaging reads — acts on the same choice.
        "allow_incomplete": bool(allow_incomplete),
        "out_dir": str(out_dir),
        "backend_dir": str(BACKEND_DIR),
        "raw_dir": str(raw_dir),
        "manifest": str(manifest),
        "mask_dir": str(mask_dir),
        "tile_dir": str(tile_dir),
        "dataset_name": tile_dataset_name,
        "tiling": dict(tiling_params),
        "packaging": {
            "output_root": str(output_root),
            "h5_dataset_name": h5_dataset_name,
            "h5_path": str(paths["h5"]),
            "marker": marker,
            "split": "train",
            "tile_size": 224,
            "threads_per_process": 4,
        },
        "extraction": {
            "hpl_repo_dir": str(sfe.HPL_REPO_DIR),
            "checkpoint": checkpoint,
            "model": model,
            "marker": marker,
            "dataset_name": h5_dataset_name,
            "z_dim": 128,
            "img_size": 224,
            "batch_size": sfe._DEFAULT_BATCH_SIZE,
            "singularity_bin": sfe.SINGULARITY_BIN,
            "singularity_image": str(sfe.SINGULARITY_IMAGE),
            "extras_dir": str(sfe.CONTAINER_EXTRAS),
            "shards": int(extraction_shards),
            "output_path": str(paths["projections"]),
            "gres": gres,
            "gres_reason": gres_reason,
        },
        "assignment": {
            "reference": str(reference_path),
            "reference_rows": reference_info["reference_rows"],
            "n_clusters": reference_info["n_clusters"],
            "rep_key": "z_latent",
            "k": resolved_vote["k"],
            "vote_flags": vote,
            "vote_preset": vote_preset or sca.DEFAULT_VOTE_PRESET,
            "vote": vote_description,
            "batch_size": 16_384,
            "shards": int(assignment_shards),
            "device": device,
            "device_reason": device_reason,
            "singularity_bin": sfe.SINGULARITY_BIN,
            "singularity_image": str(sfe.SINGULARITY_IMAGE),
            "extras_dir": str(assign_extras),
            "out_csv": str(paths["assignments"]),
        },
    }
    nf_params = {
        "max_tiling_forks": int(max_tiling_forks),
        "allow_incomplete": str(bool(allow_incomplete)).lower(),
        "cpu_partition": cpu_partition,
        "gpu_partition": sfe.GPU_PARTITION,
        "gpu_gres": gres,
        "assign_partition": assign_partition,
        "assign_device": device,
        "tiling_time": _nf_duration(TIME_LIMITS["tiling"]),
        "package_time": _nf_duration(TIME_LIMITS["package"]),
        "extract_time": _nf_duration(TIME_LIMITS["extract"]),
        "mean_time": _nf_duration(TIME_LIMITS["mean"]),
        "assign_time": _nf_duration(TIME_LIMITS["assign"]),
    }
    return config, nf_params


# --- submission ------------------------------------------------------------------

def write_run_config(config: dict) -> Path:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RUN_CONFIG_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_run_config(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / RUN_CONFIG_NAME).read_text(encoding="utf-8"))


def resume_params(config: dict, *, allow_incomplete: bool | None = None) -> tuple[dict, dict]:
    """(config, nf_params) for resuming a recorded run.

    Everything that decides what is computed, and where it runs, comes from the
    run's own recorded config — the tiling concurrency and partitions the user
    chose included. Only the GPU type is chosen again: which card looks free is
    a fact about now, not about the day the run started. allow_incomplete may
    be switched on, which is the recovery for a run stopped by slides that
    cannot be read; it cannot change anything already computed.
    """
    import submit_feature_extraction as sfe

    config = dict(config)
    nf_params = dict(config.get("nf_params") or {})
    if not nf_params:
        raise ValueError(
            "This run's config predates recorded Nextflow settings, so resuming "
            "it would silently use today's defaults. Start a new pipeline run.")
    nf_params["gpu_gres"], _ = sfe.select_gpu_gres(nf_params.get("gpu_partition") or sfe.GPU_PARTITION)
    if allow_incomplete is not None:
        config["allow_incomplete"] = bool(allow_incomplete)
        nf_params["allow_incomplete"] = str(bool(allow_incomplete)).lower()
    return config, nf_params


def build_nextflow_command(
    *,
    pipeline_dir: Path,
    config_path: Path,
    config: dict,
    nf_params: dict,
    python: str,
    profile: str = "beatson",
    resume: bool = True,
) -> list[str]:
    out_dir = Path(config["out_dir"])
    params: list[str] = []
    for key, value in nf_params.items():
        if value is None:
            continue
        params += [f"--{key}", str(value)]
    return [
        "nextflow", "-log", str(out_dir / "nextflow.log"),
        "run", str(pipeline_dir),
        "-profile", profile,
        "-work-dir", str(out_dir / "work"),
        "-ansi-log", "false",
        *(["-resume"] if resume else []),
        "--config", str(config_path),
        "--manifest", config["manifest"],
        "--outdir", str(out_dir),
        "--python", python,
        "--backend_dir", config["backend_dir"],
        *params,
    ]


def submit_pipeline(
    config: dict,
    nf_params: dict,
    *,
    pipeline_dir: Path = PIPELINE_DIR,
    supervisor: Path | None = None,
    python: str | None = None,
    profile: str = "beatson",
    resume: bool = True,
    chain: int = DEFAULT_CHAIN,
    partition: str | None = None,
    time_limit: str | None = None,
    job_name: str = "hpl_nf",
    prelude: str | None = None,
    notify_email: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Write run_config.json, then submit the supervised head job (and its chain).

    Returns {"nf_job_id", "chain_job_ids", "out_dir", "sbatch_command", ...}.
    The head job ids are also written to <out_dir>/head_job_ids, which is how
    hpl_nf_state finds the head job from a stage sentinel with no database.
    """
    if chain < 1:
        raise ValueError(f"chain must be at least 1, got {chain}")
    pipeline_dir = Path(pipeline_dir).resolve()
    if supervisor is None:
        supervisor = (Path(os.environ["HPL_NF_SUPERVISOR"]) if os.getenv("HPL_NF_SUPERVISOR")
                      else pipeline_dir / SUPERVISOR_RELATIVE_PATH)
    if not (pipeline_dir / "main.nf").is_file():
        raise ValueError(
            f"{pipeline_dir} has no main.nf. Set HPL_NF_PIPELINE_DIR to the hpl-nf "
            f"directory of this repository as copied to the cluster, and restart "
            f"the tile server."
        )
    if not Path(supervisor).is_file():
        raise ValueError(
            f"{supervisor} does not exist. The head job runs Nextflow under it so "
            f"a stalled run restarts itself instead of idling until its time "
            f"limit — hpl-nf/tools/nf_supervise.sh, a link to ANORAK's. Copy "
            f"the whole hpl-nf directory to the cluster again, or set "
            f"HPL_NF_SUPERVISOR."
        )
    drift = supervisor_drift(Path(supervisor))
    if drift:
        raise ValueError(drift)

    out_dir = Path(config["out_dir"])
    # The Nextflow params go into the run's own config, so a resume submits
    # with the same concurrency, partitions and walltimes the run started with
    # (resume_params) instead of whatever defaults are current.
    config["nf_params"] = dict(nf_params)
    config_path = write_run_config(config)
    python = python or str(Path(sys.executable).resolve())
    nextflow_command = build_nextflow_command(
        pipeline_dir=pipeline_dir, config_path=config_path, config=config,
        nf_params=nf_params, python=python, profile=profile, resume=resume,
    )
    # ANORAK's builder, not a copy of it: the copy that was here passed no
    # --stop-marker, so a real pipeline failure left nothing to stop the chain
    # and every --chain standby started, resumed and failed the same way in
    # turn; and no --signal-lead-seconds, so the TERM budget matched Slurm's
    # --signal only because both happened to be 120.
    supervised = build_supervised_command(
        supervisor=supervisor, out_dir=out_dir, work_dir=out_dir / "work",
        nextflow_command=nextflow_command,
        stall_seconds=WATCHDOG_STALL_SECONDS,
    )
    prelude = prelude if prelude is not None else os.environ.get(PRELUDE_ENV, "")
    # exec, so Slurm's --signal=B:TERM reaches the supervisor itself — see
    # submit_anorak_job, which this mirrors.
    inner = ((f"{prelude}\n" if prelude.strip() else "") + "exec " + shlex.join(supervised))
    limit = time_limit or os.getenv(HEAD_TIME_LIMIT_ENV) or HEAD_TIME_LIMIT

    def sbatch_for(name: str, dependency: str | None = None) -> list[str]:
        return head_sbatch_command(
            name, inner=inner, partition=partition, time_limit=limit,
            log_stem="hpl_nf", chdir=out_dir, notify_email=notify_email,
            dependency=dependency,
        )

    info = {
        "out_dir": str(out_dir),
        "run_config": str(config_path),
        "nextflow_log": str(out_dir / "nextflow.log"),
        "nextflow_command": shlex.join(nextflow_command),
        "sbatch_command": shlex.join(sbatch_for(job_name)),
        "time_limit": limit,
        "chain": chain,
        "nf_job_id": None,
        "chain_job_ids": [],
    }
    if dry_run:
        return info

    # A marker left by an earlier attempt stopped that attempt's chain; this
    # submission is someone deciding to go again. Cleared before sbatch — a
    # head job can start the moment it is queued, and would find it and stop.
    marker = out_dir / SUPERVISOR_STOP_MARKER
    if marker.exists():
        info["cleared_stop_marker"] = marker.read_text(encoding="utf-8",
                                                       errors="replace").strip()
        marker.unlink()

    try:
        result = _run_sbatch_with_retry(sbatch_for(job_name))
    except subprocess.CalledProcessError as error:
        reason = ((error.stderr or "").strip() or (error.stdout or "").strip()
                  or "no output from sbatch")
        raise RuntimeError(
            f"sbatch failed (exit {error.returncode}): {reason}"
            + _sbatch_advice(reason, limit, partition).replace(
                "ANORAK_HEAD_TIME_LIMIT", HEAD_TIME_LIMIT_ENV)
        ) from error
    match = re.search(r"Submitted batch job (\d+)", (result.stdout or ""))
    if not match:
        raise RuntimeError(
            f"sbatch accepted the head job but reported no job id "
            f"({(result.stdout or '').strip()!r}). Without one nothing can track "
            f"or cancel this run; check squeue for a job named {job_name}."
        )
    info["nf_job_id"] = match.group(1)

    previous = info["nf_job_id"]
    for position in range(2, chain + 1):
        try:
            follower = _run_sbatch_with_retry(
                sbatch_for(f"{job_name}_{position}", dependency=previous))
        except subprocess.CalledProcessError as error:
            info["chain_error"] = (
                f"submitted {position - 1} of {chain} head jobs; the next was "
                f"refused: {(error.stderr or error.stdout or '').strip()}")
            break
        found = re.search(r"Submitted batch job (\d+)", follower.stdout or "")
        if not found:
            info["chain_error"] = f"head job {position} reported no job id; the chain stops there."
            break
        previous = found.group(1)
        info["chain_job_ids"].append(previous)

    hpl_nf_state.write_head_job_ids(out_dir, [info["nf_job_id"], *info["chain_job_ids"]])
    return info


def all_job_ids(out_dir: Path) -> list[str]:
    """The head job and its standbys, for cancelling a run."""
    return hpl_nf_state.read_head_job_ids(out_dir)


def cancel_run(out_dir: Path, head_job_ids: list[str] | None = None) -> dict:
    """scancel the head chain, then anything still queued from this run.

    TERM to the head job lets Nextflow cancel what it submitted, but a head job
    that is itself stuck (the stall the watchdog exists for) cannot. So the
    run's task jobs are also found the way nf_supervise.sh finds them — by
    working directory, which no other run shares — and cancelled directly.
    """
    head = list(head_job_ids) if head_job_ids is not None else all_job_ids(out_dir)
    cancelled, errors = [], []
    if head:
        result = subprocess.run(["scancel", *head], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            cancelled += head
        else:
            errors.append((result.stderr or result.stdout or "").strip())

    work = str(Path(out_dir) / "work")
    user = os.environ.get("USER", "")
    listing = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i|%Z"],
                             capture_output=True, text=True, timeout=30)
    tasks = [line.split("|", 1)[0].strip()
             for line in (listing.stdout or "").splitlines()
             if "|" in line and line.split("|", 1)[1].strip().startswith(work)]
    if tasks:
        result = subprocess.run(["scancel", *tasks], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            cancelled += tasks
        else:
            errors.append((result.stderr or result.stdout or "").strip())
    return {"cancelled_job_ids": cancelled, "errors": [e for e in errors if e]}


def log_tail(out_dir: Path, lines: int = 40) -> str | None:
    """The end of nextflow.log — where a failed run says why."""
    path = Path(out_dir) / "nextflow.log"
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return "\n".join(text.splitlines()[-lines:])


__all__ = [
    "PIPELINE_DIR", "SUPERVISOR", "DEFAULT_CHAIN", "TIME_LIMITS",
    "output_paths", "refuse_foreign_outputs", "resolve_run", "submit_pipeline",
    "read_run_config", "cancel_run", "log_tail", "check_submit_from_compute_node",
]
