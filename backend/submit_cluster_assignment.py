#!/usr/bin/env python3
"""Submit a Slurm job that assigns HPL cluster IDs to a run's tile embeddings.

Stage 3. Takes the projections .h5 that feature extraction produced, runs
assign_hpc_clusters.py against the reference built by build_hpc_reference.py,
and writes a per-tile CSV of cluster IDs with two confidence columns.

No GPU. faiss's k-NN search is CPU work and the reference is a few hundred MB,
so this asks for cores and memory instead — which also means it can run on a
CPU partition while the GPU queue is busy.

Two things are worth knowing before reading further.

The reference is not optional and not inferable. Cluster IDs mean nothing
except relative to one reference (one Leiden run, one fold) plus the encoder
checkpoint the embeddings came from. Assigning against the wrong reference
produces a complete, well-formed CSV of IDs that silently do not correspond to
the hpc_dictionary the UI joins against, so the reference is checked at submit
time and recorded with the results.

And unlike feature extraction, this stage is cheap to redo — it reads
embeddings rather than images, and takes minutes rather than hours. So it does
not go to the lengths extraction does to avoid recomputation; it overwrites its
own output on request instead.

Usage:
    python submit_cluster_assignment.py
        --projections-h5 <results>/BarlowTwins_3/DS/h224_w224_n3_zdim128/hdf5_DS_he_train.h5
        --out /path/to/DS_hpc_assignments.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from submit_feature_extraction import (
    CONTAINER_EXTRAS_GPU,
    CONTAINER_EXTRAS,
    shard_ranges,
    MERGE_PARTITION,
    SINGULARITY_BIN,
    SINGULARITY_IMAGE,
    bootstrap_container_extras,
    _CONTAINER_EXTRA_PACKAGES_GPU,
    _bind_args,
    _check_container_extras,
    _check_singularity_image,
)
from submit_mask_tile_slurm import _run_sbatch_with_retry

ASSIGN_SCRIPT = "assign_hpc_clusters.py"

# Slurm walltime for the three jobs this module can submit. Named rather than
# scattered as literals because they used to be one settable default and two
# hardcoded 2-hour limits: raising the one that was settable left two that could
# still kill a long run late, after the expensive part had already succeeded.
#
# The assignment itself dominates. It is one faiss flat-L2 scan of the reference
# per query batch, so its cost is (queries x reference rows) and it grows with
# the dataset: 545,185 TCGA queries against 2.5M reference rows took about an
# hour, so a 14,000-slide cohort is a different order of magnitude -- roughly
# 11M tiles, which extrapolates to about 20 hours unsharded. Four days is not an
# estimate, it is headroom -- a job that finishes early costs nothing, and one
# that hits the wall at 95% has to be redone from the start because the assigner
# has no resume. Sharding divides this: N shards each do 1/N of the work, so the
# same limit covers a proportionally worse-than-expected run.
#
# Check it fits before raising it further. Nothing here verifies the walltime
# against the partition's MaxTime, despite what --time-limit's help used to
# claim; sbatch either rejects the job outright or -- with EnforcePartLimits=NO,
# which is the more confusing case -- accepts it and leaves it pending forever
# with reason PartitionTimeLimit. Neither is silent, but the second is easy to
# read as a busy queue. `sinfo -o "%P %l"` is the check.
ASSIGN_TIME_LIMIT = "4-00:00:00"
# One full pass over the projections .h5 to compute the shared query mean. Same
# I/O as the assignment without the search, so much cheaper -- but it scales with
# the same input, which is why it is no longer 2 hours.
MEAN_TIME_LIMIT = "08:00:00"
# Concatenating the shard CSVs. Pure I/O over the outputs, but a 14,000-slide
# cohort's CSVs are tens of gigabytes.
MERGE_TIME_LIMIT = "04:00:00"

# Default location of the reference artifact, matching the constant
# build_hpc_reference.py writes to. Imported lazily in _reference_path()
# because that module is a sibling script, not a package, and a hard import at
# module scope would make this file unimportable wherever it is absent.
_REFERENCE_ENV = "HPC_REFERENCE_PATH"

# Modules assign_hpc_clusters.py needs at runtime. faiss is a hard requirement
# — there is no fallback search path — so it belongs here rather than in a
# separate optional probe.
_REQUIRED_MODULES = ("numpy", "pandas", "h5py", "faiss")

# Exactly what build_hpc_reference.save() writes. Listed rather than guessed:
# an earlier version of this check looked for a "labels" key that the builder
# has never written, so a perfectly good reference was rejected as malformed.
# test_reference_keys_match_the_builder keeps the two in step by round-tripping
# through the real save().
_REFERENCE_KEYS = {"reference", "components", "codes", "categories", "n_neighbors", "meta"}


def _reference_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.getenv(_REFERENCE_ENV)
    if env:
        return Path(env)
    try:
        from build_hpc_reference import HPC_REFERENCE_PATH
        return HPC_REFERENCE_PATH
    except ImportError:
        return Path(__file__).resolve().with_name("hpc_reference_leiden_2p5_fold2.npz")


def check_reference(reference: Path) -> dict:
    """Refuse to submit against a reference that is missing or not one.

    Returns what the .npz says about itself, so the submitter can record which
    reference produced an assignment. Cluster IDs from two references are
    indistinguishable once they are in the registry, which is the whole reason
    tile_registry has an hpc_reference column.
    """
    if not reference.is_file():
        raise FileNotFoundError(
            f"HPC reference not found: {reference}. Build it once from the Leiden "
            f".h5ad, e.g.\n"
            f"  python build_hpc_reference.py \\\n"
            f"      --h5ad '<...>/cluster reference/LATTICeA_5x_he_complete_surv_sex"
            f"_filtered_leiden_2p5__fold2_subsample.h5ad' \\\n"
            f"      --out {reference}\n"
            f"Or set {_REFERENCE_ENV} to an existing one."
        )

    # The .h5ad is the *source* for the reference, not the reference. Pointing
    # at it is the natural mistake — it is the file you have, its name contains
    # "leiden_2p5__fold2", and the UI asks for a path. numpy's own complaint
    # ("This file contains pickled (object) data") describes the byte format it
    # failed to parse and says nothing about the missing build step, so the case
    # is detected here and named.
    with reference.open("rb") as fh:
        magic = fh.read(8)
    if magic == b"\x89HDF\r\n\x1a\n" or reference.suffix == ".h5ad":
        raise ValueError(
            f"{reference} is an HDF5/.h5ad file, not a reference .npz. The .h5ad is "
            f"what the reference is *built from* — convert it once:\n"
            f"  python build_hpc_reference.py \\\n"
            f"      --h5ad {reference} \\\n"
            f"      --out {_reference_path().name}\n"
            f"then leave the reference field blank to use it, or give the .npz path."
        )

    import json

    import numpy as np
    try:
        with np.load(reference, allow_pickle=False) as npz:
            keys = set(npz.files)
            missing = _REFERENCE_KEYS - keys
            if missing:
                raise ValueError(
                    f"{reference} is missing {sorted(missing)} (it has {sorted(keys)}) "
                    f"— it does not look like a build_hpc_reference.py artifact. "
                    f"Rebuild it."
                )
            meta = {}
            try:
                meta = json.loads(str(npz["meta"]))
            except (ValueError, TypeError):
                pass
            info = {
                "reference_path": str(reference),
                "reference_rows": int(npz["reference"].shape[0]),
                "reference_dims": int(npz["reference"].shape[1]),
                # The PCA basis is (input dims, components), so this is the
                # width of a raw embedding — 128 where reference_dims is 127.
                # A query mean is subtracted from raw embeddings *before*
                # projection, so it is this number a mean must match, not the
                # component count.
                "embedding_dims": int(npz["components"].shape[0]),
                "n_clusters": int(len(npz["categories"])),
                "groupby": meta.get("groupby"),
                "k": int(npz["n_neighbors"]),
                # No stored mean means --centering reference is unavailable, so
                # sharding has to go through --query-mean rather than the
                # shard-independent reference frame. Worth surfacing, since the
                # alternative is finding out from a failed job.
                "has_mean": "mean" in keys,
            }
    except (OSError, ValueError) as e:
        if isinstance(e, ValueError) and "does not look like" in str(e):
            raise
        raise ValueError(f"Could not read {reference} as an .npz: {e}") from e
    return info


def check_projections(path: Path, rep_key: str = "z_latent") -> int:
    """Confirm the input holds embeddings of the expected kind, and count them.

    Named datasets rather than any-h5-will-do because the failure otherwise
    lands inside assign_hpc_clusters.py after a queue wait, reported as a
    KeyError against a name the reader has no reason to recognise.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Projections file not found: {path}. Feature extraction (Stage 2) "
            f"produces this; check it finished."
        )
    try:
        with h5py.File(path, "r") as f:
            matches = [k for k in f.keys() if k.endswith(rep_key)]
            if not matches:
                raise KeyError(
                    f"{path} has no dataset ending in '{rep_key}' (found: "
                    f"{sorted(f.keys())}). This is the encoder's output file, so "
                    f"either extraction wrote something unexpected or this is the "
                    f"packaged input .h5 rather than the projections."
                )
            rows = int(f[matches[0]].shape[0])
    except OSError as e:
        raise ValueError(f"Could not open {path} as HDF5: {e}") from e
    if rows == 0:
        raise ValueError(f"{path} holds zero embeddings — nothing to assign.")
    return rows


def _import_check_python(reference_in_job: str) -> str:
    """Import probe the job runs before touching the reference.

    Same reasoning as the extraction job's: these are container-provided, and
    finding out after the queue wait that pandas is absent costs an allocation
    to learn a one-line fact.
    """
    return (
        "import sys\n"
        "missing = []\n"
        f"for name in {list(_REQUIRED_MODULES)!r}:\n"
        "    try:\n"
        "        __import__(name)\n"
        "    except ImportError as e:\n"
        "        missing.append(f'{name} ({e})')\n"
        "if missing:\n"
        "    print('FATAL: packages missing inside the container: ' + ', '.join(missing),\n"
        "          file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "import faiss\n"
        "print('faiss:', faiss.__version__ if hasattr(faiss, '__version__') else 'present',\n"
        "      flush=True)\n"
        "print('container packages: ok', flush=True)\n"
    )


# Named vote configurations. Defined once, here, because the server, the API
# client and the UI all have to mean the same thing by "the tuned one" -- six
# loose numbers copied into four places is how a run ends up with five of them
# right, which produces a complete well-formed CSV of slightly different cluster
# IDs and nothing to say so.
#
# Accuracies are leave-one-out on the production reference
# (hpc_reference_leiden_2p5_fold2.npz) at 200,000 tiles, seed 0. They are not
# agreement with Kai's TCGA transfer, which is a different measurement that can
# move the other way -- see CLASSIFIER_TUNING_2026-08-13.md section 19.
VOTE_PRESETS: dict[str, dict] = {
    "tuned": {
        "label": "Tuned (97.27%)",
        "accuracy": 0.9727,
        "summary": "k=10, distance^3, re-vote at k=25 below margin 0.15",
        "why": (
            "The settled configuration. Distance weighting at power 3, plus a "
            "second vote at k=25 for the ~5% of tiles whose first vote was "
            "nearly tied. Costs no extra search: the wider neighbours come out "
            "of the same scan."
        ),
        "flags": {
            "k": 10,
            "distance_weighted": True,
            "distance_power": 3.0,
            "class_weighted": False,
            "local_scaling": 0,
            "adaptive_margin": 0.15,
            "adaptive_k": 25,
        },
    },
    "legacy": {
        "label": "Legacy unweighted (96.78%)",
        "accuracy": 0.9678,
        "summary": "plain majority vote at the reference's own n_neighbors",
        "why": (
            "What every Stage 4 job submitted before 2026-08-21 actually ran, "
            "because the submitter forwarded no vote setting at all. Here so an "
            "existing assignment can be reproduced exactly."
        ),
        "flags": {
            "k": None,
            "distance_weighted": False,
            "distance_power": 1.0,
            "class_weighted": False,
            "local_scaling": 0,
            "adaptive_margin": 0.0,
            "adaptive_k": 0,
        },
    },
}

DEFAULT_VOTE_PRESET = "tuned"


def resolve_vote(preset: str | None = None, **overrides) -> dict:
    """The vote settings for a preset, with any explicit overrides applied.

    Overrides exist for the same reason --reference is exposed in the UI:
    comparing two configurations is a real thing to want. They are applied on
    top of a named preset rather than onto bare defaults, so an override always
    starts from something that was actually measured.

    An override of None means "not specified" and leaves the preset's value
    alone -- otherwise every optional field in an HTTP request body would erase
    the preset it was sent alongside.
    """
    name = preset or DEFAULT_VOTE_PRESET
    if name not in VOTE_PRESETS:
        raise SystemExit(
            f"Unknown vote preset {name!r}. Known: "
            f"{', '.join(sorted(VOTE_PRESETS))}."
        )
    resolved = dict(VOTE_PRESETS[name]["flags"])
    unknown = set(overrides) - set(resolved)
    if unknown:
        raise SystemExit(
            f"Not vote settings: {sorted(unknown)}. "
            f"Known: {sorted(resolved)}."
        )
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def describe_vote(resolved: dict, preset: str | None = None) -> str:
    """One line naming the configuration, for a run record and a UI caption.

    Says which preset it came from AND whether it still matches it, because a
    preset name alone would be a lie once anything was overridden.
    """
    name = preset or DEFAULT_VOTE_PRESET
    flags = " ".join(vote_flags(**resolved)) or "(plain unweighted vote)"
    k = resolved.get("k")
    detail = f"k={k}" if k is not None else "k=reference n_neighbors"
    matches = (name in VOTE_PRESETS
               and resolved == VOTE_PRESETS[name]["flags"])
    suffix = "" if matches else " (modified)"
    return f"{name}{suffix}: {detail} {flags}".strip()


def vote_flags(
    *,
    k: int | None,
    distance_weighted: bool,
    distance_power: float,
    class_weighted: bool,
    local_scaling: int,
    adaptive_margin: float,
    adaptive_k: int,
) -> list[str]:
    """The vote configuration, as flags, refusing combinations that do nothing.

    These used not to be forwarded at all, so every Stage 4 job ran the plain
    unweighted vote whatever was measured offline. Forwarding them one at a
    time is worse than not forwarding them: a partial configuration produces a
    complete, well-formed CSV of different cluster IDs with nothing to say it
    was not the configuration asked for. So they travel together and the
    inert combinations are refused here, before the queue, rather than being
    dropped silently inside the container four hours later.
    """
    flags: list[str] = []
    if distance_power != 1.0 and not distance_weighted:
        raise SystemExit(
            f"--distance-power {distance_power:g} is ignored without "
            f"--distance-weighted: every neighbour would weigh exactly 1. "
            f"Pass both, or neither."
        )
    if distance_weighted:
        flags.append("--distance-weighted")
        flags.append(f"--distance-power {distance_power:g}")
    if class_weighted:
        flags.append("--class-weighted")
    if local_scaling:
        flags.append(f"--local-scaling {local_scaling}")

    if adaptive_margin > 0:
        if k is None:
            raise SystemExit(
                "--adaptive-margin needs an explicit --k. Without one the base "
                "neighbourhood is whatever the reference's n_neighbors turns "
                "out to be, so whether --adaptive-k is actually wider than it "
                "cannot be checked until the job is already running."
            )
        if adaptive_k <= k:
            raise SystemExit(
                f"--adaptive-k {adaptive_k} is not wider than --k {k}; the "
                f"re-vote would see the same neighbours and change nothing."
            )
        flags.append(f"--adaptive-margin {adaptive_margin:g}")
        flags.append(f"--adaptive-k {adaptive_k}")
    elif adaptive_k:
        raise SystemExit(
            f"--adaptive-k {adaptive_k} does nothing without a positive "
            f"--adaptive-margin to gate it."
        )
    return flags


def parse_slurm_walltime(value: str) -> int | None:
    """Slurm walltime to seconds. None means unlimited.

    Accepts the forms both sides of this use: "4-00:00:00" from our own
    constants, "2-00:00:00" / "12:00:00" / "infinite" from `sinfo -o %l`, and
    "MM:SS" for completeness.
    """
    text = (value or "").strip().lower()
    if not text or text in ("infinite", "unlimited", "n/a"):
        return None
    days, _, clock = text.partition("-")
    if not clock:
        days, clock = "0", days
    parts = [int(p) for p in clock.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    else:
        raise ValueError(f"Unrecognised Slurm walltime: {value!r}")
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def partition_time_limit(partition: str) -> int | None:
    """The partition's own maximum walltime in seconds, or None if unknown.

    None for "could not ask" as well as for "unlimited", deliberately: this is
    used to produce a better error message, never to refuse on its own, so a
    cluster where sinfo is unavailable must not lose the ability to submit.
    """
    try:
        result = subprocess.run(["sinfo", "-h", "-p", partition, "-o", "%l"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    limits = []
    for line in (result.stdout or "").splitlines():
        try:
            seconds = parse_slurm_walltime(line)
        except ValueError:
            continue
        if seconds is None:
            return None                      # an unlimited row caps nothing
        limits.append(seconds)
    return max(limits) if limits else None


def check_time_limit(partition: str, requested: str) -> None:
    """Refuse a walltime the partition cannot grant, before sbatch does.

    sbatch's own refusal — "Requested time limit is invalid (missing or exceeds
    some limit)" — is correct and nearly useless: it names neither the limit nor
    the value, and it arrives at the bottom of a traceback holding a
    3,000-character --wrap string. It also arrives *after* the mean job has been
    submitted, leaving an orphan queued against a dependency that will never
    exist.

    The 4-day default was set for the GPU partition, which allows five. `compute`
    allows two, so submitting a CPU run with the defaults always failed here.
    """
    limit = partition_time_limit(partition)
    if limit is None:
        return
    try:
        wanted = parse_slurm_walltime(requested)
    except ValueError as e:
        raise ValueError(str(e)) from e
    if wanted is None or wanted <= limit:
        return
    raise ValueError(
        f"--time-limit {requested} exceeds partition {partition!r}'s maximum of "
        f"{limit // 86400}-{limit % 86400 // 3600:02d}:"
        f"{limit % 3600 // 60:02d}:{limit % 60:02d}. sbatch would refuse this "
        f"after the mean job had already been queued. Pass a shorter "
        f"--time-limit — a 32-shard assignment is hours per shard, so "
        f"1-00:00:00 is ample — or submit to a partition with a longer limit."
    )


def jobs_in_flight_named(job_name: str) -> list[str]:
    """Ids of this user's queued or running jobs whose name starts with
    job_name. Empty when squeue cannot be reached — see refuse_if_already_queued.
    """
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    found = []
    for line in (result.stdout or "").splitlines():
        job_id, _, name = line.partition("|")
        # The mean and merge steps are named <job_name>_mean / _merge, so a
        # prefix match catches a whole pipeline rather than only its array.
        if name.strip().startswith(job_name):
            found.append(job_id.strip())
    return found


def refuse_if_already_queued(job_name: str, *, force: bool = False) -> None:
    """Refuse a second identical pipeline while the first is still in flight.

    Two runs of the same submission collide on every path they use: one
    query_mean.npy, one set of 32 shard part files, one output CSV — and the
    second pipeline's merge runs --cleanup, deleting parts the first one's tasks
    are still writing. Two processes writing one part file is the shape of
    failure this codebase is written against: the row count can come out right
    while the contents interleave.

    A retype of the same command is how this happens, so the guard is on the job
    name rather than on any flag. squeue being unreachable is not a refusal:
    this prevents an accident, and must not become a new way to be blocked.
    """
    if force:
        return
    existing = jobs_in_flight_named(job_name)
    if not existing:
        return
    raise ValueError(
        f"{len(existing)} job(s) named {job_name!r} are already queued or "
        f"running: {', '.join(existing)}. A second pipeline would write the "
        f"same query_mean.npy, the same shard parts and the same output CSV, "
        f"and its merge would delete parts the first one is still writing.\n\n"
        f"Cancel those first (scancel {' '.join(existing)}), or pass "
        f"--force-duplicate if you genuinely intend two runs — in which case "
        f"give the second one a different --out."
    )


def partition_has_gpus(partition: str) -> bool | None:
    """Whether the partition advertises any generic resources (GPUs).

    None means "could not ask", which never blocks a submission — same posture
    as partition_time_limit.
    """
    try:
        result = subprocess.run(["sinfo", "-h", "-p", partition, "-o", "%G"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [l.strip() for l in (result.stdout or "").splitlines() if l.strip()]
    if not lines:
        return None
    return any(l not in ("(null)", "N/A") for l in lines)


def resolve_device(device: str, extras_gpu: Path = CONTAINER_EXTRAS_GPU) -> tuple[str, str]:
    """(device, why) for a submission. "auto" is decided here, not in the job.

    The job's own "auto" can look at the GPU in front of it; a submission cannot,
    because the sbatch flags — --nv, --gres, which extras directory to bind —
    have to be chosen before any node is allocated. The observable that decides
    it is whether the GPU extras have been bootstrapped: without them the
    container has faiss-cpu and nothing else, so asking Slurm for a GPU would
    queue behind every real GPU job to run a CPU search.

    Reported rather than silent, because "why is this on the CPU partition" is a
    question the answer should already be on screen for.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"Unknown device: {device!r}. Use 'auto', 'cpu' or 'gpu'.")
    if device != "auto":
        return device, "requested explicitly"
    if extras_gpu.is_dir() and (extras_gpu / "faiss").is_dir():
        return "gpu", f"GPU extras present at {extras_gpu}"
    return "cpu", (f"no GPU faiss at {extras_gpu} — run "
                   f"`submit_feature_extraction.py --bootstrap-extras-gpu` to "
                   f"use one")


def _build_assignment_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    extras_dir: Path,
    assign_script: Path,
    reference: Path,
    projections_h5: Path,
    out_csv: Path,
    rep_key: str,
    k: int | None,
    batch_size: int,
    validate_against: Path | None,
    query_mean: Path | None = None,
    shard_bounds: list[tuple[int, int]] | None = None,
    vote: list[str] | None = None,
    threads: int = 1,
    device: str = "cpu",
    row_range: tuple[int, int] | None = None,
) -> str:
    """Shell command the Slurm --wrap runs.

    shard_bounds is for a Slurm array (each task picks its range by
    SLURM_ARRAY_TASK_ID); row_range is one fixed range, for a caller that runs
    every shard as its own job with no array index — the Nextflow pipeline.
    Both reach the container by the same SINGULARITYENV_ route.

    --nv only for device="gpu". The search is CPU work by default, and asking
    for the GPU runtime then would queue the job behind every real GPU job for
    no benefit. With device="gpu" it is the opposite: without --nv the container
    sees no driver, and faiss would refuse at startup (Searcher verifies the GPU
    index against the CPU one rather than falling back silently).
    """
    paths = [assign_script.parent, reference.parent, projections_h5, out_csv.parent,
             singularity_image, extras_dir]
    if validate_against is not None:
        paths.append(validate_against.parent)
    binds = _bind_args(*paths)

    real = os.path.realpath
    args = [
        f"--reference {shlex.quote(real(reference))}",
        f"--h5 {shlex.quote(real(projections_h5))}",
        f"--out {shlex.quote(real(out_csv))}",
        f"--rep-key {shlex.quote(rep_key)}",
        f"--batch-size {batch_size}",
        # Progress every N tiles, so a long run is visibly alive in the log
        # rather than silent until it finishes.
        "--progress 50000",
        f"--device {device}",
    ]
    if k is not None:
        args.append(f"--k {k}")
    # The vote configuration, already validated as a whole by vote_flags().
    args.extend(vote or [])
    if validate_against is not None:
        args.append(f"--validate-against {shlex.quote(real(validate_against))}")

    # Sharding. The mean is supplied rather than computed per task because
    # --centering query centres on the mean of *all* queries: a task that
    # computed its own would project into a slightly different space and emit a
    # well-formed CSV of different cluster IDs. assign_hpc_clusters.py refuses
    # the combination outright, so this is belt and braces on a guard that
    # already exists.
    # The shard bounds are resolved OUTSIDE the container and handed in through
    # the environment, because SLURM_ARRAY_TASK_ID is one more thing
    # `singularity exec --cleanenv` wipes — the same mechanism that silently
    # pinned every assignment to one core, except here it is not silent: under
    # `set -u` the array index aborts the task the instant the import check
    # finishes, the whole array dies, and the merge is left on
    # DependencyNeverSatisfied with no clue in the log beyond where it stops.
    #
    # SINGULARITYENV_/APPTAINERENV_ are the documented way through --cleanenv,
    # and both prefixes are set because the binary may be either.
    outer_preamble = ""
    env_parts: list[str] = []
    if device == "gpu":
        # Slurm tells each task which physical GPU is its own through
        # CUDA_VISIBLE_DEVICES, and --cleanenv deletes it like everything else.
        # Without this, every task on a node sees all the cards and
        # index_cpu_to_gpu(res, 0, ...) puts them all on GPU 0: N shards
        # contending for one device while the rest idle, at no point failing.
        # Third instance of this class, after the thread count and the array
        # index.
        #
        # ":-0" so a hand-run job outside Slurm still works, where the variable
        # is legitimately unset and device 0 is the only sensible default.
        env_parts += [
            'SINGULARITYENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
            'APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        ]
    if query_mean is not None:
        args.append(f"--query-mean {shlex.quote(real(query_mean))}")
    if shard_bounds is not None and row_range is not None:
        raise ValueError("Pass shard_bounds (a Slurm array) or row_range (one "
                         "fixed range), not both.")
    if row_range is not None:
        lo, hi = (int(v) for v in row_range)
        outer_preamble = (
            "set -euo pipefail; "
            f"ROW_START={lo}; ROW_STOP={hi}; "
            'echo "=== Rows $ROW_START-$ROW_STOP ==="; '
        )
    elif shard_bounds is not None:
        starts = " ".join(str(lo) for lo, _ in shard_bounds)
        stops = " ".join(str(hi) for _, hi in shard_bounds)
        outer_preamble = (
            "set -euo pipefail; "
            f"SHARD_STARTS=({starts}); SHARD_STOPS=({stops}); "
            # Still `set -u` on the index, which is the check worth keeping: an
            # unset SLURM_ARRAY_TASK_ID out here means a sharded command really
            # was submitted as a plain job, and one task silently encoding the
            # wrong range is worse than a refusal.
            'ROW_START="${SHARD_STARTS[$SLURM_ARRAY_TASK_ID]}"; '
            'ROW_STOP="${SHARD_STOPS[$SLURM_ARRAY_TASK_ID]}"; '
            'echo "=== Shard $SLURM_ARRAY_TASK_ID: rows $ROW_START-$ROW_STOP ==="; '
        )
    if shard_bounds is not None or row_range is not None:
        env_parts += [
            'SINGULARITYENV_ROW_START="$ROW_START"',
            'SINGULARITYENV_ROW_STOP="$ROW_STOP"',
            'APPTAINERENV_ROW_START="$ROW_START"',
            'APPTAINERENV_ROW_STOP="$ROW_STOP"',
        ]
        args.append('--row-start "$ROW_START" --row-stop "$ROW_STOP"')

    inner = (
        "set -euo pipefail; "
        f"export PYTHONPATH={shlex.quote(real(extras_dir))}${{PYTHONPATH:+:$PYTHONPATH}}; "
        'export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-$$"; mkdir -p "$MPLCONFIGDIR"; '
        # Thread count baked in at submit time rather than read from
        # SLURM_CPUS_PER_TASK here. This string is expanded INSIDE the
        # container, and `singularity exec --cleanenv` has already wiped the
        # environment by then, so the Slurm variable does not exist and the
        # fallback silently won: every assignment ran on one core. A 2.5M-row
        # reference at 127 dims came to 49 tiles/s, which is ~31 GFLOP/s — a
        # single core's fp32 rate — turning a few hours into three days.
        #
        # Still pinned rather than left to faiss, which sizes its pool to the
        # machine rather than to the cpuset Slurm gave us; oversubscribing a
        # shared node is slower than running serially, as well as antisocial.
        f"export OMP_NUM_THREADS={int(threads)}; "
        'export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"; '
        'export MKL_NUM_THREADS="$OMP_NUM_THREADS"; '
        "echo '=== Container packages ==='; "
        f"python -c {shlex.quote(_import_check_python(real(reference)))}; "
        f"echo '=== Cluster assignment ({device}) ==='; "
        f"python {shlex.quote(real(assign_script))} {' '.join(args)}"
    )
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""
    return outer_preamble + env_prefix + " ".join([
        shlex.quote(singularity_bin), "exec", "--cleanenv",
        *(["--nv"] if device == "gpu" else []), *binds,
        shlex.quote(str(singularity_image)), "bash", "-lc", shlex.quote(inner),
    ])


def _build_simple_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    extras_dir: Path,
    script: Path,
    args: list[str],
    extra_binds: list[Path],
    banner: str,
    threads: int = 1,
) -> str:
    """One-liner container invocation, shared by the mean and merge jobs.

    Both are small, CPU-only, single-purpose steps around the array; giving each
    its own bespoke builder would be three near-identical functions.
    """
    binds = _bind_args(script.parent, extras_dir, singularity_image, *extra_binds)
    real = os.path.realpath
    inner = (
        "set -euo pipefail; "
        f"export PYTHONPATH={shlex.quote(real(extras_dir))}${{PYTHONPATH:+:$PYTHONPATH}}; "
        # See _build_assignment_command: --cleanenv means SLURM_CPUS_PER_TASK
        # is not readable from in here.
        f"export OMP_NUM_THREADS={int(threads)}; "
        'export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"; '
        f"echo '=== {banner} ==='; "
        f"python {shlex.quote(real(script))} {' '.join(args)}"
    )
    return " ".join([
        shlex.quote(singularity_bin), "exec", "--cleanenv", *binds,
        shlex.quote(str(singularity_image)), "bash", "-lc", shlex.quote(inner),
    ])


def submit_cluster_assignment_job(
    projections_h5: Path,
    out_csv: Path,
    *,
    reference: Path | None = None,
    depends_on_job_id: str | None = None,
    rep_key: str = "z_latent",
    k: int | None = None,
    batch_size: int = 16_384,
    validate_against: Path | None = None,
    shards: int = 1,
    centering: str = "query",
    # The vote, as a named preset plus optional overrides. A preset rather than
    # six loose numbers because a partially-applied vote configuration is this
    # codebase's signature failure: it produces a complete, well-formed CSV of
    # slightly different cluster IDs, and nothing downstream can tell.
    #
    # None means "legacy", not "tuned", so no existing caller changes behaviour
    # by being upgraded. The UI and the server endpoints ask for "tuned"
    # explicitly, which is what makes the change visible at the call site.
    vote_preset: str | None = None,
    k_override: int | None = None,
    distance_weighted: bool | None = None,
    distance_power: float | None = None,
    class_weighted: bool | None = None,
    local_scaling: int | None = None,
    adaptive_margin: float | None = None,
    adaptive_k: int | None = None,
    partition: str = MERGE_PARTITION,
    cpus: int = 16,
    memory: str = "64G",
    time_limit: str = ASSIGN_TIME_LIMIT,
    mean_time_limit: str = MEAN_TIME_LIMIT,
    merge_time_limit: str = MERGE_TIME_LIMIT,
    job_name: str = "hpl_cluster_assign",
    notify_email: str | None = None,
    singularity_image: Path = SINGULARITY_IMAGE,
    singularity_bin: str = SINGULARITY_BIN,
    extras_dir: Path | None = None,
    device: str = "auto",
    overwrite: bool = False,
    force_duplicate: bool = False,
    query_mean: Path | None = None,
) -> dict:
    """Submit Stage 3 for one projections file.

    depends_on_job_id chains this behind extraction (or behind a shard merge),
    so the whole pipeline can be queued in one go. afterok rather than afterany:
    assigning clusters to a projections file that extraction failed to finish
    would read zero-filled rows and produce confident-looking nonsense.
    """
    reference = _reference_path(reference)
    reference_info = check_reference(reference)

    # First, before any filesystem or Slurm work: an inert or contradictory
    # vote configuration is a refusal, not something to discover in a log.
    #
    # `k` is both a plain argument of this function and part of a vote preset,
    # so it needs one resolution order rather than two: an explicit k always
    # wins, then k_override, then the preset's. Without this a caller passing
    # k=15 alongside the tuned preset would silently get the preset's k=10.
    resolved_vote = resolve_vote(
        vote_preset or "legacy",
        k=k if k is not None else k_override,
        distance_weighted=distance_weighted,
        distance_power=distance_power,
        class_weighted=class_weighted,
        local_scaling=local_scaling,
        adaptive_margin=adaptive_margin,
        adaptive_k=adaptive_k,
    )
    k = resolved_vote["k"]
    vote = vote_flags(**resolved_vote)
    vote_description = describe_vote(resolved_vote, vote_preset or "legacy")

    # Skipped when chained: the file will not exist yet, because the job that
    # writes it has not run. The dependency is what guarantees it later.
    rows = None
    if depends_on_job_id is None:
        rows = check_projections(projections_h5, rep_key)

    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{out_csv} already exists. This stage is cheap to redo — delete it, or "
            f"resubmit with overwrite enabled, if you mean to replace it."
        )

    _check_singularity_image(singularity_image, singularity_bin)
    device, device_reason = resolve_device(device)
    print(f"Device:           {device}  ({device_reason})")
    if device == "gpu":
        # A --gres=gpu:1 on a partition with no GPUs pends forever as
        # ReqNodeNotAvail, which reads like a busy queue rather than a
        # misconfiguration. Refuse instead, at submit time.
        has_gpus = partition_has_gpus(partition)
        if has_gpus is False:
            raise ValueError(
                f"--device gpu asks Slurm for a GPU, but partition "
                f"{partition!r} advertises none, so the job would pend "
                f"indefinitely as ReqNodeNotAvail. Submit to a GPU partition "
                f"(--partition), or use --device cpu. Note a GPU partition is "
                f"preemptible here, so pair it with --shards."
            )
    # The GPU search reads a different extras directory, because faiss-cpu and
    # a GPU faiss are both imported as `faiss` and cannot share a PYTHONPATH.
    if extras_dir is None:
        extras_dir = CONTAINER_EXTRAS_GPU if device == "gpu" else CONTAINER_EXTRAS
    _check_container_extras(extras_dir, singularity_image, singularity_bin)
    # Before the mean job, so a bad walltime does not leave an orphan queued
    # against a dependency that never appears.
    check_time_limit(partition, time_limit)
    refuse_if_already_queued(job_name, force=force_duplicate)

    if shards > 1 and depends_on_job_id is not None:
        # The array size has to be known at sbatch time, and it comes from the
        # projections file's row count — which does not exist yet when this is
        # chained behind extraction. Refused rather than guessed.
        raise ValueError(
            "Cannot combine --shards with --depends-on-job-id: the number of rows "
            "to split is read from the projections file, which the job it depends "
            "on has not written yet. Either submit unsharded and chained, or wait "
            "for extraction to finish and then submit sharded."
        )

    script_path = Path(__file__).resolve()
    backend_dir = script_path.parent
    log_dir = backend_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- sharding -----------------------------------------------------------
    # Three jobs rather than one: compute the query mean over the whole file,
    # then an array of assign tasks all handed that same mean, then a merge.
    #
    # The mean job exists solely because --centering query centres on the mean of
    # every query. Letting each task compute its own would put each shard in a
    # slightly different space and produce a well-formed CSV of different cluster
    # IDs — the one failure here that no downstream check would catch.
    shard_bounds = None
    mean_path = None
    mean_job_id = None
    if shards > 1:
        shard_bounds = shard_ranges(rows, shards)
        mean_path = out_csv.with_name(f"{out_csv.stem}.query_mean.npy")

        if centering == "query" and query_mean is not None:
            # An already-computed mean, so no mean job and no dependency. The
            # step is one streamed pass over the projections and needs neither
            # faiss nor the container — `assign_hpc_clusters.py
            # --precompute-mean` runs it anywhere — so a mean job that will not
            # start should not hold up an array, and a retry should not repeat a
            # pass it already has.
            #
            # Validated here rather than trusted: every shard centres on this
            # file, so a truncated or wrong-width one produces 32 well-formed
            # CSVs of wrong cluster IDs, which is the failure mode with no
            # downstream check.
            mean_path = Path(query_mean)
            if not mean_path.is_file():
                raise FileNotFoundError(f"No such query-mean file: {mean_path}")
            try:
                loaded = np.load(mean_path)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{mean_path} is not a readable .npy: {e}") from e
            # The raw-embedding width, not the component count. project()
            # subtracts this mean from the embeddings and *then* multiplies by
            # the (input dims, components) basis, so a correct mean for a
            # 2.5M x 127 reference is 128 wide. Checking against 127 rejected
            # the right file.
            expected = int(reference_info["embedding_dims"])
            if loaded.shape != (expected,):
                raise ValueError(
                    f"{mean_path} holds shape {loaded.shape}, but this "
                    f"reference's PCA basis takes {expected}-dimensional "
                    f"embeddings (and produces "
                    f"{reference_info['reference_dims']} components). A query "
                    f"mean is subtracted from raw embeddings before projection, "
                    f"so it must be {expected} wide. A mean of the wrong width "
                    f"projects every tile into a different space and produces a "
                    f"complete CSV of wrong cluster IDs."
                )
            if not np.isfinite(loaded).all():
                raise ValueError(
                    f"{mean_path} contains non-finite values — it was probably "
                    f"written by an interrupted job. Delete it and recompute.")
            print(f"Query mean:       {mean_path} (reused, {expected}-d embeddings)")
        elif centering == "query":
            mean_command = _build_simple_command(
                singularity_bin=singularity_bin, singularity_image=singularity_image,
                extras_dir=extras_dir, script=backend_dir / ASSIGN_SCRIPT,
                args=[
                    f"--reference {shlex.quote(os.path.realpath(reference))}",
                    f"--h5 {shlex.quote(os.path.realpath(projections_h5))}",
                    f"--rep-key {shlex.quote(rep_key)}",
                    f"--precompute-mean {shlex.quote(os.path.realpath(mean_path.parent) + '/' + mean_path.name)}",
                ],
                extra_binds=[projections_h5, reference.parent, out_csv.parent],
                banner="Query mean",
                # Matches --cpus-per-task=4 on the mean sbatch below. The two
                # numbers have to be written together: the container cannot
                # read the Slurm one.
                threads=4,
            )
            mean_sbatch = [
                "sbatch", f"--job-name={job_name}_mean",
                f"--partition={partition}", "--cpus-per-task=4", "--mem=16G",
                f"--time={mean_time_limit}",
                f"--output={log_dir}/hpl_assign_mean_%j.out",
                f"--error={log_dir}/hpl_assign_mean_%j.err",
                f"--chdir={backend_dir}",
                "--wrap", f"bash -lc {shlex.quote(mean_command)}",
            ]
            try:
                mean_result = _run_sbatch_with_retry(mean_sbatch)
            except subprocess.CalledProcessError as e:
                reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output"
                raise RuntimeError(f"Could not submit the query-mean job: {reason}") from e
            match = re.search(r"Submitted batch job (\d+)", mean_result.stdout or "")
            mean_job_id = match.group(1) if match else None
            if not mean_job_id:
                raise RuntimeError(
                    "The query-mean job was submitted but sbatch reported no job ID, "
                    "so the shard array cannot be made to depend on it. Without that "
                    "dependency the shards would read a mean file that does not exist "
                    "yet."
                )
        else:
            # reference / none centering needs no shared mean: both are
            # shard-independent by construction.
            mean_path = None

    command = _build_assignment_command(
        vote=vote,
        singularity_bin=singularity_bin,
        singularity_image=singularity_image,
        extras_dir=extras_dir,
        assign_script=backend_dir / ASSIGN_SCRIPT,
        reference=reference,
        projections_h5=projections_h5,
        out_csv=out_csv,
        rep_key=rep_key,
        k=k,
        batch_size=batch_size,
        validate_against=validate_against,
        query_mean=mean_path,
        shard_bounds=shard_bounds,
        threads=cpus,
        device=device,
    )

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        # One device per task. The search holds the whole reference in device
        # memory — 2.5M x 127 float32 is about 1.3 GB, so any of this cluster's
        # cards is ample — and a shard array asks for one each.
        *(["--gres=gpu:1"] if device == "gpu" else []),
        *([f"--array=0-{shards - 1}"] if shard_bounds else []),
        # afterok on the mean job when sharding: a shard that ran before the mean
        # file existed would fail on a missing --query-mean, and one that somehow
        # read a stale mean would silently disagree with its siblings.
        *([f"--dependency=afterok:{mean_job_id or depends_on_job_id}"]
          if (mean_job_id or depends_on_job_id) else []),
        # %A_%a for an array, %j otherwise. %j in an array task expands to that
        # task's OWN JobId — a number that appears nowhere in `squeue`, which
        # shows 1241672_0 — so the logs existed under names nobody could
        # predict, and every attempt to tail a shard's log hit "no such file".
        *([f"--output={log_dir}/hpl_assign_%A_%a.out",
           f"--error={log_dir}/hpl_assign_%A_%a.err"] if shard_bounds else
          [f"--output={log_dir}/hpl_assign_%j.out",
           f"--error={log_dir}/hpl_assign_%j.err"]),
        f"--chdir={backend_dir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        "--wrap", f"bash -lc {shlex.quote(command)}",
    ]

    info = {
        "projections_h5": str(projections_h5),
        "out_csv": str(out_csv),
        "embeddings": rows,
        "assignment_job_id": None,
        "shards": shards,
        "shard_bounds": shard_bounds,
        "mean_job_id": mean_job_id,
        "merge_job_id": None,
        "sbatch_command": shlex.join(sbatch_command),
        # Two CSVs from the same reference but different vote settings are not
        # interchangeable, and nothing in the CSV itself distinguishes them.
        # Recording the flags here at least puts them in the run record next to
        # the job ID that produced the file.
        "vote_flags": " ".join(vote),
        "vote_preset": vote_preset or "legacy",
        "vote": vote_description,
        **reference_info,
    }

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as e:
        reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output from sbatch"
        raise RuntimeError(f"sbatch failed (exit {e.returncode}): {reason}") from e

    stdout = (result.stdout or "").strip()
    info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info["assignment_job_id"] = match.group(1)

    if shard_bounds:
        if not info["assignment_job_id"]:
            raise RuntimeError(
                "The shard array was submitted but sbatch reported no job ID, so the "
                "merge cannot depend on it. Merge by hand once the array finishes:\n"
                f"  python {backend_dir / 'merge_assignment_shards.py'} "
                f"--output {out_csv} --expected-rows {rows} --cleanup"
            )
        merge_command = _build_simple_command(
            singularity_bin=singularity_bin, singularity_image=singularity_image,
            extras_dir=extras_dir, script=backend_dir / "merge_assignment_shards.py",
            args=[
                f"--output {shlex.quote(os.path.realpath(out_csv.parent) + '/' + out_csv.name)}",
                # Checked against the projections file rather than believed from
                # the parts: a missing final shard leaves no gap to detect.
                f"--expected-rows {rows}",
                "--cleanup",
            ],
            extra_binds=[out_csv.parent],
            banner="Merging shards",
            threads=2,          # matches --cpus-per-task=2 below
        )
        merge_sbatch = [
            "sbatch", f"--job-name={job_name}_merge",
            f"--partition={partition}", "--cpus-per-task=2", "--mem=8G",
            f"--time={merge_time_limit}",
            # afterok, not afterany: concatenating around a failed shard is the
            # silent corruption this whole path is built to avoid.
            f"--dependency=afterok:{info['assignment_job_id']}",
            f"--output={log_dir}/hpl_assign_merge_%j.out",
            f"--error={log_dir}/hpl_assign_merge_%j.err",
            f"--chdir={backend_dir}",
            "--wrap", f"bash -lc {shlex.quote(merge_command)}",
        ]
        try:
            merge_result = _run_sbatch_with_retry(merge_sbatch)
        except subprocess.CalledProcessError as e:
            reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output"
            raise RuntimeError(
                f"The shard array is job {info['assignment_job_id']}, but the merge "
                f"job could not be submitted: {reason}. Merge by hand once it finishes."
            ) from e
        merge_match = re.search(r"Submitted batch job (\d+)", merge_result.stdout or "")
        if merge_match:
            info["merge_job_id"] = merge_match.group(1)

    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a Slurm job assigning HPL cluster IDs to tile embeddings.",
    )
    parser.add_argument("--projections-h5", type=Path, required=True,
                        help="Feature extraction's output for this dataset.")
    parser.add_argument("--out", dest="out_csv", type=Path, required=True,
                        help="Where the per-tile assignment CSV goes.")
    parser.add_argument("--reference", type=Path, default=None,
                        help=f"Reference .npz. Defaults to ${_REFERENCE_ENV} or the "
                             f"path build_hpc_reference.py writes.")
    parser.add_argument("--depends-on-job-id", type=str, default=None)
    parser.add_argument("--rep-key", type=str, default="z_latent")
    parser.add_argument("--vote-preset", default="legacy",
                        choices=sorted(VOTE_PRESETS),
                        help="Named vote configuration. "
                             + "; ".join(f"{name}: {spec['summary']}"
                                         for name, spec in sorted(VOTE_PRESETS.items()))
                             + ". Defaults to legacy so this CLI keeps doing "
                               "what it did; the UI asks for tuned explicitly.")
    # Overrides. None means "leave the preset alone", so an unset flag cannot
    # silently erase the preset it was passed alongside.
    parser.add_argument("--distance-weighted", dest="distance_weighted",
                        action="store_true", default=None,
                        help="Weight each neighbour by 1/(distance+eps)**power "
                             "instead of one vote each. Overrides the preset.")
    parser.add_argument("--no-distance-weighted", dest="distance_weighted",
                        action="store_false",
                        help="Force the plain unweighted vote even under a "
                             "preset that weights.")
    parser.add_argument("--distance-power", type=float, default=None,
                        help="Exponent on the distance weight. Needs distance "
                             "weighting on; refused without it, since it would "
                             "otherwise be silently ignored.")
    parser.add_argument("--class-weighted", dest="class_weighted",
                        action="store_true", default=None,
                        help="Also scale each neighbour's vote by 1/(its "
                             "cluster's reference count). Never measured to help.")
    parser.add_argument("--local-scaling", type=int, default=None, metavar="R",
                        help="Judge distances relative to each neighbour's own "
                             "local density (its R-th nearest reference point).")
    parser.add_argument("--adaptive-margin", type=float, default=None,
                        metavar="MARGIN",
                        help="Re-vote tiles whose base-k margin falls below this "
                             "at the wider --adaptive-k. Needs an explicit k.")
    parser.add_argument("--adaptive-k", type=int, default=None, metavar="K",
                        help="The wider neighbourhood low-margin tiles are "
                             "re-voted at. Must exceed k.")
    parser.add_argument("--k", type=int, default=None,
                        help="Neighbours to poll. Defaults to the reference's own "
                             "Leiden n_neighbors, which is what ingest used.")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--validate-against", type=Path, default=None,
                        help="A CSV of known labels to check the assignment reproduces. "
                             "Use Kai's TCGA transfer as an acceptance test.")
    parser.add_argument("--partition", type=str, default=MERGE_PARTITION)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--memory", type=str, default="64G")
    parser.add_argument("--time-limit", type=str, default=ASSIGN_TIME_LIMIT,
                        help=f"Slurm walltime for the assignment job. Default "
                             f"{ASSIGN_TIME_LIMIT}. The assigner has no resume, "
                             f"so hitting the wall means redoing the whole run — "
                             f"prefer headroom. Not checked against the partition's "
                             f"MaxTime here: sbatch rejects it, or leaves the job "
                             f"pending with reason PartitionTimeLimit. Check with "
                             # %% because argparse %-interpolates help text: a literal
                             # "%P" made --help itself raise ValueError.
                             f"`sinfo -o \"%%P %%l\"` before raising it.")
    parser.add_argument("--mean-time-limit", type=str, default=MEAN_TIME_LIMIT,
                        help=f"Walltime for the shared query-mean job, submitted "
                             f"only when sharding with --centering query. Default "
                             f"{MEAN_TIME_LIMIT}.")
    parser.add_argument("--merge-time-limit", type=str, default=MERGE_TIME_LIMIT,
                        help=f"Walltime for the shard-merge job. Default "
                             f"{MERGE_TIME_LIMIT}.")
    parser.add_argument("--notify-email", type=str, default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "gpu"],
                        help="Where the exact flat search runs. GPU is the same "
                             "exhaustive scan on faster hardware, not an "
                             "approximation, and the job verifies its GPU index "
                             "against a CPU one at startup rather than trusting "
                             "it. auto (default) uses a GPU when the GPU extras "
                             "have been bootstrapped and CPU otherwise, printing "
                             "which and why; gpu requests one and refuses if the "
                             "partition has none; cpu never asks. A GPU means a "
                             "GPU partition, which is preemptible here, so pair "
                             "it with --shards — a shard is the checkpoint that "
                             "makes a preemption cost one task rather than the "
                             "run.")
    parser.add_argument("--shards", type=int, default=1,
                        help="Split the assignment across N array tasks. A mean job "
                             "runs first so every shard centres identically, then a "
                             "merge job concatenates the parts.")
    parser.add_argument("--centering", type=str, default="query",
                        choices=["query", "reference", "none"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output CSV.")
    parser.add_argument("--bootstrap-gpu-faiss", action="store_true",
                        help="Install a GPU faiss into the GPU extras directory "
                             "and exit, so --device gpu has something to use. "
                             "One-time, on a login node (needs PyPI access). "
                             "This installs a package for THIS stage: it runs "
                             "no feature extraction and no GPU job. The same "
                             "flag exists on submit_feature_extraction.py only "
                             "because that file owns the container's package "
                             "list.")
    parser.add_argument("--query-mean", type=Path, default=None,
                        help="Reuse an existing query-mean .npy instead of "
                             "submitting the mean job. Compute one with "
                             "`assign_hpc_clusters.py --precompute-mean <path>`, "
                             "which needs no container and no faiss. Its width "
                             "is checked against the reference before anything "
                             "is queued.")
    parser.add_argument("--force-duplicate", action="store_true",
                        help="Submit even though a pipeline with this job name "
                             "is already queued or running. Only with a "
                             "different --out: two runs sharing an output path "
                             "overwrite each other's shard parts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.bootstrap_gpu_faiss:
        # Candidates one at a time: given three names pip installs whichever it
        # resolves first and reports success, hiding which one landed — and that
        # decides whether the job has a usable GPU faiss at all.
        errors = []
        for package in _CONTAINER_EXTRA_PACKAGES_GPU:
            print(f"\nTrying {package} ...", flush=True)
            try:
                bootstrap_container_extras(
                    CONTAINER_EXTRAS_GPU,
                    singularity_image=args.singularity_image
                    if hasattr(args, "singularity_image") else SINGULARITY_IMAGE,
                    singularity_bin=args.singularity_bin
                    if hasattr(args, "singularity_bin") else SINGULARITY_BIN,
                    packages=(package,),
                    verify="faiss",
                )
            except (FileNotFoundError, RuntimeError) as e:
                errors.append(f"{package}: {e}")
                print(f"  {package} did not install: {e}", file=sys.stderr)
                continue
            print(f"\nGPU faiss installed from {package} into "
                  f"{CONTAINER_EXTRAS_GPU}.")
            print("--device now resolves to gpu on its own. The job verifies the "
                  "GPU index against a CPU one at startup and refuses if they "
                  "disagree, so a wheel that imports but does not work costs a "
                  "refusal rather than wrong cluster IDs.")
            raise SystemExit(0)
        print("\nNo GPU faiss wheel installed for this container's Python:",
              file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        print("This stage still runs on CPU; --shards is the CPU-side lever and "
              "needs nothing installed.", file=sys.stderr)
        raise SystemExit(1)

    try:
        info = submit_cluster_assignment_job(
            projections_h5=args.projections_h5,
            out_csv=args.out_csv,
            reference=args.reference,
            depends_on_job_id=args.depends_on_job_id,
            rep_key=args.rep_key,
            k=args.k,
            batch_size=args.batch_size,
            validate_against=args.validate_against,
            shards=args.shards,
            device=args.device,
            force_duplicate=args.force_duplicate,
            query_mean=args.query_mean,
            centering=args.centering,
            vote_preset=args.vote_preset,
            distance_weighted=args.distance_weighted,
            distance_power=args.distance_power,
            class_weighted=args.class_weighted,
            local_scaling=args.local_scaling,
            adaptive_margin=args.adaptive_margin,
            adaptive_k=args.adaptive_k,
            partition=args.partition,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            mean_time_limit=args.mean_time_limit,
            merge_time_limit=args.merge_time_limit,
            notify_email=args.notify_email,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        print(f"Not submitted: {e}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as e:
        # sbatch refused. The reason is already extracted; a traceback here just
        # buries it under the whole --wrap string.
        print(f"Not submitted: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Reference:        {info['reference_path']}")
    print(f"  rows x dims     {info['reference_rows']:,} x {info['reference_dims']}")
    print(f"  clusters        {info['n_clusters']}")
    if info["embeddings"] is not None:
        print(f"Embeddings:       {info['embeddings']:,}")
    print(f"Output CSV:       {info['out_csv']}")
    if info["shards"] > 1:
        b = info["shard_bounds"]
        print(f"Shards:           {info['shards']}  "
              f"(rows {b[0][0]}-{b[0][1]} ... {b[-1][0]}-{b[-1][1]})")
        print(f"Mean job ID:      {info['mean_job_id']}")
        print(f"Merge job ID:     {info['merge_job_id']}")
    print(f"Slurm job ID:     {info['assignment_job_id']}")


if __name__ == "__main__":
    main()
