#!/usr/bin/env python3
"""Submit a Slurm GPU job that runs a packaged .h5 through Kai's HPL-LATTICeA
self-supervised encoder (run_representationspathology_projection.py) to get
per-tile feature embeddings.

Settings below must match your actual HPC setup — none can be guessed
correctly from this codebase alone. Each has an environment-variable override
so it can be corrected on the cluster without editing this file:

    HPL_REPO_DIR            — where you've cloned K-Rakovic/HPL-LATTICeA
    HPL_GPU_PARTITION       — GPU partition name (here: gpu)
    HPL_GPU_GRES            — legacy fixed request, kept for callers that pass
                              --gres explicitly; ignored otherwise
    HPL_GPU_PREFERENCE      — comma-separated GPU types, fastest first. The
                              submitter picks the best one that looks free, so a
                              busy H200 queue falls back rather than stalling.
                              Bare gpu:1 is still avoided: an untyped request
                              lets Slurm hand out a card the container cannot
                              drive, which is how a run once spent 8.6 hours on
                              CPU
    HPL_SINGULARITY_IMAGE   — NGC TensorFlow 1.15 SIF with a Hopper-capable
                              CUDA stack (not the host's hpl_tf15 conda env,
                              which cannot register an H200)
    HPL_SINGULARITY_BIN     — singularity/apptainer binary
    HPL_CONTAINER_EXTRAS    — writable directory holding the handful of Python
                              packages the HPL repo imports and the NGC image
                              does not ship (see bootstrap_container_extras)

Why Singularity, not the conda env: hpl_tf15 is TensorFlow 1.15 built against
CUDA 10 / cuDNN 7. On an H200 the driver sees the card, but TF cannot load
libcudart.so.10.0 and silently falls back to CPU — which is exactly the
multi-hour "GPU job" that never used a GPU. The NGC 23.03-tf1-py3 image
ships TF 1.15.5 + CUDA 12.1 and supports Hopper.

The job also fails fast if TensorFlow does not register a GPU after
allocation: better to burn one minute of queue than the two-day walltime on
CPU. That guard is worth more now than it was at a twelve-hour limit.
It fails fast on a missing container package for the same reason — the NGC
image is not built for this repo and is missing scikit-image, which the repo
imports at module scope (models/data_augmentation.py). That import happens
*after* the GPU probe passes, so without a check it costs a full queue wait
and allocation to learn a package is absent.

One upstream behaviour worth knowing, since it shapes this whole module:
real_encode_contrastive_from_checkpoint() in the HPL repo skips encoding
entirely when its output file already exists, and the skip path then raises
UnboundLocalError (it reads `key_shape`, which is only assigned in the
encoding branch). Any output left by an interrupted attempt therefore makes
every retry fail instantly. See validate_extraction_output().

Usage:
    python submit_feature_extraction.py
        --real-hdf5 /path/to/model_input/Radiogenomics/hdf5_Radiogenomics_he_train.h5
        --checkpoint /hpc-home/home/users/vpandya/long-term-scratch/Vaidehi/weights/BarlowTwins_3.ckt
        --dataset-name Radiogenomics
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

from slide_naming import tiles_missing_suffix
from submit_mask_tile_slurm import _run_sbatch_with_retry

# The entry point the job runs. Used to tell "HPL_REPO_DIR points at the repo"
# apart from "HPL_REPO_DIR points at *a* directory" — see _check_hpl_repo_dir.
HPL_ENTRY_SCRIPT = "run_representationspathology_projection.py"

# Datasets our packaged .h5 carries besides the images (see make_hpl_hdf5.py).
# The encoder copies every non-image dataset through to the projections file
# under the same name, so a complete output has all of these plus the latents.
_CARRIED_DATASETS = ("samples", "slides", "tiles")

# Cluster filesystem roots that almost every input/output path lives under.
# Binding these wholesale keeps Singularity able to see the packaged .h5, the
# checkpoint, and the results directory without enumerating every leaf path.
_CLUSTER_BIND_ROOTS = (
    Path("/mnt/cephfs-lts"),
    Path("/hpc-home"),
)

# --- Placeholders: fill these in for your HPC setup ------------------------
# Overridable by environment variable so they can be corrected on the HPC
# without editing this file (the server imports these at module level, so it
# must be restarted for a change to take effect either way).
#
# HPL_REPO_DIR must be the *directory* the repo is cloned into, not a file
# inside it — the job does `cd $HPL_REPO_DIR && python
# run_representationspathology_projection.py`, and results land under
# $HPL_REPO_DIR/results/, so it also has to be writable with room to spare.
HPL_REPO_DIR = Path(
    os.getenv("HPL_REPO_DIR", "/hpc-home/home/users/vpandya/long-term-scratch/Work/HPL-LATTICeA")
)
GPU_PARTITION = os.getenv("HPL_GPU_PARTITION", "gpu")
# Where the merge job runs. It needs no GPU, so point this at a CPU queue if
# one exists; defaulting to the GPU partition just means it does not request a
# --gres there rather than sitting in a queue that may not accept it.
MERGE_PARTITION = os.getenv("HPL_MERGE_PARTITION", GPU_PARTITION)
# Named GPU type, not bare gpu:1. On this cluster that is what distinguishes
# an H200 from an H100 / A100 — see `sinfo -p gpu -o "%N %G"`.
GPU_GRES = os.getenv("HPL_GPU_GRES", "gpu:nvidia_h200:1")
# GPU types in descending order of preference, used when no --gres is given.
# H200 first, then whatever is next best, so a busy H200 queue does not stall a
# run that an H100 or A100 would serve just as well — see select_gpu_gres for
# why "just as well" is close to literal here.
# This cluster's actual GRES type names, from `sinfo -p gpu -o "%N %G %t %D"`.
# They matter exactly: an earlier version guessed "nvidia_h100"/"nvidia_a100",
# neither of which exists here, so the fallback silently never engaged and every
# run queued for the busy H200s just as before. A preference list that does not
# match the cluster is inert, not approximate — hence _match_gpu_type's
# substring fallback and the warning when a listed type is absent.
#
# Ordered by memory bandwidth, what a decode-bound encoder would notice first:
# H200 (HBM3e) > H100 SXM (HBM3) > H100 PCIe (HBM2e) > A100 PCIe. All four
# exceed the ~1.4k tiles/s the read path sustains, so the ordering is close to
# academic and the real point is availability.
GPU_PREFERENCE = tuple(
    t.strip() for t in os.getenv(
        "HPL_GPU_PREFERENCE",
        "nvidia_h200,nvidia_h100_80gb_hbm3,nvidia_h100_pcie,nvidia_a100_80gb_pcie",
    ).split(",") if t.strip()
)
SINGULARITY_BIN = os.getenv("HPL_SINGULARITY_BIN", "/usr/bin/singularity")
SINGULARITY_IMAGE = Path(
    os.getenv(
        "HPL_SINGULARITY_IMAGE",
        "/hpc-home/home/users/vpandya/long-term-scratch/Work/containers/"
        "tensorflow-23.03-tf1-py3.sif",
    )
)
# Kept only so older CLI invocations that pass --conda-env don't break; the
# job no longer activates a conda env.
CONDA_ENV_NAME = os.getenv("HPL_CONDA_ENV", "hpl_tf15")
# Where the packages the NGC image lacks are installed. Defaults to a sibling
# of the SIF so the two halves of the runtime live together; the job binds it
# and puts it on PYTHONPATH. It cannot live *inside* the image (read-only) or
# under $HOME (not writable from inside the container on this cluster — that
# is what the "matplotlib cache directory is not writable" line in the job log
# is telling you, and it rules out `pip install --user` too).
CONTAINER_EXTRAS = Path(
    os.getenv("HPL_CONTAINER_EXTRAS", str(SINGULARITY_IMAGE.parent / "extras-py38"))
)
# -----------------------------------------------------------------------

# The gap between what HPL-LATTICeA imports and what NGC 23.03-tf1-py3 ships.
# Only scikit-image is actually wanted; the rest are its runtime dependencies
# that the image also lacks. Deliberately installed with --no-deps: a plain
# `pip install scikit-image` resolves numpy and scipy too, and because
# PYTHONPATH takes precedence over the image's site-packages, those copies
# would shadow the numpy TensorFlow 1.15 was built against. numpy, scipy,
# Pillow, matplotlib, scikit-learn and h5py all come from the image.
#
# scikit-image is pinned; the others are not. 0.19.3 is the last release with
# cp38 wheels that are relaxed about the numpy/scipy floor, which matters
# precisely because --no-deps means pip will not check compatibility for us.
# The pure-Python deps are left to float so pip picks whatever still supports
# the image's Python rather than us guessing per-package cutoffs.
_CONTAINER_EXTRA_PACKAGES = (
    "scikit-image==0.19.3",
    "networkx",
    "imageio",
    "tifffile",
    "PyWavelets",
    "packaging",
    # For Stage 4. Measured on the real reference shape (360,667 x 128, k=250),
    # faiss's selection kernels run the k-NN search ~43x faster than the exact
    # NumPy fallback — 59 tiles/s becomes 2,500. It is the whole difference
    # between that stage taking minutes and taking hours, and it is the only
    # optimisation there that does not change an assignment.
    "faiss-cpu",
)

# Stage 4 can run its exact flat search on a GPU instead, which is the same
# exhaustive scan on faster hardware (see Searcher._verify_matches_cpu, which
# refuses a build that disagrees with the CPU index). It lives in a SEPARATE
# extras directory on purpose: faiss-cpu and a GPU faiss both install as the
# module `faiss`, and with PYTHONPATH taking precedence over the image's
# site-packages, having both on one path means whichever sorts first wins —
# silently, and differently depending on the directory listing.
CONTAINER_EXTRAS_GPU = Path(
    os.getenv("HPL_CONTAINER_EXTRAS_GPU",
              str(SINGULARITY_IMAGE.parent / "extras-py38-gpu"))
)
# Tried in order, first one that installs wins. The image is CUDA 12, so
# faiss-gpu-cu12 is the match — but it is a young package and the container's
# Python is 3.8, so a cp38 wheel may not exist for it. faiss-gpu-cu11 works
# against a CUDA 12 driver (minor-version compatibility), and the legacy
# faiss-gpu is the last resort with the widest cp38 coverage.
#
# Whichever lands, Searcher._verify_matches_cpu decides whether it can be
# trusted: it searches a sample of the reference against both the GPU and CPU
# indexes and refuses on disagreement. So a wheel that installs but does not
# work costs a startup refusal, not a cohort of wrong cluster IDs.
_CONTAINER_EXTRA_PACKAGES_GPU = (
    "faiss-gpu-cu12",
    "faiss-gpu-cu11",
    "faiss-gpu",
)

# Import-checked in the job before encoding. Module name, not package name:
# scikit-image installs as `skimage`, and the point is to check the thing the
# repo actually imports.
_REQUIRED_CONTAINER_MODULES = ("skimage.color", "skimage.io")

# Raised from the upstream default of 64 after measuring: read throughput is
# flat from 64 to 512 because each tile is its own gzip chunk, so the batch
# size buys nothing on the read side — but it is free, and it does cut the
# number of session.run round trips. Provably inert for the embeddings: the
# projection graph is built with is_train=False (BarlowTwins.py), so batch
# normalisation uses stored moving averages rather than batch statistics, and
# every tile's output is independent of what else is in its batch.
_DEFAULT_BATCH_SIZE = 256

# Slurm walltime for the encode job, in Slurm's D-HH:MM:SS form. Two days
# rather than twelve hours because the encoder has no resume: it opens its
# output with mode='w' and starts from row zero, so a run killed by the wall
# clock at 99% has produced nothing and costs the entire allocation again.
# Overshooting the limit is close to free by comparison — Slurm bills what is
# used, and the only real cost is a worse position in the backfill queue.
#
# Sharded runs inherit this per array task, which is the right unit: each task
# encodes 1/N of the input, so N shards do not need N times the walltime.
_DEFAULT_TIME_LIMIT = "2-00:00:00"


def discover_gpu_types(partition: str, timeout: int = 20) -> dict[str, dict[str, int]]:
    """GPU types on a partition and how free they look, from sinfo.

    Returns {type: {"total": nodes, "idle": nodes, "mix": nodes}}. Node counts,
    not GPU counts: sinfo's per-node GRES-used reporting varies enough between
    Slurm versions that parsing it reliably is not worth it, and node state is
    enough to rank types by "something is probably free here".

    Empty dict when sinfo cannot be reached or says nothing useful, which the
    caller must treat as "no information" rather than "no GPUs" — the same
    distinction the tiling status code draws for sacct.
    """
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-p", partition, "-o", "%G|%t|%D"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}

    found: dict[str, dict[str, int]] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        gres_field, state, node_count = parts
        try:
            nodes = int(node_count)
        except ValueError:
            continue
        # "gpu:nvidia_h200:4(S:0-1),gpu:nvidia_h100:2" and "(null)" both occur.
        # So does the same type listed twice on one line — this cluster reports
        # both "gpu:X:4(S:0-1)" and "gpu:X:X:4" — so types are collected per line
        # and counted once. Without that every node is tallied twice and the
        # printed "N idle node(s)" overstates what is actually free.
        line_types = set()
        for spec in gres_field.split(","):
            spec = spec.strip()
            if not spec.startswith("gpu:"):
                continue
            fields = spec.split(":")
            if len(fields) < 3:
                continue  # bare "gpu:4" names no type, so it cannot be preferred
            gpu_type = fields[1]
            if gpu_type in line_types:
                continue
            line_types.add(gpu_type)
            entry = found.setdefault(gpu_type, {"total": 0, "idle": 0, "mix": 0})
            entry["total"] += nodes
            # sinfo suffixes carry the part that matters most here. '*' means
            # the node is not responding, so an "idle*" node is idle and
            # unschedulable — counting it as free is how the preference would
            # keep choosing a dead H200 over a live A100. It still counts toward
            # `total`, so the type remains queueable.
            flags = state.strip()
            base = flags.rstrip("*~#$@+").lower()
            if "*" in flags:
                continue
            if base == "idle":
                entry["idle"] += nodes
            elif base in ("mix", "mixed"):
                entry["mix"] += nodes
    return found


def _match_gpu_type(preferred: str, available: dict) -> str | None:
    """Resolve a preference entry against the cluster's actual type names.

    Exact match first. Failing that, a *unique* substring match, so a list
    written as "nvidia_a100" still finds "nvidia_a100_80gb_pcie" on a cluster
    that spells it out. Ambiguous substrings resolve to nothing rather than
    guessing between cards of different speeds — "nvidia_h100" against both
    nvidia_h100_pcie and nvidia_h100_80gb_hbm3 is a tie this cannot break, and
    picking either would be a silent performance decision.
    """
    if preferred in available:
        return preferred
    matches = [name for name in available if preferred in name]
    return matches[0] if len(matches) == 1 else None


def select_gpu_gres(
    partition: str,
    preference: tuple[str, ...] = GPU_PREFERENCE,
    count: int = 1,
    explicit: str | None = None,
) -> tuple[str, str]:
    """Pick a --gres value, preferring the fastest GPU that looks free.

    Returns (gres, reason) — the reason is printed at submit time, because
    "which GPU did this run actually get, and why" is otherwise only
    recoverable from sacct after the fact.

    The preference list is ordered fastest-first. Within it, a type with an idle
    node beats one with only a partially-allocated node, which beats one that
    merely exists. If nothing is free anywhere, the first *existing* preferred
    type is requested and the job queues for it — queueing for the best GPU is
    usually better than not running, and this function has no way to compare
    queue depths.

    Worth knowing before tuning this: after the read-path work, feature
    extraction is decode-bound at roughly 1.4k tiles/s, well under what any
    modern GPU encodes. So falling back from an H200 to an H100 or A100 costs
    little or nothing in wall clock — the GPU stopped being the bottleneck. The
    fallback exists to avoid *waiting*, not to trade away speed.
    """
    if explicit:
        return explicit, "explicitly requested"

    available = discover_gpu_types(partition)
    if not available:
        # No information is not the same as no GPUs. Ask for the first
        # preference and let Slurm accept or reject it.
        return (
            f"gpu:{preference[0]}:{count}",
            f"sinfo unavailable — requesting {preference[0]} without checking",
        )

    resolved = [(p, _match_gpu_type(p, available)) for p in preference]
    absent = [p for p, name in resolved if name is None]
    if absent:
        # Loud, because this is exactly how the feature goes inert: a list naming
        # types the cluster does not have falls straight through to "queue for
        # the first one", which is the behaviour the fallback exists to replace.
        print(
            f"NOTE: {absent} not present on partition {partition!r}, which has "
            f"{sorted(available)}. Set HPL_GPU_PREFERENCE to match.",
            file=sys.stderr,
        )

    for stage, key in (("idle", "idle"), ("partly free", "mix")):
        for _, gpu_type in resolved:
            if gpu_type and available[gpu_type][key] > 0:
                return (
                    f"gpu:{gpu_type}:{count}",
                    f"{gpu_type}: {available[gpu_type][key]} {stage} node(s)",
                )

    for _, gpu_type in resolved:
        if gpu_type:
            return (
                f"gpu:{gpu_type}:{count}",
                f"{gpu_type} exists but nothing is free — queueing for it",
            )

    # The partition has typed GPUs, none of them ours. Take the first it does
    # have rather than requesting a type this cluster has never heard of, which
    # sbatch rejects outright.
    fallback = sorted(available)[0]
    return (
        f"gpu:{fallback}:{count}",
        f"none of {list(preference)} exist here; falling back to {fallback}. "
        f"Set HPL_GPU_PREFERENCE to rank this cluster's types.",
    )


def expected_extraction_output_path(
    hpl_repo_dir: Path,
    model: str,
    dataset_name: str,
    real_hdf5_path: Path,
    *,
    z_dim: int = 128,
    img_size: int = 224,
) -> Path:
    """Where feature extraction's output lands, mirroring the path
    real_encode_contrastive_from_checkpoint() itself computes. Pulled out as
    its own function (not just inlined in submit_feature_extraction_job) so
    callers can compute it without submitting a job — e.g. to check whether
    a Slurm job already produced this output before deciding to resubmit.
    """
    res = f"h{img_size}_w{img_size}_n3_zdim{z_dim}"
    return hpl_repo_dir / "results" / model / dataset_name / res / real_hdf5_path.name


def validate_extraction_output(
    path: Path, expected_rows: int | None = None
) -> tuple[bool, str]:
    """Confirm a projections .h5 is a complete set of embeddings rather than a
    file that merely exists at the right path.

    Feature extraction has no resume: the encoder creates its output with
    h5py.File(mode='w') *before* encoding anything, so any attempt that dies —
    OOM, timeout, scancel, a bad checkpoint — leaves a file behind that looks
    finished to an existence check. Worse, that leftover is not inert: the
    encoder skips its whole encoding path when the output already exists (see
    the module docstring in this file and real_encode_contrastive_from_checkpoint
    in the HPL repo), so a stale file makes every subsequent attempt fail
    instantly instead of redoing the work.

    Mirrors _validate_h5 in tile_server_v2_.py — same reasoning, different
    schema. Returns (ok, reason) so callers can say *why* they rejected it.
    """
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            z_keys = [k for k in keys if k.endswith("_z_latent")]
            h_keys = [k for k in keys if k.endswith("_h_latent")]
            if not z_keys or not h_keys:
                return False, (
                    f"no latent datasets (found {keys or 'nothing'}) — the encoder "
                    f"created the file but never wrote embeddings"
                )

            rows = f[z_keys[0]].shape[0]
            if rows == 0:
                return False, "contains zero embeddings"

            # The encoder writes the latents first and copies the metadata
            # datasets afterwards, so metadata missing entirely is the
            # signature of a run that died partway rather than a schema
            # difference.
            absent = [name for name in _CARRIED_DATASETS if name not in f]
            if absent:
                return False, (
                    f"missing carried-through dataset(s) {absent} — encoding finished "
                    f"but the run died before copying the tile metadata"
                )

            mismatched = {
                name: f[name].shape[0]
                for name in (*z_keys, *h_keys, *_CARRIED_DATASETS)
                if f[name].shape[0] != rows
            }
            if mismatched:
                return False, f"dataset lengths disagree with {rows}: {mismatched}"

            # Note on tile names: the encoder copies `tiles` through unchanged,
            # so a projections file made before the tile-name fix carries names
            # without the ".jpeg" suffix. That is deliberately NOT a failure
            # here. These embeddings are correct — they were computed from the
            # right tile images — and cluster assignment reads them without
            # caring what the name column says. The suffix only matters once the
            # name becomes a Knowledge Bank join key, and migrate_tile_names.py
            # fixes it there without recomputing anything.

            # A row count that doesn't match the input is the one failure this
            # can catch that reading rows cannot. Every other check here is
            # internal consistency, which a partial run passes: the encoder
            # writes whatever it got through, and a file holding a third of the
            # slides has the right schema, the right dtypes and no gaps.
            #
            # Short and over-long are different faults and get different
            # messages, because the fix differs — a short output is a run to
            # resume or resubmit, an over-long one is the wrong file entirely.
            if expected_rows is not None and rows != expected_rows:
                if rows < expected_rows:
                    missing = expected_rows - rows
                    return False, (
                        f"has {rows:,} embeddings but the input .h5 has "
                        f"{expected_rows:,} tiles — {missing:,} missing "
                        f"({missing / expected_rows * 100:.1f}%). The encoder did not "
                        f"get through the whole input: an array task that died, a "
                        f"timeout, or a sharded run merged from only the shards that "
                        f"finished"
                    )
                return False, (
                    f"has {rows:,} embeddings but the input .h5 has only "
                    f"{expected_rows:,} tiles — this output cannot have come from "
                    f"that input, so it is a leftover from a different .h5"
                )

            # Same reasoning as _validate_h5: HDF5 validates the superblock on
            # open, so a file truncated partway through the data opens cleanly.
            # Reading the last row is what forces the missing chunk to resolve.
            f[z_keys[0]][0]
            f[z_keys[0]][rows - 1]
            f[_CARRIED_DATASETS[0]][rows - 1]
    except (OSError, KeyError, ValueError) as e:
        return False, f"unreadable HDF5: {e}"
    return True, ""


def _input_h5_rows(real_hdf5_path: Path) -> int | None:
    """Tile count of a packaged input .h5, or None if it can't be read. Only
    used to strengthen validate_extraction_output — never to gate anything on
    its own, since the input is validated properly by the caller."""
    try:
        with h5py.File(real_hdf5_path, "r") as f:
            return int(f["img"].shape[0])
    except (OSError, KeyError, ValueError):
        return None


def _check_hpl_repo_dir(hpl_repo_dir: Path) -> None:
    """Fail fast, with an actionable message, on a misconfigured HPL_REPO_DIR.

    Checking for the entry script rather than just is_dir() catches the case
    that actually happened: a path that was *nearly* right (the repo path with
    a filename stuck on the end, or a parent directory) and would otherwise
    have burned a GPU queue wait before failing inside the job.
    """
    if not hpl_repo_dir.is_dir():
        hint = ""
        if hpl_repo_dir.suffix == ".py":
            hint = (
                f" It looks like a filename got appended to the clone path — try "
                f"{hpl_repo_dir.parent} instead."
            )
        raise NotADirectoryError(
            f"HPL_REPO_DIR does not exist or isn't set: {hpl_repo_dir}. "
            "Set the HPL_REPO_DIR environment variable (or update the default at "
            "the top of this file) to the directory where you've cloned "
            f"K-Rakovic/HPL-LATTICeA on the HPC.{hint}"
        )
    if not (hpl_repo_dir / HPL_ENTRY_SCRIPT).is_file():
        raise NotADirectoryError(
            f"HPL_REPO_DIR is a directory but doesn't look like the HPL-LATTICeA "
            f"clone: {hpl_repo_dir} has no {HPL_ENTRY_SCRIPT}. Point HPL_REPO_DIR "
            "at the repository root itself."
        )


def _check_singularity_image(image: Path, binary: str) -> None:
    """Refuse to submit if the GPU runtime image is missing.

    The host conda env cannot use an H200 (missing CUDA 10 libs, and CUDA 10
    could not drive Hopper even if they were present). Submitting without the
    SIF would recreate the silent-CPU failure this module exists to prevent.
    """
    if not Path(binary).is_file() and shutil_which(binary) is None:
        raise FileNotFoundError(
            f"Singularity binary not found: {binary}. Set HPL_SINGULARITY_BIN "
            "or install singularity/apptainer on the submit host."
        )
    if not image.is_file():
        raise FileNotFoundError(
            f"HPL_SINGULARITY_IMAGE not found: {image}. Pull it once on the "
            "login node, e.g.\n"
            f"  mkdir -p {image.parent} && cd {image.parent}\n"
            "  singularity pull tensorflow-23.03-tf1-py3.sif "
            "docker://nvcr.io/nvidia/tensorflow:23.03-tf1-py3\n"
            "Then set HPL_SINGULARITY_IMAGE to that .sif path."
        )


def _extras_bootstrap_hint(
    extras_dir: Path, singularity_image: Path, singularity_bin: str
) -> str:
    """The exact command that populates extras_dir, for error messages."""
    return (
        f"  python {Path(__file__).name} --bootstrap-extras\n"
        f"or equivalently, by hand on the login node:\n"
        f"  mkdir -p {extras_dir}\n"
        f"  {singularity_bin} exec {singularity_image} pip install --no-cache-dir "
        f"--no-deps --target {extras_dir} {' '.join(_CONTAINER_EXTRA_PACKAGES)}"
    )


def _check_container_extras(
    extras_dir: Path, singularity_image: Path, singularity_bin: str
) -> None:
    """Refuse to submit until the container's missing packages are installed.

    Checked from the submit host by looking for the installed `skimage`
    directory rather than by importing it: this process is the wrong Python
    (host, not container) and the wrong architecture question. Presence of the
    directory is what PYTHONPATH will act on, so it is the right thing to test.
    """
    if (extras_dir / "skimage").is_dir():
        return
    what = (
        f"is empty of the expected packages"
        if extras_dir.is_dir()
        else "does not exist"
    )
    raise FileNotFoundError(
        f"Container extras directory {what}: {extras_dir}. The NGC image does "
        f"not ship scikit-image, which HPL-LATTICeA imports at module scope "
        f"(models/data_augmentation.py), so the job would reach the GPU and "
        f"then die on ModuleNotFoundError. Populate it once with:\n"
        f"{_extras_bootstrap_hint(extras_dir, singularity_image, singularity_bin)}"
    )


def bootstrap_container_extras(
    extras_dir: Path = CONTAINER_EXTRAS,
    *,
    singularity_image: Path = SINGULARITY_IMAGE,
    singularity_bin: str = SINGULARITY_BIN,
    packages: tuple[str, ...] | None = None,
    verify: str | None = None,
) -> None:
    """One-time install of the packages the NGC image lacks into extras_dir.

    Run on the login node, which is where outbound network access lives — a
    compute node generally cannot reach PyPI. pip runs *inside* the container
    so it resolves wheels for the container's Python and ABI, not the submit
    host's; running it outside would install cp310 wheels the job cannot load.
    """
    _check_singularity_image(singularity_image, singularity_bin)
    extras_dir.mkdir(parents=True, exist_ok=True)

    target = os.path.realpath(extras_dir)
    command = [
        singularity_bin,
        "exec",
        *_bind_args(extras_dir, singularity_image),
        str(singularity_image),
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "--target",
        target,
        *(packages if packages is not None else _CONTAINER_EXTRA_PACKAGES),
    ]
    print(f"Installing into {extras_dir}:\n  {shlex.join(command)}", flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed (exit {result.returncode}). If it could not reach "
            f"PyPI, run this on a login node rather than a compute node, or "
            f"download the wheels and pip install them from a local directory."
        )
    expected = verify if verify is not None else "skimage"
    if not (extras_dir / expected).is_dir():
        raise RuntimeError(
            f"pip reported success but {extras_dir / expected} is not there. "
            f"Check whether --target landed somewhere else."
        )
    print(f"Container extras ready: {extras_dir}")


def _check_checkpoint(checkpoint: str) -> None:
    """Refuse a checkpoint the GPU job would only fail on later.

    This is the one input a human types by hand into a free-text box, and it
    is worth an extra 40 seconds of nobody's time to get wrong: without this,
    a typo costs a queue wait, a GPU allocation, and a container start before
    anything notices.

    Relative paths are rejected outright rather than resolved. There is no cwd
    here worth resolving against — the server's happens to be backend/, so a
    path that merely lost its leading slash silently became
    .../Work/backend/mnt/cephfs-lts/... , a path that never existed and reads
    at a glance like one that does.

    A TensorFlow checkpoint is a *prefix*, not a file: BarlowTwins_3.ckt names
    the set .ckt.index / .ckt.data-00000-of-00001 / .ckt.meta and frequently
    does not exist by itself. Requiring is_file() here would reject working
    checkpoints, so a sibling matching the prefix counts as present.
    """
    if not checkpoint.strip():
        raise FileNotFoundError("Checkpoint path is required.")

    path = Path(checkpoint)
    if not path.is_absolute():
        raise FileNotFoundError(
            f"Checkpoint path must be absolute, got: {checkpoint!r}. It looks "
            f"like a leading '/' is missing — try /{checkpoint.lstrip('/')}"
        )

    if path.is_file() or (path.parent.is_dir() and any(path.parent.glob(path.name + ".*"))):
        return

    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {path.parent} (from "
            f"{checkpoint})."
        )

    # The directory is real, so list what is actually in it. A checkpoint that
    # is one directory or one typo away is the common case, and naming the
    # candidates beats making someone ls it themselves.
    nearby = sorted({p.name.split(".ckt")[0] + ".ckt"
                     for p in path.parent.iterdir() if ".ckt" in p.name})
    hint = f" Found in that directory: {', '.join(nearby)}" if nearby else \
           " No .ckt checkpoints found in that directory at all."
    raise FileNotFoundError(
        f"No checkpoint at {checkpoint} (and nothing matching "
        f"{path.name}.* alongside it).{hint}"
    )


def shutil_which(cmd: str) -> str | None:
    """Local which() so we don't import shutil only for one call."""
    from shutil import which
    return which(cmd)


def _bind_args(*paths: Path | str) -> list[str]:
    """--bind flags covering every path the job needs to see.

    Each path is bound twice: as written, and as its realpath. That is not
    belt-and-braces, it is the whole point. On this cluster
    /hpc-home/home/users/<user>/long-term-scratch is a symlink into
    /mnt/cephfs-lts, and a symlink inside a container is just a string — it
    resolves against the *container's* filesystem, where the target is absent
    unless something bound it. Binding only the /hpc-home side produced a
    directory that listed fine from the login node and did not exist inside
    the job, which is exactly how a run reached the GPU, registered an H200,
    and then died on `cd`.

    Binding a directory whose parent is already bound is redundant but
    harmless, and cheaper than reasoning about which of the two forms of every
    path is the one that will resolve.
    """
    binds: list[str] = []
    seen: set[str] = set()

    def add(target: Path) -> None:
        # Absolute first. A relative path here becomes "--bind .:." and
        # Singularity refuses it: it resolves the source against the job's cwd
        # but leaves the destination relative, so the error names an absolute
        # source and complains the destination is not absolute -- pointing
        # nowhere near the relative argument that caused it. Reached by passing
        # any path as a bare filename, e.g. --validate-against on a CSV in the
        # working directory.
        #
        # abspath, not resolve(): resolve() would follow the symlink and collapse
        # the two candidates into one, losing the /hpc-home side that the whole
        # bind-it-twice rule above exists for.
        target = Path(os.path.abspath(target))
        for candidate in (target, Path(os.path.realpath(target))):
            key = str(candidate)
            # "/" would bind the host root over the image.
            if key in seen or key == "/" or not candidate.exists():
                continue
            seen.add(key)
            binds.extend(["--bind", f"{key}:{key}"])

    for root in _CLUSTER_BIND_ROOTS:
        add(root)

    for raw in paths:
        path = Path(raw)
        add(path if path.is_dir() else path.parent)
    return binds


def shard_ranges(total_rows: int, shards: int) -> list[tuple[int, int]]:
    """Split [0, total_rows) into `shards` contiguous, non-overlapping ranges.

    The remainder is spread one row at a time over the leading shards rather
    than dumped on the last one: with 4 shards over 1,000,003 tiles the naive
    split leaves the final task three rows of extra work, which is harmless,
    but the same arithmetic at 4 shards over 7 tiles leaves an empty range,
    which the encoder exits on. Even sizes avoid the special case entirely.
    """
    if shards < 1:
        raise ValueError(f"shards must be at least 1, got {shards}")
    if shards > total_rows:
        raise ValueError(
            f"Asked for {shards} shards of {total_rows} tiles. There is no point "
            f"splitting past one tile per shard, and empty shards fail the job."
        )
    base, extra = divmod(total_rows, shards)
    ranges, cursor = [], 0
    for i in range(shards):
        size = base + (1 if i < extra else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def shard_output_path(final_path: Path, lo: int, hi: int) -> Path:
    """Where the encoder writes the part covering [lo, hi).

    Mirrors the name the patched real_encode_contrastive_from_checkpoint
    builds, so stale parts can be found and cleared without running anything.
    """
    return final_path.with_name(f"{final_path.name[:-3]}.rows{lo}-{hi}.h5")


def _gpu_probe_python() -> str:
    """Inline TF check the job runs before encoding anything.

    Printed to the Slurm .err/.out so a failed registration is obvious in the
    first few log lines rather than after hours of CPU work.
    """
    return (
        "import sys, tensorflow as tf\n"
        "print('TensorFlow:', tf.__version__, flush=True)\n"
        "ok = tf.test.is_gpu_available(cuda_only=True)\n"
        "print('GPU available:', ok, flush=True)\n"
        "print('GPU device:', tf.test.gpu_device_name() or '(none)', flush=True)\n"
        "if not ok:\n"
        "    print(\n"
        "        'FATAL: TensorFlow did not register a GPU. Refusing to run on CPU. '\n"
        "        'Check the --gres in this job\\'s sbatch line names a GPU type the '\n"
        "        'container can drive, that singularity got --nv, and that this SIF '\n"
        "        'is the NGC TF1 image (CUDA 12).',\n"
        "        file=sys.stderr,\n"
        "    )\n"
        "    sys.exit(1)\n"
    )


def _build_extraction_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    hpl_repo_dir: Path,
    real_hdf5_path: Path,
    checkpoint: str,
    dataset_name: str,
    model: str,
    marker: str,
    z_dim: int,
    img_size: int,
    batch_size: int,
    extras_dir: Path,
    shard_bounds: list[tuple[int, int]] | None = None,
    row_range: tuple[int, int] | None = None,
) -> str:
    """Shell command the Slurm --wrap runs: probe GPU, check imports, encode.

    --cleanenv keeps the host's broken CUDA stubs out of LD_LIBRARY_PATH;
    --nv is what actually injects the driver's libcuda into the container.

    shard_bounds is for a Slurm array: each task picks its range by
    SLURM_ARRAY_TASK_ID. row_range is one fixed range, for a caller that runs
    each shard as its own job with no array index — the Nextflow pipeline,
    where every shard is a separate task. Exactly one or neither.
    """
    if shard_bounds is not None and row_range is not None:
        raise ValueError("Pass shard_bounds (a Slurm array) or row_range (one "
                         "fixed range), not both.")
    binds = _bind_args(
        hpl_repo_dir, real_hdf5_path, checkpoint, singularity_image, extras_dir
    )
    probe = _gpu_probe_python()

    # Realpaths inside the container, for the reason given in _bind_args: the
    # symlinked /hpc-home spelling of these is not resolvable in there, and a
    # bare `cd` failure names only the path it was handed, not the fact that
    # it was a link.
    repo_in_job = os.path.realpath(hpl_repo_dir)
    h5_in_job = os.path.realpath(real_hdf5_path)
    checkpoint_in_job = os.path.realpath(checkpoint)
    extras_in_job = os.path.realpath(extras_dir)

    # --cleanenv wipes the environment, so PYTHONPATH is set inside the
    # container rather than exported around it. Appending to any existing value
    # rather than replacing it keeps this correct if the image ever sets one.
    #
    # MPLCONFIGDIR is set for the same reason the extras directory exists:
    # $HOME is not writable in here, and matplotlib (imported by the repo)
    # otherwise spends the first seconds of every job rebuilding its font cache
    # in a fresh temp directory and warning about it.
    environment = (
        f"export PYTHONPATH={shlex.quote(extras_in_job)}${{PYTHONPATH:+:$PYTHONPATH}}; "
        'export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig-$$"; '
        'mkdir -p "$MPLCONFIGDIR"; '
    )

    # The repo imports scikit-image at module scope, several seconds and one
    # GPU allocation into the job. Importing it up front turns "the extras bind
    # didn't carry" / "the install is for the wrong Python" into a message that
    # says so, instead of a bare ModuleNotFoundError from inside a third-party
    # import chain.
    import_check = (
        "python -c "
        + shlex.quote(
            "import sys\n"
            "missing = []\n"
            f"for name in {list(_REQUIRED_CONTAINER_MODULES)!r}:\n"
            "    try:\n"
            "        __import__(name)\n"
            "    except ImportError as e:\n"
            "        missing.append(f'{name} ({e})')\n"
            "if missing:\n"
            "    print(\n"
            "        'FATAL: packages missing inside the container: '\n"
            "        + ', '.join(missing)\n"
            f"        + '. Expected them on PYTHONPATH from {extras_in_job} — '\n"
            "        'check that directory exists, is bound into the job, and was "
            "populated by --bootstrap-extras using this same image.',\n"
            "        file=sys.stderr,\n"
            "    )\n"
            "    sys.exit(1)\n"
            "print('container packages: ok', flush=True)\n"
        )
    )

    # The row range, resolved OUTSIDE the container and handed in through the
    # environment. It used to be resolved in here, from SLURM_ARRAY_TASK_ID —
    # which `singularity exec --cleanenv` has already wiped by then, so under
    # `set -u` every array task aborted the moment the import check finished.
    # That is the bug CLAUDE.md records for Stage 4, found there first; this is
    # the same fix. SINGULARITYENV_/APPTAINERENV_ are the documented route
    # through --cleanenv, and both prefixes are set because the binary may be
    # either.
    #
    # For an array the bounds are baked in as shell arrays rather than
    # recomputed in the job: the split has to be identical to the one the
    # merge step will check against, and recomputing it in two places is how
    # those drift apart. The index is still read under `set -u`, out here,
    # where an unset one really does mean a sharded command was submitted as
    # a plain job — and one task silently encoding the wrong range is worse
    # than a refusal.
    outer_preamble = ""
    env_prefix = ""
    row_args = ""
    if shard_bounds is not None or row_range is not None:
        if shard_bounds is not None:
            starts = " ".join(str(lo) for lo, _ in shard_bounds)
            stops = " ".join(str(hi) for _, hi in shard_bounds)
            outer_preamble = (
                "set -euo pipefail; "
                f"SHARD_STARTS=({starts}); "
                f"SHARD_STOPS=({stops}); "
                'ROW_START="${SHARD_STARTS[$SLURM_ARRAY_TASK_ID]}"; '
                'ROW_STOP="${SHARD_STOPS[$SLURM_ARRAY_TASK_ID]}"; '
                'echo "=== Shard $SLURM_ARRAY_TASK_ID: rows $ROW_START-$ROW_STOP ==="; '
            )
        else:
            lo, hi = (int(v) for v in row_range)
            outer_preamble = (
                "set -euo pipefail; "
                f"ROW_START={lo}; ROW_STOP={hi}; "
                'echo "=== Rows $ROW_START-$ROW_STOP ==="; '
            )
        env_prefix = (
            'SINGULARITYENV_ROW_START="$ROW_START" '
            'SINGULARITYENV_ROW_STOP="$ROW_STOP" '
            'APPTAINERENV_ROW_START="$ROW_START" '
            'APPTAINERENV_ROW_STOP="$ROW_STOP" '
        )
        row_args = ' --row_start "$ROW_START" --row_stop "$ROW_STOP"'

    encode = (
        f"cd {shlex.quote(repo_in_job)} && "
        f"python {HPL_ENTRY_SCRIPT} "
        f"--dataset {shlex.quote(dataset_name)} "
        f"--marker {shlex.quote(marker)} "
        f"--checkpoint {shlex.quote(checkpoint_in_job)} "
        f"--model {shlex.quote(model)} "
        f"--z_dim {z_dim} "
        f"--img_size {img_size} "
        f"--batch_size {batch_size} "
        f"--real_hdf5 {shlex.quote(h5_in_job)}"
        + row_args
    )
    # Checked rather than left to `cd`, so a bind that didn't carry says so in
    # those words instead of as "No such file or directory" against a path the
    # reader can see perfectly well from the login node.
    #
    # The checkpoint is checked by its directory, not itself: a TF checkpoint
    # is a prefix (.ckt.index, .ckt.data-*, .ckt.meta) that often has no file
    # at the bare path. Whether the checkpoint is the right one is settled at
    # submit time by _check_checkpoint against the same filesystem; the only
    # question left in here is whether the bind carried, and the directory
    # answers that without inventing a failure for a valid prefix.
    preflight = "; ".join(
        f'if [ ! -e {shlex.quote(p)} ]; then '
        f'echo "FATAL: {label} not visible inside the container: {p} — '
        f'check the --bind flags in this job'"'"'s sbatch command." >&2; exit 1; fi'
        for label, p in (
            ("HPL repo", repo_in_job),
            ("input .h5", h5_in_job),
            ("checkpoint directory", os.path.dirname(checkpoint_in_job)),
            ("container extras directory", extras_in_job),
        )
    )

    # One singularity invocation for probe + encode so --nv / binds are
    # identical for both; set -e so a failed probe aborts before encode.
    inner = (
        "set -euo pipefail; "
        f"{environment}"
        "echo '=== GPU allocation ==='; "
        "nvidia-smi -L || true; "
        "echo '=== TensorFlow GPU probe ==='; "
        f"python -c {shlex.quote(probe)}; "
        "echo '=== Paths ==='; "
        f"{preflight}; "
        "echo 'all inputs visible'; "
        "echo '=== Container packages ==='; "
        f"{import_check}; "
        "echo '=== Feature extraction ==='; "
        f"{encode}"
    )
    return outer_preamble + env_prefix + " ".join([
        shlex.quote(singularity_bin),
        "exec",
        "--nv",
        "--cleanenv",
        *binds,
        shlex.quote(str(singularity_image)),
        "bash",
        "-lc",
        shlex.quote(inner),
    ])


def _build_merge_command(
    *,
    singularity_bin: str,
    singularity_image: Path,
    extras_dir: Path,
    merge_script: Path,
    final_output: Path,
    real_hdf5_path: Path,
) -> str:
    """Shell command for the job that reassembles the shards.

    Run inside the same container as the encoder, for one reason: it is the
    Python that is known to have a working h5py here. It needs no GPU, so no
    --nv and no --gres.

    --input-h5 is passed so the merge verifies the parts cover every tile in
    the input, rather than believing whatever total the parts happen to add up
    to. A missing final shard otherwise merges perfectly into a short file.
    """
    binds = _bind_args(
        merge_script.parent, final_output.parent, real_hdf5_path, singularity_image, extras_dir
    )
    inner = (
        "set -euo pipefail; "
        f"export PYTHONPATH={shlex.quote(os.path.realpath(extras_dir))}${{PYTHONPATH:+:$PYTHONPATH}}; "
        "echo '=== Merging shards ==='; "
        f"python {shlex.quote(os.path.realpath(merge_script))} "
        f"--output {shlex.quote(os.path.realpath(final_output))} "
        f"--input-h5 {shlex.quote(os.path.realpath(real_hdf5_path))} "
        "--cleanup"
    )
    return " ".join([
        shlex.quote(singularity_bin), "exec", "--cleanenv", *binds,
        shlex.quote(str(singularity_image)), "bash", "-lc", shlex.quote(inner),
    ])


def submit_feature_extraction_job(
    real_hdf5_path: Path,
    checkpoint: str,
    dataset_name: str,
    *,
    depends_on_job_id: str | None = None,
    conda_env: str = CONDA_ENV_NAME,  # unused; retained for call-site compat
    hpl_repo_dir: Path = HPL_REPO_DIR,
    singularity_image: Path = SINGULARITY_IMAGE,
    singularity_bin: str = SINGULARITY_BIN,
    extras_dir: Path = CONTAINER_EXTRAS,
    model: str = "BarlowTwins_3",
    marker: str = "he",
    z_dim: int = 128,
    img_size: int = 224,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shards: int = 1,
    merge_partition: str = MERGE_PARTITION,
    partition: str = GPU_PARTITION,
    gres: str | None = None,
    cpus: int = 8,
    memory: str = "64G",
    time_limit: str = _DEFAULT_TIME_LIMIT,
    job_name: str = "hpl_feature_extraction",
    notify_email: str | None = None,
    clear_stale_output: bool = True,
) -> dict:
    """Submit the actual model-input step: run Kai's frozen self-supervised
    encoder over a packaged .h5, producing per-tile embeddings.

    real_hdf5_path is read directly (verified against the actual encoding
    function in models/evaluation/features.py — it never goes through Kai's
    nested datasets/<dataset>/<marker>/patches_h*_w*/ directory-discovery
    logic), so our flat model_input/<dataset_name>/hdf5_....h5 layout works
    as-is with no restructuring.

    If depends_on_job_id is given, this is deferred via Slurm's own
    --dependency until that job (e.g. the .h5 packaging job) finishes.

    Raises FileExistsError if this run's output already exists and validates —
    resubmitting over it would waste a GPU allocation to produce nothing, since
    the encoder skips its work when the output is already there.
    """
    del conda_env  # host conda cannot drive an H200; Singularity is mandatory
    _check_hpl_repo_dir(hpl_repo_dir)
    _check_checkpoint(checkpoint)

    gres, gres_reason = select_gpu_gres(partition, explicit=gres)
    print(f"GPU request:      {gres}  ({gres_reason})")
    if ":" not in gres.rsplit(":", 1)[0]:
        # A bare "gpu:1" names no type, which is the misconfig that let a run
        # land on a card whose CUDA the container could not use and silently
        # fall back to CPU for 8.6 hours. Still a warning rather than a reject:
        # some clusters do not type their GPUs at all.
        print(
            f"WARNING: --gres={gres!r} names no GPU type, so Slurm may hand out "
            f"any card on the partition. Check `sinfo -p {partition} -o '%N %G'`.",
            file=sys.stderr,
        )
    elif not any(t in gres for t in ("h200", "h100")):
        # Not an error — the fallback is deliberate, and extraction is
        # decode-bound rather than GPU-bound so the cost is small. But a
        # smaller-memory card may not hold the default batch.
        print(
            f"NOTE: running on {gres} rather than an H200. Throughput should be "
            f"similar (extraction is read-bound), but if the job OOMs on the GPU, "
            f"lower --batch-size.",
            file=sys.stderr,
        )

    script_path = Path(__file__).resolve()
    backend_dir = script_path.parent
    log_dir = backend_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    expected_output_path = expected_extraction_output_path(
        hpl_repo_dir, model, dataset_name, real_hdf5_path, z_dim=z_dim, img_size=img_size,
    )

    info = {
        "expected_output_path": str(expected_output_path),
        "sbatch_command": None,
        "extraction_job_id": None,
        "cleared_stale_output": None,
        "gres": gres,
        "singularity_image": str(singularity_image),
        "extras_dir": str(extras_dir),
        "shards": shards,
        "shard_bounds": None,
        "merge_job_id": None,
        "cleared_stale_parts": [],
    }

    # Deal with any leftover output *before* submitting (and before we insist
    # on the Singularity image existing). The encoder treats an existing
    # output file as "already done" and skips encoding entirely — and in the
    # version of the repo we run, that skip path then crashes on an unbound
    # local (features.py: `num_samples = key_shape.shape[0]`, where key_shape
    # is only ever assigned in the encoding branch). So a leftover from a
    # killed attempt doesn't just waste an allocation, it makes every retry
    # fail in seconds with an error that points nowhere near the cause.
    if expected_output_path.exists():
        ok, reason = validate_extraction_output(
            expected_output_path, expected_rows=_input_h5_rows(real_hdf5_path)
        )
        if ok:
            raise FileExistsError(
                f"Feature extraction has already completed for this input — "
                f"{expected_output_path} holds a complete set of embeddings. Delete it "
                f"first if you mean to regenerate it."
            )
        if not clear_stale_output:
            raise FileExistsError(
                f"A previous feature-extraction attempt left an unusable file at "
                f"{expected_output_path} ({reason}). The encoder will skip encoding and "
                f"fail while this file exists — remove it, or resubmit with stale-output "
                f"clearing enabled."
            )
        # Only ever unlinks a file this function itself computed the path for
        # and just proved is incomplete. There is nothing to preserve: feature
        # extraction has no resume, so the next run regenerates all of it.
        expected_output_path.unlink()
        info["cleared_stale_output"] = f"{expected_output_path} ({reason})"
        print(f"Cleared stale extraction output before resubmitting: {expected_output_path} ({reason})")

    # After the already-done / stale-output decisions: requiring the SIF up
    # front would hide those clearer answers behind "image not found".
    _check_singularity_image(singularity_image, singularity_bin)
    _check_container_extras(extras_dir, singularity_image, singularity_bin)

    # Sharding: N array tasks each encode a disjoint row range of the same
    # input, and a dependent job merges the parts. This is the only way to
    # parallelise the gzip decode that bounds extraction — the reads are
    # single-threaded inside one process because h5py holds a global lock, so
    # more processes is the lever, not more threads.
    shard_bounds = None
    if shards > 1:
        total_rows = _input_h5_rows(real_hdf5_path)
        if total_rows is None:
            raise FileNotFoundError(
                f"Cannot shard {real_hdf5_path}: its tile count is unreadable, and "
                f"splitting a file whose length is unknown would silently drop or "
                f"duplicate rows. Submit without --shards, or repair the input."
            )
        shard_bounds = shard_ranges(total_rows, shards)
        info["shard_bounds"] = shard_bounds

        # Leftover parts have to go for the same reason the whole output does:
        # the encoder skips any output file that already exists and then dies
        # on an unbound local. One stale part fails one array task, and the
        # merge then refuses the whole run for a gap.
        for lo, hi in shard_bounds:
            part = shard_output_path(expected_output_path, lo, hi)
            if part.exists():
                part.unlink()
                info["cleared_stale_parts"].append(part.name)
        if info["cleared_stale_parts"]:
            print(f"Cleared {len(info['cleared_stale_parts'])} stale shard part(s) "
                  f"before resubmitting.")

    extraction_command = _build_extraction_command(
        singularity_bin=singularity_bin,
        singularity_image=singularity_image,
        hpl_repo_dir=hpl_repo_dir,
        real_hdf5_path=real_hdf5_path,
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        model=model,
        marker=marker,
        z_dim=z_dim,
        img_size=img_size,
        batch_size=batch_size,
        extras_dir=extras_dir,
        shard_bounds=shard_bounds,
    )

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--gres={gres}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        *([f"--array=0-{shards - 1}"] if shard_bounds else []),
        *([f"--dependency=afterany:{depends_on_job_id}"] if depends_on_job_id else []),
        f"--output={log_dir}/hpl_features_%j.out",
        f"--error={log_dir}/hpl_features_%j.err",
        f"--chdir={backend_dir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        "--wrap", f"bash -lc {shlex.quote(extraction_command)}",
    ]
    info["sbatch_command"] = shlex.join(sbatch_command)

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as e:
        reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output from sbatch"
        raise RuntimeError(f"sbatch failed (exit {e.returncode}): {reason}") from e

    stdout = result.stdout.strip()
    info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info["extraction_job_id"] = match.group(1)

    if shard_bounds:
        if not info["extraction_job_id"]:
            raise RuntimeError(
                "The shard array was submitted but sbatch did not report a job ID, so "
                "the merge cannot be made to depend on it. Submit the merge by hand "
                "once the array finishes:\n"
                f"  python {Path(__file__).with_name('merge_projection_shards.py')} "
                f"--output {expected_output_path} --input-h5 {real_hdf5_path} --cleanup"
            )
        merge_command = _build_merge_command(
            singularity_bin=singularity_bin,
            singularity_image=singularity_image,
            extras_dir=extras_dir,
            merge_script=Path(__file__).resolve().with_name("merge_projection_shards.py"),
            final_output=expected_output_path,
            real_hdf5_path=real_hdf5_path,
        )
        merge_sbatch = [
            "sbatch",
            f"--job-name={job_name}_merge",
            f"--partition={merge_partition}",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=04:00:00",
            # afterok, not afterany: merging after a failed shard is exactly the
            # silent corruption this whole path is built to avoid. If one task
            # dies the merge never runs, and the parts stay put for a re-run.
            f"--dependency=afterok:{info['extraction_job_id']}",
            f"--output={log_dir}/hpl_merge_%j.out",
            f"--error={log_dir}/hpl_merge_%j.err",
            f"--chdir={backend_dir}",
            *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
            "--wrap", f"bash -lc {shlex.quote(merge_command)}",
        ]
        info["merge_sbatch_command"] = shlex.join(merge_sbatch)
        try:
            merge_result = _run_sbatch_with_retry(merge_sbatch)
        except subprocess.CalledProcessError as e:
            reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output"
            raise RuntimeError(
                f"The shard array was submitted as job {info['extraction_job_id']}, but "
                f"the merge job could not be submitted (exit {e.returncode}): {reason}. "
                f"Run the merge by hand once the array finishes."
            ) from e
        merge_match = re.search(r"Submitted batch job (\d+)", merge_result.stdout or "")
        if merge_match:
            info["merge_job_id"] = merge_match.group(1)

    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a Slurm GPU job to run a packaged .h5 through Kai's HPL encoder."
    )
    # Required to submit, but not to --bootstrap-extras, which needs only the
    # image and the extras directory. Enforced in main() rather than by
    # required=True so the two modes can share one parser.
    parser.add_argument("--real-hdf5", type=Path)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--depends-on-job-id", type=str, default=None)
    parser.add_argument(
        "--conda-env",
        type=str,
        default=CONDA_ENV_NAME,
        help="Ignored. Retained for CLI compatibility; extraction runs in Singularity.",
    )
    parser.add_argument("--hpl-repo-dir", type=Path, default=HPL_REPO_DIR)
    parser.add_argument("--singularity-image", type=Path, default=SINGULARITY_IMAGE)
    parser.add_argument("--singularity-bin", type=str, default=SINGULARITY_BIN)
    parser.add_argument(
        "--extras-dir",
        type=Path,
        default=CONTAINER_EXTRAS,
        help="Directory of Python packages the NGC image lacks, bound into the job "
             "and placed on PYTHONPATH.",
    )
    parser.add_argument(
        "--bootstrap-extras",
        action="store_true",
        help="Install the missing packages into --extras-dir and exit without "
             "submitting. Run once, on a login node (needs PyPI access).",
    )
    parser.add_argument(
        "--bootstrap-extras-gpu",
        action="store_true",
        help="Install a GPU faiss into a SEPARATE extras directory "
             "(HPL_CONTAINER_EXTRAS_GPU) for Stage 4's --device gpu, and exit. "
             "Separate because faiss-cpu and GPU faiss are both imported as "
             "`faiss`, so one path cannot hold both. Login node, needs PyPI.",
    )
    parser.add_argument("--model", type=str, default="BarlowTwins_3")
    parser.add_argument("--marker", type=str, default="he")
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--shards", type=int, default=1,
        help="Split the input across N GPU array tasks, then merge. This is the "
             "lever that actually scales extraction: reads are single-threaded "
             "within a process because h5py holds a global lock, so parallelism "
             "has to come from separate processes.",
    )
    parser.add_argument("--merge-partition", type=str, default=MERGE_PARTITION)
    parser.add_argument("--partition", type=str, default=GPU_PARTITION)
    parser.add_argument(
        "--gres", type=str, default=None,
        help="Override the GPU request. Omit to pick the best free type from "
             f"HPL_GPU_PREFERENCE ({','.join(GPU_PREFERENCE)}).",
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory", type=str, default="64G")
    parser.add_argument("--time-limit", type=str, default=_DEFAULT_TIME_LIMIT)
    parser.add_argument("--notify-email", type=str, default=None)
    parser.add_argument(
        "--keep-stale-output",
        action="store_true",
        help="Refuse to submit if a previous attempt left an incomplete output file, "
             "instead of deleting it and resubmitting.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.bootstrap_extras:
        try:
            bootstrap_container_extras(
                args.extras_dir,
                singularity_image=args.singularity_image,
                singularity_bin=args.singularity_bin,
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Bootstrap failed: {e}", file=sys.stderr)
            raise SystemExit(1)
        return

    if args.bootstrap_extras_gpu:
        # One candidate at a time, because pip given three names installs the
        # first it resolves and reports success — and which of the three landed
        # decides whether the job has a GPU faiss at all. Trying them
        # individually means the failure of the preferred wheel is visible
        # rather than hidden behind a fallback.
        errors = []
        for package in _CONTAINER_EXTRA_PACKAGES_GPU:
            print(f"\nTrying {package} ...", flush=True)
            try:
                bootstrap_container_extras(
                    CONTAINER_EXTRAS_GPU,
                    singularity_image=args.singularity_image,
                    singularity_bin=args.singularity_bin,
                    packages=(package,),
                    verify="faiss",
                )
            except (FileNotFoundError, RuntimeError) as e:
                errors.append(f"{package}: {e}")
                print(f"  {package} did not install: {e}", file=sys.stderr)
                continue
            print(f"\nGPU faiss installed from {package}.")
            print("Submit with --device gpu. The job verifies the GPU index "
                  "against a CPU one at startup and refuses if they disagree, "
                  "so a wheel that imports but does not work costs a refusal "
                  "rather than wrong cluster IDs.")
            return
        print("\nNo GPU faiss wheel installed for this container's Python:",
              file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        print("Stage 4 still runs on CPU; --device gpu is what needs this. "
              "Sharding (--shards) is the CPU-side alternative and needs "
              "nothing installed.", file=sys.stderr)
        raise SystemExit(1)

    missing = [
        flag
        for flag, value in (
            ("--real-hdf5", args.real_hdf5),
            ("--checkpoint", args.checkpoint),
            ("--dataset-name", args.dataset_name),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    try:
        info = submit_feature_extraction_job(
            real_hdf5_path=args.real_hdf5,
            checkpoint=args.checkpoint,
            dataset_name=args.dataset_name,
            depends_on_job_id=args.depends_on_job_id,
            conda_env=args.conda_env,
            hpl_repo_dir=args.hpl_repo_dir,
            singularity_image=args.singularity_image,
            singularity_bin=args.singularity_bin,
            extras_dir=args.extras_dir,
            model=args.model,
            marker=args.marker,
            z_dim=args.z_dim,
            img_size=args.img_size,
            batch_size=args.batch_size,
            shards=args.shards,
            merge_partition=args.merge_partition,
            partition=args.partition,
            gres=args.gres,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            notify_email=args.notify_email,
            clear_stale_output=not args.keep_stale_output,
        )
    except (NotADirectoryError, FileExistsError, FileNotFoundError) as e:
        # Configuration and already-done cases are the expected outcomes here,
        # not crashes — a traceback buries the one line that says what to fix.
        print(f"Not submitted: {e}", file=sys.stderr)
        raise SystemExit(1)

    if info["cleared_stale_output"]:
        print(f"Cleared stale:    {info['cleared_stale_output']}")
    print(f"GRES request:     {info['gres']}")
    print(f"Singularity:      {info['singularity_image']}")
    print(f"Container extras: {info['extras_dir']}")
    if info["shards"] > 1:
        bounds = info["shard_bounds"]
        print(f"Shards:           {info['shards']}  "
              f"(rows {bounds[0][0]}-{bounds[0][1]} ... {bounds[-1][0]}-{bounds[-1][1]})")
        print(f"Merge job ID:     {info['merge_job_id']}")
    if info["cleared_stale_parts"]:
        print(f"Cleared parts:    {len(info['cleared_stale_parts'])}")
    print(f"Expected output:  {info['expected_output_path']}")
    print(f"Slurm job ID:     {info['extraction_job_id']}")
    print(f"sbatch stdout:    {info['sbatch_stdout']}")


if __name__ == "__main__":
    main()
