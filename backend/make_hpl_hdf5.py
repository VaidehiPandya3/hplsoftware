#!/usr/bin/env python3
"""Package saved JPEG tiles into the .h5 format Kai's HPL pipeline expects
as model input: img (N,224,224,3) uint8, plus samples/slides/tiles as
fixed-length byte strings, all aligned by index. See make_hdf5.py in
K-Rakovic/HPL-LATTICeA for the reference format this replicates.

Usage:
    python make_hpl_hdf5.py
        --manifest /path/to/slurm_manifests/wsi_manifest_20260714_134135.txt
        --tile-dir /path/to/processed_tiles
        --output-root /path/to/datasets
        --dataset-name Radiogenomics
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from slide_naming import slide_id_from_raw_path
from tile_metadata import CORRUPT, read_tile_metadata, tile_metadata_path

# HDF5 chunk cache for the output file. One tile is one chunk (see the
# create_dataset call in package_slides_to_h5), i.e. 224*224*3 = 147 KiB, so
# this holds roughly 430 of them. The HDF5 default is 1 MiB, which cannot hold
# even a full chunk-batch and forces an evict-and-refetch on essentially every
# write.
_CHUNK_CACHE_BYTES = 64 * 1024 * 1024

# Applied to the "img" dataset only — the byte-string metadata datasets are
# too small for a filter to matter. Recorded in run_identity below (not just
# passed to create_dataset) so a checkpoint written under a different setting
# is treated as incompatible instead of silently resumed into.
_IMG_COMPRESSION = "gzip"
_IMG_COMPRESSION_OPTS = 6

# Identifies what the "tiles" dataset contains, for run_identity. "col_row" was
# the original, storing "18_15"; "col_row.jpeg" stores "18_15.jpeg", which is
# what Kai's reference CSVs and the Knowledge Bank both use. The two produce
# byte-identical checkpoint labels, so nothing else distinguishes a checkpoint
# written by one from a resume under the other.
_TILE_NAME_FORMAT = "col_row.jpeg"


def _tile_name(tile_row) -> str:
    """The tile's name as stored in the .h5 and on disk.

    One definition, because it is consumed three times — to size the
    fixed-length `tiles` column, to build the file path, and to write the
    value — and the three must agree. Sizing from a shorter form than the one
    written truncates it back to the shorter form on write, silently.
    """
    return f"{tile_row.col}_{tile_row.row}.jpeg"





    
    
def hpl_h5_output_path(
    output_root: Path, dataset_name: str, marker: str = "he",
    split: str = "train", tile_size: int = 224,
) -> Path:
    """Where the packaged .h5 lands: <output_root>/<dataset_name>/hdf5_..._<split>.h5
    — flat, not Kai's Data class's nested datasets/<dataset>/<marker>/patches_h*_w*/
    structure. Shared with submit_mask_tile_slurm.py so the two never compute
    this path differently. If you point Kai's pipeline directly at this
    output, its Data class won't find it automatically without adjusti
    --dbs_path or restructuring — this is just where we stage it for now.
    """
    return output_root / dataset_name / f"hdf5_{dataset_name}_{marker}_{split}.h5"


# --- Resume checkpointing -----------------------------------------------
#
# A packaging run over a large dataset can take hours, and up to now a
# TIMEOUT/OOM/node failure/Ctrl-C at any point meant starting completely
# over — every tile decoded so far thrown away, even minutes before the
# end. These three sidecar files (never touched by anything else, always
# named off output_h5_path so they can't collide between datasets) are
# what let a later call detect "this exact output was left mid-run" and
# pick up only the tiles that don't already have a durable outcome,
# instead of blindly re-decoding everything.
#
# completed.txt / skipped.txt are plain one-label-per-line append logs
# (not JSON) specifically so writing to them scales to millions of tiles —
# appending a line is O(1) regardless of how large the file already is,
# where rewriting a JSON array on every update would get slower as the
# run progresses. run_config.json is small (a handful of scalars) and
# genuinely needs to be read back as structured data, so it's the one
# actual JSON file of the three.
def _checkpoint_paths(output_h5_path: Path) -> dict[str, Path]:
    base = str(output_h5_path)
    return {
        "completed": Path(base + ".completed.txt"),
        "skipped": Path(base + ".skipped.txt"),
        "config": Path(base + ".run_config.json"),
    }


def _load_checkpoint_labels(path: Path) -> set[str]:
    """Tile labels already given a durable outcome by an interrupted run.
    A malformed/truncated last line (the process could have died mid-write
    of that exact line) is simply dropped, not treated as an error — losing
    one label just means that one tile gets redecoded on resume, which is
    always safe; it's never treated as done when it might not be.
    """
    if not path.is_file():
        return set()
    labels = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.add(line)
    return labels


def _clear_checkpoint(paths: dict[str, Path]) -> None:
    for p in paths.values():
        p.unlink(missing_ok=True)


def _h5_resumable(output_h5_path: Path) -> bool:
    """Whether an interrupted run's .h5 can actually be reopened and appended
    to, rather than merely existing on disk.

    is_file() alone is not enough, and this is the exact case this whole
    checkpoint mechanism exists for: a TIMEOUT/OOM/node failure kills the job
    with SIGTERM mid-write, which routinely leaves a *truncated* .h5 that HDF5
    then refuses to open at all ("OSError: unable to open file (truncated
    file: eof = ...)"). Since the checkpoint sidecars are still sitting right
    next to it and the run's identity still matches, resume used to engage and
    immediately raise on the h5py.File(..., "r+") below — on that attempt and
    on every retry after it, permanently locking the run out of ever
    packaging again. Probing the file here instead means an unopenable one
    falls back to starting fresh (mode "w", which overwrites it) rather than
    crashing.

    Opened in the same "r+" mode the real write path uses, so this can't pass
    a file that would then fail for a reason "r" wouldn't surface (e.g. a
    read-only file). Also checks the four expected datasets are present — an
    interruption before create_dataset finished leaves an openable but
    unusable file, which would otherwise fail one line later on hdf5["img"].
    """
    if not output_h5_path.is_file():
        return False
    try:
        with h5py.File(output_h5_path, "r+") as probe:
            return all(name in probe for name in ("img", "samples", "slides", "tiles"))
    except (OSError, KeyError):
        return False


def sample_from_slide_id(slide_id: str) -> str:
    """First whitespace-separated token of the slide_id, e.g.
    'BB232001 20C2 - 2023-08-29 2020.38.24' -> 'BB232001'.

    Assumes the dataset's filenames put the patient/sample ID first,
    space-separated from everything else (block/section, date, time) —
    confirmed for Radiogenomics; re-check this if pointed at a dataset
    with a different naming convention.
    """
    return slide_id.split(" ")[0]


def read_manifest(manifest_path: Path) -> list[str]:
    """Manifest is one raw slide path per line — the same format
    submit_mask_tile_slurm.py already writes to slurm_manifests/.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]


def load_slide_tiles(tile_dir: Path, tile_dataset_name: str, slide_id: str) -> pd.DataFrame:
    """This slide's tile rows, or an empty frame if it has none worth packaging.

    Delegates the actual classification to tile_metadata.read_tile_metadata so
    find_missing_slides.py applies identical rules — see that module's
    docstring for why a mismatch between the two silently loses slides.

    Only EmptyDataError used to be caught here, which left a genuinely corrupt
    CSV (killed mid-write) raising pandas' ParserError straight out of the
    metadata-loading thread pool and aborting the entire run — every other
    slide's work lost because of one bad file. Corrupt now degrades to "this
    slide has no usable tiles," reported via slides_missing_metadata, and is
    logged loudly so it isn't a silent omission.
    """
    meta = read_tile_metadata(tile_metadata_path(tile_dir, tile_dataset_name, slide_id))
    if meta.status == CORRUPT:
        print(f"WARNING: [{slide_id}] {meta.detail} — excluded from this .h5")
    return meta.frame if meta.usable else pd.DataFrame()


def load_tile_image(tile_path: Path, tile_size: int) -> np.ndarray | None:
    """Load a tile safely.

    Returns:
        np.ndarray if the tile is valid.
        None if the tile is missing, empty, corrupt, truncated,
        unreadable, or has the wrong dimensions.
    """

    try:
        # Missing file
        if not tile_path.exists():
            return None

        # Empty file
        if tile_path.stat().st_size == 0:
            return None

        with Image.open(tile_path) as img:
            img = img.convert("RGB")
            img_array = np.asarray(img, dtype=np.uint8)

        if img_array.shape != (tile_size, tile_size, 3):
            return None

        return img_array

    except (FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            ValueError,
            EOFError):
        # Any corrupted/truncated image
        return None

    except Exception as e:
        # Don't fail the whole dataset because of one bad tile
        print(f"WARNING: Failed to read {tile_path}: {e}")
        return None


def _load_tile_paths(args: tuple[list[str], int, int]) -> list[np.ndarray | None]:
    """Runs inside one worker PROCESS (via ProcessPoolExecutor below).
    Reads its assigned tile paths concurrently with a thread pool — these
    are individually-small reads off a network filesystem, so more
    concurrency than the process count alone hides that I/O latency well,
    and Pillow's JPEG decode also releases the GIL during the actual
    C-level decode. Returns results in the same order as tile_paths, so
    the caller can zip them straight back up with the rest of each tile's
    metadata without needing to track indices through two levels of
    parallelism.
    """
    tile_paths, tile_size, threads_per_process = args
    with ThreadPoolExecutor(max_workers=threads_per_process) as pool:
        return list(pool.map(lambda p: load_tile_image(Path(p), tile_size), tile_paths))


def package_to_h5(
    manifest_path: Path,
    tile_dir: Path,
    tile_dataset_name: str,
    output_root: Path,
    dataset_name: str,
    marker: str = "he",
    split: str = "train",
    tile_size: int = 224,
    n_processes: int | None = None,
    threads_per_process: int = 4,
    batch_size: int = 2000,
    resume: bool | None = None,
) -> dict:
    """Thin wrapper over package_slides_to_h5 for manifest-file callers
    (the CLI, the dataset-wide Slurm pipeline).

    resume is passed straight through: None keeps the historical behaviour of
    continuing whatever checkpoint is on disk, True demands one, False discards
    it. See package_slides_to_h5."""
    return package_slides_to_h5(
        raw_paths=read_manifest(manifest_path),
        tile_dir=tile_dir,
        tile_dataset_name=tile_dataset_name,
        output_root=output_root,
        dataset_name=dataset_name,
        marker=marker,
        split=split,
        tile_size=tile_size,
        n_processes=n_processes,
        threads_per_process=threads_per_process,
        batch_size=batch_size,
        resume=resume,
    )


def package_slides_to_h5(
    raw_paths: list[str],
    tile_dir: Path,
    tile_dataset_name: str,
    output_root: Path,
    dataset_name: str,
    marker: str = "he",
    split: str = "train",
    tile_size: int = 224,
    n_processes: int | None = None,
    threads_per_process: int = 4,
    batch_size: int = 2000,
    slide_ids: list[str] | None = None,
    resume: bool | None = None,
) -> dict:
    """Core packaging logic, taking raw slide paths directly rather than a
    manifest file — lets a single-slide caller package one slide without
    writing a throwaway one-line manifest to disk first.

    tile_dataset_name is where tiles actually live on disk
    (tile_dir/<tile_dataset_name>/<slide_id>/, e.g. "TCGA" or
    "Radiogenomics"); dataset_name is only for the .h5 output's own name/path
    and can differ (e.g. a subset run's "TCGA_subset_1").

    slide_ids, if given, must be the same length and order as raw_paths and
    overrides slide_id_from_raw_path() per slide — needed by a caller that
    already knows which slide this is (e.g. the tile server's ad-hoc uploads,
    where the id is the one the user chose). Without it, such a caller's tiles
    (written under that chosen id — see tile_slide_from_mask's own slide_id
    override) would be looked for under whatever the stored filename derives
    to, which is not the same thing.

    Reading and decoding tiles (not writing — that stays single-threaded
    and sequential, h5py isn't safe for concurrent writes) is parallelized
    two ways: n_processes worker processes for genuine CPU parallelism
    (bypasses the GIL for JPEG decoding), each further using
    threads_per_process threads for concurrent file reads within its own
    chunk (these are small individual reads off a network filesystem, so
    I/O latency — not decode time — is likely the bigger win).

    Chunks (~batch_size/n_processes tiles each) are kept in flight as a
    rolling window — one freshly submitted the instant a worker's previous
    chunk completes — rather than dispatched in synchronized waves of
    batch_size tiles. The wave approach used to mean every worker sat idle
    twice per batch: waiting for the slowest chunk in the wave to finish,
    then again while the whole batch got written to h5 (necessarily
    single-threaded) before the next wave was even submitted. The rolling
    window keeps decoding the next chunk while the current one's already-
    decoded images are being written, and total in-flight tiles stays
    bounded at roughly batch_size either way — this changes *when* work is
    submitted, not how much memory is held at once. Tiles are written to
    the h5 datasets in whatever order their chunks finish decoding, not
    original tile_jobs order — safe here since every dataset (img,
    samples, slides, tiles) is written at the same index together, and
    nothing downstream reads this file in row order, only by index.
    """
    if slide_ids is not None and len(slide_ids) != len(raw_paths):
        raise ValueError("slide_ids must be the same length as raw_paths.")
    slide_ids = slide_ids or [slide_id_from_raw_path(p) for p in raw_paths]

    # First pass: load each slide's tile metadata and find the total tile
    # count up front, so the h5 datasets can be created at their final size
    # once — Kai's own make_hdf5.py instead holds every decoded image in a
    # Python list before stacking, which doesn't scale to a full dataset.
    #
    # This used to be a plain sequential loop — for a dataset with
    # thousands of slides, that's thousands of small, sequential reads off
    # a (likely network-mounted) filesystem, one slide's CSV at a time,
    # with the actual tile-decode parallelism further down unable to start
    # until every single one finished. Parsing one of these CSVs is fast;
    # the per-file round-trip latency is what dominates, so this is a
    # thread pool (I/O-bound, no CPU-bound work to fight the GIL over)
    # rather than the process pool used for image decoding below — clears
    # this phase in roughly (total time / worker count) instead of the sum.
    metadata_workers = min(32, max(4, len(slide_ids)))
    per_slide_rows: dict[str, pd.DataFrame] = {}
    missing_metadata: list[str] = []
    total_tiles = 0
    with ThreadPoolExecutor(max_workers=metadata_workers) as pool:
        loaded = pool.map(
            lambda sid: (sid, load_slide_tiles(tile_dir, tile_dataset_name, sid)), slide_ids
        )
        for slide_id, df in loaded:
            if df.empty:
                missing_metadata.append(slide_id)
                continue
            per_slide_rows[slide_id] = df
            total_tiles += len(df)

    if total_tiles == 0:
        raise RuntimeError(
            f"No tiles found for any of the {len(slide_ids)} requested slides "
            f"under {tile_dir}. Missing metadata for: {missing_metadata[:10]}"
        )

    # Size the fixed-length byte-string fields from the actual data instead
    # of copying Kai's hardcoded S8 (samples/tiles) / S30 (slides) widths,
    # which silently truncate anything longer — our slide_ids can easily
    # exceed those (e.g. "BB232001 20C2 - 2023-08-29 2020.38.24" is 37 bytes).
    max_sample_len = max(len(sample_from_slide_id(sid).encode("utf-8")) for sid in per_slide_rows)
    max_slide_len = max(len(sid.encode("utf-8")) for sid in per_slide_rows)
    # Sized from the name actually stored, suffix included. These are
    # fixed-length byte strings, so a width computed from the bare "24_10"
    # would silently truncate "24_10.jpeg" back to "24_10" on write — the
    # stored value would be exactly the bug this suffix was added to fix, with
    # nothing anywhere reporting a problem.
    max_tile_len = max(
        len(_tile_name(tile_row).encode("utf-8"))
        for df in per_slide_rows.values()
        for tile_row in df.itertuples()
    )

    sample_dtype = f"S{max_sample_len}"
    slide_dtype = f"S{max_slide_len}"
    tile_dtype = f"S{max_tile_len}"

    # Everything is written to a sibling ".partial" file and moved into place
    # only once the run has genuinely finished (see os.replace at the end).
    #
    # Writing straight to the final path meant that path existed — created by
    # h5py the instant packaging *started* — for the entire multi-hour run,
    # so "the .h5 is there" said nothing about whether it was complete. A
    # TIMEOUT/OOM kill left a truncated file sitting at exactly the name
    # downstream consumers read, and worse, a *re-run* clobbered a previous
    # good .h5 at that path the moment it started, destroying a working
    # dataset in exchange for an attempt that might then fail. With the
    # rename, the final path only ever holds a completed file: the previous
    # good one stays intact and readable until the new one is fully written,
    # and existence becomes a meaningful readiness signal for the tile
    # server's own checks.
    #
    # os.replace is atomic within a filesystem, and the ".partial" sibling is
    # in the same directory by construction, so there's no window where the
    # final path holds a half-moved file.
    output_h5_path = hpl_h5_output_path(output_root, dataset_name, marker, split, tile_size)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    partial_h5_path = output_h5_path.with_name(output_h5_path.name + ".partial")
    # Checkpoints stay keyed off the FINAL path, not the partial — the whole
    # point is that they survive the interruption that leaves a partial
    # behind, and keying them off a name that only exists mid-run would be
    # needlessly fragile.
    ckpt = _checkpoint_paths(output_h5_path)

    # A run's "identity" for resume purposes — every field here is
    # data-affecting (determines *what* goes into the .h5, not how fast),
    # so if any of it differs from what an interrupted attempt's checkpoint
    # recorded, the underlying dataset changed since then and resuming into
    # those same .h5 datasets isn't safe (total_tiles is baked into their
    # maxshape at creation and can never grow past it; the *_dtype
    # fixed-width byte-string sizes are equally fixed once created) —
    # falls back to starting fresh rather than guessing. Deliberately
    # excludes n_processes/threads_per_process/batch_size: those are pure
    # performance tuning, safe to change freely between attempts (e.g.
    # lowering n_processes to work around an OOM) without invalidating a
    # checkpoint. This is also the full record resume_packaging.py reads
    # back to replay this exact call without the caller needing to retype
    # every argument.
    run_identity = {
        "total_tiles": total_tiles,
        "sample_dtype": sample_dtype,
        "slide_dtype": slide_dtype,
        "tile_dtype": tile_dtype,
        "raw_paths": raw_paths,
        "slide_ids": slide_ids,
        "tile_dir": str(tile_dir),
        "tile_dataset_name": tile_dataset_name,
        "output_root": str(output_root),
        "dataset_name": dataset_name,
        "marker": marker,
        "split": split,
        "tile_size": tile_size,
        "img_compression": _IMG_COMPRESSION,
        "img_compression_opts": _IMG_COMPRESSION_OPTS,
        # What the `tiles` dataset holds. A checkpoint records tile *labels*
        # ("<slide>/<col>_<row>.jpeg"), which did not change when the stored
        # tile name gained its ".jpeg" suffix — so without this key a resume
        # across that change would skip every already-written row and leave the
        # file holding bare "18_15" for those and "18_15.jpeg" for the rest.
        # That .h5 would pass every structural check while being unjoinable for
        # part of its contents. Bump this whenever the meaning of a stored
        # column changes without its label changing.
        "tile_name_format": _TILE_NAME_FORMAT,
    }

    resuming = ckpt["completed"].is_file() or ckpt["skipped"].is_file()

    # resume=False is an explicit instruction to discard an earlier attempt's
    # progress, not a hint. It exists because resuming used to be decided
    # entirely here: a checkpoint on disk meant the next submission silently
    # continued it, and the caller had no way to say "no, rebuild this from
    # scratch" — which is the right answer whenever the tiles themselves have
    # been re-made, or when an attempt is suspected of having written bad data
    # that a resume would preserve forever.
    #
    # The .partial goes too. Leaving it would make the fresh run reopen a file
    # whose contents belong to a discarded attempt (mode "w" truncates it, but
    # only after _h5_resumable has already probed it), and its size on disk is
    # the largest thing being thrown away.
    if resuming and resume is False:
        print(
            f"[{output_h5_path}] Discarding the earlier attempt's checkpoint at the "
            f"caller's explicit request — packaging every tile from scratch."
        )
        _clear_checkpoint(ckpt)
        partial_h5_path.unlink(missing_ok=True)
        resuming = False
    elif not resuming and resume is True:
        # Asked to resume with nothing to resume from. Refuse rather than
        # quietly doing a full run: the caller told the user how much work
        # would be skipped, and a silent full re-run makes that a lie.
        raise RuntimeError(
            f"resume=True was requested for {output_h5_path}, but no checkpoint from a "
            f"previous attempt exists at {ckpt['completed']}. Re-run without resume to "
            f"package from scratch."
        )

    if resuming:
        try:
            prior_identity = json.loads(ckpt["config"].read_text())
        except Exception:
            prior_identity = None
        # Resume reopens the *partial*, which is where an interrupted attempt's
        # actual progress lives. A completed .h5 sitting at the final path is
        # irrelevant here — if one exists it's from a previous finished run,
        # and it stays untouched until this run completes and replaces it.
        if prior_identity != run_identity or not _h5_resumable(partial_h5_path):
            print(
                f"[{output_h5_path}] Found a checkpoint from an earlier attempt, but it "
                "doesn't match this run (tile set changed, or the .h5 itself is gone, "
                "truncated, or otherwise unreadable) — starting over instead of risking "
                "a corrupt resume."
            )
            resuming = False

    if not resuming:
        _clear_checkpoint(ckpt)
        ckpt["config"].write_text(json.dumps(run_identity))

    completed_labels: set[str] = _load_checkpoint_labels(ckpt["completed"]) if resuming else set()
    skipped_labels: set[str] = _load_checkpoint_labels(ckpt["skipped"]) if resuming else set()
    already_done = completed_labels | skipped_labels
    if resuming:
        print(
            f"[{output_h5_path}] Resuming: {len(completed_labels)} tiles already written, "
            f"{len(skipped_labels)} already known-bad, {total_tiles - len(already_done)} remaining."
        )

    # Datasets are created resizable (maxshape == the upper bound from
    # metadata row counts) so a bad tile can be skipped instead of aborting
    # the whole multi-hour job — a single corrupt/truncated JPEG (leftover
    # from an earlier interrupted tiling run) used to crash packaging after
    # hours of progress, on a dataset this size that's a near-certainty to
    # hit repeatedly, one tile at a time, on every retry. Trimmed down to
    # the real write count at the end if any tiles were skipped.
    # Flatten every slide's tiles into one ordered list up front, skipping
    # anything a prior attempt already gave a durable outcome to. Keeping
    # this flat (rather than nested per-slide) is what lets the batches
    # below cut across slide boundaries and stay evenly sized regardless of
    # how many tiles any one slide has.
    tile_jobs: list[tuple[str, str, str, str, str]] = []  # (path, label, sample, slide_id, tile_name)
    for slide_id, df in per_slide_rows.items():
        sample = sample_from_slide_id(slide_id)
        slide_dir = tile_dir / tile_dataset_name / slide_id
        for tile_row in df.itertuples():
            # The stored tile name keeps its extension. Kai's reference CSVs use
            # "18_15.jpeg", the KB's tile_coordinates/tile_registry store the
            # same, and assign_hpc_clusters.py's --validate-against merges on
            # (slides, tiles) against exactly that form. Writing the bare
            # "18_15" here made the packaged .h5 — and so every downstream
            # assignments CSV — disagree with all three: the KB load matched
            # 0.0% of 38,892 Radiogenomics tiles, and the acceptance test would
            # have merged zero rows.
            tile_name = _tile_name(tile_row)
            tile_path = slide_dir / tile_name
            tile_label = f"{slide_id}/{tile_name}"
            if tile_label in already_done:
                continue
            tile_jobs.append((str(tile_path), tile_label, sample, slide_id, tile_name))

    resolved_n_processes = n_processes or min(os.cpu_count() or 4, 8)

    skipped_tiles: list[str] = list(skipped_labels)
    h5_mode = "r+" if resuming else "w"

    # The worker pool is created BEFORE the HDF5 file is opened, and this
    # ordering is load-bearing rather than stylistic. HDF5 is not fork-safe:
    # with the pool created inside the open-file block (as it was), every
    # forked worker inherited this process's open HDF5 file descriptor and
    # h5py's atexit close handler, so a worker exiting could flush or close a
    # file it does not own. Forking first means there is no HDF5 state in
    # existence for a child to inherit.
    with ProcessPoolExecutor(max_workers=resolved_n_processes) as executor, \
         h5py.File(
             partial_h5_path, h5_mode,
             # The default chunk cache is 1 MiB, which cannot even hold a
             # handful of the 147 KiB chunks below — every write would evict
             # and re-read. 64 MiB holds ~430 of them; nslots is prime and
             # ~10x the chunk count, per HDF5's own guidance.
             rdcc_nbytes=_CHUNK_CACHE_BYTES, rdcc_nslots=4001,
         ) as hdf5:
        if resuming:
            img_ds = hdf5["img"]
            sample_ds = hdf5["samples"]
            slide_ds = hdf5["slides"]
            tile_ds = hdf5["tiles"]
        else:
            img_shape = (total_tiles, tile_size, tile_size, 3)
            # chunks is set explicitly, and this is the single biggest
            # throughput factor in this function. Passing maxshape without
            # chunks forces a chunked layout and leaves h5py to guess the
            # shape — and its guess targets ~1 MiB chunks by subdividing the
            # *pixel* dimensions, e.g. (250, 14, 28, 1) for 4k tiles or
            # (3125, 7, 14, 1) for 100k. Each such chunk spans thousands of
            # images but a tiny patch of one channel, so writing one image
            # became a read-modify-write across 384-1536 separate chunks:
            # ~250x the necessary I/O at 4k tiles, ~3125x at 100k, and worse
            # as the dataset grows. That is why packaging time exploded on
            # large runs rather than scaling linearly.
            #
            # One tile per chunk makes a single image write exactly one chunk
            # write, and is also the right shape for reads, since training
            # reads individual tiles.
            #
            # gzip level 6 trades some write CPU for a meaningful cut in the
            # on-disk size (raw 224x224x3 uint8 tiles compress ~2x with
            # gzip — real photographic detail, not the ~7x a lossy JPEG gets,
            # but free of any read-side change: h5py decompresses a chunk
            # transparently on access, so training's plain
            # img[i:i+n]-style slicing (see HPL's data_manipulation/dataset.py)
            # doesn't need to know this dataset is compressed at all).
            img_ds = hdf5.create_dataset(
                "img", img_shape, maxshape=img_shape, dtype="uint8",
                chunks=(1, tile_size, tile_size, 3),
                compression=_IMG_COMPRESSION, compression_opts=_IMG_COMPRESSION_OPTS,
            )
            sample_ds = hdf5.create_dataset(
                "samples", (total_tiles,), maxshape=(total_tiles,), dtype=sample_dtype
            )
            slide_ds = hdf5.create_dataset(
                "slides", (total_tiles,), maxshape=(total_tiles,), dtype=slide_dtype
            )
            tile_ds = hdf5.create_dataset(
                "tiles", (total_tiles,), maxshape=(total_tiles,), dtype=tile_dtype
            )

        # Fixed-size chunks across the remaining tile list (already
        # excludes anything a prior attempt finished), not per-batch —
        # batch_size still controls the total tiles held in flight at once
        # (chunk_size * n_processes ≈ batch_size), but chunks are now
        # submitted from one continuous queue instead of being re-split
        # per synchronized wave. See the docstring above for why.
        chunk_size = max(1, -(-batch_size // resolved_n_processes))  # ceil div
        chunks = [tile_jobs[i:i + chunk_size] for i in range(0, len(tile_jobs), chunk_size)]

        # write_index picks up exactly where a resumed run's completed count
        # left off — this loop only ever appends at one running index
        # regardless of which chunk a tile came from, so every prior write
        # filled indices 0..len(completed_labels)-1 contiguously, and
        # continuing from there is safe. Skipped tiles never occupied an
        # index in the first place.
        write_index = len(completed_labels)

        with open(ckpt["completed"], "a", encoding="utf-8") as completed_f, \
             open(ckpt["skipped"], "a", encoding="utf-8") as skipped_f:
            chunk_iter = iter(chunks)

            def _submit(chunk):
                paths = [job[0] for job in chunk]
                return executor.submit(_load_tile_paths, (paths, tile_size, threads_per_process))

            # Prime the pool: one chunk in flight per worker to start.
            in_flight: dict[Future, list] = {
                _submit(chunk): chunk
                for chunk in itertools.islice(chunk_iter, resolved_n_processes)
            }

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    chunk = in_flight.pop(future)
                    results = future.result()

                    # Collect this chunk's successful tiles, then write them
                    # as contiguous slices rather than one row at a time.
                    # Every tile in a chunk lands at consecutive indices by
                    # construction (write_index only ever advances), so a
                    # slice assignment is equivalent to the per-row loop that
                    # used to be here — but it replaces up to chunk_size
                    # separate HDF5 calls, each with its own chunk lookup and
                    # cache round trip, with four.
                    imgs: list[np.ndarray] = []
                    labels: list[str] = []
                    samples: list[bytes] = []
                    slides: list[bytes] = []
                    tiles: list[bytes] = []
                    for (_, tile_label, sample, slide_id, tile_name), img_array in zip(chunk, results):
                        if img_array is None:
                            skipped_tiles.append(tile_label)
                            skipped_f.write(tile_label + "\n")
                            continue
                        imgs.append(img_array)
                        labels.append(tile_label)
                        samples.append(sample.encode("utf-8"))
                        slides.append(slide_id.encode("utf-8"))
                        tiles.append(tile_name.encode("utf-8"))

                    if imgs:
                        end = write_index + len(imgs)
                        img_ds[write_index:end] = np.stack(imgs)
                        sample_ds[write_index:end] = np.array(samples, dtype=sample_dtype)
                        slide_ds[write_index:end] = np.array(slides, dtype=slide_dtype)
                        tile_ds[write_index:end] = np.array(tiles, dtype=tile_dtype)
                        write_index = end

                        # Push HDF5's own cache out BEFORE recording these
                        # tiles as complete. This ordering is the whole
                        # correctness argument for resume, and it is easy to
                        # get backwards: returning from the assignments above
                        # only means the data reached the chunk cache, which
                        # this file deliberately sizes at 64 MiB (~430 tiles).
                        # Recording the labels first would let the checkpoint
                        # durably claim tiles were written while their chunks
                        # were still only in memory — and since resume trusts
                        # the label count to set write_index, a kill in that
                        # window would leave it skipping straight past rows
                        # that never reached disk: permanent zero-filled holes
                        # in the .h5 that nothing would ever re-decode,
                        # because every one of those tiles is on the completed
                        # list. Flushing first makes the only possible
                        # inconsistency the harmless direction — data durable
                        # but not yet recorded, so resume redoes at most one
                        # chunk and overwrites those rows.
                        #
                        # Scope: H5Fflush hands the data to the filesystem, so
                        # this covers the failure that actually happens here —
                        # the process being killed (SIGTERM on TIMEOUT, OOM)
                        # while the node stays up. It is not an fsync and so
                        # does not promise durability across a node or power
                        # failure; recovering from that still means discarding
                        # the checkpoint and repackaging.
                        hdf5.flush()

                        completed_f.write("".join(label + "\n" for label in labels))

                    # Flushed per chunk, not per tile. Skipped labels carry no
                    # ordering constraint — they describe tiles that were
                    # never written and occupy no index, so losing one just
                    # means it gets re-read and re-skipped on resume.
                    completed_f.flush()
                    skipped_f.flush()

                    # Immediately backfill this worker's slot — it's free
                    # the instant its own chunk is done, regardless of
                    # whether any other in-flight chunk has finished yet.
                    next_chunk = next(chunk_iter, None)
                    if next_chunk is not None:
                        in_flight[_submit(next_chunk)] = next_chunk

        if write_index < total_tiles:
            img_ds.resize((write_index, tile_size, tile_size, 3))
            sample_ds.resize((write_index,))
            slide_ds.resize((write_index,))
            tile_ds.resize((write_index,))

    # Reached only if the block above completed without raising — i.e. the
    # entire remaining tile list got a durable outcome (written or skipped),
    # the h5 file is closed and flushed, and nothing is left to resume. An
    # interruption anywhere above leaves both the partial and the checkpoint
    # files in place on purpose, so whatever ran this again (see
    # resume_packaging.py) picks up from here instead of redoing the whole
    # thing.
    #
    # Publish before clearing the checkpoint, not after: if the process dies
    # between the two, the completed .h5 is already in place and the stale
    # checkpoint is harmless (the next run finds no partial to resume, so it
    # starts fresh). The reverse order could lose a finished run's output.
    os.replace(partial_h5_path, output_h5_path)
    _clear_checkpoint(ckpt)

    skip_percentage = (

        100 * len(skipped_tiles) / total_tiles

        if total_tiles > 0

        else 0.0

    )

    if skip_percentage > 0.1:

        print(

            f"WARNING: {len(skipped_tiles)} tiles "

            f"({skip_percentage:.3f}%) were skipped."

        )

    return {

        "output_h5_path": str(output_h5_path),

        "total_tiles": write_index,

        "slides_packaged": len(per_slide_rows),

        "slides_missing_metadata": missing_metadata,

        "skipped_tiles": skipped_tiles,

        "skip_percentage": skip_percentage,

        "sample_dtype": sample_dtype,

        "slide_dtype": slide_dtype,

        "tile_dtype": tile_dtype,

        "resumed": resuming,

        "resumed_tiles_already_done": len(already_done) if resuming else 0,

    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package saved JPEG tiles into the .h5 format Kai's HPL pipeline expects."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Slurm manifest (one raw slide path per line) from submit_mask_tile_slurm.py.",
    )
    parser.add_argument(
        "--tile-dir", type=Path, required=True,
        help="Root directory containing <tile_dataset_name>/<slide_id>/ tile folders "
             "(auto_tile_from_mask.py's output).",
    )
    parser.add_argument(
        "--tile-dataset-name", type=str, required=True,
        help="Folder under --tile-dir the tiles actually live in, e.g. TCGA or Radiogenomics "
             "(may differ from --dataset-name for a subset run's .h5).",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root 'datasets' directory Kai's pipeline expects "
             "(will contain <dataset_name>/<marker>/patches_h*_w*/).",
    )
    parser.add_argument("--dataset-name", type=str, required=True, help="e.g. Radiogenomics")
    parser.add_argument("--marker", type=str, default="he")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument(
        "--processes", type=int, default=None,
        help="Worker processes for reading/decoding tiles. Defaults to min(cpu count, 8) — "
             "should roughly match --cpus-per-task on the Slurm submission.",
    )
    parser.add_argument(
        "--threads-per-process", type=int, default=4,
        help="Threads per worker process for concurrent file reads (I/O-bound, so this can "
             "reasonably exceed the CPU count without hurting anything).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2000,
        help="Tiles decoded and held in memory at once before writing to the .h5 — bounds "
             "memory usage on datasets with millions of tiles.",
    )
    # Mutually exclusive so "--resume --fresh" is rejected by argparse rather
    # than silently resolved to one of them.
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", dest="resume", action="store_true", default=None,
        help="Require continuing a previous attempt's checkpoint, and fail if there "
             "isn't one. Without either flag, a checkpoint is continued if present.",
    )
    resume_group.add_argument(
        "--fresh", dest="resume", action="store_false",
        help="Discard any previous attempt's checkpoint and .partial, and package every "
             "tile again from scratch.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    info = package_to_h5(
        manifest_path=args.manifest,
        tile_dir=args.tile_dir,
        tile_dataset_name=args.tile_dataset_name,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        n_processes=args.processes,
        threads_per_process=args.threads_per_process,
        batch_size=args.batch_size,
        marker=args.marker,
        split=args.split,
        tile_size=args.tile_size,
        resume=args.resume,
    )
    print(f"Output:                {info['output_h5_path']}")
    print(f"Total tiles packaged:  {info['total_tiles']}")
    print(f"Slides packaged:       {info['slides_packaged']}")
    if info["slides_missing_metadata"]:
        print(f"Slides with no tile metadata (skipped): {info['slides_missing_metadata']}")
    if info["skipped_tiles"]:
        print(
            f"Tiles skipped (missing/corrupt on disk): "
            f"{len(info['skipped_tiles'])} "
            f"({info['skip_percentage']:.4f}%) "
            f"— e.g. {info['skipped_tiles'][:10]}"
        )


if __name__ == "__main__":
    main()
