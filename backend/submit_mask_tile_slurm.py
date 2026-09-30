#!/usr/bin/env python3
"""Submit and run a dataset-wide WSI masking + tiling Slurm array.

Submission mode:
    python submit_mask_tile_slurm.py --raw-dir /path/to/dataset

Worker mode is invoked automatically by Slurm.
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
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from slide_naming import slide_id_from_raw_path
from make_hpl_hdf5 import hpl_h5_output_path

SUPPORTED_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn"}


def discover_slides(raw_dir: Path) -> list[Path]:
    """Find all supported WSI files recursively and return stable ordering."""
    return sorted(
        (
            path.resolve()
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def validate_unique_slide_ids(slides: list[Path]) -> None:
    """Prevent different files from writing into the same output folder."""
    by_id: dict[str, list[Path]] = {}
    for slide in slides:
        by_id.setdefault(slide_id_from_raw_path(slide), []).append(slide)

    duplicates = {sid: paths for sid, paths in by_id.items() if len(paths) > 1}
    if duplicates:
        details = "\n".join(
            f"  {sid}:\n" + "\n".join(f"    - {p}" for p in paths)
            for sid, paths in duplicates.items()
        )
        raise RuntimeError(
            "Duplicate slide IDs were detected. These would overwrite each other's outputs:\n"
            + details
        )


def select_slides(
    slides: list[Path],
    *,
    sample_size: int | None = None,
    slide_names: list[str] | None = None,
    random_seed: int | None = None,
) -> list[Path]:
    """Narrow a discovered slide list down to a requested subset.

    At most one of sample_size / slide_names should be set — sample_size
    picks that many at random, slide_names matches by exact filename
    (with or without extension) or derived slide_id, so a non-coder can
    paste in whatever they recognize from a file browser without knowing
    the slide_id convention.

    Raises ValueError if sample_size exceeds the pool, rather than narrowing
    the request to fit — see the comment at that check.
    """
    if sample_size is not None and slide_names:
        raise ValueError("Specify either sample_size or slide_names, not both.")

    if slide_names:
        wanted = {name.strip() for name in slide_names if name.strip()}
        matched: list[Path] = []
        matched_keys: set[str] = set()
        for slide in slides:
            keys = {slide.name, slide.stem, slide_id_from_raw_path(slide)}
            if keys & wanted:
                matched.append(slide)
                matched_keys |= keys & wanted
        missing = wanted - matched_keys
        if missing:
            raise ValueError(
                f"Could not find these requested slides: {sorted(missing)}"
            )
        return matched

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if sample_size > len(slides):
            # Refused, not quietly narrowed to fit. Returning every available
            # slide whenever the request overshot the pool was indistinguishable
            # from success: asking a 30-slide manifest for 3500 produced a
            # 30-slide .h5 with no error and nothing in the logs to explain the
            # gap. It also hid *why* — the pool here is usually a run's
            # manifest, fixed at submission time, so no value of sample_size
            # could ever have widened it. The caller has to choose a different
            # pool, and can only know that if this says so.
            raise ValueError(
                f"Asked for {sample_size} slides but only {len(slides)} are "
                f"available to choose from. Ask for at most {len(slides)}, or "
                f"widen the pool — package with scope='tiled' to take every "
                f"slide with tiles on disk, or start a new run over the full "
                f"directory."
            )
        if sample_size == len(slides):
            return slides
        return random.Random(random_seed).sample(slides, sample_size)

    return slides


def write_manifest(slides: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(f"{slide}\n" for slide in slides),
        encoding="utf-8",
    )


def tiling_output_complete(
    metadata_path: Path,
    summary_path: Path,
    slide_tile_dir: Path,
    verify_row_count: bool = True,
) -> bool:
    """Whether a slide's tiling output is genuinely finished and self-consistent.

    This is the "should I skip this slide?" test, and it replaces a check that
    only asked whether the two files existed and were non-empty. That was too
    weak in the one case that matters: a tiling job killed mid-write (TIMEOUT,
    OOM, node failure) can leave a complete-looking summary next to a metadata
    CSV that stopped short, and the old check skipped such a slide forever
    while packaging then went looking for tiles the CSV promised and the disk
    never had.

    Cross-checking summary["saved_tiles"] against the CSV's row count is what
    closes that: the summary is written once at the end, the CSV grows as tiles
    are saved, so a disagreement is exactly the signature of an interrupted run.

    saved_tiles == 0 returns True deliberately. A slide with no tissue above
    the min-tissue threshold ran to completion and produced nothing, and
    re-tiling it on every resubmission would burn a full slide read to arrive
    at the same empty answer. Callers that care about the distinction should
    read saved_tiles themselves rather than treating "incomplete" as a proxy.

    verify_row_count=False skips reading the CSV, keeping the JSON checks. The
    row count costs a full parse of a file with one line per tile — negligible
    for the one slide a worker is deciding about, but O(dataset) when something
    sweeps thousands of slides at once (see _tiled_coverage in
    tile_server_v2_.py). Note the asymmetry that creates: a sweep may call a
    slide complete that a worker would then re-tile. That direction is the safe
    one — the worker gets the final say and redoes the work — but it does mean
    the two answers are not guaranteed identical.

    Any unreadable/corrupt file answers False: the cost of re-tiling a slide is
    bounded, while trusting a damaged summary corrupts the packaged .h5.
    """
    if not metadata_path.is_file() or not summary_path.is_file():
        return False

    try:
        # A directory replaced by a file (or a broken symlink) would otherwise
        # only surface further down as a confusing parse error.
        if not slide_tile_dir.is_dir():
            return False

        if metadata_path.stat().st_size == 0 or summary_path.stat().st_size == 0:
            return False

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_tiles = int(summary.get("saved_tiles", -1))

        if saved_tiles < 0:
            return False

        # Zero tiles can be a legitimate completed result.
        if saved_tiles == 0:
            return True

        if not verify_row_count:
            return True

        metadata = pd.read_csv(metadata_path)

        if len(metadata) != saved_tiles:
            return False

        return True

    except (OSError, ValueError, TypeError, json.JSONDecodeError, pd.errors.ParserError):
        # TypeError included for a summary whose saved_tiles is null or a list —
        # int() raises TypeError there, not ValueError.
        return False


def read_manifest_slide(manifest_path: Path, task_id: int) -> Path:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == task_id:
                slide_path = Path(line.strip())
                if not slide_path.is_file():
                    raise FileNotFoundError(f"Slide does not exist: {slide_path}")
                return slide_path

    raise IndexError(
        f"Array task {task_id} has no matching slide in {manifest_path}"
    )


def slide_output_paths(
    slide_path: Path, mask_dir: Path, tile_dir: Path, dataset_name: str
) -> dict[str, Path]:
    """Where one slide's mask and tiles land. Shared by the Slurm-array worker
    and the Nextflow pipeline's task (hpl-nf/bin/hpl_tile.py), so the two cannot
    disagree about which files make a slide "done".

    Scoped by dataset_name (not flat under mask_dir directly) so two
    different datasets — or a dataset and an ad-hoc single-slide upload —
    whose slide_id-derivation happens to collide can never silently share
    (and overwrite) the same tissue mask. tile_dir already gets this same
    treatment; mask_dir didn't, which meant a colliding slide_id here wouldn't
    just misname a file — the "mask already exists, skip" check would treat a
    wholly different dataset's leftover mask as this slide's own and tile
    against it, silently.
    """
    slide_id = slide_id_from_raw_path(slide_path)
    dataset_mask_dir = mask_dir / dataset_name
    dataset_tile_dir = tile_dir / dataset_name
    slide_tile_dir = dataset_tile_dir / slide_id
    return {
        "dataset_mask_dir": dataset_mask_dir,
        "dataset_tile_dir": dataset_tile_dir,
        "mask": dataset_mask_dir / f"{slide_id}_tissue_mask.png",
        "overlay": dataset_mask_dir / f"{slide_id}_tissue_overlay.png",
        "slide_tile_dir": slide_tile_dir,
        "metadata": slide_tile_dir / f"{slide_id}_tile_metadata.csv",
        "summary": slide_tile_dir / f"{slide_id}_tiling_summary.json",
    }


def tile_one_slide(
    slide_path: Path,
    *,
    mask_dir: Path,
    tile_dir: Path,
    dataset_name: str,
    mask_max_size: int,
    mask_saturation: float,
    mask_value: float,
    target_mpp: float,
    target_tile_px: int,
    min_tissue: float,
    level: int,
    jpeg_quality: int,
) -> dict:
    """Mask and tile one slide, skipping whichever half is already done.

    Raises if the slide still is not completely tiled afterwards, so a caller
    that returns normally has a slide tiling_output_complete() accepts — the
    same test packaging's coverage check and a resubmission's skip use.
    """
    from tile_mask import run_tissue_detection
    from auto_tile_from_mask import tile_slide_from_mask

    slide_id = slide_id_from_raw_path(slide_path)
    paths = slide_output_paths(slide_path, mask_dir, tile_dir, dataset_name)
    paths["dataset_mask_dir"].mkdir(parents=True, exist_ok=True)
    paths["dataset_tile_dir"].mkdir(parents=True, exist_ok=True)
    mask_path, overlay_path = paths["mask"], paths["overlay"]

    if mask_path.stat().st_size > 0 if mask_path.exists() else False:
        mask_ok = overlay_path.stat().st_size > 0 if overlay_path.exists() else False
    else:
        mask_ok = False

    if mask_ok:
        print("[SKIP] Mask and overlay already exist.", flush=True)
    else:
        print("[RUN] Creating tissue mask...", flush=True)
        run_tissue_detection(
            slide_path=str(slide_path),
            output_dir=str(paths["dataset_mask_dir"]),
            max_size=mask_max_size,
            saturation_threshold=mask_saturation,
            value_threshold=mask_value,
        )

    if not mask_path.exists() or mask_path.stat().st_size == 0:
        raise RuntimeError(f"Expected mask was not created: {mask_path}")

    tile_complete = tiling_output_complete(
        paths["metadata"], paths["summary"], paths["slide_tile_dir"]
    )

    if tile_complete:
        print("[SKIP] Slide already tiled.", flush=True)
    else:
        print("[RUN] Tiling slide from mask...", flush=True)
        tile_slide_from_mask(
            slide_path=str(slide_path),
            mask_path=str(mask_path),
            output_dir=str(paths["dataset_tile_dir"]),
            target_mpp=target_mpp,
            target_tile_px=target_tile_px,
            min_tissue_percent=min_tissue,
            level=level,
            jpeg_quality=jpeg_quality,
        )
        # Checked again rather than assumed. The tiler returning is not the
        # same claim as the slide being complete, and the next stage packages
        # whatever the metadata CSV promises.
        if not tiling_output_complete(
            paths["metadata"], paths["summary"], paths["slide_tile_dir"]
        ):
            raise RuntimeError(
                f"Tiling {slide_id} returned, but its output is not complete: "
                f"{paths['summary']} and {paths['metadata']} disagree or are missing."
            )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(f"[DONE] {slide_id}", flush=True)
    return {
        "slide_id": slide_id,
        "slide_path": str(slide_path),
        "saved_tiles": int(summary.get("saved_tiles", 0)),
        "skipped": tile_complete,
    }


def run_worker(args: argparse.Namespace) -> None:
    """Mask and tile the slide assigned to this Slurm array task."""
    task_id_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not defined")

    task_id = int(task_id_text)
    slide_path = read_manifest_slide(args.manifest, task_id)
    slide_id = slide_id_from_raw_path(slide_path)

    print("=" * 70, flush=True)
    print(f"Job ID:      {os.environ.get('SLURM_JOB_ID', 'unknown')}", flush=True)
    print(f"Array task:  {task_id}", flush=True)
    print(f"Slide:       {slide_path}", flush=True)
    print(f"Slide ID:    {slide_id}", flush=True)
    print(f"Dataset:     {args.dataset_name}", flush=True)
    print("=" * 70, flush=True)

    tile_one_slide(
        slide_path,
        mask_dir=args.mask_dir,
        tile_dir=args.tile_dir,
        dataset_name=args.dataset_name,
        mask_max_size=args.mask_max_size,
        mask_saturation=args.mask_saturation,
        mask_value=args.mask_value,
        target_mpp=args.target_mpp,
        target_tile_px=args.target_tile_px,
        min_tissue=args.min_tissue,
        level=args.level,
        jpeg_quality=args.jpeg_quality,
    )


def _run_sbatch_with_retry(
    sbatch_command: list[str], max_attempts: int = 6, initial_backoff_seconds: float = 10.0
) -> subprocess.CompletedProcess:
    """Run sbatch, retrying on Slurm's own transient "temporarily unable to
    accept job" / "resource temporarily unavailable" controller-busy
    message — this is what actually happened submitting ~14,000 slides in
    one array, and kept happening even after splitting into ~1000-slide
    batches, meaning the congestion is more sustained than a brief blip.
    A real rejection (invalid partition, bad GRES, etc.) won't contain that
    phrase and is raised immediately instead of wasting time retrying
    something that will never succeed.

    Backoff doubles each attempt (10s, 20s, 40s, 80s, 160s — about 5 minutes
    total across 6 attempts) rather than a fixed short delay, to actually
    ride out sustained controller congestion instead of just a momentary one.
    """
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return subprocess.run(sbatch_command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").lower()
            transient = "temporarily unable to accept job" in stderr or "resource temporarily unavailable" in stderr
            if not transient:
                raise
            last_error = e
            if attempt < max_attempts:
                time.sleep(initial_backoff_seconds * (2 ** (attempt - 1)))
    raise last_error


def _submit_one_array(
    slides: list[Path],
    *,
    manifest_dir: Path,
    log_dir: Path,
    backend_dir: Path,
    script_path: Path,
    python_executable: Path,
    mask_dir: Path,
    tile_dir: Path,
    dataset_name: str,
    max_concurrent: int,
    partition: str | None,
    cpus: int,
    memory: str,
    time_limit: str,
    job_name: str,
    mask_max_size: int,
    mask_saturation: float,
    mask_value: float,
    target_mpp: float,
    target_tile_px: int,
    min_tissue: float,
    level: int,
    jpeg_quality: int,
    notify_email: str | None,
    manifest_suffix: str,
    dry_run: bool,
) -> dict:
    """Write one manifest and submit one Slurm array for it. Internal helper
    — submit_array() calls this once per batch."""
    manifest_path = manifest_dir / f"wsi_manifest_{manifest_suffix}.txt"
    write_manifest(slides, manifest_path)

    worker_command = [
        str(python_executable),
        str(script_path),
        "--worker",
        "--manifest", str(manifest_path),
        "--mask-dir", str(mask_dir),
        "--tile-dir", str(tile_dir),
        "--dataset-name", dataset_name,
        "--mask-max-size", str(mask_max_size),
        "--mask-saturation", str(mask_saturation),
        "--mask-value", str(mask_value),
        "--target-mpp", str(target_mpp),
        "--target-tile-px", str(target_tile_px),
        "--min-tissue", str(min_tissue),
        "--level", str(level),
        "--jpeg-quality", str(jpeg_quality),
    ]

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        *([f"--partition={partition}"] if partition else []),
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--array=0-{len(slides) - 1}%{max_concurrent}",
        f"--output={log_dir}/mask_tile_%A_%a.out",
        f"--error={log_dir}/mask_tile_%A_%a.err",
        f"--chdir={backend_dir}",
        # Without ARRAY_TASKS, Slurm sends one summary email for the whole
        # array on END/FAIL rather than one per slide.
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        "--wrap",
        shlex.join(worker_command),
    ]

    batch_info = {
        "manifest_path": str(manifest_path),
        "slides_in_batch": len(slides),
        "sbatch_command": shlex.join(sbatch_command),
        "job_id": None,
    }

    if dry_run:
        return batch_info

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as e:
        # CalledProcessError's default message ("returned non-zero exit
        # status 1") drops sbatch's actual stderr, which is the only useful
        # part — surface it explicitly instead.
        reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output from sbatch"
        raise RuntimeError(f"sbatch failed (exit {e.returncode}): {reason}") from e

    stdout = result.stdout.strip()
    batch_info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        batch_info["job_id"] = match.group(1)
    return batch_info


def submit_array(
    raw_dir: Path,
    mask_dir: Path,
    tile_dir: Path,
    *,
    max_concurrent: int = 12,
    partition: str | None = None,
    # One CPU per task, because tiling is single-threaded: neither
    # auto_tile_from_mask.py nor tile_mask.py uses a thread pool, a process
    # pool, or any cpu_count-driven parallelism, so the previous default of 4
    # held four cores per slide to keep one busy. Beyond the waste, a 1-CPU
    # task is far easier for the scheduler to backfill, so more of them run
    # concurrently for the same share of the partition — which matters more
    # for wall-clock than per-task speed, since throughput here is bounded by
    # how many slides run at once, not how fast any one of them goes.
    cpus: int = 1,
    memory: str = "24G",
    time_limit: str = "2-00:00:00",
    job_name: str = "wsi_mask_tile",
    mask_max_size: int = 2000,
    mask_saturation: float = 0.08,
    mask_value: float = 0.95,
    target_mpp: float = 1.8,
    target_tile_px: int = 224,
    min_tissue: float = 30.0,
    level: int = 0,
    jpeg_quality: int = 90,
    sample_size: int | None = None,
    slide_names: list[str] | None = None,
    random_seed: int | None = None,
    notify_email: str | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
    dataset_name: str | None = None,
    on_planned: Callable[[dict], None] | None = None,
) -> dict:
    """Discover slides under raw_dir, write a manifest, and submit a Slurm array.

    Importable so both the CLI and the tile server's dataset-job endpoints
    can trigger a run the same way. Returns a dict describing the submission;
    "job_ids" is empty when dry_run is True.

    dataset_name controls the folder tiles land in under tile_dir (so TCGA
    and Radiogenomics slides don't mix). Defaults to raw_dir's own folder
    name; pass it explicitly to reuse an existing dataset folder or name a
    new one that doesn't match raw_dir (e.g. the UI letting someone pick
    "TCGA" regardless of what the raw upload directory happens to be called).

    By default every discovered slide is submitted. Pass sample_size to
    randomly submit only that many, or slide_names (filenames or slide_ids)
    to submit only specific slides — see select_slides().

    A single sbatch call building thousands of array-task records at once
    can trip Slurm's controller ("temporarily unable to accept job" — this
    is what actually happened submitting ~14,000 slides in one array).
    Above batch_size, slides get split into multiple smaller array
    submissions instead of one giant one — each batch gets its own manifest
    and job_id, but a single combined manifest listing every selected slide
    is also written (manifest_path) so downstream packaging can still treat
    the whole run as one dataset regardless of how many batches it took.

    on_planned, if given, is called once with {"manifest_path",
    "dataset_name", "slides_found", "selected_slide_ids"} right after the
    combined manifest is written but before any sbatch call — i.e. before
    the part of this function that can take minutes (each batch retries
    through Slurm controller congestion with exponential backoff, see
    _run_sbatch_with_retry, and a large dataset can mean several batches).
    A caller tracking this submission in its own store (e.g. a DB row) can
    use this to persist the plan immediately, so a crash partway through a
    multi-batch submission — some jobs actually submitted to Slurm, this
    function never returning to report their IDs — doesn't also lose the
    manifest itself. Without it, the only thing pointing back at a crashed
    run's now-orphaned Slurm jobs is whatever the caller manages to
    reconstruct after the fact.
    """
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {raw_dir}")

    # Tiles land under tile_dir/<dataset_name>/<slide_id>/ rather than flat
    # under tile_dir/<slide_id>/, so TCGA and Radiogenomics (and any other
    # dataset) stay in their own folder instead of mixed together. Falls back
    # to raw_dir's own folder name when the caller doesn't pass one — the
    # same convention packaging already uses for the .h5 output folder.
    dataset_name = (dataset_name or "").strip() or raw_dir.name

    script_path = Path(__file__).resolve()
    backend_dir = script_path.parent
    manifest_dir = backend_dir / "slurm_manifests"
    log_dir = backend_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_slides = discover_slides(raw_dir)
    if not all_slides:
        raise FileNotFoundError(f"No supported WSI files found under: {raw_dir}")

    slides = select_slides(
        all_slides,
        sample_size=sample_size,
        slide_names=slide_names,
        random_seed=random_seed,
    )
    if not slides:
        raise ValueError("No slides left to submit after filtering.")
    validate_unique_slide_ids(slides)

    mask_dir = mask_dir.expanduser().resolve()
    tile_dir = tile_dir.expanduser().resolve()
    python_executable = Path(sys.executable).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combined manifest listing every selected slide, regardless of batching
    # — this is what packaging reads later, so the .h5 output stays "one
    # dataset in, one file out" even when tiling itself took several
    # separate array submissions.
    combined_manifest_path = manifest_dir / f"wsi_manifest_{timestamp}.txt"
    write_manifest(slides, combined_manifest_path)

    if on_planned:
        on_planned({
            "manifest_path": str(combined_manifest_path),
            "dataset_name": dataset_name,
            "slides_found": len(slides),
            "selected_slide_ids": [slide_id_from_raw_path(s) for s in slides],
        })

    batches = [slides[i:i + batch_size] for i in range(0, len(slides), batch_size)]

    # Slurm's "%N" array throttle applies per array job, not across a whole
    # submission — so with the limit passed through unchanged to every batch,
    # max_concurrent=12 across 14 batches meant up to 168 concurrent tasks,
    # not 12. The effective ceiling silently scaled with dataset size (and,
    # multiplied by cpus-per-task, with the core count requested), which made
    # it both unpredictable and a good way to starve a shared partition.
    # Divide it instead, so max_concurrent means what it says: the total
    # number of tiling tasks running at once across every batch. Floor of 1
    # per batch, since 0 would throttle a batch to nothing.
    per_batch_concurrent = max(1, max_concurrent // len(batches))

    # A brief pause between submissions, on top of the retry-with-backoff
    # inside each one — firing many sbatch calls back-to-back with zero
    # gap can itself look like a burst to the controller, separate from how
    # big any single array is.
    inter_batch_delay_seconds = 5.0

    batch_results = []
    for i, batch_slides in enumerate(batches):
        if i > 0 and not dry_run:
            time.sleep(inter_batch_delay_seconds)

        suffix = f"{timestamp}_batch{i + 1}of{len(batches)}" if len(batches) > 1 else timestamp
        try:
            batch_result = _submit_one_array(
                batch_slides,
                manifest_dir=manifest_dir,
                log_dir=log_dir,
                backend_dir=backend_dir,
                script_path=script_path,
                python_executable=python_executable,
                mask_dir=mask_dir,
                tile_dir=tile_dir,
                dataset_name=dataset_name,
                max_concurrent=per_batch_concurrent,
                partition=partition,
                cpus=cpus,
                memory=memory,
                time_limit=time_limit,
                job_name=f"{job_name}_{i + 1}" if len(batches) > 1 else job_name,
                mask_max_size=mask_max_size,
                mask_saturation=mask_saturation,
                mask_value=mask_value,
                target_mpp=target_mpp,
                target_tile_px=target_tile_px,
                min_tissue=min_tissue,
                level=level,
                jpeg_quality=jpeg_quality,
                notify_email=notify_email,
                manifest_suffix=suffix,
                dry_run=dry_run,
            )
        except (RuntimeError, subprocess.CalledProcessError) as e:
            # One batch failing even after retries shouldn't lose whatever
            # other batches already submitted successfully — record the
            # failure and keep going, rather than aborting the whole run.
            batch_result = {
                "manifest_path": None,
                "slides_in_batch": len(batch_slides),
                "job_id": None,
                "error": str(e),
            }
        batch_results.append(batch_result)

    job_ids = [b["job_id"] for b in batch_results if b.get("job_id")]
    failed_batches = [b for b in batch_results if b.get("error")]

    return {
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        "total_discovered": len(all_slides),
        "slides_found": len(slides),
        "selected_slide_ids": [slide_id_from_raw_path(s) for s in slides],
        "manifest_path": str(combined_manifest_path),
        "mask_dir": str(mask_dir),
        "tile_dir": str(tile_dir),
        "max_concurrent": max_concurrent,
        # What each array job was actually throttled to, so the effective
        # total (this x batch_count) is visible rather than having to be
        # inferred from the sbatch command.
        "per_batch_concurrent": per_batch_concurrent,
        "cpus_per_task": cpus,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "batches": batch_results,
        "failed_batch_count": len(failed_batches),
        "dry_run": dry_run,
        "job_ids": job_ids,
    }


def submit_packaging_job(
    manifest_path: Path,
    tile_dir: Path,
    dataset_name: str,
    tile_dataset_name: str,
    *,
    depends_on_job_ids: list[str],
    output_root: Path | None = None,
    marker: str = "he",
    split: str = "train",
    tile_size: int = 224,
    partition: str | None = None,
    cpus: int = 8,
    memory: str = "32G",
    time_limit: str = "2-00:00:00",
    threads_per_process: int = 4,
    job_name: str = "hpl_h5_package",
    notify_email: str | None = None,
    sample_size: int | None = None,
    slide_names: list[str] | None = None,
    random_seed: int | None = None,
    allow_incomplete: bool = False,
    # None leaves make_hpl_hdf5.py to continue a checkpoint if it finds one
    # (its historical behaviour). True requires one and fails without it; False
    # discards it and repackages from scratch. Passed through as an explicit
    # flag so the decision is the caller's — and visible in the recorded sbatch
    # command — rather than an implicit consequence of what's on disk.
    resume: bool | None = None,
) -> dict:
    """Submit a Slurm job that packages a tiling run's output into the .h5
    format Kai's HPL pipeline expects, deferred via Slurm's own --dependency
    mechanism until every tiling task across every batch has finished.
    depends_on_job_ids can be one job (a small dataset, single array) or
    several (a large dataset split into multiple array submissions) — Slurm's
    dependency syntax accepts multiple colon-separated job IDs natively,
    waiting on all of them.

    Defaults to afterok: packaging runs only if every tiling task actually
    succeeded. This used to be afterany, justified by not wanting a slide that
    legitimately produced zero tiles to block packaging everything else — but
    that justification doesn't hold, because run_worker() exits 0 for a
    zero-tile slide (it writes an empty metadata CSV and a summary, then
    returns normally). It only exits non-zero when tiling genuinely failed.
    So afterany wasn't protecting the zero-tile case at all; it was silently
    packaging datasets with real, unnoticed holes in them — a slide killed by
    OOM or the array time limit contributed nothing to the .h5, and nothing
    downstream ever said so.

    Pass allow_incomplete=True to go back to afterany when packaging a
    knowingly-partial dataset is the actual intent (e.g. a test sample, or
    accepting a handful of dead slides in a 14,000-slide run rather than
    re-running them). Making that an explicit choice is the point.

    --kill-on-invalid-dep=yes is set because afterok makes an unsatisfiable
    dependency reachable: if any tiling task fails, this job's dependency can
    never be met, and Slurm's default is to leave it sitting in the queue
    indefinitely as DependencyNeverSatisfied rather than telling anyone. This
    makes it fail visibly instead.

    dataset_name and tile_dataset_name are deliberately separate: a subset
    run (random sample / specific slides) gets its own "_subset_N"-suffixed
    dataset_name so its .h5 doesn't clobber a previous full run's — but its
    tiles were still written under the plain, unsuffixed dataset folder
    (tile_dataset_name), since tiling doesn't know or care that this run is
    a subset. Passing dataset_name to both would make packaging look for
    tiles in a folder that was never created.

    time_limit defaults to 48h (was 4h, then 24h) — a ~14,000-slide dataset
    hit TIMEOUT at 4h (back when packaging was single-threaded), which
    SLURM enforces with SIGTERM mid-write, leaving a truncated/corrupt .h5
    that then fails with an HDF5-level error the next time anything tries
    to read it. A later attempt was still running past 19h even with the
    process-pool version, so 24h wasn't leaving enough margin either.
    Still override this if 48h isn't enough, or if your cluster's
    partition caps below that.

    depends_on_job_ids may be empty, which submits with no --dependency at
    all. That is not a way to skip the afterok safety check — it is for the
    case where tiling has *already* finished, so there is nothing left to
    wait on. It matters because sbatch resolves dependencies against
    slurmctld, which forgets a job MinJobAge seconds after it completes
    (default 300), while sacct keeps it for weeks. So a caller can confirm
    via sacct that every tiling task COMPLETED and still have sbatch reject
    the submission with "Job dependency problem", because slurmctld no
    longer recognises those IDs. With --kill-on-invalid-dep=yes that lands
    as a submit-time failure rather than a job that queues forever, which is
    the right behaviour for a dependency that might yet be satisfied and the
    wrong one for a dependency that is already moot. Deciding whether the
    tiling jobs are terminal is the caller's job, not this function's — it
    needs sacct state the caller has usually already fetched.
    """

    script_path = Path(__file__).resolve()
    backend_dir = script_path.parent
    log_dir = backend_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if sample_size is not None or slide_names:
        # Package only a subset of the run's manifest — e.g. to test
        # packaging (and a downstream feature-extraction checkpoint)
        # against a handful of slides before committing to a multi-hour
        # run over the whole dataset. dataset_name should already reflect
        # that this is a sample (the caller's job, not this function's —
        # e.g. "<name>_test_sample") so its .h5 never collides with the
        # real run's.
        all_slides = [
            Path(line.strip())
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        selected = select_slides(
            all_slides, sample_size=sample_size, slide_names=slide_names, random_seed=random_seed,
        )
        if not selected:
            raise ValueError("No slides left to package after filtering.")
        manifest_dir = backend_dir / "slurm_manifests"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = manifest_dir / f"wsi_manifest_{timestamp}_package_sample.txt"
        write_manifest(selected, manifest_path)

    if output_root is None:
        output_root = backend_dir.parent / "model_input"

    python_executable = Path(sys.executable).resolve()

    package_command = [
        str(python_executable),
        str(backend_dir / "make_hpl_hdf5.py"),
        "--manifest", str(manifest_path),
        "--tile-dir", str(tile_dir),
        "--tile-dataset-name", tile_dataset_name,
        "--output-root", str(output_root),
        "--dataset-name", dataset_name,
        "--marker", marker,
        "--split", split,
        "--tile-size", str(tile_size),
        # Match the worker-process count to what's actually being
        # requested from Slurm below, so the process pool has real cores
        # to run on instead of oversubscribing (or undersubscribing) cpus.
        "--processes", str(cpus),
        "--threads-per-process", str(threads_per_process),
        *([] if resume is None else ["--resume"] if resume else ["--fresh"]),
    ]

    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        *([f"--partition={partition}"] if partition else []),
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        *([
            f"--dependency={'afterany' if allow_incomplete else 'afterok'}:{':'.join(depends_on_job_ids)}",
            "--kill-on-invalid-dep=yes",
        ] if depends_on_job_ids else []),
        f"--output={log_dir}/hpl_h5_%j.out",
        f"--error={log_dir}/hpl_h5_%j.err",
        f"--chdir={backend_dir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"] if notify_email else []),
        "--wrap", shlex.join(package_command),
    ]

    info = {
        "h5_output_path": str(hpl_h5_output_path(output_root, dataset_name, marker, split, tile_size)),
        "sbatch_command": shlex.join(sbatch_command),
        "h5_job_id": None,
        # None, not the afterok/afterany the flags *would* have used, when no
        # dependency was actually set — this is what a caller inspects to see
        # whether the job is gated on anything, and reporting a gate that
        # isn't on the sbatch line would misreport an immediately-eligible
        # job as still waiting on tiling.
        "dependency_type": (
            ("afterany" if allow_incomplete else "afterok") if depends_on_job_ids else None
        ),
        "depends_on_job_ids": list(depends_on_job_ids),
        "allow_incomplete": allow_incomplete,
    }

    try:
        result = _run_sbatch_with_retry(sbatch_command)
    except subprocess.CalledProcessError as e:
        reason = (e.stderr or "").strip() or (e.stdout or "").strip() or "no output from sbatch"
        raise RuntimeError(f"sbatch failed (exit {e.returncode}): {reason}") from e

    stdout = result.stdout.strip()
    info["sbatch_stdout"] = stdout
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info["h5_job_id"] = match.group(1)
    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a Slurm array to mask and tile a complete WSI dataset."
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    # Populated automatically from raw_dir.name by submit_array() and passed
    # to worker mode via --wrap; not meant to be set by hand.
    parser.add_argument("--dataset-name", type=str, help=argparse.SUPPRESS)

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi/raw"),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks"),
    )
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=Path("/hpc-home/home/users/vpandya/long-term-scratch/processed_tiles"),
    )

    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument(
        "--partition", default=None,
        help="Omit to let Slurm use the cluster's default partition.",
    )
    # Matches submit_array()'s default — tiling is single-threaded, see there.
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory", default="24G")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--job-name", default="wsi_mask_tile")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--mask-max-size", type=int, default=2000)
    parser.add_argument("--mask-saturation", type=float, default=0.08)
    parser.add_argument("--mask-value", type=float, default=0.95)
    parser.add_argument("--target-mpp", type=float, default=1.8)
    parser.add_argument("--target-tile-px", type=int, default=224)
    parser.add_argument("--min-tissue", type=float, default=30.0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=90)

    parser.add_argument(
        "--dataset-folder-name", type=str, default=None, dest="dataset_folder_name",
        help="Folder tiles land in under --tile-dir (e.g. TCGA, Radiogenomics). "
             "Defaults to --raw-dir's own folder name.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Randomly submit only this many slides instead of the whole dataset.",
    )
    parser.add_argument(
        "--slide-names", default=None,
        help="Comma-separated filenames or slide IDs to submit specific slides only.",
    )
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--notify-email", default=None,
        help="Email Slurm's own END/FAIL notification to this address when the array finishes.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Split into multiple Slurm array submissions above this many slides "
             "(a single very large array can trip Slurm's controller).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.worker:
        if args.manifest is None:
            parser.error("--manifest is required in worker mode")
        if args.dataset_name is None:
            parser.error("--dataset-name is required in worker mode")
        run_worker(args)
        return

    slide_names = (
        [name.strip() for name in args.slide_names.split(",") if name.strip()]
        if args.slide_names else None
    )

    info = submit_array(
        raw_dir=args.raw_dir,
        mask_dir=args.mask_dir,
        tile_dir=args.tile_dir,
        max_concurrent=args.max_concurrent,
        partition=args.partition,
        cpus=args.cpus,
        memory=args.memory,
        time_limit=args.time_limit,
        job_name=args.job_name,
        mask_max_size=args.mask_max_size,
        mask_saturation=args.mask_saturation,
        mask_value=args.mask_value,
        target_mpp=args.target_mpp,
        target_tile_px=args.target_tile_px,
        min_tissue=args.min_tissue,
        level=args.level,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.sample_size,
        slide_names=slide_names,
        random_seed=args.random_seed,
        notify_email=args.notify_email,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        dataset_name=args.dataset_folder_name,
    )

    print(f"Dataset:          {info['raw_dir']}")
    print(f"Dataset folder:   {info['dataset_name']}")
    print(f"Slides discovered:{info['total_discovered']}")
    print(f"Slides submitted: {info['slides_found']}")
    print(f"Combined manifest:{info['manifest_path']}")
    print(f"Mask directory:   {info['mask_dir']}")
    print(f"Tile directory:   {info['tile_dir']}")
    print(f"Concurrent tasks: {info['max_concurrent']}")
    print(f"Batches:          {info['batch_count']} (batch size {info['batch_size']})")

    for i, batch in enumerate(info["batches"]):
        print(f"\n--- Batch {i + 1}/{info['batch_count']}: {batch['slides_in_batch']} slides ---")
        if batch.get("error"):
            print(f"FAILED: {batch['error']}")
            continue
        print(f"Manifest: {batch['manifest_path']}")
        if info["dry_run"]:
            print("Dry run; Slurm command:")
            print(batch["sbatch_command"])
        else:
            print(batch["sbatch_stdout"])

    if not info["dry_run"]:
        print(f"\nSucceeded: {len(info['job_ids'])}/{info['batch_count']} batches")
        print(f"Job IDs: {', '.join(info['job_ids'])}")
        if info["failed_batch_count"]:
            print(f"WARNING: {info['failed_batch_count']} batch(es) failed to submit — see above.")


if __name__ == "__main__":
    main()
