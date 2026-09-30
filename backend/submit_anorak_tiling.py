#!/usr/bin/env python3
"""Submit a Slurm array that tiles selected slides for ANORAK.

ANORAK (Pan et al., Nature Cancer 2024; https://github.com/xi11/AIgrading) does
its own tiling, at its own resolution, into its own layout. Its
`generating_tile/main_tiles.py` globs a directory and loops over every slide in
it, in one process, with no resume and no scheduler. This wraps that same code
in the shape the rest of this pipeline uses: a slide list, one array task per
slide, a completeness check that lets a resubmission skip what is already whole.

    python submit_anorak_tiling.py \
        --slides-csv radiogenomics_tumour_slides_min10.csv \
        --raw-dir /mnt/cephfs-lts/long-term-scratch/users/vpandya/Radiogenomics \
        --out-dir /mnt/cephfs-lts/long-term-scratch/users/vpandya/anorak/Radiogenomics \
        --anorak-dir /mnt/cephfs-lts/long-term-scratch/users/vpandya/AIgrading \
        --dry-run

Worker mode is invoked by Slurm, not by hand.

**The slide list is the point.** ANORAK only means anything on tumour, and
`select_tumour_slides.py` + `filter_slides_by_tile_count.py` are what say which
slides those are. Pointing upstream's globbing script at the cohort directory
would tile all 14,042 — roughly twice the slides, for an answer nobody reads on
the ones that carry no malignant tissue.

**`--output-mpp` is not the output resolution; twice it is.** Upstream's flag is
named for the scanner's ×40 reference, and `cws_generator.py` works out
`cws_objective_value = 20·(objective_power/40)·(in_mpp/out_mpp)`, then rescales
by `objective_power/cws_objective_value`. Multiply it through and the effective
output is exactly `2 × out_mpp` for any scanner — so upstream's 0.22 default
produces 0.44 um/px (×20), which is what the paper's inference section
specifies ("2,000 × 2,000 pixels … downsampled to ×20, approximately 0.45 um per
pixel"). The number is printed at submit time, resolved rather than quoted,
because a factor of two here is a whole extra magnification step and it would
survive to the end looking like nothing at all.

**What is refused, and why here rather than later.** A slide whose mpp
openslide cannot report does not fail upstream: `cws_generator.py:41-61` catches
it, warns, and falls back to plain objective-power scaling — a slide tiled at a
silently different resolution to the rest of the cohort, in the right layout,
with the right file names, which then gets segmented by a model trained at ×20.
The worker refuses such a slide outright, and `--metadata-sample` checks a
sample at submit time so a systematic problem (a cohort with no mpp in its
headers at all) surfaces before 7,000 tasks are queued rather than inside 7,000
logs. An unsupported extension is refused for the same reason: upstream's
`save_cws.single_file_run` dispatches on file type and silently does nothing at
all for one it does not recognise.

The output layout is upstream's and is deliberately left alone —
`<out-dir>/cws_tiling/<full filename with extension>/Da0.jpg`, beside
`Ss1.jpg`, `param.p` and `FinalScan.ini`. `inference_slide/main_gp.py` finds
slides by globbing that directory with a `*.svs`-style pattern and stitches
masks back by tile index, so the directory name carrying the extension, and the
tile numbering being dense from zero, are both load-bearing.

**Not the layout the pipeline uses, so its output is never reused.** The
Nextflow pipeline (`anorak-nf/main.nf`, via `bin/anorak_tile.py`) publishes
`<outdir>/cws_tiling/<slide_id>/<file with extension>/` — one level deeper,
keyed by slide id — while this script writes `<out-dir>/cws_tiling/<file>/`.
Nothing reads this script's output: pointing the pipeline at the same outdir
re-tiles every slide, and nothing calls this script. It is kept as a standalone
tiler; if the pipeline is the only route to ANORAK, it is a candidate for
deletion rather than for a second layout.

**What "already tiled" means.** A slide is skipped only if its directory holds
`anorak_tiling_complete.json` (`COMPLETION_RECORD`), written last, under a
`.tmp` name and renamed, and only after the tiles were counted — and only if
that record names the same `--output-mpp` and slide geometry as this attempt
and the directory holds *exactly* the tile count this attempt expects. Upstream
writes no parameters anywhere this can check, so without the record a slide
tiled at 0.22 was "already tiled" for a resubmission at 0.25; and because its
marker files survive a re-tile killed half-way, a mix of old and new tiles
looked finished too. Anything short of a matching record is re-tiled from an
emptied directory, which includes output written before the record existed.

**Concurrency is across the whole submission.** Batches are chained with
`--dependency=afterany`, so `--max-concurrent` is the number of slides being
tiled at once in total, not per batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slide_naming import slide_id_from_raw_path  # noqa: E402
from submit_mask_tile_slurm import _run_sbatch_with_retry  # noqa: E402

#: Extensions upstream's `save_cws.single_file_run` actually dispatches on.
#: Anything else reaches it and returns having done nothing, so the list is
#: copied here to refuse instead. Keep it in step with that function.
SUPPORTED_EXTENSIONS = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".png", ".qptiff"}

#: Upstream's own default, and the paper's inference resolution once doubled
#: (see the module docstring). Changing it changes the magnification the model
#: sees, which is a modelling decision, not a tuning knob.
DEFAULT_OUTPUT_MPP = 0.22

#: Per-slide files upstream writes after the tiles, in `cws_all_process` order.
#: `param.p` is written by the first step that runs after every tile is saved,
#: which is what makes it usable as a finished marker — but only alongside the
#: tile count, see `tiling_output_complete`.
_COMPLETION_MARKERS = ("param.p", "Ss1.jpg", "FinalScan.ini")

#: Ours, not upstream's: what this slide was tiled with, written only once the
#: tiles have been counted. See the module docstring's "already tiled".
COMPLETION_RECORD = "anorak_tiling_complete.json"

#: The fields of the record that must match for a slide to count as done at
#: this attempt's settings. `slide` (the path) is recorded but deliberately not
#: compared, so a raw directory that moved still reuses its tiles.
_RECORD_MATCH_FIELDS = ("slide_name", "output_mpp", "objective_power", "in_mpp",
                        "slide_size", "cws_read_size", "expected_tiles")


# --- resolving the slide list ---------------------------------------------

def read_slide_ids(slides_csv: Path, column: str = "slide_id") -> list[str]:
    """The slide ids in a slide list, in file order, exactly as written.

    Read as text: pandas' type guessing turns `00123` into `123` and `NA` into
    a blank, and either is then a slide that resolves to nothing — or to a
    different file. A blank id is refused rather than skipped, since it is a
    row that meant some slide. Duplicates are not collapsed here; they are
    refused by resolve_slides(), on the files they name, where "the same
    slide" can actually be decided.
    """
    frame = pd.read_csv(slides_csv, dtype=str, keep_default_na=False)
    if column not in frame.columns:
        raise ValueError(
            f"{slides_csv} has no '{column}' column; got "
            f"{', '.join(frame.columns)}"
        )
    ids = [value.strip() for value in frame[column]]
    blank = [i for i, sid in enumerate(ids) if not sid]
    if blank:
        raise ValueError(
            f"{slides_csv} has {len(blank)} row(s) with a blank '{column}' "
            f"(data rows {[i + 1 for i in blank[:5]]}). Refusing rather than "
            f"skipping them: each is a slide the list meant, and the cohort "
            f"would be short by it."
        )
    if not ids:
        raise ValueError(f"{slides_csv} lists no slides")
    return ids


def index_raw_slides(raw_dir: Path) -> dict[str, list[Path]]:
    """Every candidate key -> the slide files answering to it.

    Three keys per file — full name, stem, and the repo's derived slide_id —
    so a list exported from the KB, from a file browser, or from
    `select_tumour_slides.py` all resolve without the caller knowing which
    convention produced it. Values stay lists so an ambiguous key is
    detectable rather than resolved by whichever file was walked first.
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        for key in {path.name, path.stem, slide_id_from_raw_path(path)}:
            index.setdefault(key, []).append(path)
    return index


def resolve_slides(slide_ids: list[str], raw_dir: Path) -> list[Path]:
    """Slide files for `slide_ids`, in list order. Refuses on any miss.

    A partial resolution is the failure this guards: 7,000 of 7,221 slides
    tiled is a complete-looking cohort whose missing fifth only shows up as a
    thinner grading distribution months later. Ambiguity is refused for the
    same reason — two files answering to one id means one of them silently
    wins, and which one is a property of directory order.
    """
    index = index_raw_slides(raw_dir)
    if not index:
        raise ValueError(
            f"No slides with a supported extension under {raw_dir} "
            f"(looked for {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    resolved: list[Path] = []
    missing: list[str] = []
    ambiguous: dict[str, list[Path]] = {}
    for slide_id in slide_ids:
        matches = index.get(slide_id)
        if not matches:
            missing.append(slide_id)
        elif len({m.resolve() for m in matches}) > 1:
            ambiguous[slide_id] = matches
        else:
            resolved.append(matches[0])

    if missing:
        shown = "\n".join(f"    {sid}" for sid in missing[:10])
        more = f"\n    ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} of {len(slide_ids)} slides in the list have no file "
            f"under {raw_dir}:\n{shown}{more}\n"
            f"Refusing rather than tiling the {len(resolved)} that resolved — a "
            f"cohort short by a slice looks exactly like a complete one."
        )
    if ambiguous:
        shown = "\n".join(
            f"    {sid}:\n" + "\n".join(f"      - {p}" for p in paths)
            for sid, paths in list(ambiguous.items())[:5]
        )
        raise ValueError(
            f"{len(ambiguous)} slide ids match more than one file:\n{shown}\n"
            f"These would tile into the same output folder and overwrite each "
            f"other."
        )

    # Two ids for one slide — "X" and "X.ndpi", or the same id twice — used to
    # pass, because ids were compared as text. Each became its own array task
    # writing the same output directory at the same time. Refused, not
    # collapsed, as main.nf's resolveSlides refuses it: a list that names a
    # slide twice was not built the way anyone thinks it was. Keyed on the
    # output directory as well as the file, since two different files with
    # the same name in different subdirectories collide there too.
    by_target: dict[tuple, list[str]] = {}
    for slide_id, path in zip(slide_ids, resolved):
        by_target.setdefault(("file", path.resolve()), []).append(slide_id)
        by_target.setdefault(("output", path.name), []).append(slide_id)
    collisions = {key: ids for key, ids in by_target.items() if len(ids) > 1}
    if collisions:
        shown = "\n".join(
            f"    {'file' if kind == 'file' else 'output folder'} {target}: "
            f"listed as {ids}"
            for (kind, target), ids in list(collisions.items())[:5]
        )
        raise ValueError(
            f"{len(collisions)} slide(s) are listed more than once:\n{shown}\n"
            f"Each listing would be its own array task writing the same output "
            f"folder concurrently. Remove the duplicates from the list."
        )
    return resolved


# --- what upstream will produce, and whether it already did ----------------

#: Upstream's `cws_read_size`, square, in output pixels.
_CWS_READ_SIZE = 2000


def tiling_geometry(slide_path: Path, output_mpp: float) -> dict:
    """Everything this attempt's tiles depend on, and how many there will be.

    The count is recomputed from the slide's own header by the same arithmetic
    as `cws_generator.generate_cws` — and as `anorak-nf/bin/anorak_tile.py`,
    which test_anorak_tiling.py holds this to — so the completeness check is
    measured against the source rather than against a marker file that a
    killed job may have written before dying. Any change to upstream's tiling
    geometry must be mirrored here or slides start looking permanently
    incomplete. The rest of the dict is what COMPLETION_RECORD stores and what
    a resubmission must match to reuse the tiles.
    """
    import openslide

    with openslide.OpenSlide(str(slide_path)) as slide:
        objective_power, in_mpp = _slide_scale(slide, slide_path)
        slide_w, slide_h = slide.level_dimensions[0]

    cws_objective_value = 20 * (objective_power / 40) * (in_mpp / output_mpp)
    rescale = objective_power / cws_objective_value
    cws_side = _CWS_READ_SIZE * rescale  # square, so one side suffices

    y_tiles = int(math.ceil((slide_h - cws_side) / cws_side + 1))
    x_tiles = int(math.ceil((slide_w - cws_side) / cws_side + 1))
    return {
        "slide_name": slide_path.name,
        "output_mpp": float(output_mpp),
        "objective_power": objective_power,
        "in_mpp": in_mpp,
        "slide_size": [int(slide_w), int(slide_h)],
        "cws_read_size": _CWS_READ_SIZE,
        "expected_tiles": y_tiles * x_tiles,
    }


def expected_tile_count(slide_path: Path, output_mpp: float) -> int:
    """How many `Da*.jpg` upstream will write for this slide."""
    return tiling_geometry(slide_path, output_mpp)["expected_tiles"]


def _slide_scale(slide, slide_path: Path) -> tuple[float, float]:
    """(objective power, mpp) from a slide's header, or a refusal.

    Both are required, and neither is inferred. Upstream degrades quietly when
    the mpp is absent — it warns and scales by objective power alone, giving
    this slide a different output resolution to every other slide in the
    cohort, in an identical layout. That is not a distinction anything
    downstream can make, so it is refused here instead.
    """
    import openslide

    try:
        objective_power = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{slide_path.name} reports no objective power. Upstream would read "
            f"it as 0 and crash, or take a supplied default that may not be this "
            f"scanner's."
        ) from None
    try:
        in_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"{slide_path.name} reports no microns-per-pixel. Upstream warns and "
            f"falls back to objective-power scaling, which tiles this slide at a "
            f"different resolution to the rest of the cohort with nothing in the "
            f"output to say so."
        ) from None

    if objective_power <= 0 or in_mpp <= 0:
        raise ValueError(
            f"{slide_path.name} reports objective power {objective_power} and "
            f"mpp {in_mpp}; both must be positive."
        )
    return objective_power, in_mpp


def tiles_on_disk_problem(slide_dir: Path, expected_tiles: int) -> str | None:
    """Why this slide's directory is not a whole tiling, or None if it is.

    Upstream has no resume: an interrupted slide leaves a prefix of `Da*.jpg`
    and no marker files, and re-running rewrites every tile, so the unit of
    work is the whole slide. The markers say the process reached the end; the
    tiles must be *exactly* Da0..Da{n-1} — a prefix is a killed run, and a
    tile beyond n is left over from a tiling at other settings, which the
    stitch would pick up by index as if it belonged.
    """
    if not slide_dir.is_dir():
        return "no output directory"
    missing_markers = [m for m in _COMPLETION_MARKERS
                       if not (slide_dir / m).is_file()]
    if missing_markers:
        return f"missing {missing_markers}"
    if expected_tiles <= 0:
        return f"expected tile count is {expected_tiles}"
    present = {p.name for p in slide_dir.glob("Da*.jpg")}
    wanted = {f"Da{i}.jpg" for i in range(expected_tiles)}
    if present != wanted:
        return (f"{len(present)} tiles on disk, {len(wanted - present)} of the "
                f"{expected_tiles} expected missing and {len(present - wanted)} "
                f"unexpected")
    return None


def read_completion_record(slide_dir: Path) -> dict | None:
    """The record a finished tiling left, or None if there is none to trust."""
    try:
        return json.loads((slide_dir / COMPLETION_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tiling_output_complete(slide_dir: Path, geometry: dict) -> bool:
    """Whether this slide was tiled whole, at exactly this attempt's settings.

    Three things, all required: a completion record exists (it is written
    last, so a killed tiling or re-tiling never has one); it names the same
    settings and geometry as `geometry`; and the directory holds exactly the
    tiles those settings produce. Marker files alone are not enough — they
    survive a re-tile killed half-way, and say nothing about the mpp.
    """
    record = read_completion_record(slide_dir)
    if record is None:
        return False
    if any(record.get(field) != geometry[field] for field in _RECORD_MATCH_FIELDS):
        return False
    return tiles_on_disk_problem(slide_dir, geometry["expected_tiles"]) is None


def _write_completion_record(slide_dir: Path, slide_path: Path, geometry: dict) -> None:
    record = {**geometry, "slide": str(slide_path.resolve()),
              "effective_mpp": 2 * geometry["output_mpp"]}
    _write_atomic(slide_dir / COMPLETION_RECORD, json.dumps(record, indent=2))


def _write_atomic(path: Path, text: str) -> None:
    """Write under a .tmp name and rename, so a reader never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def slide_output_dir(out_dir: Path, slide_path: Path) -> Path:
    """Upstream's own output location for a slide — name *with* extension."""
    return out_dir / "cws_tiling" / slide_path.name


# --- worker ----------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    """Tile the one slide this array task was given."""
    task_id_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not defined")
    task_id = int(task_id_text)

    slide_path = read_manifest_slide(args.manifest, task_id)
    out_dir = args.out_dir
    slide_dir = slide_output_dir(out_dir, slide_path)

    print("=" * 70, flush=True)
    print(f"Job ID:      {os.environ.get('SLURM_JOB_ID', 'unknown')}", flush=True)
    print(f"Array task:  {task_id}", flush=True)
    print(f"Slide:       {slide_path}", flush=True)
    print(f"Output:      {slide_dir}", flush=True)
    print("=" * 70, flush=True)

    if slide_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"{slide_path.name}: upstream's save_cws.single_file_run does not "
            f"dispatch on '{slide_path.suffix}' and would return having written "
            f"nothing, reporting success."
        )

    # Refuses on unreadable scale before any tile is written, not after.
    geometry = tiling_geometry(slide_path, args.output_mpp)
    expected_tiles = geometry["expected_tiles"]
    print(f"Expecting {expected_tiles} tiles at "
          f"{2 * args.output_mpp:.3g} um/px", flush=True)

    if tiling_output_complete(slide_dir, geometry):
        print(f"[SKIP] Already tiled: {expected_tiles} tiles present, recorded "
              f"at output-mpp {args.output_mpp}.", flush=True)
        print(f"[DONE] {slide_path.name}", flush=True)
        return

    if slide_dir.exists():
        # The record goes first, so that however far the clearing below gets
        # before a kill, what is left can never be mistaken for finished. The
        # directory is then emptied rather than overwritten: upstream rewrites
        # tiles by index and never deletes, so a re-tile at settings that give
        # fewer tiles would leave the old tail behind it.
        previous = read_completion_record(slide_dir)
        (slide_dir / COMPLETION_RECORD).unlink(missing_ok=True)
        reason = ("no completion record" if previous is None else
                  f"recorded at output-mpp {previous.get('output_mpp')}, "
                  f"{previous.get('expected_tiles')} tiles")
        print(f"[CLEAR] {slide_dir} holds output that is not this tiling "
              f"({reason}); removing it and re-tiling.", flush=True)
        shutil.rmtree(slide_dir)

    print("[RUN] Tiling...", flush=True)
    save_cws = _import_anorak(args.anorak_dir)
    save_cws.single_file_run(
        file_name=slide_path.name,
        output_dir=str(out_dir / "cws_tiling"),
        input_dir=str(slide_path.parent),
        tif_obj=40,
        cws_objective_value=20,
        in_mpp=None,
        out_mpp=args.output_mpp,
        out_mpp_target_objective=40,
        parallel=False,
    )

    # Upstream returns without a status, so the only way to know it finished is
    # to look at what it left. A slide that ran to completion and still fails
    # this has produced output nothing downstream should read.
    problem = tiles_on_disk_problem(slide_dir, expected_tiles)
    if problem:
        raise RuntimeError(
            f"{slide_path.name}: tiling did not complete — {problem}. Leaving "
            f"the partial output in place for inspection; with no completion "
            f"record it cannot be skipped, and resubmitting re-tiles this "
            f"slide from the start."
        )
    # Last, and only now: this is the one file that makes a resubmission skip.
    _write_completion_record(slide_dir, slide_path, geometry)
    print(f"[DONE] {slide_path.name}", flush=True)


def _import_anorak(anorak_dir: Path):
    """Import upstream's `save_cws`, which imports `cws_generator` flatly."""
    generating_tile = anorak_dir / "generating_tile"
    if not (generating_tile / "save_cws.py").is_file():
        raise RuntimeError(
            f"{generating_tile}/save_cws.py not found. --anorak-dir must point "
            f"at a clone of https://github.com/xi11/AIgrading."
        )
    sys.path.insert(0, str(generating_tile))
    import save_cws  # noqa: PLC0415

    return save_cws


def read_manifest_slide(manifest_path: Path, task_id: int) -> Path:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == task_id:
                slide_path = Path(line.rstrip("\n"))
                if not slide_path.is_file():
                    raise FileNotFoundError(f"Slide does not exist: {slide_path}")
                return slide_path
    raise IndexError(f"Array task {task_id} has no matching slide in {manifest_path}")


# --- submission ------------------------------------------------------------

def write_manifest(slides: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Slide names in this cohort carry spaces; one path per line, never split.
    _write_atomic(manifest_path, "".join(f"{slide}\n" for slide in slides))


def _new_submission_dir(out_dir: Path) -> Path:
    """A directory no other submission has used, for this one's manifests.

    Array tasks read their manifest when they *start*, which can be hours after
    sbatch returned. Manifests used to live at fixed names and be rewritten by
    every submission, so resubmitting a changed list while an earlier array
    was still pending re-pointed that array's task N at the new list's slide N.
    `exist_ok=False` is what makes the name a guarantee rather than a hope.
    """
    base = out_dir / "submissions"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{attempt}"
        candidate = base / f"{stamp}-{os.getpid()}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a fresh submission directory under {base}")


def check_metadata_sample(slides: list[Path], sample_size: int, output_mpp: float) -> None:
    """Read the scale off a sample of slides before queueing thousands.

    Every slide is checked again by its own worker; this is only about finding
    out now, from one process, that (say) no slide in the cohort reports an
    mpp — rather than from 7,221 identical tracebacks in 7,221 log files.
    """
    if sample_size <= 0:
        return
    step = max(1, len(slides) // sample_size)
    sample = slides[::step][:sample_size]
    failures: list[str] = []
    tile_counts: list[int] = []
    for slide in sample:
        try:
            tile_counts.append(expected_tile_count(slide, output_mpp))
        except Exception as error:  # openslide raises its own types
            failures.append(f"    {slide.name}: {error}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(sample)} sampled slides cannot be tiled at "
            f"a known resolution:\n" + "\n".join(failures[:5])
        )
    print(f"  metadata sample:  {len(sample)} slides readable, "
          f"{min(tile_counts)}-{max(tile_counts)} tiles each "
          f"(~{int(sum(tile_counts) / len(tile_counts) * len(slides)):,} total)")


def submit_array(
    slides_csv: Path,
    raw_dir: Path,
    out_dir: Path,
    anorak_dir: Path,
    *,
    slide_column: str = "slide_id",
    output_mpp: float = DEFAULT_OUTPUT_MPP,
    max_concurrent: int = 50,
    batch_size: int = 1000,
    partition: str | None = None,
    cpus: int = 1,
    memory: str = "16G",
    time_limit: str = "12:00:00",
    job_name: str = "anorak_tile",
    notify_email: str | None = None,
    metadata_sample: int = 25,
    python_executable: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Resolve the slide list, write manifests, submit one array per batch."""
    if max_concurrent < 1 or batch_size < 1:
        raise ValueError("--max-concurrent and --batch-size must be at least 1")
    backend_dir = Path(__file__).resolve().parent
    script_path = backend_dir / Path(__file__).name
    python_executable = python_executable or Path(sys.executable)

    slide_ids = read_slide_ids(slides_csv, slide_column)
    slides = resolve_slides(slide_ids, raw_dir)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"  slide list:       {slides_csv} ({len(slide_ids)} slides)")
    print(f"  resolved:         {len(slides)} files under {raw_dir}")
    print(f"  output:           {out_dir / 'cws_tiling'}")
    print(f"  output-mpp:       {output_mpp} -> tiles at {2 * output_mpp:.3g} "
          f"um/px (x{20 * DEFAULT_OUTPUT_MPP / output_mpp:.3g}), 2000x2000 px")
    check_metadata_sample(slides, metadata_sample, output_mpp)

    manifest_dir = _new_submission_dir(out_dir)
    combined_manifest = manifest_dir / "anorak_manifest.txt"
    write_manifest(slides, combined_manifest)

    batches = [slides[i:i + batch_size] for i in range(0, len(slides), batch_size)]
    record_path = manifest_dir / "submission.json"
    latest_path = out_dir / "anorak_submission.json"
    submission = {
        "slides_csv": str(slides_csv),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "anorak_dir": str(anorak_dir),
        "output_mpp": output_mpp,
        "effective_mpp": 2 * output_mpp,
        "slides_requested": len(slide_ids),
        "slides_resolved": len(slides),
        "max_concurrent": max_concurrent,
        "manifest_dir": str(manifest_dir),
        "manifest_path": str(combined_manifest),
        "record_path": str(record_path),
        "status": "dry_run" if dry_run else "submitting",
        "batches": [],
        "job_ids": [],
        "dry_run": dry_run,
    }

    def save_record() -> None:
        # After every batch, not once at the end: a later sbatch failing used
        # to raise before anything was written, leaving the batches already
        # queued running with no record of their job ids anywhere. The
        # per-submission record is the one that stays true; the top-level copy
        # is a convenience that the next submission replaces.
        text = json.dumps(submission, indent=2)
        _write_atomic(record_path, text)
        _write_atomic(latest_path, text)

    save_record()
    previous_job: str | None = None
    for index, batch in enumerate(batches):
        manifest_path = manifest_dir / f"anorak_manifest_batch{index:03d}.txt"
        write_manifest(batch, manifest_path)

        worker_command = [
            str(python_executable), str(script_path), "--worker",
            "--manifest", str(manifest_path),
            "--out-dir", str(out_dir),
            "--anorak-dir", str(anorak_dir),
            "--output-mpp", str(output_mpp),
        ]
        # Each batch waits for the one before it. `%N` throttles one array
        # only, and every batch used to be submitted at once, so eight batches
        # at the default 50 were 400 slides tiling together. Chained, the
        # limit holds across the submission. afterany rather than afterok so
        # one failed slide does not strand every batch after it; the cost is
        # that a batch's slowest slides briefly run below the limit.
        dependency = []
        if index > 0:
            dependency = [f"--dependency=afterany:"
                          f"{previous_job or f'<batch{index - 1:03d} job id>'}"]
        sbatch_command = [
            "sbatch",
            f"--job-name={job_name}",
            *([f"--partition={partition}"] if partition else []),
            f"--cpus-per-task={cpus}",
            f"--mem={memory}",
            f"--time={time_limit}",
            f"--array=0-{len(batch) - 1}%{max_concurrent}",
            *dependency,
            f"--output={log_dir}/anorak_tile_%A_%a.out",
            f"--error={log_dir}/anorak_tile_%A_%a.err",
            f"--chdir={backend_dir}",
            *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"]
              if notify_email else []),
            "--wrap", shlex.join(worker_command),
        ]

        batch_info = {
            "manifest_path": str(manifest_path),
            "slides_in_batch": len(batch),
            "sbatch_command": shlex.join(sbatch_command),
            "job_id": None,
        }
        submission["batches"].append(batch_info)
        if not dry_run:
            failure = None
            try:
                result = _run_sbatch_with_retry(sbatch_command)
            except subprocess.CalledProcessError as error:
                reason = ((error.stderr or "").strip()
                          or (error.stdout or "").strip()
                          or "no output from sbatch")
                failure = f"sbatch failed (exit {error.returncode}): {reason}"
            else:
                stdout = result.stdout.strip()
                batch_info["sbatch_stdout"] = stdout
                match = re.search(r"Submitted batch job (\d+)", stdout)
                if match:
                    previous_job = match.group(1)
                    batch_info["job_id"] = previous_job
                    submission["job_ids"].append(previous_job)
                else:
                    # Without the id the next batch cannot be chained, and
                    # submitting it unchained is the concurrency bug again.
                    failure = f"sbatch printed no job id: {stdout!r}"
            if failure:
                submission["status"] = "partial"
                submission["error"] = f"batch {index:03d}: {failure}"
                save_record()
                queued = ", ".join(submission["job_ids"]) or "none"
                raise RuntimeError(
                    f"{failure}\nBatch {index:03d} of {len(batches)} was not "
                    f"submitted; batches before it are queued (job ids: "
                    f"{queued}). Recorded in {record_path}."
                )
        save_record()

    if not dry_run:
        submission["status"] = "submitted"
    save_record()
    return submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)

    parser.add_argument("--slides-csv", type=Path,
                        help="filtered slide list (filter_slides_by_tile_count.py)")
    parser.add_argument("--slide-column", default="slide_id",
                        help="column holding the slide ids (default: slide_id)")
    parser.add_argument("--raw-dir", type=Path, help="directory of raw WSIs")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="tiles land in <out-dir>/cws_tiling/<slide file>/ "
                             "(NOT the pipeline's layout; see the module "
                             "docstring)")
    parser.add_argument("--anorak-dir", type=Path, required=True,
                        help="clone of github.com/xi11/AIgrading")
    parser.add_argument("--output-mpp", type=float, default=DEFAULT_OUTPUT_MPP,
                        help=f"upstream's scanner reference; tiles come out at "
                             f"TWICE this (default: {DEFAULT_OUTPUT_MPP} -> "
                             f"0.44 um/px, x20)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="slides tiling at once across the whole "
                             "submission; batches run one after another "
                             "(default: 50)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="slides per array submission; each batch starts "
                             "when the previous one has finished (default: "
                             "1000)")
    parser.add_argument("--partition")
    parser.add_argument("--cpus", type=int, default=1,
                        help="upstream tiles single-threaded (default: 1)")
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--job-name", default="anorak_tile")
    parser.add_argument("--notify-email")
    parser.add_argument("--metadata-sample", type=int, default=25,
                        help="slides to scale-check before submitting; 0 to skip")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve, check and write manifests; submit nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.worker:
        if args.manifest is None:
            parser.error("--worker requires --manifest")
        run_worker(args)
        return 0

    if args.slides_csv is None or args.raw_dir is None:
        parser.error("--slides-csv and --raw-dir are required to submit")
    if not args.slides_csv.is_file():
        parser.error(f"no such file: {args.slides_csv}")
    if not args.raw_dir.is_dir():
        parser.error(f"not a directory: {args.raw_dir}")
    if args.output_mpp <= 0:
        parser.error("--output-mpp must be positive")

    print("ANORAK tiling submission")
    try:
        submission = submit_array(
            slides_csv=args.slides_csv,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            anorak_dir=args.anorak_dir,
            slide_column=args.slide_column,
            output_mpp=args.output_mpp,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            partition=args.partition,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            job_name=args.job_name,
            notify_email=args.notify_email,
            metadata_sample=args.metadata_sample,
            dry_run=args.dry_run,
        )
    except (ValueError, RuntimeError) as refusal:
        # A refusal here is the designed outcome, not a crash: an unresolved
        # slide list or an unreadable cohort is something to read and fix, and
        # a traceback buries the sentence that says which.
        print(f"\nRefused: {refusal}", file=sys.stderr)
        return 1

    print(f"  batches:          {len(submission['batches'])}")
    if args.dry_run:
        print("\nDry run — nothing submitted. First batch would be:")
        print(f"  {submission['batches'][0]['sbatch_command']}")
    else:
        print(f"  job ids:          {', '.join(submission['job_ids']) or 'none'}")
    print(f"  record:           {submission['record_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
