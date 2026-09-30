# anorak-nf

Growth-pattern segmentation and IASLC grading of lung adenocarcinoma whole-slide
images, as a Nextflow pipeline.

The model, the weights and every segmentation step are **ANORAK**, from
[xi11/AIgrading](https://github.com/xi11/AIgrading) — Pan et al., *The
artificial intelligence-based model ANORAK improves histopathological grading of
lung adenocarcinoma*, Nature Cancer **5**, 347–363 (2024). This pipeline calls
that code unchanged. What it adds is what a several-thousand-slide cohort needs
and a single-process script does not.

## What it does

```
slide list ─→ TILE_SLIDE ─→ PREDICT_GP ─→ SS1_STITCH ─→ SLIDE_PROPORTIONS ─→ TUMOUR_GRADE
              (CPU)         (GPU)         (CPU)          (CPU)                (CPU, once)
```

| step | upstream code | output |
|---|---|---|
| `TILE_SLIDE` | `generating_tile/save_cws.py` | 2000×2000 JPEG tiles at 0.44 µm/px (×20) |
| `PREDICT_GP` | `inference_slide/predict_gp.py` | one colour mask PNG per tile |
| `SS1_STITCH` | `ss1_stich.py` + `ss1_final.py` | one post-processed whole-slide mask |
| `SLIDE_PROPORTIONS` | *new* | per-slide pattern pixel counts |
| `TUMOUR_GRADE` | *new* | per-tumour proportions, predominant pattern, IASLC grade |

The last two implement the paper's Methods directly: `g_j = Σᵢ Sᵢⱼ / Σᵢ Σⱼ Sᵢⱼ`
over the slides of one tumour, `P = argmax(g_j)`, and grade 1 / 2 / 3 by
predominant pattern and the 20% high-grade cutoff. Counts are pooled across a
tumour's slides before dividing, not averaged as per-slide proportions — that is
what the definition says, and it stops a small biopsy outweighing a resection.

## Running it

```bash
nextflow run . -profile beatson \
    --slides_csv  radiogenomics_tumour_slides_min10.csv \
    --raw_dir     /mnt/cephfs-lts/long-term-scratch/users/vpandya/Radiogenomics \
    --anorak_dir  /path/to/AIgrading \
    --outdir      /path/to/results
```

Add `-resume` to continue a previous run; completed slides are skipped.
`-profile stub -stub-run` exercises the whole DAG on empty files in seconds.
`--queue_size` (default 200 on `-profile beatson`) caps the jobs Nextflow keeps
queued; with the head job and any chain standbys it must stay under the
per-user `MaxSubmitJobs`, which preflight checks.

**Before the first run**, put the model checkpoint from
[zenodo.org/records/15272883](https://zenodo.org/records/15272883) at
`<anorak_dir>/models/AIgrading_anorak.h5`. `generate_gp()` builds that path from
its own location and cannot be pointed elsewhere; the workflow refuses at launch
if it is missing.

### Before the first run: the inference image's missing packages

```bash
export ANORAK_REPO_DIR=/path/to/AIgrading
tools/bootstrap_gpu_extras.sh --anorak-dir "$ANORAK_REPO_DIR"
```

The inference image is NGC TensorFlow. It has TensorFlow, CUDA and numpy, and
nothing else from upstream's `requirments.txt` — so `import cv2`, on line 3 of
`inference_slide/predict_gp.py`, ends every GPU task in seconds. `$HOME` is not
writable inside the image and pip cannot reach the internet from a compute
node, so the missing packages go in a directory on scratch (`gpu_extras` in
`conf/beatson.config`) and reach the interpreter on `PYTHONPATH`. Same
arrangement, for the same reasons, as `HPL_CONTAINER_EXTRAS` in the HPL
pipeline.

The script probes the image first and installs only what is genuinely absent,
with `--no-deps`, so nothing in the extras directory can shadow the numpy
TensorFlow was built against. What it may install is pinned — only what a GPU
task actually imports, `opencv-python-headless==4.8.1.78` and `pillow==10.4.0`
— and a module that is missing and not pinned is refused rather than fetched at
whatever version pip picks. A verified build writes `anorak_extras.lock` into
the directory (atomically, recording the image it was built against); a re-run
installs exactly the lock, `--lock FILE` rebuilds from a copy, and `--refresh`
re-resolves from the pins. Verification imports upstream's `predict_gp` inside
the image with the task's exact `PYTHONPATH` — the only check that covers
upstream's whole import list rather than the subset this repo knows about.

An extras directory built by the older, unpinned bootstrap warns about
packages the pins do not list. Rebuild it. Its package names are part of
`PREDICT_GP`'s cache key, so the rebuild re-segments — which on a fresh cohort
is what you want.

### Before the first run: preflight

```bash
export ANORAK_REPO_DIR=/path/to/AIgrading
tools/preflight.sh --slides-csv <list.csv> --raw-dir <slides/> \
    [--chain N] [--queue-size N] [--head-partition P]
```

Run from a cluster login node; every value it compares is read from the
*effective* config (`nextflow config -profile beatson -flat`), never restated.
It covers everything answerable before a queue slot: the tooling, the clone and
checkpoint (3), the extras against their lock (4), the slide list (5, 5b —
the workflow's own launch checks via `nextflow -preview`), `.mrxs` slides and
their data directories (5c), every label's largest retry request —
`min(base × (maxRetries+1), ceiling)` — against its partition's MaxTime,
memory and cpus, plus the GPU partition and exact `gpu_type` (6), the submit
limits — association MaxSubmitJobs, job QOS MaxSubmitPU, partition QOS —
against `queueSize + chain + 1` (6b), GPU isolation, failing if the effective
`envWhitelist` lacks `CUDA_VISIBLE_DEVICES` (6c), and whether a compute node can
submit jobs (7). Then, inside the real containers on a real node: whether every
input path resolves (8), whether the checkpoint loads on a GPU (9), whether
upstream's `predict_gp` imports (9b), whether a task launched as Nextflow
launches it sees exactly one GPU (9c), whether openslide opens your slides and
reads their mpp (10), and whether node-local `$TMPDIR` has room for the labels
that still use scratch (11 — GPU and stitching; tiling and publishing write
straight to the work directory).

Exit 0 means ready, 1 that a check failed, 2 that checks were skipped (they
are listed) — and 2 is not a pass. Check 9b exists because check 9 did not
catch its failure: loading the checkpoint needs TensorFlow and nothing else, so
it passed on an image where every task was about to die on `cv2`. The on-node
checks are the point: each performs the operation it asks about, because this
is the class of failure where everything looks correct from outside. Run it
again after changing a container, a config, or the cluster.

### Checking just the container



```bash
singularity exec --nv <image>.sif python3 tools/check_anorak_model.py \
    --checkpoint <anorak_dir>/models/AIgrading_anorak.h5
```

This is the one question with no cheap answer further down. ANORAK pins
TensorFlow 2.2, whose CUDA 10.1 supports compute capability up to 7.5 — so it
cannot drive an A100 (8.0) or an H100/H200 (9.0). The image that *can* talk to
those cards is TF 2.11+, and a Keras 2.4.3 `.h5` is not guaranteed to load under
it. `tools/check_anorak_model.py` settles that in a minute: the checkpoint
loads, the output is `(1, 768, 768, 7)`, and a GPU is actually visible.

If it does not load, `tools/convert_anorak_model.py` re-exports the checkpoint
as a SavedModel from a **CPU-only** TF 2.2 container — loading needs no GPU, so
CUDA 10.1 never comes into it — and a SavedModel crosses TensorFlow versions far
better than an `.h5` with `custom_objects` does. `--in-place` puts the result at
the checkpoint's own path, because `generate_gp()` builds that path from its own
location and cannot be pointed elsewhere.

`conf/beatson.config` needs the inference image filled in, plus either a tiling
image with openslide or (as here) a native environment that has it. The
inference image needs a TensorFlow that both loads the checkpoint and drives
the card; whatever else upstream imports comes from `gpu_extras` rather than
from the image. They cannot be one image: the checkpoint is a Keras
`.h5` that will not load under an arbitrary newer TensorFlow, and that pinned
stack has no reason to carry openslide.

### Inputs

`--slides_csv` is a slide list with a `slide_id` column and a column naming
the tumour each slide belongs to (`--sample_column`, default `samples`). The
output of `filter_slides_by_tile_count.py` is exactly this shape. Slide ids are
resolved against `--raw_dir` by filename or stem.
A `.mrxs` works: its pixels are in the companion directory beside it, which is
not staged, so the tiler opens the slide at its real path under `--raw_dir`.

Refused before anything is queued, because each used to be graded without
complaint: a missing sample column or a blank sample (they were pooled into one
tumour called `''`), a row whose `is_tumour` is not true (`--tumour_column ''`
turns that check off), and two ids that make the same output name once
characters outside `[A-Za-z0-9._-]` become `_` (`A B` and `A_B` used to fail at
the very last step, again on every resume).

### Editing bin/ and -resume

Nextflow keys a cached task on its script text, its inputs and its container's
*name*. None of these covers what the steps actually run — `python3
${projectDir}/bin/x.py`, the AIgrading code the wrappers import, the checkpoint,
or the bytes of an image rebuilt at the same path — so every step takes a
digest of those as an input, and an edit re-runs exactly the steps it affects
and everything after them:

| edit | re-runs |
|---|---|
| `anorak_tile.py`, `generating_tile/*.py`, `anorak_common.py` | everything |
| `anorak_predict.py`, `inference_slide/*.py`, the checkpoint, the GPU image or its extras | `PREDICT_GP` onward |
| `anorak_stitch.py` | `SS1_STITCH` onward |
| `slide_proportions.py` | `SLIDE_PROPORTIONS`, `TUMOUR_GRADE` |
| a slide's sample in the slide list | that slide's `SLIDE_PROPORTIONS`, `TUMOUR_GRADE` |
| nothing | nothing |

This used to be the other way round: the heavy steps were deliberately not
keyed on their code, so a resume after an edit kept the old outputs and ran the
new code only on slides not yet done — one cohort, two versions, nothing to say
which slide got which. It became the right trade when the cohort was restarted
from nothing (2026-09-25). A comment edit in `anorak_tile.py` now re-tiles the
cohort, so edit the heavy steps' code between cohorts, not during one.

Large files are identified by size and a digest of their first, middle and last
MiB rather than read whole, because the head job reads them on every launch;
`__pycache__` is ignored, because importing the code writes it. The conda env
the CPU steps run in natively (`native_env_bin`) is *not* in the key.
`tools/test_workflow_cache.py` proves every row of the table against real
Nextflow.

The `meta` map the heavy steps carry is only `id` and `slide_name`; the sample
joins at `SLIDE_PROPORTIONS`, which is what makes relabelling a tumour cheap.

### Failures

A wrapper that refuses (`bin/anorak_common.refuse`) exits 65: that slide's input
or output is wrong, and the run finishes — running tasks drain, nothing new is
queued. Anything else is retried twice first, including `Integer.MAX_VALUE`,
which is Nextflow's code for a `.exitcode` it could not read (node failure,
preemption, a refused `sbatch`, or CephFS slower than `exitReadTimeout`). With
`-process.errorStrategy=ignore` the run grades what arrived and lists the rest in
`anorak_missing_slides.csv`; otherwise a missing slide is a refusal, not a short
table.

## What it changes about upstream, and why

**Slides are addressed by name, never by index.** `generate_gp`, `ss1_stich` and
`ss1_final` each select their slide with `sorted(glob(dir/pattern))[nfile]`,
evaluated independently in three steps. A directory that gained or lost an entry
between them shifts every index above it, and the stitch then assembles one
slide's masks against another slide's `param.p` — producing a whole-slide mask of
the right size, in the right colours, for the wrong slide. Each task here is
given exactly one slide and passes that slide's own escaped name as the pattern,
so the glob returns one entry and `nfile=0` is correct by construction
(`bin/anorak_common.py`).

**Every step checks the previous one is whole, not merely present.** Tiling
verifies the tile count against a grid recomputed from the slide header;
prediction verifies one mask per tile; stitching refuses a short mask set,
because the gaps it would leave are background-coloured and indistinguishable
from tissue the model found nothing in. This matters because `ss1_stich` writes
its output every 20 tiles and then treats an existing file as finished.

**The colour palette is read off `ss1_final.py`, not the README.** Upstream's
README calls cribriform cyan `#00ffff`; `class_colors[1]` and `ss1_final`'s own
thresholds both use green `(0,255,0)`. The code is what produced the pixels.

**A slide whose header reports no microns-per-pixel is refused.**
`cws_generator.py` catches that, warns, and falls back to objective-power
scaling — tiling that one slide at a resolution the rest of the cohort does not
share, with nothing in the output to say so.

## Resolution

`--output_mpp` is upstream's flag and is **not** the output resolution.
`cws_objective_value = 20·(objective/40)·(in_mpp/out_mpp)` and
`rescale = objective/cws_objective_value` multiply through to an effective
output of exactly `2 × out_mpp`, for any scanner. The 0.22 default therefore
gives 0.44 µm/px (×20) — the resolution the model was trained and published at.
Changing it changes the magnification the model sees.

## Outputs

```
<outdir>/
├── cws_tiling/<slide id>/<file>/Da*.jpg     tiles (hard links)
├── ss1_final/<slide id>/<file>_Ss1.png      post-processed whole-slide masks (hard links)
├── anorak_slide_proportions.csv             per-slide pixel counts and fractions
├── anorak_tumour_grades.csv                 per-tumour proportions, predominant pattern, grade
├── anorak_missing_slides.csv                listed slides with no counts (empty unless ignore)
└── pipeline_info/                      timeline, report, trace, DAG
```

Tiles and masks are published by `PUBLISH_SLIDE`, one small task per slide on a
compute node, not by `publishDir`: the head job publishing tiles is what it was
doing when it died on 2026-09-16, and `publishDir` could leave a slide's
directory partial (head killed mid-slide, then skipped on resume) or stale (a
re-tiled slide kept its old links). `bin/publish_tree.py` compares the published
directory with the task's output by inode and, if they differ, rebuilds it
beside the old one and renames it into place — so a slide's directory is
complete and current, or absent. It is cached like any task, so a resume that
re-tiled nothing links nothing. Damage done by hand after publishing is not
visible to the cache; `--republish <any new value>` re-checks every slide and
re-links only those that differ. Hard links need `--outdir` on the work
directory's filesystem; the task refuses otherwise. A slide that never reaches
stitching has neither published.

`anorak_tumour_grades.csv` carries a `reason` column stating why each tumour got
its grade, a `tie` column when two patterns share the maximum (argmax picks the
first, which is a property of column order rather than of the tissue), and an
empty grade with a stated reason for a tumour whose slides carry no pattern
pixels at all.
