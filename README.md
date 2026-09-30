# HPL software

Software for running **histomorphological phenotype learning (HPL)** and **ANORAK
growth-pattern grading** on H&E whole-slide images at cohort scale, on a Slurm
cluster, from a web UI.

Given a directory of whole-slide images, it:

1. **tiles** every slide at the resolution the HPL model was trained on,
2. **packages** the tiles into one HDF5 file per dataset,
3. **encodes** every tile with a frozen self-supervised model (Barlow Twins, on GPU),
4. **classifies** every tile into one of the reference's histomorphological
   phenotype clusters (HPCs) by exact k-nearest-neighbour search,
5. **registers** the cohort and **loads** the cluster assignments into a Postgres
   "Knowledge Bank" that the slide viewer and a natural-language chatbot read,

and, separately or alongside,

6. **grades** lung adenocarcinoma growth patterns and the IASLC grade with
   [ANORAK](https://github.com/xi11/AIgrading).

Steps 1-4 and step 6 each run as a single [Nextflow](https://www.nextflow.io)
pipeline started with one click from the UI.

---

## Contents

- [How it works](#how-it-works)
- [Using the UI](#using-the-ui)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running from the command line](#running-from-the-command-line)
- [Testing](#testing)
- [Design principles](#design-principles)
- [Further documentation](#further-documentation)
- [Credits](#credits)
- [Licence](#licence)

---

## How it works

```
                 dataset directory (.svs / .ndpi / .mrxs / .tif ...)
                                   |
        +--------------------------+---------------------------+
        |                                                      |
   Run HPL (hpl-nf/)                                     Run ANORAK (anorak-nf/)
        |                                                      |
   1 Tiling ........... processed_tiles/<dataset>/        TILE_SLIDE   (CPU, per slide)
   2 Packaging ........ model_input/<dataset>/*.h5        PREDICT_GP   (GPU, per slide)
   3 Feature extraction  results/.../*.h5  (GPU)          SS1_STITCH   (CPU, per slide)
   4 Classification ... <dataset>_hpc_assignments.csv     SLIDE_PROPORTIONS
        |                                                 TUMOUR_GRADE -> grading table
   5 Registration  --+
   6 Knowledge Bank --+--> Postgres (hpl_kb) --> slide viewer, HPC panels, chatbot
      load
```

| Stage | Code | Output |
|---|---|---|
| 1 Tiling | `backend/submit_mask_tile_slurm.py`, `auto_tile_from_mask.py`, `tile_mask.py` | tissue masks, 224 px tiles at 1.8 µm/px, a per-slide tile metadata CSV |
| 2 Packaging | `backend/make_hpl_hdf5.py` | one gzip HDF5 of every tile in the dataset |
| 3 Feature extraction | `backend/submit_feature_extraction.py` + `HPL-LATTICeA/` | per-tile embeddings (Singularity, NGC TensorFlow 1.15, GPU) |
| 4 Classification | `backend/assign_hpc_clusters.py`, `submit_cluster_assignment.py` | per-tile cluster ID with vote margin and neighbour distance (faiss, exact search) |
| 5 Registration | `backend/register_dataset.py` | slide and tile identity rows in the Knowledge Bank |
| 6 Knowledge Bank load | `backend/load_hpc_assignments.py` | per-tile clusters and per-slide cluster profiles |
| ANORAK | `anorak-nf/`, `backend/submit_anorak_nf.py` | per-tumour growth-pattern proportions, predominant pattern and IASLC grade |

Stages 1-4 run as one Nextflow pipeline (`hpl-nf/`). Each pipeline task runs a
thin wrapper in `hpl-nf/bin/` that calls the same backend code the stages have
always used, and a stage counts as finished only once its output passes the
same validation the server applies. Stages 5 and 6 write to the shared
Knowledge Bank, so they are separate, reviewed steps: the load always shows a
dry run first and refuses below a 95% match rate.

The two Nextflow pipelines each run as one long-lived, supervised Slurm "head
job" that submits the per-slide, per-shard work itself. A watchdog
(`anorak-nf/tools/nf_supervise.sh`) restarts Nextflow with `-resume` if its
task monitor stalls, and standby head jobs take over if one reaches its
walltime.

---

## Using the UI

There are two UIs over one server: the Streamlit app (`app/app_v28.py`) and a
React front end (`frontend/`). Both have the same **Process a dataset** panel:

1. **Dataset path**: the directory of slides on the cluster's filesystem. This
   is the only input.
2. **Test on a random subset** (optional): run on N randomly chosen slides
   instead of all of them. The seed is recorded so the same sample can be
   requested again. Applies to both buttons.
3. **Run HPL**: stages 1-4 on the server's own settings, which are shown
   beside the button (tile and output locations, checkpoint, reference, vote,
   concurrency). Slides that are already tiled are reused. Complete outputs
   from an earlier run at the same paths are moved into a dated `superseded-…`
   folder, never deleted.
4. **Run ANORAK**: growth-pattern grading on its own.
   - With a **tumour-slide list** (from `backend/select_tumour_slides.py`),
     only those slides are graded.
   - Left blank, **every slide** in the directory is graded, grouped into
     tumours by the part of the slide name before the first space. The run is
     recorded as "tumour status not checked", because non-tumour slides are
     graded too.

Below the buttons are the latest HPL run and the latest ANORAK run, each with
per-stage progress, **Resume** and **Stop**, and a collapsed **History** of
every earlier run. Registration and the Knowledge Bank load appear on an HPL
run once classification is verified.

A single slide can also be uploaded (**Upload new WSI**); it is tiled and
packaged on the server and then goes through the same stages 3-6.

---

## Repository layout

```
app/                 Streamlit UI (app_v28.py is current) and its API client
frontend/            React + Vite UI over the same API
backend/             FastAPI tile server and every stage's code
  tile_server_v2_.py   the server (current, despite the name)
  submit_hpl_nf.py     submits the HPL Nextflow pipeline
  submit_anorak_nf.py  submits the ANORAK Nextflow pipeline
  hpl_nf_state.py      reads a pipeline run's per-stage state
  stage_outputs.py     output validators shared by the server and the pipeline
  tests/               the backend test suite
hpl-nf/              HPL stages 1-4 as a Nextflow pipeline (see its README)
anorak-nf/           ANORAK as a Nextflow pipeline (see its README)
HPL-LATTICeA/        the frozen HPL encoder, a git subtree of K-Rakovic/HPL-LATTICeA
docs/                architecture notes
*.md                 dated design notes and hand-offs (see Further documentation)
```

---

## Requirements

- A **Slurm** cluster with CPU and GPU partitions, and `sbatch` reachable from
  compute nodes (the Nextflow head jobs submit their own tasks).
- **Nextflow** 23.04 or newer and Java, on the head job's `PATH`. If they come
  from environment modules, load them through `HPL_NEXTFLOW_PRELUDE`.
- **Singularity or Apptainer**, with the NGC image
  `tensorflow-23.03-tf1-py3.sif` (TensorFlow 1.15, CUDA 12) for stages 3 and 4,
  and a TensorFlow 2 image for ANORAK's GPU step.
- **Python 3.10+** for the server and the pipeline tasks:
  `pip install -r backend/requirements.txt` (FastAPI, openslide-python, h5py,
  numpy, pandas, scikit-image, faiss-cpu, SQLAlchemy, psycopg2).
- **PostgreSQL** for run tracking and the Knowledge Bank.
- **Node.js 20+** for the React front end, and Streamlit for the Streamlit UI.
- The **HPL encoder checkpoint** (e.g. `BarlowTwins_3.ckt`), the **HPC
  reference** `.npz`, and for ANORAK a clone of
  [xi11/AIgrading](https://github.com/xi11/AIgrading) with its model weights.

---

## Setup

```bash
# 1. Python dependencies (the same environment runs the server and the pipeline tasks)
pip install -r backend/requirements.txt

# 2. Database: build the run-tracking and Knowledge Bank tables. This creates 13 of
#    the 17 tables; the four hpc_* reference tables describing the clusters are
#    loaded separately (see CLAUDE.md, "The Knowledge Bank").
psql -d hpl_kb -f backend/migrate_all.sql

# 3. Packages the NGC image lacks, installed once into a bound directory (login node)
python backend/submit_feature_extraction.py --bootstrap-extras
python backend/submit_feature_extraction.py --bootstrap-extras-gpu   # optional: GPU k-NN

# 4. The classification reference, built once from the Leiden .h5ad
python backend/build_hpc_reference.py --h5ad <leiden .h5ad> --out hpc_reference_leiden_2p5_fold2.npz

# 5. Check the cluster is ready for the HPL pipeline
python hpl-nf/tools/preflight.py --checkpoint /path/to/BarlowTwins_3.ckt
```

`preflight.py` checks Nextflow, the effective cluster config, the watchdog, the
task interpreter's imports, the HPL repository, the container and its extras,
the GPU type, the reference, walltimes against partition limits, that a compute
node can run `sbatch`, the submit limit, and that the results directory is
writable. ANORAK has its own `anorak-nf/tools/preflight.sh`.

Then start the services:

```bash
cd backend && python tile_server_v2_.py        # API on :8000
streamlit run app/app_v28.py                   # Streamlit UI
cd frontend && npm install && npm run dev      # or the React UI, on :5173
```

`hpl-nf/` and `backend/` must reach the cluster together: the pipeline's
wrappers import from `backend/` and refuse a copy that predates them.

---

## Configuration

Everything site-specific is an environment variable read by the server at
start-up; restart it after changing one.

| Variable | Purpose |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_NAME`, `DB_NAME_TEST` | Postgres: run tracking and the production and test Knowledge Banks |
| `HPL_JOB_DB_HOST` | the database host as seen from a compute node, for Slurm-backed KB writes |
| `PROCESSED_TILES_DIR`, `TISSUE_MASK_DIR`, `HPL_DATASETS_ROOT` | where tiles, masks and packaged `.h5` files are written |
| `HPL_REPO_DIR` | the cluster's clone of HPL-LATTICeA (the encoder) |
| `HPL_SINGULARITY_IMAGE`, `HPL_SINGULARITY_BIN` | the NGC TensorFlow image and the singularity/apptainer binary |
| `HPL_CONTAINER_EXTRAS`, `HPL_CONTAINER_EXTRAS_GPU` | the bound package directories the image lacks |
| `HPL_GPU_PARTITION`, `HPL_MERGE_PARTITION`, `HPL_GPU_PREFERENCE` | where GPU and CPU work is queued, and which GPU types to prefer |
| `HPL_CHECKPOINT` | the encoder checkpoint every one-click HPL run uses |
| `HPC_REFERENCE_PATH` | the classification reference `.npz` |
| `HPL_NF_RESULTS_ROOT`, `HPL_NF_PIPELINE_DIR`, `HPL_NF_PROFILE` | HPL pipeline run directories, pipeline location, Nextflow profile |
| `HPL_NF_MAX_TILING`, `HPL_NF_CHAIN`, `HPL_NF_HEAD_TIME_LIMIT` | slides tiled at once, head jobs per run, head job walltime |
| `ANORAK_REPO_DIR`, `ANORAK_PIPELINE_DIR`, `ANORAK_RESULTS_ROOT` | the AIgrading clone, the ANORAK pipeline, its run directories |
| `HPL_NEXTFLOW_PRELUDE` | shell run before `nextflow` in a head job (e.g. `module load`) |
| `TILE_SERVER_URL`, `TILE_SERVER_BROWSER_URL` | how the Streamlit app, and the browser, reach the server |
| `VITE_API_BASE_URL` | how the React UI reaches the server |
| `LLM_ENABLED`, `OLLAMA_MODEL`, `USE_LLM_PLANNER` | the chatbot's optional local LLM |

Cluster-specific Nextflow settings (partitions, GPU requests, executor timing)
live in `hpl-nf/conf/beatson.config` and `anorak-nf/conf/beatson.config`.

---

## Running from the command line

Every stage can also be run without the UI:

```bash
# HPL stages 1-4 as one Nextflow run, locally in stub mode (no slides, no GPU): a quick wiring check
nextflow run hpl-nf -profile stub -stub --config run_config.json --manifest slides.txt \
    --outdir out --python "$(which python)" --backend_dir "$PWD/backend"

# Individual stages, each submitted to Slurm
python backend/submit_mask_tile_slurm.py --raw-dir /path/to/slides
python backend/submit_feature_extraction.py --real-hdf5 <packaged .h5> --checkpoint <.ckt> --dataset-name <name>
python backend/submit_cluster_assignment.py --projections-h5 <projections .h5> --out assignments.csv

# Knowledge Bank: registration, then a dry run of the load, then the load
python backend/register_dataset.py --help
python backend/load_hpc_assignments.py --help            # dry run by default; --commit to write

# ANORAK over a tumour-slide list
python backend/submit_anorak_nf.py --slides-csv tumours.csv --raw-dir /path/to/slides \
    --out-dir /path/to/out --pipeline-dir anorak-nf --anorak-dir /path/to/AIgrading
```

Validation of the classifier:

```bash
# Does k-NN recover the reference's own labels?
python backend/validate_reference.py --reference hpc_reference_leiden_2p5_fold2.npz

# End to end: reproduce the published TCGA cluster transfer (below 99% agreement is a defect)
python backend/submit_cluster_assignment.py --projections-h5 <TCGA .h5> --out check.csv \
    --validate-against TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv
```

---

## Testing

```bash
python -m pytest backend/tests hpl-nf/tools anorak-nf/tools -q    # everything (~13 min)
python -m pytest hpl-nf/tools -q                                  # the HPL pipeline, incl. real Nextflow stub runs
cd frontend && npm test                                           # the React UI
```

Every Python suite also runs standalone (`python backend/tests/test_x.py`),
which is how they run on the cluster.

---

## Design principles

Almost nothing in this pipeline fails by crashing. A wrong reference, a
slide-naming mismatch, a shard computed with its own mean, a half-merged
output or a stale aggregate each produce a file or table of the right shape
with no missing values, which passes every completeness check while every
cluster ID is attached to the wrong tile. So:

- **Refuse before the queue.** Every input a stage needs is checked at
  submission, not hours into a job.
- **Validate against the source.** Outputs are checked against the row count
  of the stage before them, not against their own parts.
- **Write under a temporary name.** Outputs are renamed into place only when
  whole.
- **Dry-run anything that writes to the shared database.** The Knowledge Bank
  load refuses below a 95% match rate.
- **Make each guard's test able to fail.** Tests check that a validator can
  come out bad, not only that it comes out good.

`CLAUDE.md` records the invariants that cost the most to rediscover: the
container's `--cleanenv` boundary, exact GPU type names, shared query-mean
centring, and the Nextflow task-monitor stall. Read it before changing the
pipeline.

---

## Further documentation

- `hpl-nf/README.md`: the HPL Nextflow pipeline, its per-stage checks and its preflight
- `anorak-nf/README.md`: the ANORAK pipeline and its first-run setup
- `CLAUDE.md`: architecture, the Knowledge Bank schema, and hard-won invariants
- `PIPELINE_HARDENING_2026-08.md`, `CLASSIFIER_TUNING_2026-08-13.md`,
  `KB_TABLE_COVERAGE_2026-08-26.md`, `JS_FRONTEND_PORT_2026-08-24.md`,
  `HANDOFF_2026-08-26.md`: dated design notes
- `docs/HPL_Pipeline_Architecture.pdf`: an architecture overview

---

## Credits

- **HPL and the frozen encoder**: [K-Rakovic/HPL-LATTICeA](https://github.com/K-Rakovic/HPL-LATTICeA),
  included here as a git subtree with a small I/O patch
  (`backend/patches/hpl-encode-io.patch`).
- **ANORAK**: Pan et al., *The artificial intelligence-based model ANORAK
  improves histopathological grading of lung adenocarcinoma*, Nature Cancer
  **5**, 347–363 (2024). Model and reference implementation:
  [xi11/AIgrading](https://github.com/xi11/AIgrading), called unchanged.

## Licence

No licence has been chosen for this repository yet. Until one is added, the
code may be viewed but not reused. `HPL-LATTICeA/` is third-party code under
its upstream authors' terms.
