# hpl-nf

HPL Stages 1-4 — tiling, packaging, feature extraction and cluster
classification of H&E whole-slide images — as one Nextflow pipeline. It is what
the UI's **Run pipeline (Stages 1-4)** button submits, and it runs the way
[`anorak-nf`](../anorak-nf/README.md) does: one supervised head job on Slurm
that submits a job per slide, per shard and per step itself.

It does not reimplement any stage. Every task runs a wrapper in `bin/` that
calls the same tiler, packager and container command builders from `backend/`
that the per-stage Slurm submitters use, so the science, the bind rules and the
`--cleanenv` workarounds exist once.

Stages 5 and 6 — Knowledge Bank registration and load — are deliberately not
here. They write the shared Knowledge Bank, and the load never commits without
a dry run a person has looked at. The pipeline stops at a verified assignments
CSV and the UI takes it from there.

## What it does

```
manifest ─→ TILE ─→ TILING_GATE ─→ PACKAGE ─→ EXTRACT_PLAN ─→ EXTRACT_SHARD ─→ EXTRACT_FINISH
            (CPU,    (CPU, once)    (CPU)      (CPU)           (GPU, ×N)         (CPU, merge)
            per slide)
                                              ─→ ASSIGN_PLAN ─→ ASSIGN_SHARD ─→ ASSIGN_FINISH
                                                 (CPU, mean)    (CPU/GPU, ×N)   (CPU, merge)
```

| step | wrapper | backend code | writes |
|---|---|---|---|
| `TILE` | `hpl_tile.py tile` | `submit_mask_tile_slurm.tile_one_slide` | mask + tiles + `_tile_metadata.csv` per slide |
| `TILING_GATE` | `hpl_tile.py gate` | `tiling_output_complete(verify_row_count=True)` | `manifest.packaged.txt`, `stages/tiling.done.json` |
| `PACKAGE` | `hpl_package.py` | `make_hpl_hdf5.package_to_h5` | the packaged `.h5` |
| `EXTRACT_*` | `hpl_extract.py plan/shard/finish` | `_build_extraction_command`, `merge_projection_shards` | the projections `.h5` |
| `ASSIGN_*` | `hpl_assign.py plan/shard/finish` | `compute_query_mean`, `_build_assignment_command`, `merge_assignment_shards` | the assignments CSV |

Outputs land where the per-stage path puts them — `processed_tiles/`,
`model_input/`, the HPL repo's `results/` — not in Nextflow's `work/`, so
registration, the viewer and the KB load find them where they always have.

## Every stage is checked, not just run

- **A stage is done only when its output validates.** Each stage's last task
  runs the validator the server's own gate uses (`backend/stage_outputs.py`,
  `validate_extraction_output`), bound to the row count of the stage before —
  every tile CSV against its summary; the `.h5`'s datasets against each other;
  embeddings against the packaged tile count; assignments against the
  embeddings — and only then writes `stages/<stage>.done.json`. The server reads
  a stage as COMPLETED on that marker alone, and still re-validates the output
  before any later stage may read it. A head job that exits 0 without a marker
  reads FAILED.
- **Refusals exit 65** (`bin/hpl_common.REFUSAL_EXIT_CODE`): the run finishes,
  because a retry would read the same bytes. Everything else — OOM, preemption,
  an `.exitcode` CephFS was slow to show — is retried twice.
- **Every step is safe to run twice.** Each checks whether its own output
  already validates and keeps it; a shard retried after a crash clears the
  half-written part first, because the encoder treats any existing output as
  finished and crashes on it. A fresh run refuses to start over another run's
  complete outputs (`submit_hpl_nf.refuse_foreign_outputs`), which is what makes
  "it validates" mean "this run made it".
- **Shards share one query mean.** With more than one assignment shard,
  `ASSIGN_PLAN` computes the mean over every tile once, so each shard centres
  identically (CLAUDE.md, "--centering query").
- **An unreadable slide stops the run by default.** With *Package without
  slides that fail to tile* on — at submission or when resuming — the gate
  leaves those slides out of `manifest.packaged.txt` and names them on the
  tiling step instead.

## Running it

Normally from the UI. The server resolves and checks everything, writes the
run's `run_config.json`, and submits the head job; its directory is
`$HPL_NF_RESULTS_ROOT/<submission_id>/`. By hand, the same thing:

```bash
nextflow run hpl-nf -profile beatson \
    --config      <run dir>/run_config.json \
    --manifest    <run dir>/manifest.txt \
    --outdir      <run dir> \
    --python      /path/to/the/tile/server/python \
    --backend_dir /path/to/repo/backend \
    -resume
```

`-profile stub -stub` runs the whole DAG in seconds with no slides and no GPU;
`--stub_extract_shards N` / `--stub_assign_shards N` fan the shards out (0
takes the "already finished" path).

### Before the first run

```bash
python hpl-nf/tools/preflight.py --checkpoint /path/to/BarlowTwins_3.ckt
```

checks nextflow, the effective `-profile beatson` config, the watchdog, the
task interpreter's imports (openslide, h5py), the HPL repo, the NGC image and
its extras, the GPU type, the reference, walltimes against partitions, that a
compute node can run `sbatch`, the submit limit, and the results root. Exit 0
passed, 1 failed, 2 skipped something — 2 is not a pass. The container extras
are the same ones the per-stage path uses:
`python backend/submit_feature_extraction.py --bootstrap-extras`.

## Layout

```
main.nf              the DAG
nextflow.config      resources, retry policy, profiles, provenance reports
conf/beatson.config  partitions, GRES, executor timing for this cluster
bin/                 one wrapper per stage + hpl_common.py
tools/preflight.py   the checks above
tools/nf_supervise.sh  the stall watchdog — a link to anorak-nf's, one script
tools/test_*.py      the per-stage guards, and real Nextflow in stub mode
```

The head job, its chain of standbys and the watchdog are shared with ANORAK
(`backend/submit_anorak_nf.head_sbatch_command`, `tools/nf_supervise.sh`); why
the head job looks the way it does is in `backend/submit_anorak_nf.py` and
CLAUDE.md. **Copy `hpl-nf/` and `backend/` to the cluster together**: the
wrappers import from `backend/`, and refuse a `backend/` that predates them.

## Tests

```bash
python -m pytest hpl-nf/tools -q          # per-stage guards + stub runs
python hpl-nf/tools/test_workflow_steps.py   # standalone, as on the cluster
```
